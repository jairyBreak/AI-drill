import sys
import os
import time
import json
import math
import pandas as pd
import numpy as np
import joblib
import logging
import subprocess
import warnings
import collections
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

# 隱藏所有警告
warnings.filterwarnings("ignore")

# 載入 P4-Utils
P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI
from all_controller import TopologyAnalyzer
from topo_independent_helper import transform_to_topo_independent

# 設定日誌 (只顯示 ERROR 以上)
logging.basicConfig(level=logging.ERROR)

# ==========================================
# 配置參數
# ==========================================
CONTROL_LEAF = "l1"
TARGET_LEAF = "l2"
TARGET_IP = "10.0.2.2"
SRC_ADD = 1
PORTS = list(range(2, 10))
CAPACITY = {2: 0.48, 3: 0.48, 4: 0.64, 5: 0.64,
            6: 0.80, 7: 0.80, 8: 0.96, 9: 0.96}

LATENCY_THRESHOLD_MS = 200.0
LOSS_THRESHOLD_PCT   = 2.0
COOLDOWN_SEC         = 4
WEIGHT_MIN           = 1
WEIGHT_MAX           = 8
UTIL_THRESHOLD       = 0.6
MIN_UTIL_TO_REBALANCE = 0.1

MODELS = {
    "latency": "rf_model_latency_1s.pkl",
    "loss":    "rf_model_loss_1s.pkl",
    "anomaly": "rf_model_anomaly_1s.pkl"
}

FEATURE_NAMES = [
    "Is_Rehash_Event", "Time_Since_Last_Rehash_s", "Rehash_Impact",
    "Total_Util_Sum", "Max_Util_Diff", "Group_Imbalance",
    "Max_QDepth", "Total_QDepth", "QDepth_Imbalance",
    "Over_Capacity_Sum", "Max_Q_Ratio", "Q_Danger_Flag", "Q_Danger_Count",
    "Total_QDepth_Trend",
    "Total_Actual_Mbps", "Expected_Over_Capacity_Sum",
    "Overflow_Intensity", "Queue_Full_And_Over_Cap"
]
for _k in range(3):
    FEATURE_NAMES.extend([
        f"top{_k+1}_qdepth",
        f"top{_k+1}_mbps", f"top{_k+1}_weight",
        f"top{_k+1}_norm_load", f"top{_k+1}_expected_util",
        f"top{_k+1}_qdepth_trend", f"top{_k+1}_mbps_trend"
    ])


class MLController:
    def __init__(self):
        self.topo = load_topo("topology.json")
        self.api_telemetry = SimpleSwitchThriftAPI(self.topo.get_thrift_port(TARGET_LEAF))

        # 載入模型 (安全載入，缺少模型時發出警告)
        self.models = {}
        for k, v in MODELS.items():
            if os.path.exists(v):
                self.models[k] = joblib.load(v)
                print(f"   - 載入模型 {v} 成功")
            else:
                print(f"   - 警告: 找不到模型檔案 {v}")

        self.prev_bytes = {p: 0 for p in PORTS}

        # 預測值平滑緩存
        self.smoothed_latency = 20.0
        self.smoothed_loss    = 0.0

        # 1s 採集器狀態 (對應 realtime_1s_predictor_topo_indep)
        self.raw_history      = collections.deque(maxlen=100)
        self.start_time       = time.time()
        self.last_rehash_time = time.time()
        self.is_rehash_event  = 0
        self.prev_weights_1s  = {p: 1 for p in PORTS}
        self.prev_l1_enq      = {p: 0 for p in PORTS}
        self.prev_l2_ingress  = {p: 0 for p in PORTS}

        self.init_baseline()

        self._api_control   = SimpleSwitchThriftAPI(self.topo.get_thrift_port(CONTROL_LEAF))
        self._hw_rules      = self._load_hw_rules()
        self._grp_handle    = None
        self._mbr_handles   = []
        self._last_adj_time = 0.0
        self._last_weights  = []

        # 初始化 l1 enqueue 計數器基準 (在 _api_control 建立後)
        with open(os.devnull, 'w') as _f, redirect_stdout(_f), redirect_stderr(_f):
            for p in PORTS:
                try:
                    self.prev_l1_enq[p] = self._api_control.counter_read('cnt_enq', p)[0]
                except: pass

    def init_baseline(self):
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            for p in PORTS:
                try:
                    self.prev_bytes[p]      = self.api_telemetry.counter_read('port_bytes_counter', p)[0]
                    self.prev_l2_ingress[p] = self.api_telemetry.counter_read('cnt_ingress', p)[0]
                except: pass

    def _load_hw_rules(self):
        with open('p4app.json') as f:
            p4app = json.load(f)
        analyzer = TopologyAnalyzer(p4app, self.topo)
        _, rules = analyzer.get_ecmp_weights_and_rules(CONTROL_LEAF, TARGET_LEAF)
        return rules

    def compute_weights(self, feats):
        scores = []
        for rule in self._hw_rules:
            port = rule['ports_and_macs'][0][0]
            util   = feats.get(f'src1_port{port}_load_util', 0.1)
            qdepth = feats.get(f'src1_port{port}_qdepth_max', 0)
            free_bw    = CAPACITY[port] * max(0.0, 1.0 - util)
            q_headroom = max(0.0, 64 - qdepth) / 64.0
            scores.append(free_bw * q_headroom + 0.01)
        min_s = min(scores)
        return [max(WEIGHT_MIN, min(WEIGHT_MAX, round(s / min_s))) for s in scores]

    def apply_weights(self, weights_list):
        sel = "w_ecmp_selector"
        act = "assign_component"
        thrift_port = self.topo.get_thrift_port(CONTROL_LEAF)

        with open(os.devnull, 'w') as devnull, \
             redirect_stdout(devnull), redirect_stderr(devnull):

            if self._grp_handle is not None:
                # Subsequent calls: rewire members inside the existing group so the
                # table entry never changes and other l1 routes are undisturbed.
                try:
                    for m in self._mbr_handles:
                        self._api_control.act_prof_remove_member_from_group(sel, m, self._grp_handle)
                        self._api_control.act_prof_delete_member(sel, m)
                except Exception:
                    pass
                self._mbr_handles = []
                for idx, rule in enumerate(self._hw_rules):
                    comp_id = str(rule['comp_id'])
                    for _ in range(weights_list[idx]):
                        m = self._api_control.act_prof_create_member(sel, act, [comp_id])
                        self._api_control.act_prof_add_member_to_group(sel, m, self._grp_handle)
                        self._mbr_handles.append(m)
            else:
                # First call: replace all_controller.py's entry for TARGET_IP only.
                # table_clear is unavoidable here, but runs once and only once.
                subprocess.run(
                    ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
                    input="table_clear w_ecmp_table\n",
                    text=True, capture_output=True
                )
                grp = self._api_control.act_prof_create_group(sel)
                self._mbr_handles = []
                for idx, rule in enumerate(self._hw_rules):
                    comp_id = str(rule['comp_id'])
                    for _ in range(weights_list[idx]):
                        m = self._api_control.act_prof_create_member(sel, act, [comp_id])
                        self._api_control.act_prof_add_member_to_group(sel, m, grp)
                        self._mbr_handles.append(m)
                self._grp_handle = grp
                subprocess.run(
                    ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
                    input=f"table_indirect_add_with_group w_ecmp_table {TARGET_IP} => {grp}\n",
                    text=True, capture_output=True
                )

    def control_step(self, feats, preds):
        if time.time() - self._last_adj_time < COOLDOWN_SEC:
            return
        max_util = max(feats.get(f'src1_port{p}_load_util', 0) for p in PORTS)
        if max_util < MIN_UTIL_TO_REBALANCE:
            return
        reactive = max_util > UTIL_THRESHOLD or feats.get('qdepth_max_imbalance', 0) > 15
        trigger = (
            preds.get('anomaly', 0) == 1
            or reactive
            or self.smoothed_latency > LATENCY_THRESHOLD_MS
            or self.smoothed_loss    > LOSS_THRESHOLD_PCT
        )
        if not trigger:
            return
        new_weights = self.compute_weights(feats)
        if new_weights == self._last_weights:
            return
        try:
            self.apply_weights(new_weights)
        except Exception:
            self._api_control = SimpleSwitchThriftAPI(self.topo.get_thrift_port(CONTROL_LEAF))
            self._grp_handle  = None
            self._mbr_handles = []
            return
        self._last_adj_time = time.time()
        self._last_weights  = new_weights
        port_weights = {}
        for rule, w in zip(self._hw_rules, new_weights):
            for port, _ in rule['ports_and_macs']:
                port_weights[port] = w
        print(f"\n[CTRL] Weight change → " +
              "  ".join(f"s{p-1}:{w}" for p, w in sorted(port_weights.items())),
              flush=True)

    def get_current_weights(self):
        weights = {p: 1 for p in PORTS}
        try:
            entries = self._api_control.table_get_entries("w_ecmp_table", False)
            if not entries: return weights
            grp_handle = entries[0].action_data.action_params[0]
            grp_info = self._api_control.act_prof_get_group("w_ecmp_selector", grp_handle)
            members = grp_info.member_handles
            comp_counts = {}
            for m_handle in members:
                mbr = self._api_control.act_prof_get_member("w_ecmp_selector", m_handle)
                comp_id = int(mbr.action_params[0])
                comp_counts[comp_id] = comp_counts.get(comp_id, 0) + 1
            nh_entries = self._api_control.table_get_entries("ecmp_group_to_nhop", False)
            for entry in nh_entries:
                c_id = int(entry.match_key[0].data)
                port = int(entry.action_data.action_params[1])
                if c_id in comp_counts and port in weights:
                    weights[port] = comp_counts[c_id]
        except: pass
        return weights

    def collect_1s_data(self):
        """採集 1 秒的數據點與硬體真實數據"""
        start_t = time.time()
        time.sleep(1.0)
        dt = time.time() - start_t

        current_time    = time.time()
        row             = {}
        qdepths         = []
        max_hw_latency  = 0
        total_delta_enq = 0
        total_drops     = 0

        with open(os.devnull, 'w') as _dn, redirect_stdout(_dn), redirect_stderr(_dn):
            weights = self.get_current_weights()
            if weights != self.prev_weights_1s:
                self.is_rehash_event  = 1
                self.last_rehash_time = current_time
                self.prev_weights_1s  = weights

            total_weight = sum(weights.values())
            if total_weight == 0: total_weight = 1

            for p in PORTS:
                reg_idx         = SRC_ADD * 16 + p
                q               = self.api_telemetry.register_read('path_max_queue_depth_reg', reg_idx)
                raw_acc_q_delay = self.api_telemetry.register_read('path_acc_q_delay_reg', reg_idx)

                self.api_telemetry.register_write('path_max_queue_depth_reg', reg_idx, 0)
                self.api_telemetry.register_write('path_acc_q_delay_reg', reg_idx, 0)

                cnt  = self.api_telemetry.counter_read('port_bytes_counter', p)[0]
                db   = cnt - self.prev_bytes[p]
                mbps = (max(0, db) * 8) / (dt * 1_000_000)
                self.prev_bytes[p] = cnt

                row[f"src1_port{p}_qdepth"] = q
                row[f"src1_port{p}_mbps"]   = mbps
                row[f"Weight_Port{p}"]      = weights[p] / total_weight

                qdepths.append(q)

                if raw_acc_q_delay > max_hw_latency:
                    max_hw_latency = raw_acc_q_delay

                try:
                    l1_enq_pkts     = self._api_control.counter_read('cnt_enq', p)[0]
                    l2_ingress_pkts = self.api_telemetry.counter_read('cnt_ingress', p)[0]

                    delta_enq     = l1_enq_pkts - self.prev_l1_enq[p]
                    delta_ingress = l2_ingress_pkts - self.prev_l2_ingress[p]

                    self.prev_l1_enq[p]     = l1_enq_pkts
                    self.prev_l2_ingress[p] = l2_ingress_pkts

                    if delta_enq > 0:
                        drops            = max(0, delta_enq - delta_ingress)
                        total_delta_enq += delta_enq
                        total_drops     += drops
                except: pass

        row['Real_HW_Latency_ms']         = max_hw_latency / 1000.0
        row['Real_HW_Loss_Percent']       = (total_drops / total_delta_enq * 100) if total_delta_enq > 0 else 0.0
        row["Time_Since_Traffic_Start_s"] = current_time - self.start_time
        row["Is_Rehash_Event"]            = self.is_rehash_event
        self.is_rehash_event              = 0
        row["Time_Since_Last_Rehash_s"]   = current_time - self.last_rehash_time
        row["Rehash_Impact"]              = math.exp(-row["Time_Since_Last_Rehash_s"])

        return row

    def run(self):
        print("\n" + "="*125)
        print(" [ML 智能監控 v4.3] 啟動 - 1s 拓樸無關採集模式")
        print("="*125 + "\n")

        try:
            while True:
                data_row = self.collect_1s_data()
                self.raw_history.append(data_row)

                df_history     = pd.DataFrame(list(self.raw_history))
                df_transformed = transform_to_topo_independent(df_history, PORTS, CAPACITY, K=3)
                last_row       = df_transformed.iloc[-1]
                X              = pd.DataFrame([last_row[FEATURE_NAMES]])

                preds = {}
                for k, m in self.models.items():
                    p = m.predict(X)[0]
                    if k == "latency": p = np.expm1(p)
                    preds[k] = p

                self.smoothed_latency = preds.get('latency', self.smoothed_latency)
                self.smoothed_loss    = preds.get('loss',    self.smoothed_loss)

                # feats dict for control_step / compute_weights
                feats = {}
                for p in PORTS:
                    feats[f'src1_port{p}_load_util']  = data_row[f'src1_port{p}_mbps'] / CAPACITY[p]
                    feats[f'src1_port{p}_qdepth_max'] = data_row[f'src1_port{p}_qdepth']
                feats['qdepth_max_imbalance'] = last_row['QDepth_Imbalance']
                feats['Total_Util_Sum']       = last_row['Total_Util_Sum']

                status  = "NORMAL" if preds.get('anomaly', 0) == 0 else "\033[91mANOMALY\033[0m"
                hw_lat  = data_row['Real_HW_Latency_ms']
                hw_loss = data_row['Real_HW_Loss_Percent']

                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {status:7} | "
                      f"Lat: {self.smoothed_latency:5.1f}/{hw_lat:5.1f}ms | "
                      f"Loss: {self.smoothed_loss:4.1f}/{hw_loss:4.1f}% | "
                      f"Util: {feats['Total_Util_Sum']:4.2f}",
                      end='', flush=True)
                self.control_step(feats, preds)
        except KeyboardInterrupt: print("\n停止。")


if __name__ == "__main__":
    ctrl = MLController()
    ctrl.run()
