import sys
import os
import time
import json
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

# 設定日誌 (只顯示 ERROR 以上)
logging.basicConfig(level=logging.ERROR)

# ==========================================
# 配置參數
# ==========================================
CONTROL_LEAF = "l1"
TARGET_LEAF = "l2"
TARGET_IP = "10.0.2.2"
SRC_ADD = 1
PORTS = [2, 3, 4, 5]
CAPACITY = {2: 0.64, 3: 0.80, 4: 0.96, 5: 1.12}
# Stage 2 — control thresholds
LATENCY_THRESHOLD_MS = 200.0
JITTER_THRESHOLD_MS  = 30.0
LOSS_THRESHOLD_PCT   = 2.0
COOLDOWN_SEC         = 4
WEIGHT_MIN           = 1
WEIGHT_MAX           = 8
UTIL_THRESHOLD = 0.6

MODELS = {
    "latency": "rf_model_latency_simplified.pkl",
    "loss": "rf_model_loss_simplified.pkl",
    "jitter": "rf_model_jitter_simplified.pkl",
    "anomaly": "rf_model_anomaly_simplified.pkl"
}

FEATURE_NAMES = [
    "Total_Util_Sum",
    "Max_Util_Diff",
    "Group_Imbalance",
    "Norm_Load_P2", "Norm_Load_P3", "Norm_Load_P4", "Norm_Load_P5",
    "idx_load_balance",
    "mbps_imbalance",
    "max_qdepth_p99",
    "total_qdepth_p99",
    "total_qdepth_max",
    "qdepth_max_imbalance",
    "qdepth_fft_max_all",
    "Weight_Port2", "Weight_Port3", "Weight_Port4", "Weight_Port5",
    "src1_port3_mbps_cv", "src1_port5_mbps_cv", "src1_port4_mbps_cv", "src1_port2_mbps_cv",
    "src1_port5_load_util", "src1_port3_load_util", "src1_port4_load_util", "src1_port2_load_util",
    "src1_port3_qdepth_max", "src1_port5_qdepth_max", "src1_port4_qdepth_max", "src1_port2_qdepth_max",
    "qdepth_sq", "qdepth_slope"
]

class RankECDF:
    def __init__(self): self._sorted = None
    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if self._sorted is None or len(self._sorted) == 0: return np.zeros_like(x)
        return np.searchsorted(self._sorted, x, side="right") / len(self._sorted)

class MLController:
    def __init__(self):
        self.topo = load_topo("topology.json")
        self.api_telemetry = SimpleSwitchThriftAPI(self.topo.get_thrift_port(TARGET_LEAF))
        
        # 載入模型與 ECDF
        self.models = {k: joblib.load(v) for k, v in MODELS.items()}
        sys.modules['__main__'].RankECDF = RankECDF
        self.ecdf_objs = joblib.load("ecdf_objects.pkl")
        
        self.prev_bytes = {p: 0 for p in PORTS}

        self.telemetry_buffer = collections.deque(maxlen=100)  # 10s @ 10Hz

        # 預測值平滑緩存
        self.smoothed_latency = 20.0
        self.smoothed_loss = 0.0
        self.smoothed_jitter = 0.0

        self._prev_qdepth_p99 = 0.0

        self.init_baseline()

        # Stage 2 — control state
        self._api_control  = SimpleSwitchThriftAPI(self.topo.get_thrift_port(CONTROL_LEAF))
        self._hw_rules     = self._load_hw_rules()
        self._grp_handle   = None
        self._mbr_handles  = []
        self._last_adj_time = 0.0
        self._last_weights  = []

    def init_baseline(self):
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            for p in PORTS:
                try: self.prev_bytes[p] = self.api_telemetry.counter_read('port_bytes_counter', p)[0]
                except: pass

    def _load_hw_rules(self):
        with open('p4app.json') as f:
            p4app = json.load(f)
        analyzer = TopologyAnalyzer(p4app, self.topo)
        _, rules = analyzer.get_ecmp_weights_and_rules(CONTROL_LEAF, TARGET_LEAF)
        return rules  # 4 dicts: comp_id, num_nhops, ports_and_macs

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

            # 1. Always clear the forwarding table first (removes any entry from
            #    all_controller.py on first call, or from a prior apply thereafter)
            subprocess.run(
                ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
                input="table_clear w_ecmp_table\n",
                text=True, capture_output=True
            )
            if self._grp_handle is not None:
                try:
                    for m in self._mbr_handles:
                        self._api_control.act_prof_remove_member_from_group(sel, m, self._grp_handle)
                        self._api_control.act_prof_delete_member(sel, m)
                    self._api_control.act_prof_delete_group(sel, self._grp_handle)
                except Exception:
                    pass
                self._grp_handle  = None
                self._mbr_handles = []

            # 2. Build new group
            grp = self._api_control.act_prof_create_group(sel)
            mbrs = []
            for idx, rule in enumerate(self._hw_rules):
                comp_id = str(rule['comp_id'])
                for _ in range(weights_list[idx]):
                    m = self._api_control.act_prof_create_member(sel, act, [comp_id])
                    self._api_control.act_prof_add_member_to_group(sel, m, grp)
                    mbrs.append(m)
            self._grp_handle  = grp
            self._mbr_handles = mbrs

            # 3. Point forwarding table at the new group
            subprocess.run(
                ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
                input=f"table_indirect_add_with_group w_ecmp_table {TARGET_IP} => {grp}\n",
                text=True, capture_output=True
            )

    def control_step(self, feats, preds):
        if time.time() - self._last_adj_time < COOLDOWN_SEC:
            return
        max_util = max(feats.get(f'src1_port{p}_load_util', 0) for p in PORTS)
        reactive = max_util > UTIL_THRESHOLD or feats.get('qdepth_max_imbalance', 0) > 15
        trigger = (
            preds['anomaly'] == 1
            or reactive
            or self.smoothed_latency > LATENCY_THRESHOLD_MS
            or self.smoothed_jitter  > JITTER_THRESHOLD_MS
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
        port_weights = {rule['ports_and_macs'][0][0]: w
                        for rule, w in zip(self._hw_rules, new_weights)}
        print(f"\n[CTRL] Weight change → " +
              "  ".join(f"s{p-1}:{w}" for p, w in sorted(port_weights.items())))

    def collect_window(self, duration=1.0, interval=0.1):
        """採集新樣本並更新滑動視窗"""
        start_t = time.time()
        prev_sample_t = start_t
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            while time.time() - start_t < duration:
                time.sleep(interval)
                now = time.time()
                dt = now - prev_sample_t
                row = {}
                for p in PORTS:
                    reg_idx = SRC_ADD * 16 + p
                    q = self.api_telemetry.register_read('path_max_queue_depth_reg', reg_idx)
                    self.api_telemetry.register_write('path_max_queue_depth_reg', reg_idx, 0)
                    row[f'qdepth_{p}'] = min(64, q)
                    cnt = self.api_telemetry.counter_read('port_bytes_counter', p)[0]
                    db = cnt - self.prev_bytes[p]
                    if db < 0: db = 0
                    
                    # 計算原始 Mbps
                    raw_mbps = ((db * 8) / (dt * 1_000_000)) if dt > 0 else 0
                    
                    # --- 爆發過濾器 (Burst Filter) ---
                    # 如果計算出的 Mbps 超過物理頻寬的 2 倍，判定為工具初始化雜訊或緩衝區排空
                    # 將其限幅在 2.0x Capacity，既保留了「擁塞」的特徵，又去除了「天文數字」的噪音
                    limit = CAPACITY[p] * 2.0
                    row[f'mbps_{p}'] = min(limit, raw_mbps)
                    
                    self.prev_bytes[p] = cnt
                self.telemetry_buffer.append(row)
                prev_sample_t = now
        return pd.DataFrame(list(self.telemetry_buffer))

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

    def extract_features(self, df):
        feats = {}
        current_weights = self.get_current_weights()
        for p in PORTS:
            feats[f'Weight_Port{p}'] = current_weights.get(p, 1)

        q_p99s, m_means, m_cvs, q_maxes, utils = [], [], [], [], []
        for p in PORTS:
            q, m = df[f'qdepth_{p}'].values, df[f'mbps_{p}'].values
            p99, m_mean, m_std = np.percentile(q, 99), np.mean(m), np.std(m)
            m_cv = m_std / m_mean if m_mean > 0.001 else 0
            q_max = np.max(q)
            util = m_mean / CAPACITY[p]
            
            feats[f'src1_port{p}_qdepth_max'] = q_max
            feats[f'src1_port{p}_mbps_cv'] = m_cv
            feats[f'src1_port{p}_load_util'] = util
            feats[f'Norm_Load_P{p}'] = m_mean / max(1, feats[f'Weight_Port{p}'])
            
            q_p99s.append(p99); m_means.append(m_mean); m_cvs.append(m_cv); q_maxes.append(q_max); utils.append(util)

        feats["Total_Util_Sum"] = sum(utils)
        feats["Max_Util_Diff"] = max(utils) - min(utils)

        load_a = m_means[0] + m_means[1]
        weight_a = feats['Weight_Port2'] + feats['Weight_Port3']
        load_b = m_means[2] + m_means[3]
        weight_b = feats['Weight_Port4'] + feats['Weight_Port5']
        feats["Group_Imbalance"] = np.abs((load_a / max(1, weight_a)) - (load_b / max(1, weight_b)))

        feats["mbps_imbalance"] = np.std(m_means)
        feats["max_qdepth_p99"] = np.max(q_p99s)
        feats["total_qdepth_p99"] = np.sum(q_p99s)
        feats["total_qdepth_max"] = np.sum(q_maxes)
        feats["qdepth_max_imbalance"] = np.max(q_maxes) - np.min(q_maxes)
        feats["qdepth_sq"] = feats["total_qdepth_p99"] ** 2
        feats["qdepth_slope"] = feats["total_qdepth_p99"] - self._prev_qdepth_p99
        self._prev_qdepth_p99 = feats["total_qdepth_p99"]
        
        fft_mags = []
        for p in PORTS:
            q = df[f'qdepth_{p}'].values
            fft_mags.append(np.max(np.abs(np.fft.rfft(q - np.mean(q)))) if len(q) > 2 else 0)
        feats["qdepth_fft_max_all"] = np.max(fft_mags)

        # ECDF transformations
        utils_ecdf = []
        for p in PORTS:
            col, val = f"src1_port{p}_load_util", feats[f'src1_port{p}_load_util']
            utils_ecdf.append(self.ecdf_objs[col].transform([val])[0] if col in self.ecdf_objs else 0.5)
        mb_ecdf = self.ecdf_objs["mbps_imbalance"].transform([feats["mbps_imbalance"]])[0] if "mbps_imbalance" in self.ecdf_objs else 0.5
        idx_lb = 1.0
        for u in utils_ecdf: idx_lb *= (1.1 * u)
        idx_lb *= (1.1 * mb_ecdf)
        feats["idx_load_balance"] = idx_lb ** 2

        vector = [feats.get(name, 0) for name in FEATURE_NAMES]
        return pd.DataFrame([vector], columns=FEATURE_NAMES), feats

    def run(self):
        print("\n" + "="*125)
        print(" [ML 智能監控 v4.0] 啟動 - 滑動視窗模式 (10s 尺度對齊)")
        print("="*125 + "\n")
        
        # 預熱
        print(" [系統] 正在預熱資料緩存 (10s)...", end='', flush=True)
        for _ in range(10):
            self.collect_window(duration=1.0)
            print(".", end='', flush=True)
        print(" 完成！\n")

        try:
            while True:
                df = self.collect_window(duration=1.0)
                X, feats = self.extract_features(df)
                preds = {}
                for k, m in self.models.items():
                    p = m.predict(X)[0]
                    if k != "anomaly": p = np.expm1(p)
                    preds[k] = p
                
                self.smoothed_latency = (0.6 * preds['latency']) + (0.4 * self.smoothed_latency)
                self.smoothed_jitter = (0.6 * preds['jitter']) + (0.4 * self.smoothed_jitter)
                self.smoothed_loss = (0.6 * preds['loss']) + (0.4 * self.smoothed_loss)
                
                status = "NORMAL" if preds['anomaly'] == 0 else "\033[91mANOMALY\033[0m"

                sys.stdout.write("\033[K")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status:7} | "
                      f"Lat: {self.smoothed_latency:6.1f}ms | "
                      f"Jit: {self.smoothed_jitter:5.1f}ms | "
                      f"Loss: {self.smoothed_loss:4.1f}% | "
                      f"Util: {feats['Total_Util_Sum']:4.2f}", end='\r')
                sys.stdout.flush()
                self.control_step(feats, preds)
        except KeyboardInterrupt: print("\n停止。")

if __name__ == "__main__":
    ctrl = MLController()
    ctrl.run()
