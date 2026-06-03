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
from all_controller import TopologyAnalyzer, install_ecmp_drill_rules
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
COOLDOWN_SEC         = 6
WEIGHT_MIN           = 1
WEIGHT_MAX           = 8
WEIGHT_SMOOTHING     = 0.35   # 權重 EMA 平滑係數 (0=凍結, 1=即時)；越小越穩、擺盪越少
WEIGHT_AVG           = 4      # 等分時每個 component 的基準權重 (4 components -> 平均 4)
UTIL_THRESHOLD       = 0.6
MIN_UTIL_TO_REBALANCE = 0.1
BALANCE_UTIL_TOLERANCE = 0.15  # 各 component 利用率差距 < 此值 -> 視為已平衡，不動權重

# 結果 CSV (與 baseline / plot_1s_metrics 相同欄位，供 plot_result.py 三方比較)
OUTPUT_CSV = "research_results/data/validation/comparison_ml.csv"

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

        # 啟動時重裝 4-component W-ECMP+DRILL 轉發 (覆蓋任何 baseline 殘留設定)。
        # 否則若上一個跑的是 baseline_ecmp/wecmp (8 個單埠 component)，本控制器發出的
        # comp_id 1-4 會對應到錯誤的單埠映射，導致流量只走 2-3 個 spine。
        print("[init] 重新安裝 4-component W-ECMP+DRILL 轉發 (覆蓋 baseline 設定)...")
        with open('p4app.json') as _f:
            _p4app = json.load(_f)
        with open(os.devnull, 'w') as _dn, redirect_stdout(_dn), redirect_stderr(_dn):
            install_ecmp_drill_rules(_p4app, self.topo, clear_first=True, verbose=False)
        print("[init] 轉發規則安裝完成")

        self.api_telemetry = SimpleSwitchThriftAPI(self.topo.get_thrift_port(TARGET_LEAF))

        # 載入模型 (安全載入，缺少模型時發出警告)
        self.models = {}
        for k, v in MODELS.items():
            if os.path.exists(v):
                mdl = joblib.load(v)
                # 單列推論用不到平行運算；關掉 joblib worker 可消除 sklearn 平行警告並略為加速
                try:
                    mdl.set_params(n_jobs=1)
                except Exception:
                    try: mdl.n_jobs = 1
                    except Exception: pass
                self.models[k] = mdl
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
        self._smoothed_weights = None   # 連續權重 EMA 狀態，避免硬切換造成擺盪

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
        self._prev_sample_t = time.time()   # 上次取樣時間，用於計算真實取樣間隔
        self._cum_enq  = 0                   # 自開始累積的 l1 enqueue 封包數
        self._cum_recv = 0                   # 自開始累積的 l2 ingress 封包數 (用於端到端丟包率)

    def _load_hw_rules(self):
        with open('p4app.json') as f:
            p4app = json.load(f)
        analyzer = TopologyAnalyzer(p4app, self.topo)
        _, rules = analyzer.get_ecmp_weights_and_rules(CONTROL_LEAF, TARGET_LEAF)
        return rules

    def _component_utils(self, feats):
        """每個 component 的容量加權平均利用率 (load/cap)。"""
        cu = []
        for rule in self._hw_rules:
            ports = [p for p, _ in rule['ports_and_macs']]
            cap   = sum(CAPACITY[p] for p in ports)
            load  = sum(feats.get(f'src1_port{p}_load_util', 0.0) * CAPACITY[p] for p in ports)
            cu.append(load / cap if cap > 0 else 0.0)
        return cu

    def is_balanced(self, feats):
        """各 component 利用率差距夠小 -> 已平衡 (重配權重幫不上忙)。"""
        cu = self._component_utils(feats)
        return (max(cu) - min(cu)) < BALANCE_UTIL_TOLERANCE if cu else True

    def compute_weights(self, feats):
        # 1. 每個 component 算一個「可用度」分數 (跨該 component 的所有埠取最差值)
        raw = []
        for rule in self._hw_rules:
            ports  = [p for p, _ in rule['ports_and_macs']]
            util   = max(feats.get(f'src1_port{p}_load_util', 0.1) for p in ports)
            qdepth = max(feats.get(f'src1_port{p}_qdepth_max', 0)  for p in ports)
            cap        = sum(CAPACITY[p] for p in ports)
            free_bw    = cap * max(0.0, 1.0 - util)
            q_headroom = max(0.0, 64 - qdepth) / 64.0
            raw.append(free_bw * q_headroom + 0.05)

        # 2. 比例正規化 (除以總和，而非最小值) -> 目標連續權重，平均落在 WEIGHT_AVG
        n      = len(raw)
        total  = sum(raw) or 1.0
        target = [(r / total) * n * WEIGHT_AVG for r in raw]

        # 3. EMA 平滑：在連續權重上做指數平滑，逐步逼近目標，避免在 1/8 之間硬擺
        if self._smoothed_weights is None or len(self._smoothed_weights) != n:
            self._smoothed_weights = list(target)
        else:
            a = WEIGHT_SMOOTHING
            self._smoothed_weights = [a * t + (1 - a) * s
                                      for t, s in zip(target, self._smoothed_weights)]

        # 4. 四捨五入並夾在 [WEIGHT_MIN, WEIGHT_MAX]
        return [max(WEIGHT_MIN, min(WEIGHT_MAX, round(w))) for w in self._smoothed_weights]

    def apply_weights(self, weights_list):
        sel = "w_ecmp_selector"
        act = "assign_component"
        thrift_port = self.topo.get_thrift_port(CONTROL_LEAF)

        with open(os.devnull, 'w') as devnull, \
             redirect_stdout(devnull), redirect_stderr(devnull):

            # 1. 建立一個全新的群組並填入新權重對應的成員，避免動態修改正在使用的群組成員導致 BMv2 當機
            new_grp = self._api_control.act_prof_create_group(sel)
            new_mbr_handles = []
            for idx, rule in enumerate(self._hw_rules):
                comp_id = str(rule['comp_id'])
                for _ in range(weights_list[idx]):
                    m = self._api_control.act_prof_create_member(sel, act, [comp_id])
                    self._api_control.act_prof_add_member_to_group(sel, m, new_grp)
                    new_mbr_handles.append(m)

            # 2. 獲取現有轉發表中 TARGET_IP 對應的 Entry Handle
            try:
                self._api_control.load_table_entries_match_to_handle()
                entry_handle = self._api_control.get_handle_from_match("w_ecmp_table", [TARGET_IP])
            except Exception:
                entry_handle = None

            # 3. 如果 entry_handle 存在，原子性地修改表項指向新群組
            if entry_handle is not None:
                try:
                    self._api_control.client.bm_mt_indirect_ws_modify_entry(0, "MyIngress.w_ecmp_table", entry_handle, new_grp)
                except Exception:
                    # 如果 Thrift RPC 修改失敗，退回到刪除並重新添加的方案
                    try:
                        self._api_control.table_delete("MyIngress.w_ecmp_table", entry_handle)
                    except Exception:
                        pass
                    entry_handle = None

            # 4. 如果 entry_handle 不存在（首次執行或之前被刪除），則直接添加表項
            if entry_handle is None:
                subprocess.run(
                    ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
                    input=f"table_indirect_add_with_group w_ecmp_table {TARGET_IP} => {new_grp}\n",
                    text=True, capture_output=True
                )

            # 5. 安全地刪除舊群組與舊成員（因為它們現在已經沒有被任何表項引用了，這在 BMv2 中是完全安全的）
            if self._grp_handle is not None:
                try:
                    self._api_control.act_prof_delete_group(sel, self._grp_handle)
                    for m in self._mbr_handles:
                        self._api_control.act_prof_delete_member(sel, m)
                except Exception:
                    pass

            # 6. 更新實體變數參考
            self._grp_handle = new_grp
            self._mbr_handles = new_mbr_handles

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

        # 觸發了 (偵測到問題)：先評估目前是否已平衡，再決定是否真的動權重
        new_weights = self.compute_weights(feats)
        if self.is_balanced(feats) or new_weights == self._last_weights:
            # 已平衡 (或目標與現況相同) -> 不動權重，套用冷卻避免每秒重印
            self._last_adj_time = time.time()
            print("\n[CTRL] weight not changed", flush=True)
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
            import socket
            entries = self._api_control.client.bm_mt_get_entries(0, "MyIngress.w_ecmp_table")
            if not entries:
                return weights
            
            target_ip_bytes = socket.inet_aton(TARGET_IP)
            target_entry = None
            for entry in entries:
                if entry.match_key and entry.match_key[0].exact and entry.match_key[0].exact.key == target_ip_bytes:
                    target_entry = entry
                    break
            
            if target_entry is not None:
                grp_handle = target_entry.action_entry.grp_handle
                if grp_handle > 0:
                    grp_info = self._api_control.client.bm_mt_act_prof_get_group(0, "MyIngress.w_ecmp_selector", grp_handle)
                    members = grp_info.mbr_handles
                    
                    comp_counts = {}
                    for m_handle in members:
                        mbr = self._api_control.client.bm_mt_act_prof_get_member(0, "MyIngress.w_ecmp_selector", m_handle)
                        if mbr.action_data:
                            comp_id = int(mbr.action_data[0].hex(), 16)
                            comp_counts[comp_id] = comp_counts.get(comp_id, 0) + 1
                    
                    nh_entries = self._api_control.client.bm_mt_get_entries(0, "MyIngress.ecmp_group_to_nhop")
                    for entry in nh_entries:
                        if entry.match_key and entry.match_key[0].exact:
                            c_id = int(entry.match_key[0].exact.key.hex(), 16)
                            if entry.action_entry and entry.action_entry.action_data:
                                port = int(entry.action_entry.action_data[1].hex(), 16)
                                if c_id in comp_counts and port in weights:
                                    weights[port] = comp_counts[c_id]
        except Exception as e:
            pass
        return weights

    def collect_1s_data(self):
        """採集 1 秒的數據點與硬體真實數據"""
        time.sleep(1.0)
        now_t        = time.time()
        # dt = 與上次取樣的真實間隔 (含處理時間)；不可只用 sleep(1.0)，否則處理較慢時吞吐量會被高估
        dt           = now_t - self._prev_sample_t
        self._prev_sample_t = now_t

        current_time    = now_t
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

                    # 瞬時 (每秒) 估計 — 受佇列堆積/延遲影響會偏高，僅供時間序列參考
                    if delta_enq > 0:
                        drops            = max(0, delta_enq - delta_ingress)
                        total_delta_enq += delta_enq
                        total_drops     += drops

                    # 累積總量 (不夾值，全埠) — 用於端到端正確丟包率，堆積/排空會自然抵消
                    self._cum_enq  += delta_enq
                    self._cum_recv += delta_ingress
                except: pass

        row['Real_HW_Latency_ms']         = max_hw_latency / 1000.0
        row['Real_HW_Loss_Percent']       = (total_drops / total_delta_enq * 100) if total_delta_enq > 0 else 0.0
        row['Cum_Enq']                    = self._cum_enq
        row['Cum_Recv']                   = self._cum_recv
        row["Time_Since_Traffic_Start_s"] = current_time - self.start_time
        row["Is_Rehash_Event"]            = self.is_rehash_event
        self.is_rehash_event              = 0
        row["Time_Since_Last_Rehash_s"]   = current_time - self.last_rehash_time
        row["Rehash_Impact"]              = math.exp(-row["Time_Since_Last_Rehash_s"])

        return row

    def run(self, duration=None):
        print("\n" + "="*125)
        dur_txt = f"{duration}s" if duration else "持續 (Ctrl-C 停止)"
        print(f" [ML 智能監控 v4.3] 啟動 - 1s 拓樸無關採集模式 | 時長: {dur_txt}")
        print("="*125 + "\n")

        results = []
        start = time.time()
        try:
            while True:
                if duration is not None and time.time() - start >= duration:
                    break
                data_row = self.collect_1s_data()
                self.raw_history.append(data_row)

                df_history     = pd.DataFrame(list(self.raw_history))
                df_transformed = transform_to_topo_independent(df_history, PORTS, CAPACITY, K=3)
                last_row       = df_transformed.iloc[-1]
                X              = pd.DataFrame([last_row[FEATURE_NAMES]])

                preds = {}
                with open(os.devnull, 'w') as _dn, redirect_stderr(_dn):
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

                # 記錄一列 (與 baseline / plot_1s_metrics 完全相同欄位 + 各交換機 util)
                results.append({
                    'Timestamp':  datetime.now(),
                    'Pred_Lat':   self.smoothed_latency,
                    'Real_Lat':   hw_lat,
                    'Pred_Loss':  self.smoothed_loss,
                    'Real_Loss':  hw_loss,
                    'Util_Sum':   feats['Total_Util_Sum'],
                    'Total_Mbps': last_row['Total_Actual_Mbps'],
                    'Cum_Enq':    data_row['Cum_Enq'],
                    'Cum_Recv':   data_row['Cum_Recv'],
                    **{f'util_s{p-1}': feats[f'src1_port{p}_load_util'] for p in PORTS},
                })

                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {status:7} | "
                      f"Lat: {self.smoothed_latency:5.1f}/{hw_lat:5.1f}ms | "
                      f"Loss: {self.smoothed_loss:4.1f}/{hw_loss:4.1f}% | "
                      f"Util: {feats['Total_Util_Sum']:4.2f}",
                      end='', flush=True)
                self.control_step(feats, preds)
        except KeyboardInterrupt: print("\n停止。")
        finally:
            if results:
                os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
                pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
                print(f"已寫入 {len(results)} 列 -> {OUTPUT_CSV}")


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ctrl = MLController()
    ctrl.run(duration)
