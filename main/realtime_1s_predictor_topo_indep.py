import sys
import os
import time
import json
import pandas as pd
import numpy as np
import joblib
import logging
import warnings
import collections
import math
from datetime import datetime

warnings.filterwarnings("ignore")

# load P4-Utils
P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

from topo_independent_helper import transform_to_topo_independent

logging.basicConfig(level=logging.ERROR)

# ---- config (8-spine topology) ----
CONTROL_LEAF = "l1"
TARGET_LEAF = "l2"
PORTS = list(range(2, 10))
CAPACITY = {2: 0.48, 3: 0.56, 4: 0.64, 5: 0.72, 6: 0.80, 7: 0.88, 8: 0.96, 9: 1.04}
SRC_ID = 1

MODELS = {
    "latency": "rf_model_latency_1s.pkl",
    "loss": "rf_model_loss_1s.pkl",
    "anomaly": "rf_model_anomaly_1s.pkl"
}

# 39 topology-independent model features
FEATURE_NAMES = [
    "Is_Rehash_Event", "Time_Since_Last_Rehash_s", "Rehash_Impact",
    "Total_Util_Sum", "Max_Util_Diff", "Group_Imbalance", 
    "Max_QDepth", "Total_QDepth", "QDepth_Imbalance",
    "Over_Capacity_Sum", "Max_Q_Ratio", "Q_Danger_Flag", "Q_Danger_Count",
    "Total_QDepth_Trend",
    "Total_Actual_Mbps", "Expected_Over_Capacity_Sum",
    "Overflow_Intensity", "Queue_Full_And_Over_Cap"
]
for k in range(3):
    FEATURE_NAMES.extend([
        f"top{k+1}_qdepth",
        f"top{k+1}_mbps", f"top{k+1}_weight",
        f"top{k+1}_norm_load", f"top{k+1}_expected_util",
        f"top{k+1}_qdepth_trend", f"top{k+1}_mbps_trend"
    ])

class Realtime1sPredictorTopoIndep:
    def __init__(self):
        print(" [系統] 正在初始化 1s 拓樸無關即時預測器...")
        self.topo = load_topo("topology.json")
        self.api_telemetry = SimpleSwitchThriftAPI(self.topo.get_thrift_port(TARGET_LEAF))
        self.api_control = SimpleSwitchThriftAPI(self.topo.get_thrift_port(CONTROL_LEAF))
        
        # load models
        self.models = {}
        for k, v in MODELS.items():
            if os.path.exists(v):
                self.models[k] = joblib.load(v)
                print(f"   - loaded model {v}")
            else:
                print(f"   - warning: model not found {v}")

        self.prev_bytes = {p: 0 for p in PORTS}
        self.prev_l1_enq = {p: 0 for p in PORTS}
        self.prev_l2_ingress = {p: 0 for p in PORTS}

        # history for trend features
        self.raw_history = collections.deque(maxlen=100)
        self.start_time = time.time()

        # weight-change / rehash tracking
        self.last_rehash_time = time.time()
        self.is_rehash_event = 0
        self.prev_weights = {p: 1 for p in PORTS}
        
        self.init_baseline()

    def init_baseline(self):
        """Read initial counter baselines."""
        for p in PORTS:
            try:
                self.prev_bytes[p] = self.api_telemetry.counter_read('port_bytes_counter', p)[0]
                self.prev_l1_enq[p] = self.api_control.counter_read('cnt_enq', p)[0]
                self.prev_l2_ingress[p] = self.api_telemetry.counter_read('cnt_ingress', p)[0]
            except: pass
        self._prev_sample_t = time.time()   # last sample time (for true interval)
        self._cum_enq = 0                    # cumulative l1 enqueue packets
        self._cum_recv = 0                   # cumulative l2 ingress packets (for E2E loss)

    def get_current_weights(self):
        """Read live weights from l1's action profile."""
        weights = {p: 1 for p in PORTS}
        try:
            import socket
            entries = self.api_control.client.bm_mt_get_entries(0, "MyIngress.w_ecmp_table")
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
                    grp_info = self.api_control.client.bm_mt_act_prof_get_group(0, "MyIngress.w_ecmp_selector", grp_handle)
                    members = grp_info.mbr_handles
                    
                    comp_counts = {}
                    for m_handle in members:
                        mbr = self.api_control.client.bm_mt_act_prof_get_member(0, "MyIngress.w_ecmp_selector", m_handle)
                        if mbr.action_data:
                            comp_id = int(mbr.action_data[0].hex(), 16)
                            comp_counts[comp_id] = comp_counts.get(comp_id, 0) + 1
                    
                    nh_entries = self.api_control.client.bm_mt_get_entries(0, "MyIngress.ecmp_group_to_nhop")
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

    def collect_1s_data(self, sleep_before=True):
        """Collect one 1s data point + hardware ground truth (sleep_before=False: caller paces)."""
        if sleep_before:
            time.sleep(1.0)
        now_t = time.time()
        # dt = true interval since last sample (incl. processing); not just sleep(1.0)
        dt = now_t - self._prev_sample_t
        self._prev_sample_t = now_t

        current_time = now_t
        row = {}

        # 1. read weights, detect rehash event
        weights = self.get_current_weights()
        if weights != self.prev_weights:
            self.is_rehash_event = 1
            self.last_rehash_time = current_time
            self.prev_weights = weights
            
        total_weight = sum(weights.values())
        if total_weight == 0: total_weight = 1
        
        # 2. telemetry
        qdepths = []

        # hardware ground-truth latency / loss
        max_hw_latency = 0
        total_delta_enq = 0
        total_drops = 0

        # bulk read: one RPC for the whole register array (cuts ~32 RPCs/s to 4)
        q_arr   = self.api_telemetry.register_read('path_max_queue_depth_reg')
        acc_arr = self.api_telemetry.register_read('path_acc_q_delay_reg')
        # reset after read; measurement window = interval between resets
        self.api_telemetry.register_reset('path_max_queue_depth_reg')
        self.api_telemetry.register_reset('path_acc_q_delay_reg')

        for p in PORTS:
            reg_idx = SRC_ID * 16 + p
            q = q_arr[reg_idx]
            raw_acc_q_delay = acc_arr[reg_idx]

            cnt = self.api_telemetry.counter_read('port_bytes_counter', p)[0]
            db = cnt - self.prev_bytes[p]
            mbps = ((max(0, db) * 8) / (dt * 1_000_000))
            self.prev_bytes[p] = cnt
            
            row[f"src1_port{p}_qdepth"] = q
            row[f"src1_port{p}_mbps"] = mbps
            row[f"Weight_Port{p}"] = weights[p] / total_weight
            
            qdepths.append(q)

            # max hardware latency
            if raw_acc_q_delay > max_hw_latency:
                max_hw_latency = raw_acc_q_delay

            # hardware drops (l1_enq - l2_ingress)
            try:
                l1_enq_pkts = self.api_control.counter_read('cnt_enq', p)[0]
                l2_ingress_pkts = self.api_telemetry.counter_read('cnt_ingress', p)[0]
                
                delta_enq = l1_enq_pkts - self.prev_l1_enq[p]
                delta_ingress = l2_ingress_pkts - self.prev_l2_ingress[p]
                
                self.prev_l1_enq[p] = l1_enq_pkts
                self.prev_l2_ingress[p] = l2_ingress_pkts

                # instantaneous per-sec estimate (biased; time-series only)
                if delta_enq > 0:
                    drops = max(0, delta_enq - delta_ingress)
                    total_delta_enq += delta_enq
                    total_drops += drops

                # cumulative totals -> correct E2E loss rate
                self._cum_enq += delta_enq
                self._cum_recv += delta_ingress
            except: pass

        row['Real_HW_Latency_ms'] = max_hw_latency / 1000.0
        row['Real_HW_Loss_Percent'] = (total_drops / total_delta_enq * 100) if total_delta_enq > 0 else 0.0
        row['Cum_Enq'] = self._cum_enq
        row['Cum_Recv'] = self._cum_recv

        # time features (match dataset schema)
        row["Time_Since_Traffic_Start_s"] = current_time - self.start_time
        row["Is_Rehash_Event"] = self.is_rehash_event
        self.is_rehash_event = 0  # reset after firing
        row["Time_Since_Last_Rehash_s"] = current_time - self.last_rehash_time
        row["Rehash_Impact"] = math.exp(-row["Time_Since_Last_Rehash_s"])
        
        return row

    def run(self):
        print("\n" + "="*120)
        print(" [1s 拓樸無關預測測試] 啟動 - 監控 8-Spine 硬體指標")
        print("="*120 + "\n")
        
        try:
            while True:
                data_row = self.collect_1s_data()
                self.raw_history.append(data_row)
                
                df_history = pd.DataFrame(list(self.raw_history))
                df_transformed = transform_to_topo_independent(df_history, PORTS, CAPACITY, K=3)
                last_row = df_transformed.iloc[-1]
                X = pd.DataFrame([last_row[FEATURE_NAMES]])
                
                preds = {}
                for k, m in self.models.items():
                    p = m.predict(X)[0]
                    if k == "latency":
                        p = np.expm1(p)
                    preds[k] = p
                
                status = "NORMAL" if preds['anomaly'] == 0 else "\033[91mANOMALY\033[0m"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status:7} | "
                      f"Lat(預測/真實): {preds['latency']:5.1f} / {data_row['Real_HW_Latency_ms']:5.1f} ms | "
                      f"Loss(預測/真實): {preds['loss']:4.1f} / {data_row['Real_HW_Loss_Percent']:4.1f} % | "
                      f"Max Q: {last_row['Max_QDepth']:2.0f} | "
                      f"Total: {last_row['Total_Actual_Mbps']:4.2f}M")
                
        except KeyboardInterrupt:
            print("\n預測停止。")

if __name__ == "__main__":
    predictor = Realtime1sPredictorTopoIndep()
    predictor.run()
