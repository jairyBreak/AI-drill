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
import math
from datetime import datetime

warnings.filterwarnings("ignore")

# load P4-Utils
P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

logging.basicConfig(level=logging.ERROR)

# ---- config (aligned with 1s training script) ----
CONTROL_LEAF = "l1"
TARGET_LEAF = "l2"
PORTS = list(range(2, 10))
CAPACITY = {2: 0.8, 3: 0.8, 4: 0.8, 5: 0.8, 6: 1.2, 7: 1.2, 8: 1.2, 9: 1.2}
SRC_ID = 1

MODELS = {
    "latency": "rf_model_latency_1s.pkl",
    "loss": "rf_model_loss_1s.pkl",
    "anomaly": "rf_model_anomaly_1s.pkl"
}

# full 1s model feature list
FEATURE_NAMES = [
    "Is_Rehash_Event", "Time_Since_Last_Rehash_s", "Rehash_Impact",
    "Total_Util_Sum", "Max_Util_Diff", "Group_Imbalance", 
    "Max_QDepth", "Total_QDepth", "QDepth_Imbalance",
    "Over_Capacity_Sum", "Max_Q_Ratio", "Q_Danger_Flag", "Q_Danger_Count",
    "Total_QDepth_Trend",
    "Total_Actual_Mbps", "Expected_Over_Capacity_Sum",
    "Overflow_Intensity", "Queue_Full_And_Over_Cap"
]
for p in range(2, 10):
    FEATURE_NAMES.extend([
        f"src1_port{p}_qdepth", f"src1_port{p}_mbps", f"Weight_Port{p}",
        f"Norm_Load_P{p}", f"QDepth_Trend_P{p}", f"Mbps_Trend_P{p}",
        f"Expected_Util_P{p}"
    ])

class Realtime1sPredictor:
    def __init__(self):
        print(" [系統] 正在初始化 1s 即時預測器...")
        self.topo = load_topo("topology.json")
        # l1 for weights/drop stats, l2 for telemetry
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
        self.prev_state = None # for trend calc

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

    def get_current_weights(self):
        """Read live weights from l1's action profile."""
        weights = {p: 1 for p in PORTS} # default
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

    def collect_1s_data(self):
        """Collect one 1s data point + hardware ground truth."""
        start_t = time.time()
        time.sleep(1.0)
        dt = time.time() - start_t
        
        current_time = time.time()
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
        mbps_list = []

        # hardware ground-truth latency / loss
        max_hw_latency = 0
        total_delta_enq = 0
        total_drops = 0
        
        for p in PORTS:
            reg_idx = SRC_ID * 16 + p
            q = self.api_telemetry.register_read('path_max_queue_depth_reg', reg_idx)
            raw_acc_q_delay = self.api_telemetry.register_read('path_acc_q_delay_reg', reg_idx)
            
            self.api_telemetry.register_write('path_max_queue_depth_reg', reg_idx, 0)
            self.api_telemetry.register_write('path_acc_q_delay_reg', reg_idx, 0)
            
            cnt = self.api_telemetry.counter_read('port_bytes_counter', p)[0]
            db = cnt - self.prev_bytes[p]
            mbps = ((max(0, db) * 8) / (dt * 1_000_000))
            self.prev_bytes[p] = cnt
            
            row[f"src1_port{p}_qdepth"] = q
            row[f"src1_port{p}_mbps"] = mbps
            row[f"Weight_Port{p}"] = weights[p] / total_weight
            
            qdepths.append(q)
            mbps_list.append(mbps)
            
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
                
                if delta_enq > 0:
                    drops = max(0, delta_enq - delta_ingress)
                    total_delta_enq += delta_enq
                    total_drops += drops
            except: pass

        row['Real_HW_Latency_ms'] = max_hw_latency / 1000.0
        row['Real_HW_Loss_Percent'] = (total_drops / total_delta_enq * 100) if total_delta_enq > 0 else 0.0

        # 3. derived features
        row["Is_Rehash_Event"] = self.is_rehash_event
        self.is_rehash_event = 0 # reset after firing
        row["Time_Since_Last_Rehash_s"] = current_time - self.last_rehash_time
        row["Rehash_Impact"] = math.exp(-row["Time_Since_Last_Rehash_s"])
        
        for i, p in enumerate(PORTS):
            row[f"Norm_Load_P{p}"] = mbps_list[i] / max(0.01, row[f"Weight_Port{p}"])
            row[f"Util_P{p}"] = mbps_list[i] / CAPACITY[p]

        total_actual_mbps = sum(mbps_list)
        row["Total_Actual_Mbps"] = total_actual_mbps
        row["Total_Util_Sum"] = sum(mbps_list[i] / CAPACITY[p] for i, p in enumerate(PORTS))
        row["Max_Util_Diff"] = max(row[f"Util_P{p}"] for p in PORTS) - min(row[f"Util_P{p}"] for p in PORTS)
        
        # Traffic Projection
        expected_over_cap = 0
        for p in PORTS:
            exp_mbps = total_actual_mbps * row[f"Weight_Port{p}"]
            row[f"Expected_Util_P{p}"] = exp_mbps / CAPACITY[p]
            expected_over_cap += max(0, exp_mbps - CAPACITY[p])
        row["Expected_Over_Capacity_Sum"] = expected_over_cap
        
        row["Max_QDepth"] = max(qdepths)
        row["Total_QDepth"] = sum(qdepths)
        row["QDepth_Imbalance"] = max(qdepths) - min(qdepths)
        row["Max_Q_Ratio"] = row["Max_QDepth"] / 64.0
        row["Q_Danger_Flag"] = 1 if row["Max_QDepth"] > 40 else 0
        row["Q_Danger_Count"] = sum(1 for q in qdepths if q > 40)
        
        # group imbalance
        load_a = sum(row[f"src1_port{p}_mbps"] for p in [2, 3, 4, 5])
        weight_a = sum(row[f"Weight_Port{p}"] for p in [2, 3, 4, 5])
        load_b = sum(row[f"src1_port{p}_mbps"] for p in [6, 7, 8, 9])
        weight_b = sum(row[f"Weight_Port{p}"] for p in [6, 7, 8, 9])
        row["Group_Imbalance"] = abs((load_a / max(0.01, weight_a)) - (load_b / max(0.01, weight_b)))
        
        row["Over_Capacity_Sum"] = sum(max(0, row[f"src1_port{p}_mbps"] - CAPACITY[p]) for p in PORTS)
        row["Overflow_Intensity"] = row["Over_Capacity_Sum"] * row["Max_Q_Ratio"]
        row["Queue_Full_And_Over_Cap"] = row["Over_Capacity_Sum"] * row["Q_Danger_Flag"]

        # 4. temporal trends
        if self.prev_state is None:
            for p in PORTS:
                row[f"QDepth_Trend_P{p}"] = 0
                row[f"Mbps_Trend_P{p}"] = 0
            row["Total_QDepth_Trend"] = 0
        else:
            for p in PORTS:
                row[f"QDepth_Trend_P{p}"] = row[f"src1_port{p}_qdepth"] - self.prev_state[f"src1_port{p}_qdepth"]
                row[f"Mbps_Trend_P{p}"] = row[f"src1_port{p}_mbps"] - self.prev_state[f"src1_port{p}_mbps"]
            row["Total_QDepth_Trend"] = row["Total_QDepth"] - self.prev_state["Total_QDepth"]
        
        self.prev_state = row
        return row

    def run(self):
        print("\n" + "="*120)
        print(" [1s 即時預測測試] 啟動 - 正在監控硬體真實指標 (Hardware Ground Truth)")
        print("="*120 + "\n")
        
        try:
            while True:
                data_row = self.collect_1s_data()
                
                # only feed features the model knows
                X = pd.DataFrame([[data_row.get(f, 0) for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
                
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
                      f"Max Q: {data_row['Max_QDepth']:2d} | "
                      f"Total: {data_row['Total_Actual_Mbps']:4.2f}M")
                
        except KeyboardInterrupt:
            print("\n預測停止。")

if __name__ == "__main__":
    predictor = Realtime1sPredictor()
    predictor.run()
