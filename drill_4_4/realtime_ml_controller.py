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
import threading
import re
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
CAPACITY = {2: 0.8, 3: 0.8, 4: 1.2, 5: 1.2}

# 修正因子：經過校準，0.7 左右能對齊 iperf 載荷流量
HEADER_RATIO = (1400 / (1400 + 54)) * 0.73

MODELS = {
    "latency": "rf_regressor_latency_ms_simplified.pkl",
    "loss": "rf_regressor_loss_rate_simplified.pkl",
    "anomaly": "rf_anomaly_classifier_simplified.pkl"
}

# 30 個特徵（包含新加入的 Easy Features）
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
    "src1_port3_qdepth_max", "src1_port5_qdepth_max", "src1_port4_qdepth_max", "src1_port2_qdepth_max"
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
        self.real_latency = 0.0
        self.smoothed_latency = 20.0 # 初始值
        self.init_baseline()
        
        # 啟動背景 Ping 執行緒
        self.ping_thread = threading.Thread(target=self.bg_ping, daemon=True)
        self.ping_thread.start()

    def bg_ping(self):
        """每 0.2 秒執行一次 ping 來獲取真實延遲 (在 h1 命名空間內)"""
        while True:
            try:
                # 增加超時至 5.0s，防止在高擁塞時顯示 TIMEOUT
                output = subprocess.check_output(
                    ["mx", "h1", "ping", "-c", "1", "-W", "5.0", TARGET_IP],
                    stderr=subprocess.STDOUT, text=True
                )
                match = re.search(r'time=([\d\.]+)\s*ms', output)
                if match:
                    self.real_latency = float(match.group(1))
                else:
                    self.real_latency = -1.0 
            except:
                self.real_latency = -1.0
            time.sleep(0.2) # 更頻繁的採樣

    def init_baseline(self):
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            for p in PORTS:
                try: self.prev_bytes[p] = self.api_telemetry.counter_read('port_bytes_counter', p)[0]
                except: pass

    def collect_window(self, duration=1.0, interval=0.1):
        samples = []
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
                    row[f'mbps_{p}'] = ((db * 8) / (dt * 1_000_000)) if dt > 0 else 0
                    self.prev_bytes[p] = cnt
                samples.append(row)
                prev_sample_t = now
        return pd.DataFrame(samples)

    def get_current_weights(self):
        """從 Thrift API 獲取當前的 W-ECMP 權重"""
        weights = {p: 1 for p in PORTS} 
        try:
            entries = self.api_telemetry.table_get_entries("w_ecmp_table", False)
            if not entries: return weights
            grp_handle = entries[0].action_data.action_params[0]
            grp_info = self.api_telemetry.act_prof_get_group("w_ecmp_selector", grp_handle)
            members = grp_info.member_handles
            comp_counts = {}
            for m_handle in members:
                mbr = self.api_telemetry.act_prof_get_member("w_ecmp_selector", m_handle)
                comp_id = int(mbr.action_params[0])
                comp_counts[comp_id] = comp_counts.get(comp_id, 0) + 1
            nh_entries = self.api_telemetry.table_get_entries("ecmp_group_to_nhop", False)
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

        # 新增匯總指標
        feats["Total_Util_Sum"] = sum(utils)
        feats["Max_Util_Diff"] = max(utils) - min(utils)

        # Group A vs B Balance
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
        print("\n" + "="*90)
        print(" [ML 智能監控 v3] 啟動 - 模式: 自適應平滑 / 健康指標預測")
        print("="*90 + "\n")
        try:
            while True:
                df = self.collect_window()
                X, feats = self.extract_features(df)
                preds = {}
                for k, m in self.models.items():
                    p = m.predict(X)[0]
                    if k != "anomaly": p = np.expm1(p)
                    preds[k] = p
                
                # 非對稱自適應平滑
                new_pred = preds['latency']
                alpha = 0.8 if new_pred < self.smoothed_latency else 0.3
                self.smoothed_latency = (alpha * new_pred) + ((1 - alpha) * self.smoothed_latency)
                
                status = "NORMAL" if preds['anomaly'] == 0 else "\033[91mANOMALY\033[0m"
                ping_str = f"{self.real_latency:5.1f}ms" if self.real_latency > 0 else "\033[91mTIMEOUT\033[0m"
                
                sys.stdout.write("\033[K") 
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status:7} | Pred_Lat: {self.smoothed_latency:6.1f}ms | Real_Lat: {ping_str} | Util_Sum: {feats['Total_Util_Sum']:4.2f} | Diff: {feats['Max_Util_Diff']:4.2f} | Q: {feats['max_qdepth_p99']:4.1f}", end='\r')
                sys.stdout.flush()
        except KeyboardInterrupt: print("\n停止。")

if __name__ == "__main__":
    ctrl = MLController()
    ctrl.run()
