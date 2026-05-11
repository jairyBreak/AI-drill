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
import collections
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
IPERF_PORT = 5202
IPERF_LOG = "./iperf_server.log"

MODELS = {
    "latency": "rf_model_latency_simplified.pkl",
    "latency_p99": "rf_model_latency_p99_simplified.pkl",
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
        
        # 滑動視窗緩存 (對齊 10 秒訓練尺度)
        self.telemetry_buffer = collections.deque(maxlen=100) # 10s @ 10Hz
        self.lat_buffer = collections.deque(maxlen=20)       # 10s @ 2Hz
        self.jit_buffer = collections.deque(maxlen=10)       # 10s @ 1Hz
        self.loss_buffer = collections.deque(maxlen=10)      # 10s @ 1Hz
        
        # 背景採集原始值
        self._raw_latency = 20.0
        self._raw_loss = 0.0
        self._raw_jitter = 0.0
        
        # 預測值平滑緩存
        self.smoothed_latency = 20.0 
        self.smoothed_loss = 0.0
        self.smoothed_jitter = 0.0
        
        self.init_baseline()
        self.start_iperf_monitoring()

    @property
    def real_latency(self):
        valid = [v for v in self.lat_buffer if v > 0]
        return np.mean(valid) if valid else self._raw_latency

    @property
    def real_jitter(self):
        return np.mean(self.jit_buffer) if self.jit_buffer else self._raw_jitter

    @property
    def real_loss(self):
        return np.mean(self.loss_buffer) if self.loss_buffer else self._raw_loss

    def start_iperf_monitoring(self):
        """啟動 iperf3 監控 (單探針流模式)"""
        print(f" [系統] 啟動 iperf3 探針解析器 (Port: {IPERF_PORT})...")
        
        # 1. 殺掉舊的 iperf3 並清理日誌
        subprocess.run(["pkill", "-f", "iperf3"], stderr=subprocess.DEVNULL)
        try:
            if os.path.exists(IPERF_LOG): os.remove(IPERF_LOG)
        except Exception as e:
            pass
        time.sleep(1.0)

        # 2. 在 h2 啟動 server
        cmd = ["mx", "h2", "iperf3", "-s", "-i", "1", "-p", str(IPERF_PORT), "--logfile", IPERF_LOG]
        subprocess.Popen(cmd)
        
        # 3. 啟動日誌監聽執行緒
        self.log_thread = threading.Thread(target=self.bg_log_tail, daemon=True)
        self.log_thread.start()
        
        # 4. 在 h1 啟動 client (單一條 0.1M 流)
        time.sleep(2.0)
        self.client_thread = threading.Thread(target=self.bg_iperf_client, daemon=True)
        self.client_thread.start()
        
        # 5. 啟動 Ping
        self.ping_thread = threading.Thread(target=self.bg_ping, daemon=True)
        self.ping_thread.start()

    def bg_log_tail(self):
        """監控 iperf3 日誌檔案並解析單流指標"""
        pattern = re.compile(r'([\d\.]+)\s+ms\s+[\d\s\w]+/([\d\s]+)\s+\((-?[\d\.]+)%\)')
        while not os.path.exists(IPERF_LOG): time.sleep(0.5)
        with open(IPERF_LOG, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                if "sec" in line and "/" in line:
                    match = pattern.search(line)
                    if match:
                        jitter = float(match.group(1))
                        loss_pct = float(match.group(3))
                        if loss_pct < 0: loss_pct = 20.0
                        self._raw_jitter = jitter
                        self._raw_loss = loss_pct
                        self.jit_buffer.append(jitter)
                        self.loss_buffer.append(loss_pct)

    def bg_iperf_client(self):
        """在 h1 運行 iperf3 client 產生 UDP 探針 (單流 0.1M)"""
        cmd = ["mx", "h1", "iperf3", "-c", TARGET_IP, "-u", "-b", "0.1M", "-t", "3600", "-i", "1", "-p", str(IPERF_PORT)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    def bg_ping(self):
        """每 0.5 秒執行一次 ping 來獲取真實延遲"""
        while True:
            try:
                output = subprocess.check_output(
                    ["mx", "h1", "ping", "-c", "1", "-W", "5.0", TARGET_IP],
                    stderr=subprocess.STDOUT, text=True
                )
                match = re.search(r'time=([\d\.]+)\s*ms', output)
                if match:
                    val = float(match.group(1))
                    self._raw_latency = val
                    self.lat_buffer.append(val)
                else:
                    self._raw_latency = -1.0 
                    self.lat_buffer.append(-1.0)
            except:
                self._raw_latency = -1.0
                self.lat_buffer.append(-1.0)
            time.sleep(0.5)

    def init_baseline(self):
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            for p in PORTS:
                try: self.prev_bytes[p] = self.api_telemetry.counter_read('port_bytes_counter', p)[0]
                except: pass

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
                
                self.smoothed_latency = (0.4 * preds['latency']) + (0.6 * self.smoothed_latency)
                self.smoothed_jitter = (0.4 * preds['jitter']) + (0.6 * self.smoothed_jitter)
                self.smoothed_loss = (0.4 * preds['loss']) + (0.6 * self.smoothed_loss)
                
                status = "NORMAL" if preds['anomaly'] == 0 else "\033[91mANOMALY\033[0m"
                ping_str = f"{self.real_latency:5.1f}ms" if self.real_latency > 0 else "\033[91mTIMEOUT\033[0m"
                
                sys.stdout.write("\033[K") 
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status:7} | "
                      f"Lat: {self.smoothed_latency:5.1f}/{ping_str} | "
                      f"Jit: {self.smoothed_jitter:4.1f}/{self.real_jitter:4.1f}ms | "
                      f"Loss: {self.smoothed_loss:4.1f}/{self.real_loss:4.1f}% | "
                      f"Util: {feats['Total_Util_Sum']:4.2f}", end='\r')
                sys.stdout.flush()
        except KeyboardInterrupt: print("\n停止。")

if __name__ == "__main__":
    ctrl = MLController()
    ctrl.run()
