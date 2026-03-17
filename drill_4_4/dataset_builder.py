import os
import time
import random
import logging
import multiprocessing
import pandas as pd

# 載入 Phase 1 與 Phase 2 模組
from telemetry_collector import collect_telemetry
from iperf_parser import run_iperf_and_get_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [大腦] %(message)s')

# ==========================================
# 系統全域參數
# ==========================================
MASTER_CSV = "training_dataset_master.csv"
SOURCE_HOST = "h1"
SRC_ADD = 1 

TARGET_LEAF = "l2"
TARGET_IP = "10.0.2.2" #h2

DURATION = 10
FLOWS = 15

# 乾跑測試用的「當下假定拓樸頻寬」 (單位: Mbps)
CURRENT_CAPACITY = [0.8, 0.8, 1.2, 1.2] 

def simulate_set_weights():
    """模擬 SDN 控制面下發 W-ECMP 權重"""
    weights = [random.randint(1, 10) for _ in range(4)]
    logging.info(f"==> 執行 SDN 控制面：下發隨機探索權重 {weights}")
    time.sleep(0.5)
    return weights

def run_single_experiment(iteration_id):
    """執行單次測量生命週期：收集 -> 特徵聚合 -> 對齊 -> 寫入"""
    temp_x_csv = f"temp_x_{iteration_id}.csv"
    traffic_load = random.choice(["0.1M", "0.3M", "0.5M", "1M"]) 
    
    current_weights = simulate_set_weights()
    logging.info(f"--- 開始實驗 #{iteration_id} | 流量壓力: {FLOWS} * {traffic_load} ---")
    start_event = multiprocessing.Event()  # 用於同步 Phase 1 和 Phase 2 的開始時間

    # 1. 啟動 Phase 1 (X 遙測收集器)
    telemetry_proc = multiprocessing.Process(
        target=collect_telemetry,
        args=(SRC_ADD,TARGET_LEAF, DURATION, temp_x_csv, start_event)
    )
    telemetry_proc.start()
    start_event.wait()  # 等待 Phase 1 準備就緒後再啟動 Phase 2

    # 2. 啟動 Phase 2 (Y 標籤萃取器)
    latency_y, jitter_y, loss_y = run_iperf_and_get_metrics(
        SOURCE_HOST, TARGET_IP, traffic_load, DURATION, FLOWS
    )

    # 等待 Phase 1 收尾
    telemetry_proc.join()

    # 3. 特徵工程、聚合與資料對齊 (Data Fusion & Aggregation)
    if os.path.exists(temp_x_csv):
        logging.info(f"開始進行特徵聚合 (Feature Aggregation)...")
        try:
            df = pd.read_csv(temp_x_csv)
            
            # 【關鍵修改：特徵聚合邏輯】
            agg_data = {}
            for col in df.columns:
                # 排除時間戳記，不進行數值運算
                if col.lower() in ['time', 'timestamp']:
                    continue
                # 策略 A: 佇列深度 (qdepth) 取最大值 (捕捉最嚴重的壅塞點)
                if 'qdepth' in col.lower():
                    agg_data[f'{col}_max'] = round(df[col].max(), 4)
                # 策略 B: 吞吐量 (Mbps) 取平均值與標準差 (捕捉常態負載與流量震盪)
                # 計算統計特徵 (ddof=0 確保單筆資料時標準差為 0 而非 NaN)
                elif 'mbps' in col.lower():
                    agg_data[f'{col}_mean'] = round(df[col].mean(), 4)
                    agg_data[f'{col}_std'] = round(df[col].std(ddof=0), 4)
            
            # 將聚合後的字典轉換成單一列 (1 Row) 的 DataFrame
            agg_df = pd.DataFrame([agg_data])
            
            # --- 寫入系統先驗特徵與控制動作 (Action) ---
            total_mbps = float(traffic_load.replace('M', '')) * FLOWS
            agg_df['Total_Load_Mbps'] = round(total_mbps, 2)
            
            for i in range(4):
                agg_df[f'Weight_Port{i+2}'] = current_weights[i]
                agg_df[f'Capacity_Port{i+2}'] = CURRENT_CAPACITY[i]
                
            # --- 寫入客觀物理反饋 (Y Labels) ---
            agg_df['Label_Latency_ms'] = latency_y
            agg_df['Label_Jitter_ms'] = jitter_y
            agg_df['Label_Loss_Rate'] = loss_y
            
            # 輸出為單一筆訓練資料
            write_header = not os.path.exists(MASTER_CSV)
            agg_df.to_csv(MASTER_CSV, mode='a', index=False, header=write_header)
            
            os.remove(temp_x_csv)
            logging.info(f"實驗 #{iteration_id} 成功！已將 100 筆遙測壓縮為 1 筆特徵並寫入 {MASTER_CSV}\n")
            
        except Exception as e:
            logging.error(f"整合 CSV 時發生系統錯誤: {e}")
    else:
        logging.error(f"找不到特徵檔 {temp_x_csv}，本次實驗作廢。\n")

if __name__ == "__main__":
    TOTAL_ITERATIONS = 5
    logging.info(f"=== 啟動資料集生成大腦 (聚合特徵版 | 測試執行: {TOTAL_ITERATIONS} 迴圈) ===")
    
    try:
        for i in range(1, TOTAL_ITERATIONS + 1):
            run_single_experiment(i)
            logging.info("等待 3 秒讓硬體佇列冷卻歸零...")
            time.sleep(3) 
            
        logging.info(f"全自動收集完畢！請打開 {MASTER_CSV} 檢視最終機器學習矩陣。")
        
    except KeyboardInterrupt:
        logging.info("\n使用者手動中斷，安全停止大腦。")