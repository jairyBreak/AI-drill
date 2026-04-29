import os
import time
import random
import logging
import multiprocessing
import pandas as pd
import json
import sys
import subprocess
# 載入 Phase 1 與 Phase 2 模組
from telemetry_collector import collect_telemetry
from iperf_parser import run_iperf_and_get_metrics
from all_controller import TopologyAnalyzer
P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [大腦] %(message)s')

# ==========================================
# 系統全域參數
# ==========================================
MASTER_CSV = "training_dataset_master.csv"
SOURCE_HOST = "h1"
SRC_ADD = 1 
CONTROL_LEAF = "l1"

TARGET_LEAF = "l2"
TARGET_IP = "10.0.2.2" #h2

DURATION = 10
FLOWS = 12

# 乾跑測試用的「當下假定拓樸頻寬」 (單位: Mbps)
CURRENT_CAPACITY = {2: 0.8, 3: 0.8, 4: 1.2, 5: 1.2}

def apply_real_group_weights(ingress_leaf, target_leaf, target_ip):
    """透過重用 TopologyAnalyzer，對「同頻寬群組」下發真實權重"""
    
    # 1. 讀取拓樸並初始化分析器
    with open('p4app.json', 'r') as f:
        p4app_data = json.load(f)
    topo = load_topo("topology.json")
    analyzer = TopologyAnalyzer(p4app_data, topo)
    
    _ , hardware_rules = analyzer.get_ecmp_weights_and_rules(ingress_leaf, target_leaf)
    num_components = len(hardware_rules)
    
    # 3. 針對「群組 (Component)」骰隨機權重 (例如: 產生 2 個權重 [8, 3])
    group_weights = [random.randint(1, 8) for _ in range(num_components)]

    thrift_port = topo.get_thrift_port(ingress_leaf)
    try:
        api = SimpleSwitchThriftAPI(thrift_port)
    except Exception as e:
        logging.error(f"無法連線至 {ingress_leaf} 的 Thrift API: {e}")
        return {}

    selector_name = "w_ecmp_selector"
    action_name = "assign_component"

    try:
        # 建立新的 Action Profile Group
        grp_handle = api.act_prof_create_group(selector_name)

        # 根據群組權重，動態塞入對應的 Component ID
        for idx, rule in enumerate(hardware_rules):
            comp_id = str(rule['comp_id'])
            weight = group_weights[idx]
            for _ in range(weight):
                mbr_handle = api.act_prof_create_member(selector_name, action_name, [comp_id])
                api.act_prof_add_member_to_group(selector_name, mbr_handle, grp_handle)
                
    except Exception as e:
        logging.error(f"Thrift API 群組創建失敗: {e}")
        return {}

    # 5. CLI 暴力置換路由表綁定
    cli_cmds = [
        "table_clear w_ecmp_table",
        f"table_indirect_add_with_group w_ecmp_table {target_ip} => {grp_handle}"
    ]
    subprocess.run(
        ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
        input="\n".join(cli_cmds) + "\n",
        text=True, capture_output=True
    )
    time.sleep(0.1) 

    # 6. 【特徵對齊工程】：將群組權重「展開」回實體 Port 字典，方便寫入 CSV
    # 例如群組 1 (Port 4, 5) 骰到權重 8，就會回傳 {4: 8, 5: 8}
    port_weights = {}
    for idx, rule in enumerate(hardware_rules):
        weight = group_weights[idx]
        for port, _ in rule['ports_and_macs']:
            port_weights[port] = weight
            
    return port_weights

def run_single_experiment(iteration_id):
    """執行單次測量生命週期：收集 -> 特徵聚合 -> 對齊 -> 寫入"""
    temp_x_csv = f"temp_x_{iteration_id}.csv"
    traffic_load = random.choice(["0.1M", "0.2M", "0.3M", "0.4M"]) 
    
    logging.info(f"--- 開始實驗 #{iteration_id} | 流量壓力: {FLOWS} * {traffic_load} ---")

    port_weights_dict = apply_real_group_weights(CONTROL_LEAF, TARGET_LEAF, TARGET_IP)
    if not port_weights_dict:
        logging.error("權重下發失敗，跳過本次實驗")
        return

    start_event = multiprocessing.Event()  # 用於同步 Phase 1 和 Phase 2 的開始時間

    # 1. 啟動 Phase 1 (X 遙測收集器)
    telemetry_proc = multiprocessing.Process(
        target=collect_telemetry,
        args=(SRC_ADD,TARGET_LEAF, DURATION, temp_x_csv, start_event)
    )
    telemetry_proc.start()
    start_event.wait()  # 等待 Phase 1 準備就緒後再啟動 Phase 2

    # 2. 啟動 Phase 2 (Y 標籤萃取器)
    latency_y, p99_latency_y, jitter_y, loss_y = run_iperf_and_get_metrics(
        SOURCE_HOST, TARGET_IP, traffic_load, DURATION, FLOWS
    )

    # 等待 Phase 1 收尾
    telemetry_proc.join()

    # 3. 特徵工程、聚合與資料對齊 (Data Fusion & Aggregation)
    if os.path.exists(temp_x_csv):
        logging.info(f"開始進行特徵聚合 (Feature Aggregation)...")
        try:
            df = pd.read_csv(temp_x_csv)
            
            agg_data = {}
            for col in df.columns:
                # 排除時間戳記，不進行數值運算
                if col.lower() in ['time', 'timestamp']:
                    continue
                # 策略 A: 佇列深度 (qdepth) 取最大值 (捕捉最嚴重的壅塞點)
                if 'qdepth' in col.lower():
                    print(df[col])
                    agg_data[f'{col}_max'] = round(df[col].max(), 4)
                # 策略 B: 吞吐量 (Mbps) 取平均值與標準差 (捕捉常態負載與流量震盪)
                # 計算統計特徵 (ddof=0 確保單筆資料時標準差為 0 而非 NaN)
                elif 'mbps' in col.lower():
                    agg_data[f'{col}_mean'] = round(df[col].mean(), 4)
                    agg_data[f'{col}_std'] = round(df[col].std(ddof=0), 4)

            # 推導特徵（從聚合值計算，不需 raw telemetry）
            _CAPACITY = {2: 0.8, 3: 0.8, 4: 1.2, 5: 1.2}
            _port_qdepth_maxes = []
            _port_mbps_means   = []
            for _n in [2, 3, 4, 5]:
                _mean = agg_data[f'src1_port{_n}_mbps_mean']
                _std  = agg_data[f'src1_port{_n}_mbps_std']
                agg_data[f'src1_port{_n}_mbps_cv']    = round(_std / _mean if _mean > 0 else 0.0, 4)
                agg_data[f'src1_port{_n}_load_util']  = round(_mean / _CAPACITY[_n], 4)
                _port_qdepth_maxes.append(agg_data[f'src1_port{_n}_qdepth_max'])
                _port_mbps_means.append(_mean)
            agg_data['qdepth_max_imbalance'] = round(max(_port_qdepth_maxes) - min(_port_qdepth_maxes), 4)
            agg_data['mbps_imbalance']       = round(pd.Series(_port_mbps_means).std(ddof=0), 4)
            agg_data['total_qdepth_max']     = sum(_port_qdepth_maxes)

            # 將聚合後的字典轉換成單一列 (1 Row) 的 DataFrame
            agg_df = pd.DataFrame([agg_data])
            
            # --- 寫入系統先驗特徵與控制動作 (Action) ---
            total_mbps = float(traffic_load.replace('M', '')) * FLOWS
            agg_df['Total_Load_Mbps'] = round(total_mbps, 2)
            
            for i in range(2, 6):
                agg_df[f'Weight_Port{i}'] = port_weights_dict.get(i, 1)
                agg_df[f'Capacity_Port{i}'] = CURRENT_CAPACITY.get(i, 1.0)
                
            # --- 寫入客觀物理反饋 (Y Labels) ---
            agg_df['Label_Latency_ms'] = latency_y
            agg_df['Label_Latency_p99_ms'] = p99_latency_y
            agg_df['Label_Jitter_ms'] = jitter_y
            agg_df['Label_Loss_Rate'] = loss_y
            
            # 輸出為單一筆訓練資料
            write_header = not os.path.exists(MASTER_CSV)
            agg_df.to_csv(MASTER_CSV, mode='a', index=False, header=write_header)
            
            import shutil
            os.makedirs("raw_telemetry", exist_ok=True)
            shutil.move(temp_x_csv, f"raw_telemetry/experiment_{iteration_id}.csv")
            logging.info(f"實驗 #{iteration_id} 成功！已將 100 筆遙測壓縮為 1 筆特徵並寫入 {MASTER_CSV}\n")
            
        except Exception as e:
            logging.error(f"整合 CSV 時發生系統錯誤: {e}")
    else:
        logging.error(f"找不到特徵檔 {temp_x_csv}，本次實驗作廢。\n")

if __name__ == "__main__":
    TOTAL_ITERATIONS = 1500
    logging.info(f"=== 啟動資料集生成大腦 (聚合特徵版 | 測試執行: {TOTAL_ITERATIONS} 迴圈) ===")
    
    try:
        for i in range(1, TOTAL_ITERATIONS + 1):
            run_single_experiment(i)
            logging.info("等待 3 秒讓硬體佇列冷卻歸零...")
            time.sleep(3) 
            
        logging.info(f"全自動收集完畢！請打開 {MASTER_CSV} 檢視最終機器學習矩陣。")
        
    except KeyboardInterrupt:
        logging.info("\n使用者手動中斷，安全停止大腦。")