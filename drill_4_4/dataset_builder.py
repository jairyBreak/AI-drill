import os
import time
import random
import logging
import multiprocessing
import pandas as pd
import json
import sys
import subprocess
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
# 系統全域參數 (8小時強化版)
# ==========================================
MODE = "PRODUCTION_BALANCED" 
ADDITIONAL_SAMPLES = 1500 # 預計 8 小時可跑完

MASTER_CSV = "training_dataset_master.csv"
SOURCE_HOST = "h1"
SRC_ADD = 1 
CONTROL_LEAF = "l1"
TARGET_LEAF = "l2"
TARGET_IP = "10.0.2.2" 
DURATION = 10
CURRENT_CAPACITY = {2: 0.8, 3: 0.8, 4: 1.2, 5: 1.2}

def get_params(iteration_id):
    """循環 5 種狀態，確保資料集極度均衡"""
    state_idx = iteration_id % 5
    
    if state_idx == 0: # NORMAL
        logging.info(f">> 狀態 [{iteration_id}]: NORMAL")
        return [random.randint(4, 5), random.randint(4, 5)], "0.1M", random.randint(4, 8)
        
    elif state_idx == 1: # SUSTAINED_CONGESTION
        logging.info(f">> 狀態 [{iteration_id}]: SUSTAINED_CONGESTION")
        return [4, 4], "0.5M", random.randint(15, 25)
        
    elif state_idx == 2: # BURST_CONGESTION
        logging.info(f">> 狀態 [{iteration_id}]: BURST_CONGESTION")
        # 臨界負載：容易觸發佇列堆積但不會一直滿載
        return [random.randint(3, 5), random.randint(3, 5)], "0.3M", random.randint(10, 14)
        
    elif state_idx == 3: # HIGH_JITTER
        logging.info(f">> 狀態 [{iteration_id}]: HIGH_JITTER")
        # 極多流、極低單流頻寬 (觸發 Buffer 爭搶)
        load = random.choice(["0.02M", "0.04M", "0.06M"])
        return [random.randint(3, 6), random.randint(3, 6)], load, random.randint(40, 70)
        
    elif state_idx == 4: # UNBALANCED_LOAD
        logging.info(f">> 狀態 [{iteration_id}]: UNBALANCED_LOAD")
        # 極端權重，中等負載
        load = random.choice(["0.3M", "0.4M"])
        weights = random.choice([[8, 1], [1, 8], [7, 2], [2, 7]])
        return weights, load, random.randint(5, 10)

def apply_real_group_weights(ingress_leaf, target_leaf, target_ip, group_weights):
    with open('p4app.json', 'r') as f:
        p4app_data = json.load(f)
    topo = load_topo("topology.json")
    analyzer = TopologyAnalyzer(p4app_data, topo)
    _ , hardware_rules = analyzer.get_ecmp_weights_and_rules(ingress_leaf, target_leaf)
    thrift_port = topo.get_thrift_port(ingress_leaf)
    try:
        api = SimpleSwitchThriftAPI(thrift_port)
    except Exception as e:
        return {}

    selector_name = "w_ecmp_selector"
    action_name = "assign_component"
    try:
        grp_handle = api.act_prof_create_group(selector_name)
        for idx, rule in enumerate(hardware_rules):
            comp_id = str(rule['comp_id'])
            weight = group_weights[idx]
            for _ in range(weight):
                mbr_handle = api.act_prof_create_member(selector_name, action_name, [comp_id])
                api.act_prof_add_member_to_group(selector_name, mbr_handle, grp_handle)
    except Exception:
        return {}

    cli_cmds = ["table_clear w_ecmp_table", f"table_indirect_add_with_group w_ecmp_table {target_ip} => {grp_handle}"]
    subprocess.run(['simple_switch_CLI', '--thrift-port', str(thrift_port)], input="\n".join(cli_cmds) + "\n", text=True, capture_output=True)
    
    port_weights = {}
    for idx, rule in enumerate(hardware_rules):
        weight = group_weights[idx]
        for port, _ in rule['ports_and_macs']:
            port_weights[port] = weight
    return port_weights

def run_single_experiment(iteration_id):
    temp_x_csv = f"temp_x_{iteration_id}.csv"
    group_weights, traffic_load, flows = get_params(iteration_id)
    port_weights_dict = apply_real_group_weights(CONTROL_LEAF, TARGET_LEAF, TARGET_IP, group_weights)
    if not port_weights_dict: return

    start_event = multiprocessing.Event()
    telemetry_proc = multiprocessing.Process(target=collect_telemetry, args=(SRC_ADD, TARGET_LEAF, DURATION, temp_x_csv, start_event))
    telemetry_proc.start()
    start_event.wait()

    latency_y, p99_latency_y, jitter_y, loss_y = run_iperf_and_get_metrics(SOURCE_HOST, TARGET_IP, traffic_load, DURATION, flows)
    telemetry_proc.join()

    if os.path.exists(temp_x_csv):
        try:
            df = pd.read_csv(temp_x_csv)
            agg_data = {}
            for col in df.columns:
                if col.lower() in ['time', 'timestamp']: continue
                if 'qdepth' in col.lower(): agg_data[f'{col}_max'] = round(df[col].max(), 4)
                elif 'mbps' in col.lower():
                    agg_data[f'{col}_mean'] = round(df[col].mean(), 4)
                    agg_data[f'{col}_std'] = round(df[col].std(ddof=0), 4)

            _CAPACITY = {2: 0.8, 3: 0.8, 4: 1.2, 5: 1.2}
            _port_qdepth_maxes, _port_mbps_means = [], []
            for _n in [2, 3, 4, 5]:
                _mean, _std = agg_data[f'src1_port{_n}_mbps_mean'], agg_data[f'src1_port{_n}_mbps_std']
                agg_data[f'src1_port{_n}_mbps_cv'] = round(_std / _mean if _mean > 0 else 0.0, 4)
                agg_data[f'src1_port{_n}_load_util'] = round(_mean / _CAPACITY[_n], 4)
                _port_qdepth_maxes.append(agg_data[f'src1_port{_n}_qdepth_max'])
                _port_mbps_means.append(_mean)
            agg_data['qdepth_max_imbalance'] = round(max(_port_qdepth_maxes) - min(_port_qdepth_maxes), 4)
            agg_data['mbps_imbalance'] = round(pd.Series(_port_mbps_means).std(ddof=0), 4)
            agg_data['total_qdepth_max'] = sum(_port_qdepth_maxes)

            agg_df = pd.DataFrame([agg_data])
            agg_df['Total_Load_Mbps'] = round(float(traffic_load.replace('M', '')) * flows, 2)
            for i in range(2, 6):
                agg_df[f'Weight_Port{i}'] = port_weights_dict.get(i, 1)
                agg_df[f'Capacity_Port{i}'] = CURRENT_CAPACITY.get(i, 1.0)
            agg_df['Label_Latency_ms'], agg_df['Label_Latency_p99_ms'] = latency_y, p99_latency_y
            agg_df['Label_Jitter_ms'], agg_df['Label_Loss_Rate'] = jitter_y, loss_y
            
            agg_df.to_csv(MASTER_CSV, mode='a', index=False, header=not os.path.exists(MASTER_CSV))
            os.makedirs("raw_telemetry", exist_ok=True)
            import shutil
            shutil.move(temp_x_csv, f"raw_telemetry/experiment_{iteration_id}.csv")
            logging.info(f"實驗 #{iteration_id} 成功！")
        except Exception as e: logging.error(f"整合錯誤: {e}")
    time.sleep(2)

if __name__ == "__main__":
    os.makedirs("raw_telemetry", exist_ok=True)
    existing_files = os.listdir("raw_telemetry")
    existing_ids = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith("experiment_") and f.endswith(".csv")]
    start_id = max(existing_ids) + 1 if existing_ids else 1
    
    logging.info(f"=== 啟動生產級資料大腦 | 模式: {MODE} | 從 #{start_id} 開始 | 追加目標: {ADDITIONAL_SAMPLES} 筆 ===")
    try:
        for i in range(start_id, start_id + ADDITIONAL_SAMPLES):
            run_single_experiment(i)
    except KeyboardInterrupt: logging.info("\n手動中斷。")
