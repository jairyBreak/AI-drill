import os
import time
import random
import logging
import multiprocessing
import pandas as pd
import json
import sys
import subprocess
import threading
import re
import numpy as np
from datetime import datetime

# 導入現有的工具
from all_controller import TopologyAnalyzer
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [數據大腦V2] %(message)s')

# ==========================================
# 配置與路徑
# ==========================================
MASTER_CSV = "research_results/data/datasets/rolling_training_dataset.csv"
RAW_DIR = "raw_telemetry_v2"
CONTROL_LEAF = "l1"
TARGET_LEAF = "l2"
TARGET_IP = "10.0.2.2"
IPERF_PORT = 5201
IPERF_LOG = "rolling_iperf.log"
SERVER_LOG = "rolling_server.log"
PORTS = [2, 3, 4, 5]
SRC_ID = 1

class RollingDataBrain:
    def __init__(self):
        os.makedirs(os.path.dirname(MASTER_CSV), exist_ok=True)
        os.makedirs(RAW_DIR, exist_ok=True)
        
        # 預先載入拓樸與分析器
        with open('p4app.json', 'r') as f:
            self.p4app_data = json.load(f)
        self.topo = load_topo("topology.json")
        self.analyzer = TopologyAnalyzer(self.p4app_data, self.topo)
        
        self.current_weights = {p: 1 for p in PORTS}
        self.current_load = "0M"
        self.current_flows = 0
        
        self.stop_event = threading.Event()
        self.prev_bytes = {p: 0 for p in PORTS}
        self.prev_drops = {p: 0 for p in PORTS}
        
        self.prev_l1_enq = {p: 0 for p in PORTS}
        self.prev_l2_ingress = {p: 0 for p in PORTS}
        
        # 用於平滑化的變數 (EMA)
        self.smoothed_acc_delay = {p: 0.0 for p in PORTS}
        self.smoothed_max_delay = {p: 0.0 for p in PORTS}
        # 平滑係數：0.7 代表最新讀數佔 70%，歷史佔 30%
        self.delay_alpha = 0.7
        
        # 新增：用於時間特徵追蹤
        self.traffic_start_time = 0.0
        self.last_rehash_time = 0.0
        self.is_rehash_this_second = 0

    def _get_api(self, switch_name):
        thrift_port = self.topo.get_thrift_port(switch_name)
        try:
            return SimpleSwitchThriftAPI(thrift_port)
        except Exception as e:
            logging.error(f"無法連線至 {switch_name}: {e}")
            return None

    def reset_switch_stats(self):
        api_target = self._get_api(TARGET_LEAF)
        if not api_target: return
        
        for p in PORTS:
            reg_idx = SRC_ID * 16 + p
            try:
                # 抹除佇列極值暫存器與延遲暫存器
                api_target.register_write('path_max_queue_depth_reg', reg_idx, 0)
                api_target.register_write('path_max_q_delay_reg', reg_idx, 0)
                api_target.register_write('path_acc_q_delay_reg', reg_idx, 0)
                
                # 預讀 Byte Counter 作為基準
                self.prev_bytes[p] = api_target.counter_read('port_bytes_counter', p)[0]
                
                # 初始化平滑變數
                self.smoothed_acc_delay[p] = 0.0
                self.smoothed_max_delay[p] = 0.0
            except Exception as e:
                logging.debug(f"Reset 失敗 (Port {p}): {e}")

    def apply_weights(self, weights_list):
        api_ctrl = self._get_api(CONTROL_LEAF)
        if not api_ctrl: return {}

        _ , hardware_rules = self.analyzer.get_ecmp_weights_and_rules(CONTROL_LEAF, TARGET_LEAF)
        
        selector_name = "w_ecmp_selector"
        action_name = "assign_component"
        
        try:
            grp_handle = api_ctrl.act_prof_create_group(selector_name)
            
            port_weights_dict = {}
            for idx, rule in enumerate(hardware_rules):
                comp_id = str(rule['comp_id'])
                weight = weights_list[idx] if idx < len(weights_list) else 1
                
                for port, _ in rule['ports_and_macs']:
                    port_weights_dict[port] = weight

                for _ in range(weight):
                    mbr_handle = api_ctrl.act_prof_create_member(selector_name, action_name, [comp_id])
                    api_ctrl.act_prof_add_member_to_group(selector_name, mbr_handle, grp_handle)
            
            cli_cmds = [
                "table_clear w_ecmp_table",
                f"table_indirect_add_with_group w_ecmp_table {TARGET_IP} => {grp_handle}"
            ]
            thrift_port = self.topo.get_thrift_port(CONTROL_LEAF)
            subprocess.run(['simple_switch_CLI', '--thrift-port', str(thrift_port)], 
                           input="\n".join(cli_cmds) + "\n", text=True, capture_output=True)
            
            # 檢查權重是否真的有改變
            weights_changed = False
            for p in PORTS:
                if self.current_weights.get(p) != port_weights_dict.get(p, 1):
                    weights_changed = True
                self.current_weights[p] = port_weights_dict.get(p, 1)
            
            if weights_changed:
                self.last_rehash_time = time.time()
                self.is_rehash_this_second = 1
                logging.info(f"權重發生變更 (Rehash): {weights_list}")
            
            return port_weights_dict
        except Exception as e:
            logging.error(f"下發權重出錯: {e}")
            return {}

    def collect_and_save(self, duration_sec):
        """核心採樣迴圈 (每 1s 一筆數據)"""
        start_time = time.time()
        samples = []
        while time.time() - start_time < duration_sec:
            sample_start = time.time()
            time.sleep(1.0)
            dt = time.time() - sample_start
            current_time = time.time()
            
            api_target = self._get_api(TARGET_LEAF)
            api_ctrl = self._get_api(CONTROL_LEAF)
            if not api_target or not api_ctrl: continue
            
            load_val = 0.0
            try:
                load_val = float(self.current_load.replace('M','')) * self.current_flows
            except: pass

            row = {
                'Timestamp': datetime.now(),
                'Total_Load_Mbps_Config': load_val,
                'Flows': self.current_flows,
                'Time_Since_Traffic_Start_s': round(current_time - self.traffic_start_time, 2),
                'Time_Since_Last_Rehash_s': round(current_time - self.last_rehash_time, 2),
                'Is_Rehash_Event': self.is_rehash_this_second
            }
            
            # 記錄完事件後將 Flag 歸零
            self.is_rehash_this_second = 0
            
            max_overall_delay = 0
            total_delta_enq = 0
            total_drops = 0
            
            # 計算權重總和，以轉換為比例 (Probability)
            total_weight = sum(self.current_weights.values())
            if total_weight == 0: total_weight = 1
            
            for p in PORTS:
                reg_idx = SRC_ID * 16 + p
                try:
                    # 讀取佇列深度與延遲
                    q = api_target.register_read('path_max_queue_depth_reg', reg_idx)
                    raw_max_q_delay = api_target.register_read('path_max_q_delay_reg', reg_idx)
                    raw_acc_q_delay = api_target.register_read('path_acc_q_delay_reg', reg_idx)
                    
                    # Reset-on-read
                    api_target.register_write('path_max_queue_depth_reg', reg_idx, 0)
                    api_target.register_write('path_max_q_delay_reg', reg_idx, 0)
                    api_target.register_write('path_acc_q_delay_reg', reg_idx, 0)
                    
                    # === 套用輕度 EMA 平滑化 ===
                    if self.smoothed_acc_delay[p] == 0 and raw_acc_q_delay > 0:
                        self.smoothed_acc_delay[p] = raw_acc_q_delay
                        self.smoothed_max_delay[p] = raw_max_q_delay
                    else:
                        self.smoothed_acc_delay[p] = (self.delay_alpha * raw_acc_q_delay) + ((1 - self.delay_alpha) * self.smoothed_acc_delay[p])
                        self.smoothed_max_delay[p] = (self.delay_alpha * raw_max_q_delay) + ((1 - self.delay_alpha) * self.smoothed_max_delay[p])
                    
                    # 記錄平滑後的數值
                    row[f'src1_port{p}_qdepth'] = q
                    row[f'src1_port{p}_max_q_delay_us'] = round(self.smoothed_max_delay[p], 2)
                    row[f'src1_port{p}_acc_q_delay_us'] = round(self.smoothed_acc_delay[p], 2)
                    
                    # 計算吞吐量 (Bytes -> Mbps)
                    cnt_bytes = api_target.counter_read('port_bytes_counter', p)[0]
                    mbps = ((cnt_bytes - self.prev_bytes[p]) * 8) / (dt * 1_000_000)
                    row[f'src1_port{p}_mbps'] = round(mbps, 3)
                    self.prev_bytes[p] = cnt_bytes
                    
                    # 計算丟包率 (l1_enq - l2_ingress)
                    l1_enq_pkts = api_ctrl.counter_read('cnt_enq', p)[0]
                    l2_ingress_pkts = api_target.counter_read('cnt_ingress', p)[0]
                    
                    delta_enq = l1_enq_pkts - self.prev_l1_enq[p]
                    delta_ingress = l2_ingress_pkts - self.prev_l2_ingress[p]
                    
                    self.prev_l1_enq[p] = l1_enq_pkts
                    self.prev_l2_ingress[p] = l2_ingress_pkts
                    
                    drop_rate = 0.0
                    drops = 0
                    if delta_enq > 0:
                        drops = max(0, delta_enq - delta_ingress)
                        drop_rate = round((drops / delta_enq) * 100, 2)
                    
                    row[f'src1_port{p}_congestion_drop_rate_percent'] = drop_rate
                    
                    total_delta_enq += delta_enq
                    total_drops += drops
                    
                    if self.smoothed_acc_delay[p] > max_overall_delay:
                        max_overall_delay = self.smoothed_acc_delay[p]

                except Exception as e:
                    # print(e)
                    row[f'src1_port{p}_qdepth'] = 0
                    row[f'src1_port{p}_max_q_delay_us'] = 0
                    row[f'src1_port{p}_acc_q_delay_us'] = 0
                    row[f'src1_port{p}_mbps'] = 0
                    row[f'src1_port{p}_congestion_drop_rate_percent'] = 0.0
                    
                # 正規化權重，轉換為比例 (0.0 ~ 1.0)
                row[f'Weight_Port{p}'] = round(self.current_weights[p] / total_weight, 4)
            
            # 以全網最大累積延遲作為總體 Latency 標籤（可選）
            row['Label_Max_Path_Delay_ms'] = max_overall_delay / 1000.0
            
            # 計算全網總丟包率
            total_drop_rate = 0.0
            if total_delta_enq > 0:
                total_drop_rate = round((total_drops / total_delta_enq) * 100, 2)
            row['Label_Total_Drop_Rate_Percent'] = total_drop_rate
            
            samples.append(row)
            sys.stdout.write(f"\r  採集中... {len(samples)}/{duration_sec}s | Max Path Delay: {row['Label_Max_Path_Delay_ms']:.2f} ms | Total Loss: {row['Label_Total_Drop_Rate_Percent']}%")
            sys.stdout.flush()

        # 存入 Master CSV
        df = pd.DataFrame(samples)
        df.to_csv(MASTER_CSV, mode='a', index=False, header=not os.path.exists(MASTER_CSV))

    def get_diverse_params(self, iteration_id):
        state_idx = iteration_id % 10
        _ , rules = self.analyzer.get_ecmp_weights_and_rules(CONTROL_LEAF, TARGET_LEAF)
        num_comp = len(rules) if len(rules) > 0 else 1

        if state_idx < 2: 
            # 狀態 0-1: 一般負載 (Normal) - 幾乎無丟包
            weights = [random.randint(2, 4) for _ in range(num_comp)]
            load = random.uniform(0.10, 0.15)
            flows = random.randint(10, 20)
        elif state_idx == 2: 
            # 狀態 2: 極少數大流 (Elephant flows) - 偶發局部丟包
            weights = [random.randint(1, 3) for _ in range(num_comp)]
            load = random.uniform(0.30, 0.60) 
            flows = random.randint(1, 4)
        elif state_idx == 3: 
            # 狀態 3: 權重不平衡 (Weight imbalance) - 易產生局部丟包
            weights = [1] * num_comp
            if num_comp > 0: weights[random.randint(0, num_comp-1)] = 10
            load = random.uniform(0.15, 0.25)
            flows = random.randint(15, 20)
        elif state_idx == 4: 
            # 狀態 4: 臨界滿載 (Near Capacity) - 輕微丟包
            weights = [random.randint(1, 3) for _ in range(num_comp)]
            load = random.uniform(0.20, 0.25)
            flows = random.randint(15, 20)
        elif state_idx == 5: 
            # 狀態 5: ⚠️ 全面超載 (Massive Overload) - 保證嚴重丟包
            # 總負載 = 20*0.5M = 10M，遠超 4.0M 總頻寬
            weights = [random.randint(1, 2) for _ in range(num_comp)]
            load = random.uniform(0.40, 0.60)
            flows = random.randint(20, 25)
        elif state_idx == 6: 
            # 狀態 6: ⚠️ 巨型象流衝突 (Extreme Elephant Clash) - 保證嚴重局部丟包
            # 單一流負載達 2.0M，只要撞進同一個 Port 瞬間塞爆
            weights = [random.randint(1, 3) for _ in range(num_comp)]
            load = random.uniform(1.50, 2.50)
            flows = random.randint(2, 4)
        elif state_idx == 7:
            # 狀態 7: ⚠️ 大量微流衝擊 (Microburst DDoS) - 佇列瞬間滿載丟包
            weights = [random.randint(1, 4) for _ in range(num_comp)]
            load = random.uniform(0.15, 0.20)
            flows = random.randint(40, 60)
        elif state_idx == 8:
            # 狀態 8: ⚠️ 惡劣路由+高負載 (Bad Routing + High Load) - 保證嚴重丟包
            weights = [1] * num_comp
            if num_comp > 0: weights[random.randint(0, num_comp-1)] = 15
            load = random.uniform(0.30, 0.40)
            flows = random.randint(15, 25)
        else:
            # 狀態 9: 隨機混沌狀態 (Chaos)
            weights = [random.randint(1, 10) for _ in range(num_comp)]
            load = random.uniform(0.10, 0.80)
            flows = random.randint(5, 30)
        
        return weights, f"{load:.2f}M", flows

    def run_experiment(self, exp_id):
        print(f"\n=== 開始長連線實驗 #{exp_id} ===")
        
        # 標記整批流量開始的時間
        self.traffic_start_time = time.time()
        # 剛開始時沒有所謂的"上次Rehash"
        self.last_rehash_time = time.time()
        
        weights, load_str, flows = self.get_diverse_params(exp_id)
        self.apply_weights(weights)
        self.reset_switch_stats()
        
        self.current_load = load_str
        self.current_flows = flows
        
        if os.path.exists(IPERF_LOG): os.remove(IPERF_LOG)
        
        # 啟動背景流量
        iperf_cmd = ["mx", "h1", "iperf3", "-c", TARGET_IP, "-u", "-b", self.current_load, "-t", "120", "-P", str(self.current_flows), "-p", str(IPERF_PORT), "-l", "1400", "--logfile", IPERF_LOG]
        p_iperf = subprocess.Popen(iperf_cmd)
        
        print(f"  [流量] {self.current_flows} 條 x {self.current_load} | INT 遙測中")
        
        for i in range(4):
            if i > 0:
                _ , rules = self.analyzer.get_ecmp_weights_and_rules(CONTROL_LEAF, TARGET_LEAF)
                self.apply_weights([random.randint(1, 10) for _ in range(len(rules))])
            self.collect_and_save(30)
        
        p_iperf.terminate()
        subprocess.run(["sudo", "pkill", "-f", "iperf3 -c"], stderr=subprocess.DEVNULL)

def main():
    subprocess.run(["sudo", "pkill", "-f", "iperf3"], stderr=subprocess.DEVNULL)
    if os.path.exists(SERVER_LOG): os.remove(SERVER_LOG)
    
    # 背景流量伺服器 (Port 5201)
    subprocess.Popen(["mx", "h2", "iperf3", "-s", "-i", "1", "-p", str(IPERF_PORT), "--logfile", SERVER_LOG])
    time.sleep(2)

    brain = RollingDataBrain()
    
    try:
        for i in range(1, 301):
            brain.run_experiment(i)
    except KeyboardInterrupt:
        print("\n中斷。")
    finally:
        brain.stop_event.set()
        subprocess.run(["sudo", "pkill", "-f", "iperf3"], stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    main()
