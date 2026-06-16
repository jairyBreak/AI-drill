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

from all_controller import TopologyAnalyzer
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [DataBrainV2] %(message)s')

# ---- config & paths ----
MASTER_CSV = "research_results/data/datasets/rolling_training_dataset.csv"
RAW_DIR = "raw_telemetry_v2"
CONTROL_LEAF = "l1"
TARGET_LEAF = "l2"
TARGET_IP = "10.0.2.2"
IPERF_PORT = 5201
IPERF_LOG = "rolling_iperf.log"
SERVER_LOG = "rolling_server.log"
PORTS = list(range(2, 10))
SRC_ID = 1

class RollingDataBrain:
    def __init__(self):
        os.makedirs(os.path.dirname(MASTER_CSV), exist_ok=True)
        os.makedirs(RAW_DIR, exist_ok=True)
        
        # preload topology + analyzer
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

        # EMA smoothing of delay (alpha 0.7 = 70% newest reading)
        self.smoothed_acc_delay = {p: 0.0 for p in PORTS}
        self.smoothed_max_delay = {p: 0.0 for p in PORTS}
        self.delay_alpha = 0.7

        # time-feature tracking
        self.traffic_start_time = 0.0
        self.last_rehash_time = 0.0
        self.is_rehash_this_second = 0

    def _get_api(self, switch_name):
        thrift_port = self.topo.get_thrift_port(switch_name)
        try:
            return SimpleSwitchThriftAPI(thrift_port)
        except Exception as e:
            logging.error(f"cannot connect {switch_name}: {e}")
            return None

    def reset_switch_stats(self):
        api_target = self._get_api(TARGET_LEAF)
        if not api_target: return
        
        for p in PORTS:
            reg_idx = SRC_ID * 16 + p
            try:
                # clear queue-peak + delay registers
                api_target.register_write('path_max_queue_depth_reg', reg_idx, 0)
                api_target.register_write('path_max_q_delay_reg', reg_idx, 0)
                api_target.register_write('path_acc_q_delay_reg', reg_idx, 0)

                # prime byte counter baseline
                self.prev_bytes[p] = api_target.counter_read('port_bytes_counter', p)[0]

                self.smoothed_acc_delay[p] = 0.0
                self.smoothed_max_delay[p] = 0.0
            except Exception as e:
                logging.debug(f"reset failed (Port {p}): {e}")

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
            
            # detect whether weights actually changed
            weights_changed = False
            for p in PORTS:
                if self.current_weights.get(p) != port_weights_dict.get(p, 1):
                    weights_changed = True
                self.current_weights[p] = port_weights_dict.get(p, 1)

            if weights_changed:
                self.last_rehash_time = time.time()
                self.is_rehash_this_second = 1
                logging.info(f"weights changed (rehash): {weights_list}")

            return port_weights_dict
        except Exception as e:
            logging.error(f"weight push failed: {e}")
            return {}

    def collect_and_save(self, duration_sec):
        """Core sampling loop (one row per 1s)."""
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
            
            # reset flag after recording
            self.is_rehash_this_second = 0

            max_overall_delay = 0
            total_delta_enq = 0
            total_drops = 0

            # weight sum -> normalize to probability
            total_weight = sum(self.current_weights.values())
            if total_weight == 0: total_weight = 1
            
            for p in PORTS:
                reg_idx = SRC_ID * 16 + p
                try:
                    # read queue depth + delay
                    q = api_target.register_read('path_max_queue_depth_reg', reg_idx)
                    raw_max_q_delay = api_target.register_read('path_max_q_delay_reg', reg_idx)
                    raw_acc_q_delay = api_target.register_read('path_acc_q_delay_reg', reg_idx)

                    # reset-on-read
                    api_target.register_write('path_max_queue_depth_reg', reg_idx, 0)
                    api_target.register_write('path_max_q_delay_reg', reg_idx, 0)
                    api_target.register_write('path_acc_q_delay_reg', reg_idx, 0)

                    # light EMA smoothing
                    if self.smoothed_acc_delay[p] == 0 and raw_acc_q_delay > 0:
                        self.smoothed_acc_delay[p] = raw_acc_q_delay
                        self.smoothed_max_delay[p] = raw_max_q_delay
                    else:
                        self.smoothed_acc_delay[p] = (self.delay_alpha * raw_acc_q_delay) + ((1 - self.delay_alpha) * self.smoothed_acc_delay[p])
                        self.smoothed_max_delay[p] = (self.delay_alpha * raw_max_q_delay) + ((1 - self.delay_alpha) * self.smoothed_max_delay[p])
                    
                    row[f'src1_port{p}_qdepth'] = q
                    row[f'src1_port{p}_max_q_delay_us'] = round(self.smoothed_max_delay[p], 2)
                    row[f'src1_port{p}_acc_q_delay_us'] = round(self.smoothed_acc_delay[p], 2)

                    # throughput (bytes -> Mbps)
                    cnt_bytes = api_target.counter_read('port_bytes_counter', p)[0]
                    mbps = ((cnt_bytes - self.prev_bytes[p]) * 8) / (dt * 1_000_000)
                    row[f'src1_port{p}_mbps'] = round(mbps, 3)
                    self.prev_bytes[p] = cnt_bytes

                    # drop rate (l1_enq - l2_ingress)
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
                    
                # normalize weight to a fraction (0..1)
                row[f'Weight_Port{p}'] = round(self.current_weights[p] / total_weight, 4)

            # latency label = network-wide max accumulated delay
            row['Label_Max_Path_Delay_ms'] = max_overall_delay / 1000.0

            # network-wide total drop rate
            total_drop_rate = 0.0
            if total_delta_enq > 0:
                total_drop_rate = round((total_drops / total_delta_enq) * 100, 2)
            row['Label_Total_Drop_Rate_Percent'] = total_drop_rate
            
            samples.append(row)
            sys.stdout.write(f"\r  collecting... {len(samples)}/{duration_sec}s | Max Path Delay: {row['Label_Max_Path_Delay_ms']:.2f} ms | Total Loss: {row['Label_Total_Drop_Rate_Percent']}%")
            sys.stdout.flush()

        # append to master CSV
        df = pd.DataFrame(samples)
        df.to_csv(MASTER_CSV, mode='a', index=False, header=not os.path.exists(MASTER_CSV))

    def get_diverse_params(self, iteration_id):
        _ , rules = self.analyzer.get_ecmp_weights_and_rules(CONTROL_LEAF, TARGET_LEAF)
        num_comp = len(rules) if len(rules) > 0 else 1
        weights = [random.randint(1, 10) for _ in range(num_comp)]

        # pick mode 0.2 : 0.3 : 0.5
        idx = iteration_id % 10
        if idx < 2:
            mode = "--static"
        elif idx < 5:
            mode = "--dynamic"
        else:
            mode = "--elmice"

        # global load scale; 70% near nominal 3.12M
        r = random.random()
        if r < 0.70:
            scale = random.uniform(0.9, 1.1)  # 70% normal (~3.12M)
        elif r < 0.85:
            scale = random.uniform(0.4, 0.8)  # 15% light
        else:
            scale = random.uniform(1.2, 1.6)  # 15% heavy

        avg_load = (3.12 * scale) / 18
        return weights, mode, f"{avg_load:.3f}M", 18, scale

    def run_experiment(self, exp_id):
        print(f"\n=== starting long-run experiment #{exp_id} ===")
        
        self.traffic_start_time = time.time()
        self.last_rehash_time = time.time()
        
        weights, mode, load_str, flows, scale = self.get_diverse_params(exp_id)
        self.apply_weights(weights)
        self.reset_switch_stats()
        
        self.current_load = load_str
        self.current_flows = flows
        
        duration_sec = 120
        traffic_cmd = ["sudo", "python3", "traffic.py", mode, "--no-monitor", "--scale", f"{scale:.2f}", str(duration_sec)]
        p_traffic = subprocess.Popen(traffic_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"  [traffic] traffic.py {mode} | INT telemetry running")
        
        for i in range(4):
            if i > 0:
                _ , rules = self.analyzer.get_ecmp_weights_and_rules(CONTROL_LEAF, TARGET_LEAF)
                self.apply_weights([random.randint(1, 10) for _ in range(len(rules))])
            self.collect_and_save(duration_sec // 4)
        
        try:
            p_traffic.terminate()
            p_traffic.wait(timeout=5)
        except Exception:
            pass
        subprocess.run(["sudo", "pkill", "-9", "-f", "traffic.py"], stderr=subprocess.DEVNULL)
        subprocess.run(["sudo", "pkill", "-9", "-f", "iperf3"], stderr=subprocess.DEVNULL)

def main():
    subprocess.run(["sudo", "pkill", "-f", "iperf3"], stderr=subprocess.DEVNULL)
    if os.path.exists(SERVER_LOG): os.remove(SERVER_LOG)
    
    # background iperf3 server (port 5201)
    subprocess.Popen(["mx", "h2", "iperf3", "-s", "-i", "1", "-p", str(IPERF_PORT), "--logfile", SERVER_LOG])
    time.sleep(2)

    brain = RollingDataBrain()
    
    try:
        for i in range(1, 301):
            brain.run_experiment(i)
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        brain.stop_event.set()
        subprocess.run(["sudo", "pkill", "-f", "iperf3"], stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    main()
