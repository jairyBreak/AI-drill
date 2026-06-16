import sys
import os
import time
import csv
import logging
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

try:
    from p4utils.utils.helper import load_topo
    from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI
except ImportError as e:
    logging.error(f"[error] failed to load P4-Utils: {e}")
    sys.exit(1)

def collect_telemetry(src_add,target_leaf, test_duration_sec, output_csv, start_event = None):
    try:
        topo = load_topo("topology.json")
        thrift_port = topo.get_thrift_port(target_leaf)
        api = SimpleSwitchThriftAPI(thrift_port)
    except Exception as e:
        logging.error(f"[error] cannot connect {target_leaf}: {e}")
        return

    # reg index = src_id * 16 + ingress_port (per ecmp.p4); reads spine ports 2-5 only
    target_src_ids = []
    target_src_ids.append(src_add)
    spine_ports = [2, 3, 4, 5]

    logging.info(f"開始收集 {target_leaf} 的 INT 佇列特徵，持續 {test_duration_sec} 秒...")
    if start_event:
        start_event.set()
    
    prev_time = time.time()
    prev_bytes = {port: 0 for port in spine_ports}
    ema_ratio = 0.3
    prev_ema_mbps = {}

    logging.info("init hardware baselines, clearing leftover state...")
    for src in target_src_ids:
        for port in spine_ports:
            # 1. clear queue-peak register
            reg_index = src * 16 + port
            try:
                api.register_write('path_max_queue_depth_reg', reg_index, 0)
            except Exception:
                pass

            # 2. prime byte-counter baseline
            try:
                current_obj = api.counter_read('port_bytes_counter', port)
                prev_bytes[port] = current_obj[0]
            except Exception:
                prev_bytes[port] = 0
    # =================================================================

    time.sleep(0.1) # let hardware state settle

    time.sleep(0.1)

    # CSV header
    headers = ["Timestamp"]
    for src in target_src_ids:
        for port in spine_ports:
            headers.append(f"src{src}_port{port}_qdepth")
            headers.append(f"src{src}_port{port}_mbps")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        start_time = time.time()
        # sample at 10 Hz
        while time.time() - start_time < test_duration_sec:
            current_timestamp = time.time()
            time_delta = current_timestamp - prev_time
            row_data = [current_timestamp]

            for src in target_src_ids:
                for port in spine_ports:
                    # mapping formula matches ecmp.p4
                    reg_index = src * 16 + port
                    try:
                        q_depth = api.register_read('path_max_queue_depth_reg', reg_index)
                        q_depth = min(64,q_depth)
                        row_data.append(q_depth)
                        # reset-on-read so last peak doesn't bleed into the next sample
                        api.register_write('path_max_queue_depth_reg', reg_index, 0)
                        
                    except Exception:
                        row_data.append(0)
                    try:
                        current_obj = api.counter_read('port_bytes_counter', port)
                        current_bytes = current_obj[0]
                        bytes_delta = current_bytes - prev_bytes[port]
                        
                        # guard against divide-by-zero, compute Mbps
                        if time_delta > 0:
                            throughput_mbps = (bytes_delta * 8) / (time_delta * 1_000_000)
                        else:
                            throughput_mbps = 0.0

                        if port not in prev_ema_mbps: 
                            ema_mbps = throughput_mbps
                        else:
                            ema_mbps = (ema_ratio * throughput_mbps) + ((1 - ema_ratio) * prev_ema_mbps[port])
                        # round to 2 dp
                        row_data.append(round(ema_mbps, 2))

                        prev_bytes[port] = current_bytes
                        prev_ema_mbps[port] = ema_mbps

                    except Exception:
                        row_data.append(0.0)

            writer.writerow(row_data)
            prev_time = current_timestamp           
            time.sleep(0.1)

    logging.info(f"收集完成！資料已儲存至 {output_csv}")

if __name__ == "__main__":
    # egress leaf to monitor (h1 -> h2, h2 on l2)
    SRC_ADD = 1
    TARGET_LEAF = "l2"
    DURATION = 20  # total collection seconds
    CSV_FILE = "l2_ml_features_qdepth.csv"
    
    collect_telemetry(SRC_ADD,TARGET_LEAF, DURATION, CSV_FILE)