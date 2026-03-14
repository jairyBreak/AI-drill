import sys
import os
import time
import csv
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

try:
    # 統一改用 load_topo
    from p4utils.utils.helper import load_topo
    from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI
except ImportError as e:
    logging.error(f"[錯誤] 無法載入 P4-Utils: {e}")
    sys.exit(1)

def collect_telemetry(target_leaf, test_duration_sec, output_csv):
    try:
        # 使用 load_topo 建立 NetworkGraph 物件
        topo = load_topo("topology.json")
        thrift_port = topo.get_thrift_port(target_leaf)
        api = SimpleSwitchThriftAPI(thrift_port)
    except Exception as e:
        logging.error(f"[錯誤] 無法連線至 {target_leaf}: {e}")
        return

    # ================= 物理參數對應區 =================
    # 根據 ecmp.p4 邏輯：src_add = src_id * 16 + ingress_port
    # 假設 iperf 發送端是 h2 (10.0.2.2)，其 IP 結尾為 2，所以 src_id = 2
    target_src_ids = [1] 
    
    # 假設目標接收端是 h1 (接在 l1 的 port 1)
    # 那麼從 Spine (s1~s4) 進入 l1 的 ingress_port 通常是 2, 3, 4, 5
    # 請根據你的 topology.json 實際狀況微調此陣列
    spine_ports = [2, 3, 4, 5] 
    # ==================================================

    logging.info(f"開始收集 {target_leaf} 的 INT 佇列特徵，持續 {test_duration_sec} 秒...")
    
    prev_time = time.time()
    prev_bytes = {port: 0 for port in spine_ports}
    ema_ratio = 0.3
    prev_ema_mbps = {}
    time.sleep(0.1)

    # 準備 CSV 標頭
    headers = ["Timestamp"]
    for src in target_src_ids:
        for port in spine_ports:
            headers.append(f"src{src}_port{port}_qdepth")
            headers.append(f"src{src}_port{port}_mbps")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        start_time = time.time()
        # 取樣頻率：每 0.1 秒抓取一次 (10Hz)
        while time.time() - start_time < test_duration_sec:
            current_timestamp = time.time()
            time_delta = current_timestamp - prev_time
            row_data = [current_timestamp]

            for src in target_src_ids:
                for port in spine_ports:
                    # 完全對齊 ecmp.p4 的映射公式
                    reg_index = src * 16 + port
                    try:
                        q_depth = api.register_read('path_max_queue_depth_reg', reg_index)
                        q_depth = min(64,q_depth)
                        row_data.append(q_depth)
                        # 讀取完畢後強制歸零 (Reset-on-read)
                        # 避免前一次的壅塞極值污染下一秒的數據
                        api.register_write('path_max_queue_depth_reg', reg_index, 0)
                        
                    except Exception:
                        row_data.append(0)
                    try:
                        current_obj = api.counter_read('port_bytes_counter', port)
                        current_bytes = current_obj[0]
                        bytes_delta = current_bytes - prev_bytes[port]
                        
                        # 防止時間差極小導致的除以零錯誤，並計算 Mbps
                        if time_delta > 0:
                            throughput_mbps = (bytes_delta * 8) / (time_delta * 1_000_000)
                        else:
                            throughput_mbps = 0.0

                        if port not in prev_ema_mbps: 
                            ema_mbps = throughput_mbps
                        else:
                            ema_mbps = (ema_ratio * throughput_mbps) + ((1 - ema_ratio) * prev_ema_mbps[port])
                        # 取小數點後兩位，保持資料集乾淨
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
    # 設定要監聽的 Egress Leaf (假設流量打向 h1，h1 接在 l1)
    TARGET_LEAF = "l2"
    DURATION = 20 # 總收集秒數
    CSV_FILE = "l2_ml_features_qdepth.csv"
    
    collect_telemetry(TARGET_LEAF, DURATION, CSV_FILE)