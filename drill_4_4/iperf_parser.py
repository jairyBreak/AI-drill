import subprocess
import json
import logging
import threading
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_ping_measurement(source_host: str, target_ip: str, duration: int, result_dict: dict):
    """保持不變：使用 ping 測量真實的佇列延遲 (Bufferbloat RTT)"""
    cmd = ["mx", source_host, "ping", "-c", str(duration), target_ip]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        match = re.search(r'rtt min/avg/max/mdev = [\d\.]+/([\d\.]+)/[\d\.]+/[\d\.]+ ms', result.stdout)
        result_dict['latency'] = float(match.group(1)) if match else -1.0
    except Exception:
        result_dict['latency'] = -1.0

def run_iperf_and_get_metrics(source_host: str, target_ip: str, bw_per_flow_str: str, duration: int = 10, num_flows: int = 15):
    """
    使用 iperf3 執行測量，透過 TCP 控制通道確保報告必達，並解析 JSON 獲得絕對精準的丟包率。
    """
    logging.info(f"開始從 {source_host} 對 {target_ip} 發起 {num_flows} 條 iperf3 微流 (每條 {bw_per_flow_str})...")
    
    # 1. 啟動 Ping 背景測量 (負責收集 Y_latency)
    ping_result = {'latency': -1.0}
    ping_thread = threading.Thread(target=run_ping_measurement, args=(source_host, target_ip, duration, ping_result))
    ping_thread.start()
    
    # 2. 執行 iperf3 
    # 參數解析: -u (UDP), -b (單流頻寬), -t (秒數), -P (平行微流數量), -J (JSON 輸出)
    cmd = [
        "mx", source_host, 
        "iperf3", "-c", target_ip, 
        "-u", 
        "-b", bw_per_flow_str, 
        "-t", str(duration), 
        "-P", str(num_flows), 
        "-J"
    ]
    
    avg_loss_rate = 100.0 
    avg_jitter = -1.0
    
    try:
        # iperf3 會自己管理 15 條流，並在結束後匯總成一份 JSON
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 解析 JSON 輸出
        if result.returncode == 0 or "error" not in result.stdout:
            data = json.loads(result.stdout)
            
            total_lost = float(data['end']['sum']['lost_packets'])
            total_sent = float(data['end']['sum']['packets'])
            
            if total_sent > 0:
                avg_loss_rate = (total_lost / total_sent) * 100.0
            else:
                avg_loss_rate = 100.0 # didn't send anying
                
            avg_jitter = float(data['end']['sum']['jitter_ms'])
        else:
            logging.error(f"[iPerf3 執行異常] server doesn't open or something")
            logging.debug(result.stderr)

    except json.JSONDecodeError:
        logging.error(f"[JSON] iperf3 json error。")
    except Exception as e:
        logging.error(f"[error] {e}")

    # 等待 ping 結束
    ping_thread.join()
    avg_latency = ping_result['latency']
    
    logging.info(f"[Y] latency: {avg_latency} ms | jitter: {avg_jitter} ms | loss rate: {avg_loss_rate:.2f}%")
    return avg_latency, avg_jitter, avg_loss_rate

if __name__ == "__main__":
    # mx h1 iperf3 -s &
    
    TEST_SOURCE = "h2"
    TEST_TARGET_IP = "10.0.1.1"
    FLOWS = 15
    BW_PER_FLOW = "0.3M" 
    
    latency_y, jitter_y, loss_y = run_iperf_and_get_metrics(TEST_SOURCE, TEST_TARGET_IP, BW_PER_FLOW, duration=10, num_flows=FLOWS)
    print(f"\n最終萃取矩陣 Y = [Latency: {latency_y} ms, Jitter: {jitter_y} ms, Total Loss: {loss_y:.2f} %]")