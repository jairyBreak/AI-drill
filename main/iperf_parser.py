import subprocess
import json
import logging
import threading
import re
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

PING_COUNT = 100  # enough samples for stable p99 estimation

def run_ping_measurement(source_host: str, target_ip: str, duration: int, result_dict: dict):
    """Measure RTT via ping; collect 100 samples for p99 latency."""
    cmd = ["mx", source_host, "ping", "-c", str(PING_COUNT), "-i", "0.1", target_ip]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # parse individual RTT samples from each reply line
        rtts = [float(x) for x in re.findall(r'time=([\d\.]+)\s*ms', result.stdout)]

        if len(rtts) >= 10:
            result_dict['latency'] = float(np.mean(rtts))
            result_dict['p99_latency'] = float(np.percentile(rtts, 99))
        else:
            # fall back to summary line if too few individual samples parsed
            match = re.search(r'rtt min/avg/max/mdev = [\d\.]+/([\d\.]+)/[\d\.]+/[\d\.]+ ms', result.stdout)
            result_dict['latency'] = float(match.group(1)) if match else -1.0
            result_dict['p99_latency'] = -1.0
    except Exception:
        result_dict['latency'] = -1.0
        result_dict['p99_latency'] = -1.0

def run_iperf_and_get_metrics(source_host: str, target_ip: str, bw_per_flow_str: str, duration: int = 10, num_flows: int = 15):
    """Run iperf3 (UDP), parse JSON for exact loss rate. Returns (avg_latency, p99_latency, avg_jitter, avg_loss_rate)."""
    logging.info(f"starting {num_flows} iperf3 flows {source_host}->{target_ip} ({bw_per_flow_str} each)...")

    # 1. ping in the background (latency + p99)
    ping_result = {'latency': -1.0, 'p99_latency': -1.0}
    ping_thread = threading.Thread(target=run_ping_measurement, args=(source_host, target_ip, duration, ping_result))
    ping_thread.start()
    
    # 2. run iperf3: -u UDP, -b per-flow bw, -t seconds, -P parallel flows, -J JSON
    cmd = [
        "mx", source_host, 
        "iperf3", "-c", target_ip, 
        "-u", 
        "-b", bw_per_flow_str, 
        "-t", str(duration), 
        "-P", str(num_flows), 
        "-l", "1400",
        "-J"
    ]
    
    avg_loss_rate = 100.0 
    avg_jitter = -1.0
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # parse JSON output
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

    # wait for ping to finish
    ping_thread.join()
    avg_latency = ping_result['latency']
    p99_latency = ping_result['p99_latency']

    logging.info(f"[Y] latency: {avg_latency} ms | p99: {p99_latency} ms | jitter: {avg_jitter} ms | loss rate: {avg_loss_rate:.2f}%")
    return avg_latency, p99_latency, avg_jitter, avg_loss_rate

if __name__ == "__main__":
    # mx h1 iperf3 -s &
    
    TEST_SOURCE = "h2"
    TEST_TARGET_IP = "10.0.1.1"
    FLOWS = 15
    BW_PER_FLOW = "0.3M" 
    
    latency_y, p99_y, jitter_y, loss_y = run_iperf_and_get_metrics(TEST_SOURCE, TEST_TARGET_IP, BW_PER_FLOW, duration=10, num_flows=FLOWS)
    print(f"\n最終萃取矩陣 Y = [Latency: {latency_y} ms, p99: {p99_y} ms, Jitter: {jitter_y} ms, Total Loss: {loss_y:.2f} %]")