import socket
import time
import sys
import random

def send_traffic(target_ip, target_port, bandwidth_mbps, duration_sec):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    PAYLOAD_SIZE = 1400  # bytes per packet
    payload = b'X' * PAYLOAD_SIZE

    # packets per second from target bandwidth
    pps = (bandwidth_mbps * 1_000_000) / (PAYLOAD_SIZE * 8)
    inter_packet_gap = 1.0 / pps
    
    print(f"=== Python 流量產生器 ===")
    print(f"目標: {target_ip}:{target_port}")
    print(f"頻寬: {bandwidth_mbps} Mbps (~{int(pps)} PPS)")
    print(f"持續時間: {duration_sec} 秒")
    
    start_time = time.time()
    total_packets = 0
    
    try:
        while time.time() - start_time < duration_sec:
            sock.sendto(payload, (target_ip, target_port))
            total_packets += 1
            
            # pace the send rate
            time.sleep(inter_packet_gap)
            
            if total_packets % 100 == 0:
                elapsed = time.time() - start_time
                print(f"已發送: {total_packets} 封包 | 已耗時: {elapsed:.1f}s", end='\r')
                
    except KeyboardInterrupt:
        print("\n使用者中斷發送。")
    
    print(f"\n發送結束。總計發送 {total_packets} 封包。")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 traffic_gen.py <目標IP> <頻寬Mbps> [持續秒數]")
        sys.exit(1)
        
    ip = sys.argv[1]
    bw = float(sys.argv[2])
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    port = 5001
    
    send_traffic(ip, port, bw, duration)
