import socket
import time
import sys
import random

def send_traffic(target_ip, target_port, bandwidth_mbps, duration_sec):
    # 建立 UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 每個封包的大小 (1400 bytes)
    PAYLOAD_SIZE = 1400
    payload = b'X' * PAYLOAD_SIZE
    
    # 計算發包頻率
    # bits_per_second = bandwidth_mbps * 1,000,000
    # packets_per_second = bits_per_second / (PAYLOAD_SIZE * 8)
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
            
            # 精確控制時間間隔
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
    
    # 隨機選擇一個 Port 或固定使用 5001
    port = 5001
    
    send_traffic(ip, port, bw, duration)
