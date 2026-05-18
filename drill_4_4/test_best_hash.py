import subprocess
import time
TARGET_IP = "10.0.2.2"
group0 = [5201,5202,5203,5204]
group1 = [5205,5206,5207,5208,5209,5212]
def bg_iperf_client(IPERF_PORT,LOAD):
    # 在 h1 運行 iperf3 client 產生 UDP 探針 (單流 0.1M)
    cmd = ["mx", "h1", "iperf3", "-c", TARGET_IP, "-u", "-b", LOAD , "-t", "60", "-i", "1", "-p", str(IPERF_PORT),"--cport",str(IPERF_PORT)]
    subprocess.Popen(cmd)

def iperf_server(IPERF_PORT):
    cmd = ["mx", "h2", "iperf3", "-s", "-p",IPERF_PORT, "-D"]    
    subprocess.Popen(cmd)

if __name__ == "__main__":

    all_ports = group0 + group1
    
    print("[*] 階段一：啟動所有 Server...")
    for port in all_ports:
        iperf_server(str(port))
        
    # 2. 統一等待 1.5 秒，確保所有 Port 都成功 Bind
    print("[*] 等待 1.5 秒，確保所有 Server 就緒...")
    time.sleep(1.5)
    
    print("[*] 階段二：發動綁死 Source/Dest Port 的 4:6 併發流量...")
    for port in all_ports:
        bg_iperf_client(str(port), "0.1M")
    time.sleep(75) 
    print("[*] 清理背景 Server 程序...")
    time.sleep(0.5) # 給作業系統一點時間釋放 Port
   