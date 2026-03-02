import time
import sys
sys.path.append('/home/p4/p4-utils')
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

class TelemetryCollector:
    def __init__(self, switch_name, thrift_port, num_ports=5):
        self.switch_name = switch_name
        self.api = SimpleSwitchThriftAPI(thrift_port)
        self.num_ports = num_ports # 假設你的 Leaf Switch 有 5 個 Port
        
        # 儲存上一次的 Byte 數，用來計算 Utilization
        self.last_bytes = {port: 0 for port in range(num_ports)}
        self.last_time = time.time()

    def fetch_features(self):
        """讀取暫存器與計數器，回傳給 ML 模型的原始特徵"""
        current_time = time.time()
        time_delta = current_time - self.last_time
        
        queue_depths = []
        utilizations = []

        print(f"--- [{self.switch_name}] 即時遙測數據 ---")
        
        for port in range(self.num_ports):
            # 1. 讀取佇列深度 (Queue Depth)
            # register_read 回傳的是單一數值
            try:
                q_depth = self.api.register_read('q_depth_reg', port)
            except Exception:
                q_depth = 0 # 如果尚未初始化則為 0
            queue_depths.append(q_depth)

            # 2. 讀取流量計數器 (Link Utilization)
            # counter_read 回傳一個 tuple: (packets, bytes)
            try:
                packet_count, byte_count = self.api.counter_read('cnt_ingress', port)
            except Exception:
                byte_count = 0

            # 計算頻寬使用率 (Bytes per second)
            byte_delta = byte_count - self.last_bytes[port]
            bps = (byte_delta * 8) / time_delta # 轉換為 bits per second
            utilizations.append(bps)
            
            # 更新紀錄
            self.last_bytes[port] = byte_count

            print(f"  Port {port} | Queue Depth: {q_depth:4} | Ingress BPS: {bps:10.2f} bps")

        self.last_time = current_time
        
        # 3. 統整 ML 特徵
        max_q_depth = max(queue_depths)
        
        print(f"  => [ML Feature] Maximum Queue Depth: {max_q_depth}")
        print("-" * 35)
        
        return {
            "max_queue_depth": max_q_depth,
            "utilizations": utilizations
        }

if __name__ == "__main__":
    # 測試連線至 l1 (請確認 Thrift Port 是否為 9090)
    collector = TelemetryCollector("l1", 9090, num_ports=5)
    
    try:
        while True:
            # 每 1 秒輪詢一次 (你的預測區間是 1 秒)
            features = collector.fetch_features()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n遙測收集終止。")