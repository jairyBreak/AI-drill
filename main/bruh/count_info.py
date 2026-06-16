import time
import sys
sys.path.append('/home/p4/p4-utils')
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

class TelemetryCollector:
    def __init__(self, switch_name, thrift_port, num_ports):
        self.switch_name = switch_name
        self.api = SimpleSwitchThriftAPI(thrift_port)
        self.num_ports = num_ports

        # previous byte counts (for utilization)
        self.last_bytes = {port: 0 for port in range(num_ports)}
        self.last_time = time.time()

    def fetch_features(self):
        """Read registers + counters, return raw ML features."""
        current_time = time.time()
        time_delta = current_time - self.last_time
        
        queue_depths = []
        utilizations = []

        print(f"--- [{self.switch_name}] 即時遙測數據 ---")
        
        for port in range(self.num_ports):
            # 1. read queue depth (single value)
            try:
                q_depth = self.api.register_read('q_depth_reg', port)
            except Exception:
                q_depth = 0 # 0 if uninitialized
            queue_depths.append(q_depth)

            # 2. read traffic counter (packets, bytes)
            try:
                packet_count, byte_count = self.api.counter_read('cnt_ingress', port)
            except Exception:
                byte_count = 0

            # bandwidth utilization (bits per second)
            byte_delta = byte_count - self.last_bytes[port]
            bps = (byte_delta * 8) / time_delta
            utilizations.append(bps)

            self.last_bytes[port] = byte_count

            print(f"  Port {port} | Queue Depth: {q_depth:4} | Ingress BPS: {bps:10.2f} bps")

        self.last_time = current_time
        
        # 3. aggregate ML features
        max_q_depth = max(queue_depths)
        print(f"  => [ML Feature] Maximum Queue Depth: {max_q_depth}")
        print("-" * 35)
        return {
            "max_queue_depth": max_q_depth,
            "utilizations": utilizations
        }

if __name__ == "__main__":
    # test connection to l1 (Thrift port 9090)
    collector = TelemetryCollector("l1", 9090, num_ports=6)
    try:
        while True:
            # poll every 1s
            features = collector.fetch_features()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n遙測收集終止。")