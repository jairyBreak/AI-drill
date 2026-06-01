import sys
import os
import json
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

try:
    from p4utils.utils.helper import load_topo
    from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI
except ImportError as e:
    logging.error(f"無法載入 P4-Utils 模組，請確認路徑正確: {e}")
    sys.exit(1)


# ==========================================
# 模組 1: 交換機限速控制器
# ==========================================
class SwitchRateLimiter:
    def __init__(self, switch_name: str, topo_obj):
        self.switch_name = switch_name
        self.topo = topo_obj
        self.thrift_port = self.topo.get_thrift_port(switch_name)
        self.api = None
        
        try:
            self.api = SimpleSwitchThriftAPI(self.thrift_port)
        except BaseException as e:
            logging.error(f"無法連線至 {self.switch_name} (Thrift Port: {self.thrift_port}): {e}")

    def apply_link_limit(self, neighbor: str, bw_mbps: float, packet_size_bits: int, max_queue_pkts: int):
        """計算並下發單一實體 Port 的硬體限速規則"""
        if not self.api:
            return

        # 物理轉換：Mbps 轉 PPS
        bits_per_second = bw_mbps * 1_000_000
        rate_pps = int(bits_per_second / packet_size_bits)
        
        # 取得實體 Port 號碼
        port_num = self.topo.node_to_node_port_num(self.switch_name, neighbor)
        
        try:
            # 寫入硬體限制 (正確順序：數值在前，Port 在後)
            self.api.set_queue_depth(max_queue_pkts, egress_port=port_num)
            self.api.set_queue_rate(rate_pps, egress_port=port_num)
            
            logging.info(f"[{self.switch_name}] 往 {neighbor:2} (Port {port_num:2}) | 限速: {bw_mbps:4} Mbps -> {rate_pps:5} PPS")
        except Exception as e:
            logging.error(f"[{self.switch_name}] 往 {neighbor} (Port {port_num}) 限速寫入失敗: {e}")


# ==========================================
# 模組 2: 主控制迴圈
# ==========================================
if __name__ == "__main__":
    if not os.path.exists('topology.json') or not os.path.exists('p4app.json'):
        logging.error("找不到設定檔，請確認已執行 sudo p4run。")
        sys.exit(1)

    # 載入拓樸物件 (負責實體 Port 與 Thrift 映射)
    topo_json_obj = load_topo('topology.json')
    
    # 載入 p4app 原始設定 (負責讀取頻寬邏輯)
    with open('p4app.json', 'r') as f:
        p4app_data = json.load(f)

    # ================= 建立 Link BW 對照表 =================
    bw_map = {}
    for link in p4app_data.get('topology', {}).get('links', []):
        if len(link) >= 3 and 'bw' in link[2]:
            u, v = link[0], link[1]
            bw = link[2]['bw']
            # 將雙向的頻寬皆記錄下來
            bw_map[(u, v)] = bw
            bw_map[(v, u)] = bw
    # =======================================================

    # 物理參數設定
    PACKET_SIZE_BITS = 1450 * 8
    MAX_QUEUE_PKTS = 64

    logging.info("===========================================")
    logging.info(" 啟動全網硬體自適應限速控制器 (依賴 p4app.json 修正版)")
    logging.info("===========================================")

    # 實例化所有 P4 交換機的控制器
    limiters = {}
    scale = 0.8
    for sw_name in topo_json_obj.get_p4switches():
        limiters[sw_name] = SwitchRateLimiter(sw_name, topo_json_obj)

    # 走訪拓樸並套用限速
    for sw_name, limiter in limiters.items():
        if not limiter.api:
            continue
            
        for neighbor in topo_json_obj.get_neighbors(sw_name):
            # 查表驗證這條連線是否有被設定 bw
            if (sw_name, neighbor) in bw_map:

                raw_bw_mbps = bw_map[(sw_name, neighbor)] * scale 

                limiter.apply_link_limit(
                    neighbor=neighbor, 
                    bw_mbps=raw_bw_mbps, 
                    packet_size_bits=PACKET_SIZE_BITS, 
                    max_queue_pkts=MAX_QUEUE_PKTS
                )

    logging.info("-" * 50)
    logging.info("所有拓樸定義之頻寬限制，已成功寫入 BMv2 硬體層！")