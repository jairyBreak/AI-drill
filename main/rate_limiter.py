import sys
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

try:
    from p4utils.utils.helper import load_topo
    from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI
except ImportError as e:
    logging.error(f"Failed to load P4-Utils: {e}")
    sys.exit(1)


# Module 1: per-switch rate limiter
class SwitchRateLimiter:
    def __init__(self, switch_name: str, topo_obj):
        self.switch_name = switch_name
        self.topo = topo_obj
        self.thrift_port = self.topo.get_thrift_port(switch_name)
        self.api = None
        
        try:
            self.api = SimpleSwitchThriftAPI(self.thrift_port)
        except BaseException as e:
            logging.error(f"cannot connect {self.switch_name} (Thrift {self.thrift_port}): {e}")

    def apply_link_limit(self, neighbor: str, bw_mbps: float, packet_size_bits: int, max_queue_pkts: int):
        """Compute and push the hardware rate limit for one physical port."""
        if not self.api:
            return

        # Mbps -> PPS
        bits_per_second = bw_mbps * 1_000_000
        rate_pps = int(bits_per_second / packet_size_bits)

        port_num = self.topo.node_to_node_port_num(self.switch_name, neighbor)

        try:
            # value first, port second
            self.api.set_queue_depth(max_queue_pkts, egress_port=port_num)
            self.api.set_queue_rate(rate_pps, egress_port=port_num)

            logging.info(f"[{self.switch_name}] -> {neighbor:2} (Port {port_num:2}) | {bw_mbps:4} Mbps -> {rate_pps:5} PPS")
        except Exception as e:
            logging.error(f"[{self.switch_name}] -> {neighbor} (Port {port_num}) rate write failed: {e}")


# Module 2: main loop
if __name__ == "__main__":
    if not os.path.exists('topology.json') or not os.path.exists('p4app.json'):
        logging.error("config not found; run sudo p4run first.")
        sys.exit(1)

    topo_json_obj = load_topo('topology.json')

    with open('p4app.json', 'r') as f:
        p4app_data = json.load(f)

    # link bw lookup (both directions)
    bw_map = {}
    for link in p4app_data.get('topology', {}).get('links', []):
        if len(link) >= 3 and 'bw' in link[2]:
            u, v = link[0], link[1]
            bw = link[2]['bw']
            bw_map[(u, v)] = bw
            bw_map[(v, u)] = bw

    PACKET_SIZE_BITS = 1450 * 8
    MAX_QUEUE_PKTS = 64

    logging.info("=== applying per-port rate limits (from p4app.json, scale 0.8) ===")

    limiters = {}
    scale = 0.8
    for sw_name in topo_json_obj.get_p4switches():
        limiters[sw_name] = SwitchRateLimiter(sw_name, topo_json_obj)

    for sw_name, limiter in limiters.items():
        if not limiter.api:
            continue

        for neighbor in topo_json_obj.get_neighbors(sw_name):
            if (sw_name, neighbor) in bw_map:

                raw_bw_mbps = bw_map[(sw_name, neighbor)] * scale

                limiter.apply_link_limit(
                    neighbor=neighbor,
                    bw_mbps=raw_bw_mbps,
                    packet_size_bits=PACKET_SIZE_BITS,
                    max_queue_pkts=MAX_QUEUE_PKTS
                )

    logging.info("-" * 50)
    logging.info("all bandwidth limits written to BMv2.")