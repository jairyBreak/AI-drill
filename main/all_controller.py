import sys
import os
import json
import subprocess
import math
import logging
import networkx as nx
from functools import reduce
from typing import Dict, List, Tuple, Any, Set, Optional

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

# ---- capacity-clustering params (how spines are grouped into W-ECMP components) ----
# Cluster similar-capacity spines into one component so intra-group DRILL is valid
# (DRILL equalizes queue depth, only ≈ util when capacities match). Symmetric topo -> same 4 pairs;
# all-distinct bw -> adjacent pairs (still keeps a DRILL second choice, vs 8 singletons = pure W-ECMP).
GROUP_MIN_SIZE = 2     # min ports/group (>=2 = a real DRILL second choice)
GROUP_MAX_SIZE = 2     # max ports/group (=2 -> adjacent-capacity pairs)
GROUP_BW_TOL   = 1.5   # intra-group max/min capacity ratio tolerance (only when GROUP_MAX_SIZE>2)
WEIGHT_BASE_MIN = 3    # target weight of smallest group; keeps member count small (coprime caps would explode)

# Module 1: topology + hardware-mapping analyzer
class TopologyAnalyzer:
    def __init__(self, p4app_data: Dict[str, Any], topo_json_obj: Any):
        self.topo_data = p4app_data.get('topology', {})
        self.topo_json = topo_json_obj
        
        self.G = nx.DiGraph()
        self.Q = nx.DiGraph()
        self.leaf_switches: Set[str] = set()
        self.leaf_to_ip: Dict[str, str] = {}
        
        self._build_graph()
        self._map_leaf_to_ip()
        self._build_quiver()

    def _build_graph(self) -> None:
        """Parse bandwidths from p4app.json into a bidirectional graph."""
        for link in self.topo_data.get('links', []):
            if len(link) < 2:
                continue
            u, v = link[0], link[1]
            bw = link[2].get('bw', 1.0) if len(link) > 2 else 1.0
            self.G.add_edge(u, v, bw=bw)
            self.G.add_edge(v, u, bw=bw)

    def _map_leaf_to_ip(self) -> None:
        """Map each leaf switch to its host IP."""
        for host, info in self.topo_data.get('hosts', {}).items():
            ip = info.get('ip', '').split('/')[0]
            if not ip:
                continue
            for neighbor in self.G.neighbors(host):
                if str(neighbor).startswith('l'):
                    self.leaf_switches.add(neighbor)
                    self.leaf_to_ip[neighbor] = ip

    def _build_quiver(self) -> None:
        """Build a directed graph annotated with capacity factor (CF)."""
        import networkx as nx
        for u, v in self.G.edges():
            self.Q.add_edge(u, v, labels=set())
            
        for src in self.leaf_switches:
            for dst in self.leaf_switches:
                if src == dst: 
                    continue
                try:
                    for path in nx.all_shortest_paths(self.G, source=src, target=dst):
                        bottleneck_bw = float('inf')
                        for i in range(len(path) - 1):
                            a, b = path[i], path[i+1]
                            link_bw = self.G[a][b]['bw']
                            cf = float('inf') if i == 0 else bottleneck_bw / link_bw
                            bottleneck_bw = min(bottleneck_bw, link_bw)
                            self.Q[a][b]['labels'].add(f"{src}->{dst}_CF:{cf}")
                except nx.NetworkXNoPath:
                    logging.warning(f"no path: {src} -> {dst}")

    def _cluster_by_capacity(self, ordered_nhops: List[str],
                             nh_cap: Dict[str, float]) -> List[List[str]]:
        """Cut capacity-sorted next-hops into contiguous groups (optimal 1-D partition is contiguous).

        >=GROUP_MIN_SIZE ports/group; keep capacities close within GROUP_MAX_SIZE/GROUP_BW_TOL;
        a trailing singleton merges into the previous group (never a 1-port group)."""
        groups: List[List[str]] = []
        cur: List[str] = []
        for nh in ordered_nhops:
            if not cur:
                cur = [nh]
                continue
            caps  = [nh_cap[m] for m in cur] + [nh_cap[nh]]
            ratio = max(caps) / min(caps) if min(caps) > 0 else float('inf')
            if len(cur) < GROUP_MIN_SIZE:
                cur.append(nh)                       # below min size: must include
            elif len(cur) < GROUP_MAX_SIZE and ratio <= GROUP_BW_TOL:
                cur.append(nh)                       # within tolerance and below max: keep growing
            else:
                groups.append(cur)                   # close and start a new group
                cur = [nh]
        if cur:
            if len(cur) < GROUP_MIN_SIZE and groups:
                groups[-1].extend(cur)               # trailing singleton -> merge into previous
            else:
                groups.append(cur)
        return groups

    def get_ecmp_weights_and_rules(self, src_leaf: str, dst_leaf: str) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Compute W-ECMP component weights + P4 rules via capacity clustering (weight ∝ group capacity)."""
        import networkx as nx
        try:
            paths = list(nx.all_shortest_paths(self.G, source=src_leaf, target=dst_leaf))
        except nx.NetworkXNoPath:
            return [], []

        # path-bottleneck capacity per next-hop spine (min across shared shortest paths)
        nh_cap: Dict[str, float] = {}
        for path in paths:
            if len(path) < 2:
                continue
            nh = path[1]
            bott = min(self.G[path[i]][path[i+1]]['bw'] for i in range(len(path) - 1))
            nh_cap[nh] = min(nh_cap.get(nh, float('inf')), bott)
        if not nh_cap:
            return [1], []

        # sort by (capacity, port) for determinism, then cluster
        ordered = sorted(nh_cap,
                         key=lambda nh: (nh_cap[nh],
                                         self.topo_json.node_to_node_port_num(src_leaf, nh)))
        groups = self._cluster_by_capacity(ordered, nh_cap)

        weights_float: List[float] = []
        hardware_rules: List[Dict[str, Any]] = []
        for comp_idx, members in enumerate(groups):
            comp_id = comp_idx + 1
            ports_and_macs = []
            cap_sum = 0.0
            for nh in members:
                port = self.topo_json.node_to_node_port_num(src_leaf, nh)
                mac  = self.topo_json.node_to_node_mac(nh, src_leaf)
                ports_and_macs.append((port, mac))
                cap_sum += nh_cap[nh]
            ports_and_macs.sort(key=lambda x: x[0])

            weights_float.append(cap_sum)            # group weight ∝ aggregate capacity
            hardware_rules.append({
                'comp_id': comp_id,
                'num_nhops': len(ports_and_macs),
                'base_port': ports_and_macs[0][0] if ports_and_macs else 0,
                'ports_and_macs': ports_and_macs,
            })

        if not weights_float:
            return [1], []

        # reduce to smallest-integer ratio (scale by min-group cap); avoids coprime blowup (e.g. [13,17,21,25]=76 members)
        base_cap = min(weights_float)
        simplified_weights = [max(1, int(round(w / base_cap * WEIGHT_BASE_MIN)))
                              for w in weights_float]

        return simplified_weights, hardware_rules
# Module 2: switch controller (batched command buffer)
class LeafController:
    def __init__(self, switch_name, topo_json_obj):
        self.switch_name = switch_name
        self.thrift_port = topo_json_obj.get_thrift_port(switch_name)
        self.cli_command_buffer = []
        self.api = None
        try:
            self.api = SimpleSwitchThriftAPI(self.thrift_port)
        except BaseException:
            pass

    def buffer_cli_cmd(self, command):
        """Buffer a command instead of running it immediately."""
        self.cli_command_buffer.append(command)

    def commit_cli_cmds(self):
        """Flush the whole buffer through a single child process (one subprocess per switch)."""
        if not self.cli_command_buffer:
            return

        batch_script = "\n".join(self.cli_command_buffer) + "\n"

        subprocess.run(
            ['simple_switch_CLI', '--thrift-port', str(self.thrift_port)],
            input=batch_script,
            text=True,
            capture_output=True
        )

        self.cli_command_buffer.clear()

    def set_w_ecmp_weights(self, target_ip, weights_list, hardware_rules):
        if not self.api:
            return
        weight_str = ":".join(map(str, weights_list))
        print(f"  -> target IP: {target_ip} | weights = {weight_str} | paths: {len(hardware_rules)}")

        # 1. buffer P4 forwarding-table commands
        for rule in hardware_rules:
            c_id = rule['comp_id']
            self.buffer_cli_cmd(f"table_add drill_params_table run_drill {c_id} => {rule['num_nhops']}")
            for logical_idx, (port, mac) in enumerate(rule['ports_and_macs']):
                map_address = c_id * 16 + logical_idx  # (c_id * 16) + logical_idx
                try:
                    self.api.register_write("port_map_reg", map_address, port)
                except Exception as e:
                    print(f"      [error] port map register write failed: {e}")
                self.buffer_cli_cmd(f"table_add ecmp_group_to_nhop set_nhop {c_id} {port} => {mac} {port}")

        # 2. push W-ECMP selector weights via Thrift (fast RPC, kept direct)
        selector_name = "w_ecmp_selector"
        table_name = "w_ecmp_table"
        action_name = "assign_component"

        grp_handle = self.api.act_prof_create_group(selector_name)

        for comp_index, weight in enumerate(weights_list):
            comp_id = str(comp_index + 1)
            for _ in range(weight):
                mbr_handle = self.api.act_prof_create_member(selector_name, action_name, [comp_id])
                self.api.act_prof_add_member_to_group(selector_name, mbr_handle, grp_handle)

        # 3. buffer the group-binding command
        self.buffer_cli_cmd(f"table_indirect_add_with_group {table_name} {target_ip} => {grp_handle}")


# Module 3: shared installer (used by main + realtime_ml_controller at startup)
def clear_leaf_forwarding(api):
    """Clear a leaf's existing W-ECMP forwarding (3 tables + action-profile groups/members)."""
    grp_handles = set()
    try:
        for e in api.table_get_entries("w_ecmp_table", False):
            try:
                grp_handles.add(e.action_data.action_params[0])
            except Exception:
                pass
    except Exception:
        pass

    for t in ("w_ecmp_table", "drill_params_table", "ecmp_group_to_nhop"):
        try:
            api.table_clear(t)
        except Exception:
            pass

    for gh in grp_handles:
        try:
            info = api.act_prof_get_group("w_ecmp_selector", gh)
            api.act_prof_delete_group("w_ecmp_selector", gh)
            for mh in info.member_handles:
                try:
                    api.act_prof_delete_member("w_ecmp_selector", mh)
                except Exception:
                    pass
        except Exception:
            pass


def install_ecmp_drill_rules(p4app_data, topo_json_obj, clear_first=True, verbose=True):
    """Install the 4-component W-ECMP+DRILL rules on every leaf pair (single source of truth)."""
    analyzer = TopologyAnalyzer(p4app_data, topo_json_obj)
    leaves = sorted(analyzer.leaf_switches)
    controllers = {leaf: LeafController(leaf, topo_json_obj) for leaf in leaves}

    for src_leaf in leaves:
        ctrl = controllers[src_leaf]
        if ctrl.api is None:
            continue
        if clear_first:
            clear_leaf_forwarding(ctrl.api)
        for dst_leaf in leaves:
            if src_leaf == dst_leaf:
                continue
            weights_list, hardware_rules = analyzer.get_ecmp_weights_and_rules(src_leaf, dst_leaf)
            target_ip = analyzer.leaf_to_ip[dst_leaf]
            ctrl.set_w_ecmp_weights(target_ip, weights_list, hardware_rules)
        ctrl.commit_cli_cmds()
        if verbose:
            print(f"[{src_leaf}] hardware rules written (batched)")


def print_grouping_summary(p4app_data, topo_json_obj, src_leaf="l1", dst_leaf="l2",
                           out_file=".grouping_summary.txt"):
    """Print + save the W-ECMP capacity grouping for a leaf pair (read-only, writes no hardware)."""
    analyzer = TopologyAnalyzer(p4app_data, topo_json_obj)

    # port -> (spine name, link bw)
    port_info = {}
    for nh in analyzer.G.neighbors(src_leaf):
        if not str(nh).startswith('s'):
            continue
        try:
            port = analyzer.topo_json.node_to_node_port_num(src_leaf, nh)
            port_info[port] = (nh, analyzer.G[src_leaf][nh]['bw'])
        except Exception:
            pass

    weights, rules = analyzer.get_ecmp_weights_and_rules(src_leaf, dst_leaf)

    lines = ["=" * 60,
             f" W-ECMP 容量分群結果 ({src_leaf} -> {dst_leaf})  —— 組內由 DRILL 逐封包選最短佇列",
             "=" * 60]
    if not rules:
        lines.append("  (無可用路徑)")
    else:
        for w, rule in zip(weights, rules):
            parts = []
            for port, _ in rule['ports_and_macs']:
                name, bw = port_info.get(port, (f"port{port}", 0.0))
                parts.append(f"{name}({bw:g})")
            mode = "DRILL" if rule['num_nhops'] > 1 else "單埠"
            lines.append(f"  Component {rule['comp_id']} | {'  '.join(parts):28} | 權重 {w} | {mode}")
        lines.append(f"  權重比例 = {':'.join(map(str, weights))}")
    lines.append("=" * 60)

    text = "\n".join(lines)
    print("\n" + text + "\n", flush=True)
    if out_file:
        try:
            with open(out_file, "w") as f:
                f.write(text + "\n")
        except Exception:
            pass


# Module 4: main control loop (batched)
if __name__ == "__main__":
    if not os.path.exists('p4app.json') or not os.path.exists('topology.json'):
        print("[錯誤] 找不到設定檔，請確認已執行 sudo p4run。")
        exit(1)

    with open('p4app.json', 'r') as f:
        p4app_data = json.load(f)
    topo_json_obj = load_topo('topology.json')

    # --summary: read-only grouping printout, installs nothing
    if "--summary" in sys.argv:
        print_grouping_summary(p4app_data, topo_json_obj)
        sys.exit(0)

    print("\n===========================================")
    print(" 啟動全網自適應 W-ECMP 控制器 (批次寫入優化版)")
    print("===========================================\n")

    install_ecmp_drill_rules(p4app_data, topo_json_obj, clear_first=True, verbose=True)

    print("\n全網拓樸批次配置完畢！")
