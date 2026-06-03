import sys
import os
import json
import subprocess
import math
import logging
import networkx as nx
from functools import reduce
from typing import Dict, List, Tuple, Any, Set, Optional

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 避免硬編碼路徑，建議未來改用環境變數 PYTHONPATH
P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

try:
    from p4utils.utils.helper import load_topo
    from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI
except ImportError as e:
    logging.error(f"無法載入 P4-Utils 模組，請確認路徑正確: {e}")
    sys.exit(1)

# 模組 1: 拓樸與硬體映射分析器
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
        """從 p4app.json 解析頻寬並建立雙向圖"""
        for link in self.topo_data.get('links', []):
            if len(link) < 2:
                continue
            u, v = link[0], link[1]
            bw = link[2].get('bw', 1.0) if len(link) > 2 else 1.0
            self.G.add_edge(u, v, bw=bw)
            self.G.add_edge(v, u, bw=bw)

    def _map_leaf_to_ip(self) -> None:
        """建立 Leaf Switch 與 Host IP 的映射"""
        for host, info in self.topo_data.get('hosts', {}).items():
            ip = info.get('ip', '').split('/')[0]
            if not ip:
                continue
            for neighbor in self.G.neighbors(host):
                if str(neighbor).startswith('l'):
                    self.leaf_switches.add(neighbor)
                    self.leaf_to_ip[neighbor] = ip

    def _build_quiver(self) -> None:
        """建立附帶 Capacity Factor (CF) 的有向圖 """
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
                    logging.warning(f"無路徑可達: {src} -> {dst}")

    def get_ecmp_weights_and_rules(self, src_leaf: str, dst_leaf: str) -> Tuple[List[int], List[Dict[str, Any]]]:
        """計算權重比例，並生成 P4 硬體規則"""
        import networkx as nx
        components_dict: Dict[tuple, Dict[str, Any]] = {} 
        
        try:
            paths = list(nx.all_shortest_paths(self.G, source=src_leaf, target=dst_leaf))
        except nx.NetworkXNoPath:
            return [], []

        for path in paths:
            signature = []
            path_bottleneck = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_labels = tuple(sorted(list(self.Q[u][v]['labels'])))
                signature.append(edge_labels)
                path_bottleneck = min(path_bottleneck, self.G[u][v]['bw'])
                
            sig_key = (tuple(signature), path_bottleneck)
            if sig_key not in components_dict:
                components_dict[sig_key] = {'weight': 0.0, 'next_hops': set()}
            
            components_dict[sig_key]['weight'] += path_bottleneck
            if len(path) > 1:
                components_dict[sig_key]['next_hops'].add(path[1])

        weights_float: List[float] = []
        hardware_rules: List[Dict[str, Any]] = []
        
        for comp_idx, (sig_key, data) in enumerate(components_dict.items()):
            comp_id = comp_idx + 1
            weights_float.append(data['weight'])
            
            ports_and_macs = []
            for nh in data['next_hops']:
                port = self.topo_json.node_to_node_port_num(src_leaf, nh)
                mac = self.topo_json.node_to_node_mac(nh, src_leaf)
                ports_and_macs.append((port, mac))
            
            ports_and_macs.sort(key=lambda x: x[0])
            num_nhops = len(ports_and_macs)
            base_port = ports_and_macs[0][0] if num_nhops > 0 else 0
            
            hardware_rules.append({
                'comp_id': comp_id,
                'num_nhops': num_nhops,
                'base_port': base_port,
                'ports_and_macs': ports_and_macs
            })

        if not weights_float:
            return [1], []
            
        weights_int = [max(1, int(w * 10)) for w in weights_float]
        common_divisor = reduce(math.gcd, weights_int)
        simplified_weights = [w // common_divisor for w in weights_int]
        
        return simplified_weights, hardware_rules
# ==========================================
# 模組 2: 交換機控制器 (加入批次優化 Command Buffer)
# ==========================================
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
        """將指令加入緩衝區，不立即執行"""
        self.cli_command_buffer.append(command)

    def commit_cli_cmds(self):
        """一次性將緩衝區內的所有指令透過單一 Child Process 送出"""
        if not self.cli_command_buffer:
            return
            
        # 將所有指令用換行符號連接成單一腳本字串
        batch_script = "\n".join(self.cli_command_buffer) + "\n"
        
        # 透過 subprocess 的 input 參數一次性寫入，大幅減少 I/O 延遲
        subprocess.run(
            ['simple_switch_CLI', '--thrift-port', str(self.thrift_port)],
            input=batch_script,
            text=True,
            capture_output=True
        )
        
        # 執行完畢後清空緩衝區
        self.cli_command_buffer.clear()

    def set_w_ecmp_weights(self, target_ip, weights_list, hardware_rules):
        if not self.api:
            return
        weight_str = ":".join(map(str, weights_list))
        print(f"  -> 目標 IP: {target_ip} | 動態權重 = {weight_str} | 生成路徑: {len(hardware_rules)} 條")
        
        # 1. 緩衝 P4 底層轉發表指令
        for rule in hardware_rules:
            c_id = rule['comp_id']
            self.buffer_cli_cmd(f"table_add drill_params_table run_drill {c_id} => {rule['num_nhops']}")
            for logical_idx, (port, mac) in enumerate(rule['ports_and_macs']):
                # 計算 Mapping Address: (c_id * 16) + logical_idx
                map_address = c_id * 16 + logical_idx
                # 透過 Thrift 寫入 Register
                try:
                    self.api.register_write("port_map_reg", map_address, port)
                except Exception as e:
                    print(f"      [錯誤] 無法寫入 port map register: {e}")
                # 寫入下一跳 MAC 轉發表 (保持不變)
                self.buffer_cli_cmd(f"table_add ecmp_group_to_nhop set_nhop {c_id} {port} => {mac} {port}")

        # 2. 透過 Thrift API 下發 W-ECMP Selector 機率權重 (Thrift 本身是快速的 RPC，維持直接呼叫)
        selector_name = "w_ecmp_selector"
        table_name = "w_ecmp_table"
        action_name = "assign_component"

        grp_handle = self.api.act_prof_create_group(selector_name)

        for comp_index, weight in enumerate(weights_list):
            comp_id = str(comp_index + 1)
            for _ in range(weight):
                mbr_handle = self.api.act_prof_create_member(selector_name, action_name, [comp_id])
                self.api.act_prof_add_member_to_group(selector_name, mbr_handle, grp_handle)

        # 3. 緩衝綁定 Group 的指令
        self.buffer_cli_cmd(f"table_indirect_add_with_group {table_name} {target_ip} => {grp_handle}")


# ==========================================
# 模組 3: 共用安裝程序 (供主程式與 realtime_ml_controller 啟動時共用)
# ==========================================
def clear_leaf_forwarding(api):
    """清掉某 leaf 既有的 W-ECMP 轉發 (三張表 + action profile group/member)。

    讓安裝從乾淨狀態開始、且可重複執行；切換演算法 (ECMP / W-ECMP / ML) 時尤其重要，
    否則殘留的舊 component 設定會與新設定衝突。
    """
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
    """在所有 leaf pair 安裝正式的 4-component W-ECMP+DRILL 轉發規則。

    這是 4-component 設定的唯一來源 (single source of truth)，被 all_controller 主程式
    與 realtime_ml_controller 啟動時共用，確保不論前一個跑的是哪個基準演算法，狀態都正確。
    """
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
            print(f"[{src_leaf}] 批次寫入硬體規則")


# ==========================================
# 模組 4: 主控制迴圈 (批次執行版)
# ==========================================
if __name__ == "__main__":
    if not os.path.exists('p4app.json') or not os.path.exists('topology.json'):
        print("[錯誤] 找不到設定檔，請確認已執行 sudo p4run。")
        exit(1)

    with open('p4app.json', 'r') as f:
        p4app_data = json.load(f)
    topo_json_obj = load_topo('topology.json')

    print("\n===========================================")
    print(" 啟動全網自適應 W-ECMP 控制器 (批次寫入優化版)")
    print("===========================================\n")

    install_ecmp_drill_rules(p4app_data, topo_json_obj, clear_first=True, verbose=True)

    print("\n全網拓樸批次配置完畢！")
