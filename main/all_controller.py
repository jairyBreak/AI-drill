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

# ==========================================
# 容量分群參數 (W-ECMP component 如何把 spine 分組)
# ==========================================
# 把容量「相近」的 spine 分到同一個 W-ECMP component，組內交給 dataplane DRILL 逐封包選最短佇列。
# 理由：DRILL 均衡的是「佇列深度」而非「利用率」，唯有組內容量相近時「最短佇列 ≈ 最低利用率」才成
# 立，且 DRILL 能用佇列反應自動補償小幅容量差。對等拓樸 (成對相同 bw) 會還原成原本的 4 組；當 8 個
# spine 全不同 bw 時則退化為「排序後鄰接配對」，仍保住組內 DRILL —— 不像嚴格對稱 DRILL 會因為找不到
# 完全相同容量而拆成 8 個單埠組 (失去微負載平衡、等同純 W-ECMP)。
GROUP_MIN_SIZE = 2     # 每組至少幾個埠 (>=2 才有 DRILL 的第二選擇；單埠組 = 無微負載平衡)
GROUP_MAX_SIZE = 2     # 每組最多幾個埠：=2 -> 容量相鄰配對 (預設)。調大可在容差內合併更多埠 (更多 DRILL 選擇，但組內容量差變大)
GROUP_BW_TOL   = 1.5   # 組內容量 max/min 比容差 (僅 GROUP_MAX_SIZE>2 時生效，決定是否續長同一組)
WEIGHT_BASE_MIN = 3    # 權重化為最小整數比時，最小組的目標權重。讓總成員數維持在數十以內，
                       # 避免各組容量互質時 (例如 1.3:1.7:2.1:2.5) gcd 約不掉 -> 權重/成員數爆量、
                       # 每次 rehash 要建很多 selector member 而拖慢控制迴圈。

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

    def _cluster_by_capacity(self, ordered_nhops: List[str],
                             nh_cap: Dict[str, float]) -> List[List[str]]:
        """把『已依容量排序』的 next-hop 切成連續群組 (1-D 分群最佳解必為排序後的連續區間)。

        規則：每組至少 GROUP_MIN_SIZE 埠 (保住組內 DRILL)；在 GROUP_MAX_SIZE 與 GROUP_BW_TOL
        (組內 max/min 容量比) 約束下盡量讓組內容量相近。預設 GROUP_MAX_SIZE=2 -> 鄰接配對，
        對等拓樸還原成原本的成對分組；尾端落單則併入前一組，避免出現失去 DRILL 的單埠組。
        """
        groups: List[List[str]] = []
        cur: List[str] = []
        for nh in ordered_nhops:
            if not cur:
                cur = [nh]
                continue
            caps  = [nh_cap[m] for m in cur] + [nh_cap[nh]]
            ratio = max(caps) / min(caps) if min(caps) > 0 else float('inf')
            if len(cur) < GROUP_MIN_SIZE:
                cur.append(nh)                       # 還沒達最小組大小：必須收進來
            elif len(cur) < GROUP_MAX_SIZE and ratio <= GROUP_BW_TOL:
                cur.append(nh)                       # 容差內且未達上限：可續長同一組
            else:
                groups.append(cur)                   # 收尾，另起新組
                cur = [nh]
        if cur:
            if len(cur) < GROUP_MIN_SIZE and groups:
                groups[-1].extend(cur)               # 尾端落單 -> 併入前一組
            else:
                groups.append(cur)
        return groups

    def get_ecmp_weights_and_rules(self, src_leaf: str, dst_leaf: str) -> Tuple[List[int], List[Dict[str, Any]]]:
        """計算 W-ECMP component 的權重與 P4 硬體規則 (容量分群版)。

        不再以『完全相同的瓶頸簽章』分組 (那會在 8 個 bw 皆不同時拆成 8 個單埠組、失去 DRILL)，
        改為：把每個 next-hop 依路徑瓶頸容量排序，呼叫 _cluster_by_capacity 做鄰接分群，組內容量
        最相近者在一起；組權重 ∝ 組總容量。對等拓樸 (成對相同 bw) 的結果與舊版一致 ([3,4,5,6])。
        """
        import networkx as nx
        try:
            paths = list(nx.all_shortest_paths(self.G, source=src_leaf, target=dst_leaf))
        except nx.NetworkXNoPath:
            return [], []

        # 每個 next-hop (spine) 的路徑瓶頸容量 (此 leaf-spine 拓樸即上行鏈路 bw)；
        # 若多條最短路徑共用同一 next-hop，取較小的瓶頸以保守估計。
        nh_cap: Dict[str, float] = {}
        for path in paths:
            if len(path) < 2:
                continue
            nh = path[1]
            bott = min(self.G[path[i]][path[i+1]]['bw'] for i in range(len(path) - 1))
            nh_cap[nh] = min(nh_cap.get(nh, float('inf')), bott)
        if not nh_cap:
            return [1], []

        # 依 (容量, 實體 port) 排序確保決定性，再做容量鄰接分群
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

            weights_float.append(cap_sum)            # 組權重 ∝ 組總容量
            hardware_rules.append({
                'comp_id': comp_id,
                'num_nhops': len(ports_and_macs),
                'base_port': ports_and_macs[0][0] if ports_and_macs else 0,
                'ports_and_macs': ports_and_macs,
            })

        if not weights_float:
            return [1], []

        # 化為「最小整數比」：以最小組容量為基準縮放後四捨五入。不用 ×10+gcd，因為當各組容量互質
        # (例如 1.3:1.7:2.1:2.5) 時 gcd=1 約不掉，權重會變 [13,17,21,25] = 76 個 selector member，
        # 讓每次 rehash 都要建一堆 member、拖慢控制迴圈。改用此法權重恆維持小整數 (此例 -> [3,4,5,6])。
        base_cap = min(weights_float)
        simplified_weights = [max(1, int(round(w / base_cap * WEIGHT_BASE_MIN)))
                              for w in weights_float]

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


def print_grouping_summary(p4app_data, topo_json_obj, src_leaf="l1", dst_leaf="l2",
                           out_file=".grouping_summary.txt"):
    """印出 (並存檔) 某 leaf pair 的 W-ECMP 容量分群結果 (唯讀，不寫任何硬體)。

    重算 get_ecmp_weights_and_rules，列出每個 component 內含哪些 spine、各自頻寬與組權重。
    同時把同一份內容寫入 out_file，讓 start_env.sh 即使在終端被 p4run 輸出洗版時，仍有一份
    乾淨可隨時 cat 的結果 (容量相近者成組、組權重 ∝ 組總容量)。
    """
    analyzer = TopologyAnalyzer(p4app_data, topo_json_obj)

    # port -> (spine 名稱, 連結頻寬)
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

    # --summary：唯讀印出分群結果，不安裝任何規則 (供 start_env.sh 最後顯示)
    if "--summary" in sys.argv:
        print_grouping_summary(p4app_data, topo_json_obj)
        sys.exit(0)

    print("\n===========================================")
    print(" 啟動全網自適應 W-ECMP 控制器 (批次寫入優化版)")
    print("===========================================\n")

    install_ecmp_drill_rules(p4app_data, topo_json_obj, clear_first=True, verbose=True)

    print("\n全網拓樸批次配置完畢！")
