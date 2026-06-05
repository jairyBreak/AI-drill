"""
baseline_common.py — 共用邏輯：靜態 ECMP / W-ECMP 基準演算法的安裝與量測

這兩個基準演算法與正式的 W-ECMP+DRILL+ML 控制器跑在「完全相同」的 P4 程式上，
差別只在控制平面設定，確保比較公平 (相同 dataplane / INT / counter / 量測路徑)：

  * ECMP   : 8 個單埠 component，每個權重相同 (member 數=1)，5-tuple 均勻雜湊到 8 個 spine
  * W-ECMP : 8 個單埠 component，member 數正比於頻寬 (0.6/0.8/1.0/1.2 -> 3/4/5/6)

因為每個 component 只有 1 個 port (num_nhops=1)，dataplane 的 run_drill 沒有第二個
選擇，DRILL 自然被停用 —— 純粹的 (加權) ECMP 雜湊。

量測沿用既有 plot_1s_metrics.py 的 CSV 格式：
  Timestamp, Pred_Lat, Real_Lat, Pred_Loss, Real_Loss, Util_Sum, Total_Mbps
基準沒有 ML 預測，故 Pred_* 欄位為 NaN。
"""

import os
import sys
import json
import math
import time
from functools import reduce
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import networkx as nx

# 載入 P4-Utils (與其他腳本一致)
P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

from p4utils.utils.helper import load_topo

from all_controller import TopologyAnalyzer, LeafController, clear_leaf_forwarding
from realtime_1s_predictor_topo_indep import (
    Realtime1sPredictorTopoIndep, PORTS, CAPACITY,
)


# ==========================================
# 控制平面安裝：單埠 component (停用 DRILL)
# ==========================================
def build_single_port_config(analyzer, src_leaf, dst_leaf, mode):
    """為 src_leaf->dst_leaf 產生 8 個單埠 component 的規則與權重。

    mode == 'equal' : 全部權重 = 1 (純 ECMP)
    mode == 'bw'    : 權重正比於上行頻寬 (W-ECMP)
    回傳 (weights_list, hardware_rules)，格式相容於 LeafController.set_w_ecmp_weights。
    """
    try:
        paths = list(nx.all_shortest_paths(analyzer.G, source=src_leaf, target=dst_leaf))
    except nx.NetworkXNoPath:
        return [], []

    # 蒐集所有最短路徑的第一跳 (spine)，依實體 port 排序確保決定性
    next_hops = {p[1] for p in paths if len(p) > 1}
    ordered = sorted(
        next_hops,
        key=lambda nh: analyzer.topo_json.node_to_node_port_num(src_leaf, nh),
    )

    rules, weights = [], []
    for idx, nh in enumerate(ordered):
        port = analyzer.topo_json.node_to_node_port_num(src_leaf, nh)
        mac = analyzer.topo_json.node_to_node_mac(nh, src_leaf)
        bw = analyzer.G[src_leaf][nh]['bw']
        rules.append({
            'comp_id': idx + 1,
            'num_nhops': 1,                 # 單埠 -> run_drill 無第二選擇 -> DRILL 停用
            'base_port': port,
            'ports_and_macs': [(port, mac)],
        })
        weights.append(bw if mode == 'bw' else 1.0)

    if mode == 'bw':
        # 化為最簡整數比 (與 all_controller 相同手法)
        wi = [max(1, int(round(w * 10))) for w in weights]
        g = reduce(math.gcd, wi)
        weights = [w // g for w in wi]
    else:
        weights = [1 for _ in weights]

    return weights, rules


def build_drill_config(analyzer, src_leaf, dst_leaf):
    """為 src_leaf->dst_leaf 產生「單一 component 含全部 spine」的純 DRILL 規則。

    與 build_single_port_config 相反：不是每個 next-hop 一個單埠 component，而是把
    全部 next-hop 塞進一個 component (comp_id=1, num_nhops=N)，W-ECMP selector 只有
    一個 member。如此每個封包都進入這個 component，run_drill 在全部 N 個 spine 中
    隨機抽 2 個 + 記憶埠，挑佇列最短者 —— 純 DRILL，無分組、無加權。
    回傳 (weights_list, hardware_rules)，格式相容於 LeafController.set_w_ecmp_weights。
    """
    try:
        paths = list(nx.all_shortest_paths(analyzer.G, source=src_leaf, target=dst_leaf))
    except nx.NetworkXNoPath:
        return [], []

    # 蒐集所有最短路徑的第一跳 (spine)，依實體 port 排序確保決定性 (與其他 builder 一致)
    next_hops = {p[1] for p in paths if len(p) > 1}
    ordered = sorted(
        next_hops,
        key=lambda nh: analyzer.topo_json.node_to_node_port_num(src_leaf, nh),
    )
    if not ordered:
        return [], []

    ports_and_macs = [
        (analyzer.topo_json.node_to_node_port_num(src_leaf, nh),
         analyzer.topo_json.node_to_node_mac(nh, src_leaf))
        for nh in ordered
    ]

    rules = [{
        'comp_id': 1,
        'num_nhops': len(ports_and_macs),   # 全部 spine 在同一 component -> DRILL 跨全部選最短佇列
        'base_port': ports_and_macs[0][0],
        'ports_and_macs': ports_and_macs,
    }]
    weights = [1]   # selector 單一 member -> 所有 flow 都進 comp_id 1

    return weights, rules


def install_baseline(mode):
    """在所有 leaf pair 上安裝 ECMP / W-ECMP 靜態轉發 (取代 all_controller 的 4-component 設定)。"""
    if not (os.path.exists('p4app.json') and os.path.exists('topology.json')):
        print("[錯誤] 找不到 p4app.json / topology.json，請先執行 sudo p4run。")
        sys.exit(1)

    with open('p4app.json') as f:
        p4app = json.load(f)
    topo = load_topo('topology.json')
    analyzer = TopologyAnalyzer(p4app, topo)
    leaves = sorted(analyzer.leaf_switches)

    if mode == 'drill':
        label = "DRILL (純 DRILL，跨所有 spine 逐封包選最短佇列)"
        subtitle = "單一多埠 component，DRILL 啟用"
    elif mode == 'bw':
        label = "W-ECMP (頻寬加權)"
        subtitle = "單埠 component，DRILL 停用"
    else:
        label = "ECMP (等權)"
        subtitle = "單埠 component，DRILL 停用"
    print("\n" + "=" * 60)
    print(f" 安裝基準演算法：{label}  —— {subtitle}")
    print("=" * 60 + "\n")

    # Thrift API (act_prof_create_member 等) 會狂印 log，靜音之，只留我們自己的摘要
    with open(os.devnull, 'w') as _dn:
        for src in leaves:
            with redirect_stdout(_dn), redirect_stderr(_dn):
                ctrl = LeafController(src, topo)
                api_ok = ctrl.api is not None
                if api_ok:
                    clear_leaf_forwarding(ctrl.api)
                    for dst in leaves:
                        if src == dst:
                            continue
                        if mode == 'drill':
                            weights, rules = build_drill_config(analyzer, src, dst)
                        else:
                            weights, rules = build_single_port_config(analyzer, src, dst, mode)
                        if not rules:
                            continue
                        ip = analyzer.leaf_to_ip[dst]
                        ctrl.set_w_ecmp_weights(ip, weights, rules)
                    ctrl.commit_cli_cmds()

            if api_ok:
                print(f"  [{src}] 已安裝 ({len(leaves) - 1} 個目標)")
            else:
                print(f"  [警告] 無法連線 {src}，跳過")

    print("\n=== 基準演算法安裝完畢 ===\n")


# ==========================================
# 量測：沿用既有 CSV 格式 (plot_1s_metrics.py)
# ==========================================
def run_measurement(label, duration, csv_path):
    """被動量測 1s 硬體指標，寫入與 plot_1s_metrics 相同欄位的 CSV。

    基準無 ML 預測，故 Pred_Lat / Pred_Loss 留 NaN。
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 重用既有被動採集器 (collect_1s_data 不需模型；缺模型只會印警告)。
    # 其 register_read / counter_read 會狂印 Thrift log，全程靜音。
    _dn = open(os.devnull, 'w')
    with redirect_stdout(_dn), redirect_stderr(_dn):
        predictor = Realtime1sPredictorTopoIndep()

    results = []
    start = time.time()
    print("\n" + "=" * 60)
    print(f" [量測 {label}] 時長 {duration}s  ->  {csv_path}")
    print("=" * 60)
    print(f"{'時間':^10} | {'Lat(ms)':^9} | {'Loss(%)':^9} | {'Util':^6} | {'Mbps':^6}")
    print("-" * 55)

    try:
        while time.time() - start < duration:
            with redirect_stdout(_dn), redirect_stderr(_dn):
                row = predictor.collect_1s_data()
            per_switch_util = {f'util_s{p-1}': row[f'src1_port{p}_mbps'] / CAPACITY[p]
                               for p in PORTS}
            total_mbps = sum(row[f'src1_port{p}_mbps'] for p in PORTS)
            util_sum = sum(per_switch_util.values())
            now = datetime.now()

            results.append({
                'Timestamp': now,
                'Pred_Lat': np.nan,
                'Real_Lat': row['Real_HW_Latency_ms'],
                'Pred_Loss': np.nan,
                'Real_Loss': row['Real_HW_Loss_Percent'],
                'Util_Sum': util_sum,
                'Total_Mbps': total_mbps,
                'Cum_Enq': row['Cum_Enq'],
                'Cum_Recv': row['Cum_Recv'],
                **per_switch_util,
            })

            print(f"\r {now.strftime('%H:%M:%S'):^10} | "
                  f"{row['Real_HW_Latency_ms']:7.1f} | "
                  f"{row['Real_HW_Loss_Percent']:7.1f} | "
                  f"{util_sum:5.2f} | {total_mbps:5.2f}",
                  end='', flush=True)
    except KeyboardInterrupt:
        print("\n[已中止]")
    finally:
        _dn.close()

    if not results:
        print("沒有採集到任何數據。")
        return

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\n已寫入 {len(df)} 列 -> {csv_path}")


def run_baseline(mode, label, duration, csv_path):
    """一鍵：安裝基準 -> 量測 -> 寫 CSV。"""
    install_baseline(mode)
    run_measurement(label, duration, csv_path)
