"""
baseline_common.py — install + measure for the static ECMP / W-ECMP / DRILL baselines.

Same P4 program as the full controller; only the control-plane config differs (fair comparison).
  * ECMP   : 8 single-port components, equal weights
  * W-ECMP : 8 single-port components, weights ∝ bandwidth
  * DRILL  : 1 component with all spines (num_nhops=N), capacity-blind
Single-port components have no DRILL second choice, so DRILL is disabled there.
CSV format matches plot_1s_metrics.py; baselines have no ML, so Pred_* are NaN.
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

P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

from p4utils.utils.helper import load_topo

from all_controller import TopologyAnalyzer, LeafController, clear_leaf_forwarding
from realtime_1s_predictor_topo_indep import (
    Realtime1sPredictorTopoIndep, PORTS, CAPACITY,
)


# Control-plane install: single-port components (DRILL disabled)
def build_single_port_config(analyzer, src_leaf, dst_leaf, mode):
    """Build 8 single-port components. mode 'equal' -> all weight 1 (ECMP); 'bw' -> ∝ uplink bw (W-ECMP)."""
    try:
        paths = list(nx.all_shortest_paths(analyzer.G, source=src_leaf, target=dst_leaf))
    except nx.NetworkXNoPath:
        return [], []

    # first hops (spines), sorted by physical port for determinism
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
            'num_nhops': 1,                 # single port -> no DRILL second choice -> disabled
            'base_port': port,
            'ports_and_macs': [(port, mac)],
        })
        weights.append(bw if mode == 'bw' else 1.0)

    if mode == 'bw':
        # reduce to smallest integer ratio
        wi = [max(1, int(round(w * 10))) for w in weights]
        g = reduce(math.gcd, wi)
        weights = [w // g for w in wi]
    else:
        weights = [1 for _ in weights]

    return weights, rules


def build_drill_config(analyzer, src_leaf, dst_leaf):
    """Build one component containing all spines (comp_id=1, num_nhops=N) -> pure capacity-blind DRILL."""
    try:
        paths = list(nx.all_shortest_paths(analyzer.G, source=src_leaf, target=dst_leaf))
    except nx.NetworkXNoPath:
        return [], []

    # first hops (spines), sorted by physical port for determinism
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
        'num_nhops': len(ports_and_macs),   # all spines in one component -> DRILL across all
        'base_port': ports_and_macs[0][0],
        'ports_and_macs': ports_and_macs,
    }]
    weights = [1]   # single selector member -> all flows enter comp_id 1

    return weights, rules


def install_baseline(mode):
    """Install static ECMP / W-ECMP / DRILL forwarding on every leaf pair."""
    if not (os.path.exists('p4app.json') and os.path.exists('topology.json')):
        print("[error] p4app.json / topology.json not found; run sudo p4run first.")
        sys.exit(1)

    with open('p4app.json') as f:
        p4app = json.load(f)
    topo = load_topo('topology.json')
    analyzer = TopologyAnalyzer(p4app, topo)
    leaves = sorted(analyzer.leaf_switches)

    if mode == 'drill':
        label = "DRILL (pure DRILL, shortest-queue across all spines)"
        subtitle = "single multi-port component, DRILL on"
    elif mode == 'bw':
        label = "W-ECMP (bandwidth-weighted)"
        subtitle = "single-port components, DRILL off"
    else:
        label = "ECMP (equal weight)"
        subtitle = "single-port components, DRILL off"
    print("\n" + "=" * 60)
    print(f" Installing baseline: {label}  —— {subtitle}")
    print("=" * 60 + "\n")

    # silence noisy Thrift API logs, keep only our summary
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
                print(f"  [{src}] installed ({len(leaves) - 1} targets)")
            else:
                print(f"  [warn] cannot connect {src}, skipped")

    print("\n=== baseline install done ===\n")


# Measurement: same CSV format as plot_1s_metrics.py
def run_measurement(label, duration, csv_path):
    """Passively measure 1s hardware metrics; Pred_* stay NaN (no ML)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # reuse the passive collector (no model needed); silence its Thrift logs
    _dn = open(os.devnull, 'w')
    with redirect_stdout(_dn), redirect_stderr(_dn):
        predictor = Realtime1sPredictorTopoIndep()

    results = []
    start = time.time()
    print("\n" + "=" * 60)
    print(f" [measure {label}] duration {duration}s  ->  {csv_path}")
    print("=" * 60)
    print(f"{'time':^10} | {'Lat(ms)':^9} | {'Loss(%)':^9} | {'Util':^6} | {'Mbps':^6}")
    print("-" * 55)

    # same deadline cadence as the ML controller: fixed PERIOD measurement window for a fair
    # comparison; on overrun jump to the next future deadline (no catch-up RPC bursts)
    PERIOD    = 1.0
    next_tick = start + PERIOD
    try:
        while next_tick - start <= duration:
            now = time.time()
            if now < next_tick:
                time.sleep(next_tick - now)
            else:
                next_tick += (int((now - next_tick) // PERIOD) + 1) * PERIOD
                time.sleep(max(0.0, next_tick - time.time()))

            with redirect_stdout(_dn), redirect_stderr(_dn):
                row = predictor.collect_1s_data(sleep_before=False)
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

            next_tick += PERIOD
    except KeyboardInterrupt:
        print("\n[aborted]")
    finally:
        _dn.close()

    if not results:
        print("no data collected.")
        return

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {len(df)} rows -> {csv_path}")


def run_baseline(mode, label, duration, csv_path):
    """One shot: install baseline -> measure -> write CSV."""
    install_baseline(mode)
    run_measurement(label, duration, csv_path)
