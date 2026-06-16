import sys
import os
import time
import json
import math
import pandas as pd
import numpy as np
import joblib
import logging
import subprocess
import warnings
import collections
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

warnings.filterwarnings("ignore")

# load P4-Utils
P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI
from all_controller import TopologyAnalyzer, install_ecmp_drill_rules
from topo_independent_helper import transform_to_topo_independent

logging.basicConfig(level=logging.ERROR)

# ---- config ----
CONTROL_LEAF = "l1"
TARGET_LEAF = "l2"
TARGET_IP = "10.0.2.2"
SRC_ADD = 1
PORTS = list(range(2, 10))
# 8 distinct-bw asymmetric topology (p4app bw 0.6~1.3, x0.8 effective)
CAPACITY = {2: 0.48, 3: 0.56, 4: 0.64, 5: 0.72,
            6: 0.80, 7: 0.88, 8: 0.96, 9: 1.04}

# True = W-ECMP+DRILL+ML (dynamic weights); False = static W-ECMP+DRILL anchor (weights never change)
ML_WEIGHT_ENABLE = True

# Controller stability params (elephant/mice; capacity anchor + bounded correction)
LATENCY_THRESHOLD_MS = 200.0   # model alarm threshold
LOSS_THRESHOLD_PCT   = 2.0
COOLDOWN_SEC         = 6        # min spacing between rehashes
SETTLE_SEC           = 2        # post-rehash measurement blackout
PERSIST_TICKS        = 2        # ticks imbalance must persist before acting
RELAX_TICKS          = 6        # ticks a shed class stays cold before relaxing to anchor
IMBALANCE_TOL        = 0.15     # no-action band (between mouse ~0.13 and elephant ~0.21+)
WEIGHT_BOUND         = 2        # max weight deviation from anchor
CORRECTION_GAIN      = 0.5      # proportional shed/boost gain
UTIL_SAT             = 0.90     # absolute "class near its own limit" threshold
QDEPTH_HOT           = 32       # queue depth marking a class congested (cap is 64)
RATE_LIMIT_SCALE     = 0.8      # effective cap = link bw x this (matches rate_limiter.py)
Q_WEIGHT             = 0.5      # weight of queue pressure in the hotness signal
WEIGHT_MIN           = 1
WEIGHT_MAX           = 8

# Result CSV (same columns as baselines, for plot_result.py); dynamic vs static use different names
OUTPUT_CSV = ("research_results/data/validation/comparison_ml.csv" if ML_WEIGHT_ENABLE
              else "research_results/data/validation/comparison_wecmp_drill.csv")

MODELS = {
    "latency": "rf_model_latency_1s.pkl",
    "loss":    "rf_model_loss_1s.pkl",
    "anomaly": "rf_model_anomaly_1s.pkl"
}

FEATURE_NAMES = [
    "Is_Rehash_Event", "Time_Since_Last_Rehash_s", "Rehash_Impact",
    "Total_Util_Sum", "Max_Util_Diff", "Group_Imbalance",
    "Max_QDepth", "Total_QDepth", "QDepth_Imbalance",
    "Over_Capacity_Sum", "Max_Q_Ratio", "Q_Danger_Flag", "Q_Danger_Count",
    "Total_QDepth_Trend",
    "Total_Actual_Mbps", "Expected_Over_Capacity_Sum",
    "Overflow_Intensity", "Queue_Full_And_Over_Cap"
]
for _k in range(3):
    FEATURE_NAMES.extend([
        f"top{_k+1}_qdepth",
        f"top{_k+1}_mbps", f"top{_k+1}_weight",
        f"top{_k+1}_norm_load", f"top{_k+1}_expected_util",
        f"top{_k+1}_qdepth_trend", f"top{_k+1}_mbps_trend"
    ])


class MLController:
    def __init__(self):
        self.topo = load_topo("topology.json")

        # Reinstall 4-component W-ECMP+DRILL forwarding (overrides any leftover baseline config)
        print("[init] 重新安裝 4-component W-ECMP+DRILL 轉發 (覆蓋 baseline 設定)...")
        with open('p4app.json') as _f:
            _p4app = json.load(_f)
        with open(os.devnull, 'w') as _dn, redirect_stdout(_dn), redirect_stderr(_dn):
            install_ecmp_drill_rules(_p4app, self.topo, clear_first=True, verbose=False)
        print("[init] 轉發規則安裝完成")

        self.api_telemetry = SimpleSwitchThriftAPI(self.topo.get_thrift_port(TARGET_LEAF))

        # load models (warn if missing)
        self.models = {}
        for k, v in MODELS.items():
            if os.path.exists(v):
                mdl = joblib.load(v)
                # single-row inference: disable joblib workers to silence warnings
                try:
                    mdl.set_params(n_jobs=1)
                except Exception:
                    try: mdl.n_jobs = 1
                    except Exception: pass
                self.models[k] = mdl
                print(f"   - 載入模型 {v} 成功")
            else:
                print(f"   - 警告: 找不到模型檔案 {v}")

        self.prev_bytes = {p: 0 for p in PORTS}

        # smoothed prediction cache
        self.smoothed_latency = 20.0
        self.smoothed_loss    = 0.0

        # 1s collector state
        self.raw_history      = collections.deque(maxlen=100)
        self.start_time       = time.time()
        self.last_rehash_time = time.time()
        self.is_rehash_event  = 0
        self.prev_weights_1s  = {p: 1 for p in PORTS}
        self.prev_l1_enq      = {p: 0 for p in PORTS}
        self.prev_l2_ingress  = {p: 0 for p in PORTS}

        self.init_baseline()

        self._api_control   = SimpleSwitchThriftAPI(self.topo.get_thrift_port(CONTROL_LEAF))
        # anchor + per-port bw from the same source the dataplane was installed with (graph bw, not CAPACITY)
        self._base_weights, self._hw_rules, self._port_cap = self._load_hw_rules()
        self._grp_handle    = None
        self._mbr_handles   = []
        self._last_adj_time = 0.0
        self._settle_until  = 0.0
        self._hot_streak    = 0
        self._relax_streak  = 0
        self._last_state_log = ""   # dedup passive state logs
        # read back the live group/member handles + per-class weights instead of assuming install succeeded
        self._sync_from_dataplane()

        # baseline l1 enqueue counters (after _api_control exists)
        with open(os.devnull, 'w') as _f, redirect_stdout(_f), redirect_stderr(_f):
            for p in PORTS:
                try:
                    self.prev_l1_enq[p] = self._api_control.counter_read('cnt_enq', p)[0]
                except: pass

    def init_baseline(self):
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            for p in PORTS:
                try:
                    self.prev_bytes[p]      = self.api_telemetry.counter_read('port_bytes_counter', p)[0]
                    self.prev_l2_ingress[p] = self.api_telemetry.counter_read('cnt_ingress', p)[0]
                except: pass
        self._prev_sample_t = time.time()   # last sample time (for true sampling interval)
        self._cum_enq  = 0                   # cumulative l1 enqueue packets
        self._cum_recv = 0                   # cumulative l2 ingress packets (for E2E loss)

    def _load_hw_rules(self):
        """Return (anchor weights, hw rules, per-port effective bw = graph bw x RATE_LIMIT_SCALE)."""
        with open('p4app.json') as f:
            p4app = json.load(f)
        analyzer = TopologyAnalyzer(p4app, self.topo)
        weights, rules = analyzer.get_ecmp_weights_and_rules(CONTROL_LEAF, TARGET_LEAF)
        port_cap = {}
        for nh in analyzer.G.neighbors(CONTROL_LEAF):
            try:
                port = analyzer.topo_json.node_to_node_port_num(CONTROL_LEAF, nh)
                port_cap[port] = analyzer.G[CONTROL_LEAF][nh]['bw'] * RATE_LIMIT_SCALE
            except Exception:
                pass
        return weights, rules, port_cap

    def _sync_from_dataplane(self):
        """Read back the live group/member handles + per-class weights; reinstall anchor if mismatched."""
        import socket
        with open(os.devnull, 'w') as _dn, redirect_stdout(_dn), redirect_stderr(_dn):
            try:
                entries = self._api_control.client.bm_mt_get_entries(0, "MyIngress.w_ecmp_table")
                tip = socket.inet_aton(TARGET_IP)
                ent = next((e for e in entries if e.match_key and e.match_key[0].exact
                            and e.match_key[0].exact.key == tip), None)
                if ent is not None and ent.action_entry.grp_handle is not None:
                    self._grp_handle = ent.action_entry.grp_handle
                    grp = self._api_control.client.bm_mt_act_prof_get_group(
                        0, "MyIngress.w_ecmp_selector", self._grp_handle)
                    self._mbr_handles = list(grp.mbr_handles)
            except Exception:
                self._grp_handle, self._mbr_handles = None, []

            pw = self.get_current_weights()   # {port: member_count}
        cur = [pw.get(rule['ports_and_macs'][0][0], 0) for rule in self._hw_rules]
        self._last_weights = cur if cur and all(w > 0 for w in cur) else list(self._base_weights)
        if self._last_weights != self._base_weights:
            # dataplane != anchor -> reinstall anchor (clears the old group via the handle grabbed above)
            if self._safe_apply(self._base_weights):
                print(f"[init] dataplane 權重 {self._last_weights} != anchor，已補裝 {self._base_weights}")

    def _component_stats(self, feats):
        """Per class: (util = Σmbps/Σeffective_cap, qfrac = max class qdepth / QDEPTH_HOT)."""
        utils, qfracs = [], []
        for rule in self._hw_rules:
            ports = [p for p, _ in rule['ports_and_macs']]
            cap   = sum(self._port_cap.get(p, CAPACITY[p]) for p in ports)
            load  = sum(feats.get(f'src1_port{p}_mbps', 0.0) for p in ports)
            utils.append(load / cap if cap > 0 else 0.0)
            qmax  = max(feats.get(f'src1_port{p}_qdepth_max', 0) for p in ports)
            qfracs.append(qmax / QDEPTH_HOT if QDEPTH_HOT > 0 else 0.0)
        return utils, qfracs

    def _class_pressure(self, utils, qfracs):
        """Hotness signal = util + Q_WEIGHT * max(0, qfrac-1); queue pressure catches elephants early."""
        return [u + Q_WEIGHT * max(0.0, q - 1.0) for u, q in zip(utils, qfracs)]

    def is_balanced(self, feats):
        """Balanced = util spread < IMBALANCE_TOL and no queue jammed."""
        utils, qfracs = self._component_stats(feats)
        if not utils:
            return True
        return (max(utils) - min(utils)) < IMBALANCE_TOL and not any(q >= 1.0 for q in qfracs)

    def compute_weights(self, feats):
        """Bounded shed/boost around the capacity anchor by hotness signal; desired->base as imbalance->0."""
        utils, qfracs = self._component_stats(feats)
        sig    = self._class_pressure(utils, qfracs)
        mean_s = sum(sig) / len(sig) if sig else 0.0
        out = []
        for base, s in zip(self._base_weights, sig):
            d = base * (1.0 - CORRECTION_GAIN * (s - mean_s) / max(mean_s, 1e-6))
            d = max(base - WEIGHT_BOUND, min(base + WEIGHT_BOUND, d))
            out.append(max(WEIGHT_MIN, min(WEIGHT_MAX, int(round(d)))))
        return out

    def apply_weights(self, weights_list):
        sel = "w_ecmp_selector"
        act = "assign_component"
        thrift_port = self.topo.get_thrift_port(CONTROL_LEAF)

        with open(os.devnull, 'w') as devnull, \
             redirect_stdout(devnull), redirect_stderr(devnull):

            # 1. Build a fresh group (never mutate the in-use group -> can crash BMv2)
            new_grp = self._api_control.act_prof_create_group(sel)
            new_mbr_handles = []
            for idx, rule in enumerate(self._hw_rules):
                comp_id = str(rule['comp_id'])
                for _ in range(weights_list[idx]):
                    m = self._api_control.act_prof_create_member(sel, act, [comp_id])
                    self._api_control.act_prof_add_member_to_group(sel, m, new_grp)
                    new_mbr_handles.append(m)

            # 2. Get the TARGET_IP entry handle in the forwarding table
            try:
                self._api_control.load_table_entries_match_to_handle()
                entry_handle = self._api_control.get_handle_from_match("w_ecmp_table", [TARGET_IP])
            except Exception:
                entry_handle = None

            # 3. Atomically repoint the entry to the new group (hitless)
            if entry_handle is not None:
                try:
                    self._api_control.client.bm_mt_indirect_ws_modify_entry(0, "MyIngress.w_ecmp_table", entry_handle, new_grp)
                except Exception:
                    # fall back to delete-then-add if the atomic modify fails
                    try:
                        self._api_control.table_delete("MyIngress.w_ecmp_table", entry_handle)
                    except Exception:
                        pass
                    entry_handle = None

            # 4. If no entry handle (first run / deleted), add the entry directly
            if entry_handle is None:
                subprocess.run(
                    ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
                    input=f"table_indirect_add_with_group w_ecmp_table {TARGET_IP} => {new_grp}\n",
                    text=True, capture_output=True
                )

            # 5. Free the old group/members (now unreferenced -> safe in BMv2)
            if self._grp_handle is not None:
                try:
                    self._api_control.act_prof_delete_group(sel, self._grp_handle)
                    for m in self._mbr_handles:
                        self._api_control.act_prof_delete_member(sel, m)
                except Exception:
                    pass

            # 6. Update local references
            self._grp_handle = new_grp
            self._mbr_handles = new_mbr_handles

        # post-rehash measurement blackout
        self._settle_until = time.time() + SETTLE_SEC

    def _safe_apply(self, weights):
        """Apply weights + maintain state; rebuild connection and return False on Thrift failure."""
        try:
            self.apply_weights(weights)
        except Exception:
            self._api_control = SimpleSwitchThriftAPI(self.topo.get_thrift_port(CONTROL_LEAF))
            self._grp_handle  = None
            self._mbr_handles = []
            return False
        self._last_adj_time = time.time()
        self._last_weights  = list(weights)
        return True

    def _log_weights(self, tag, weights):
        port_weights = {}
        for rule, w in zip(self._hw_rules, weights):
            for port, _ in rule['ports_and_macs']:
                port_weights[port] = w
        print(f"\n[CTRL] {tag} → " +
              "  ".join(f"s{p-1}:{w}" for p, w in sorted(port_weights.items())), flush=True)

    def _state_log(self, msg):
        """Dedup passive state logs: print only when the status string changes."""
        if msg != self._last_state_log:
            print(msg, flush=True)
            self._last_state_log = msg

    def control_step(self, feats, preds):
        now = time.time()
        # 1. Settle blackout: queues still redistributing after a rehash -> don't measure transients
        if now < self._settle_until:
            return

        utils, qfracs = self._component_stats(feats)
        if not utils:
            return
        sig      = self._class_pressure(utils, qfracs)
        mean_u   = sum(utils) / len(utils)
        mean_s   = sum(sig) / len(sig)
        hot_i    = max(range(len(sig)), key=lambda i: sig[i])  # hottest class by pressure signal
        spread   = max(utils) - min(utils)
        any_qhot = any(q >= 1.0 for q in qfracs)
        balanced = spread < IMBALANCE_TOL and not any_qhot

        # Hardware safety signals (no RF model needed): real loss / jammed queue / high util
        hw_loss   = feats.get('hw_loss', 0.0)
        hw_bad    = hw_loss > LOSS_THRESHOLD_PCT or any_qhot or mean_u > UTIL_SAT
        model_bad = (preds.get('anomaly', 0) == 1
                     or self.smoothed_latency > LATENCY_THRESHOLD_MS
                     or self.smoothed_loss    > LOSS_THRESHOLD_PCT)

        is_hot = (sig[hot_i] - mean_s > IMBALANCE_TOL) or (qfracs[hot_i] >= 1.0) \
                 or (utils[hot_i] > UTIL_SAT)
        # Evidence-based relax: off-anchor, no hotspot, and a shed class has gone cold (elephant left)
        cold_relax = (self._last_weights != self._base_weights and not is_hot
                      and any(self._last_weights[i] < self._base_weights[i]
                              and utils[i] < mean_u - IMBALANCE_TOL
                              for i in range(len(utils))))

        # persistence counters filter transients
        self._hot_streak   = self._hot_streak + 1 if is_hot else 0
        self._relax_streak = self._relax_streak + 1 if cold_relax else 0

        def _stat():
            return "  ".join(f"c{i+1}:{u:.2f}/q{q:.2f}"
                             for i, (u, q) in enumerate(zip(utils, qfracs)))

        # A. Hot class -> shed: hardware evidence fires now; model-only fires after PERSIST_TICKS
        if is_hot and (hw_bad or (model_bad and self._hot_streak >= PERSIST_TICKS)):
            if now - self._last_adj_time < COOLDOWN_SEC:
                return
            new = self.compute_weights(feats)
            if new == self._last_weights:
                # at correction bound / weights can't help -> log (deduped) and apply cooldown
                self._last_adj_time = now
                self._state_log(f"\n[CTRL] at correction bound — cannot improve via weights | {_stat()}")
                return
            if self._safe_apply(new):
                self._hot_streak, self._last_state_log = 0, ""
                self._log_weights("shed/boost", new)
            return

        # B. Evidence-based relax to anchor: a shed class stays cold -> jump back in one rehash
        if cold_relax and self._relax_streak >= RELAX_TICKS \
                and now - self._last_adj_time >= COOLDOWN_SEC:
            if self._safe_apply(self._base_weights):
                self._relax_streak, self._last_state_log = 0, ""
                self._log_weights("relax → anchor", self._base_weights)
            return

        # C. Balanced but overloaded: weights can't fix aggregate overload -> log, no rehash
        if balanced and (hw_loss > LOSS_THRESHOLD_PCT or mean_u > UTIL_SAT):
            self._state_log(f"\n[CTRL] balanced overload — no weight solution | {_stat()}")
            return
        # D. Healthy / converged -> freeze

    def get_current_weights(self):
        weights = {p: 1 for p in PORTS}
        try:
            import socket
            entries = self._api_control.client.bm_mt_get_entries(0, "MyIngress.w_ecmp_table")
            if not entries:
                return weights
            
            target_ip_bytes = socket.inet_aton(TARGET_IP)
            target_entry = None
            for entry in entries:
                if entry.match_key and entry.match_key[0].exact and entry.match_key[0].exact.key == target_ip_bytes:
                    target_entry = entry
                    break
            
            if target_entry is not None:
                grp_handle = target_entry.action_entry.grp_handle
                if grp_handle > 0:
                    grp_info = self._api_control.client.bm_mt_act_prof_get_group(0, "MyIngress.w_ecmp_selector", grp_handle)
                    members = grp_info.mbr_handles
                    
                    comp_counts = {}
                    for m_handle in members:
                        mbr = self._api_control.client.bm_mt_act_prof_get_member(0, "MyIngress.w_ecmp_selector", m_handle)
                        if mbr.action_data:
                            comp_id = int(mbr.action_data[0].hex(), 16)
                            comp_counts[comp_id] = comp_counts.get(comp_id, 0) + 1
                    
                    nh_entries = self._api_control.client.bm_mt_get_entries(0, "MyIngress.ecmp_group_to_nhop")
                    for entry in nh_entries:
                        if entry.match_key and entry.match_key[0].exact:
                            c_id = int(entry.match_key[0].exact.key.hex(), 16)
                            if entry.action_entry and entry.action_entry.action_data:
                                port = int(entry.action_entry.action_data[1].hex(), 16)
                                if c_id in comp_counts and port in weights:
                                    weights[port] = comp_counts[c_id]
        except Exception as e:
            pass
        return weights

    def _local_current_weights(self):
        """Per-port weights from local state (trusts _last_weights; avoids walking selector each tick)."""
        weights = {p: 1 for p in PORTS}
        for rule, weight in zip(self._hw_rules, self._last_weights):
            for port, _ in rule['ports_and_macs']:
                weights[port] = weight
        return weights

    def collect_1s_data(self, sleep_before=True):
        """Collect one 1s data point + hardware ground truth."""
        if sleep_before:
            time.sleep(1.0)
        now_t        = time.time()
        # dt = true interval since last sample (incl. processing); not just sleep(1.0), else mbps overestimated
        dt           = now_t - self._prev_sample_t
        self._prev_sample_t = now_t

        current_time    = now_t
        row             = {}
        qdepths         = []
        max_hw_latency  = 0
        total_delta_enq = 0
        total_drops     = 0

        with open(os.devnull, 'w') as _dn, redirect_stdout(_dn), redirect_stderr(_dn):
            weights = self._local_current_weights()
            if weights != self.prev_weights_1s:
                self.is_rehash_event  = 1
                self.last_rehash_time = current_time
                self.prev_weights_1s  = weights

            total_weight = sum(weights.values())
            if total_weight == 0: total_weight = 1

            for p in PORTS:
                reg_idx         = SRC_ADD * 16 + p
                q               = self.api_telemetry.register_read('path_max_queue_depth_reg', reg_idx)
                raw_acc_q_delay = self.api_telemetry.register_read('path_acc_q_delay_reg', reg_idx)

                self.api_telemetry.register_write('path_max_queue_depth_reg', reg_idx, 0)
                self.api_telemetry.register_write('path_acc_q_delay_reg', reg_idx, 0)

                cnt  = self.api_telemetry.counter_read('port_bytes_counter', p)[0]
                db   = cnt - self.prev_bytes[p]
                mbps = (max(0, db) * 8) / (dt * 1_000_000)
                self.prev_bytes[p] = cnt

                row[f"src1_port{p}_qdepth"] = q
                row[f"src1_port{p}_mbps"]   = mbps
                row[f"Weight_Port{p}"]      = weights[p] / total_weight

                qdepths.append(q)

                if raw_acc_q_delay > max_hw_latency:
                    max_hw_latency = raw_acc_q_delay

                try:
                    l1_enq_pkts     = self._api_control.counter_read('cnt_enq', p)[0]
                    l2_ingress_pkts = self.api_telemetry.counter_read('cnt_ingress', p)[0]

                    delta_enq     = l1_enq_pkts - self.prev_l1_enq[p]
                    delta_ingress = l2_ingress_pkts - self.prev_l2_ingress[p]

                    self.prev_l1_enq[p]     = l1_enq_pkts
                    self.prev_l2_ingress[p] = l2_ingress_pkts

                    # instantaneous per-sec estimate (biased by queue buildup; time-series only)
                    if delta_enq > 0:
                        drops            = max(0, delta_enq - delta_ingress)
                        total_delta_enq += delta_enq
                        total_drops     += drops

                    # cumulative totals -> correct E2E loss rate (buildup/drain cancel out)
                    self._cum_enq  += delta_enq
                    self._cum_recv += delta_ingress
                except: pass

        row['Real_HW_Latency_ms']         = max_hw_latency / 1000.0
        row['Real_HW_Loss_Percent']       = (total_drops / total_delta_enq * 100) if total_delta_enq > 0 else 0.0
        row['Cum_Enq']                    = self._cum_enq
        row['Cum_Recv']                   = self._cum_recv
        row["Time_Since_Traffic_Start_s"] = current_time - self.start_time
        row["Is_Rehash_Event"]            = self.is_rehash_event
        self.is_rehash_event              = 0
        row["Time_Since_Last_Rehash_s"]   = current_time - self.last_rehash_time
        row["Rehash_Impact"]              = math.exp(-row["Time_Since_Last_Rehash_s"])

        return row

    def run(self, duration=None):
        print("\n" + "="*125)
        dur_txt = f"{duration}s" if duration else "持續 (Ctrl-C 停止)"
        mode_txt = ("W-ECMP+DRILL+ML (動態調權)" if ML_WEIGHT_ENABLE
                    else "W-ECMP+DRILL 靜態 (固定容量比例權重，永不變動)")
        print(f" [ML 智能監控 v4.3] 啟動 - 1s 拓樸無關採集模式 | 模式: {mode_txt} | 時長: {dur_txt}")
        print("="*125 + "\n")

        results = []
        start = time.time()
        next_tick = start
        try:
            while True:
                next_tick += 1.0
                if duration is not None and next_tick - start > duration:
                    break
                sleep_for = next_tick - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.time()

                data_row = self.collect_1s_data(sleep_before=False)
                self.raw_history.append(data_row)

                df_history     = pd.DataFrame(list(self.raw_history))
                df_transformed = transform_to_topo_independent(df_history, PORTS, CAPACITY, K=3)
                last_row       = df_transformed.iloc[-1]
                X              = pd.DataFrame([last_row[FEATURE_NAMES]])

                preds = {}
                with open(os.devnull, 'w') as _dn, redirect_stderr(_dn):
                    for k, m in self.models.items():
                        p = m.predict(X)[0]
                        if k == "latency": p = np.expm1(p)
                        preds[k] = p

                self.smoothed_latency = preds.get('latency', self.smoothed_latency)
                self.smoothed_loss    = preds.get('loss',    self.smoothed_loss)

                # feats dict for control_step / compute_weights
                feats = {}
                for p in PORTS:
                    feats[f'src1_port{p}_load_util']  = data_row[f'src1_port{p}_mbps'] / CAPACITY[p]
                    feats[f'src1_port{p}_qdepth_max'] = data_row[f'src1_port{p}_qdepth']
                    feats[f'src1_port{p}_mbps']       = data_row[f'src1_port{p}_mbps']  # raw util signal (bypasses CAPACITY)
                feats['qdepth_max_imbalance'] = last_row['QDepth_Imbalance']
                feats['Total_Util_Sum']       = last_row['Total_Util_Sum']
                feats['hw_loss']              = data_row['Real_HW_Loss_Percent']  # hardware safety trigger

                status  = "NORMAL" if preds.get('anomaly', 0) == 0 else "\033[91mANOMALY\033[0m"
                hw_lat  = data_row['Real_HW_Latency_ms']
                hw_loss = data_row['Real_HW_Loss_Percent']

                # log one row (same columns as baselines + per-switch util)
                results.append({
                    'Timestamp':  datetime.now(),
                    'Pred_Lat':   self.smoothed_latency,
                    'Real_Lat':   hw_lat,
                    'Pred_Loss':  self.smoothed_loss,
                    'Real_Loss':  hw_loss,
                    'Util_Sum':   feats['Total_Util_Sum'],
                    'Total_Mbps': last_row['Total_Actual_Mbps'],
                    'Cum_Enq':    data_row['Cum_Enq'],
                    'Cum_Recv':   data_row['Cum_Recv'],
                    **{f'util_s{p-1}': feats[f'src1_port{p}_load_util'] for p in PORTS},
                })

                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {status:7} | "
                      f"Lat: {self.smoothed_latency:5.1f}/{hw_lat:5.1f}ms | "
                      f"Loss: {self.smoothed_loss:4.1f}/{hw_loss:4.1f}% | "
                      f"Util: {feats['Total_Util_Sum']:4.2f}",
                      end='', flush=True)
                # static mode: weights never change (anchor installed at startup)
                if ML_WEIGHT_ENABLE:
                    self.control_step(feats, preds)
        except KeyboardInterrupt: print("\n停止。")
        finally:
            if results:
                os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
                pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
                print(f"已寫入 {len(results)} 列 -> {OUTPUT_CSV}")


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ctrl = MLController()
    ctrl.run(duration)
