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

# 隱藏所有警告
warnings.filterwarnings("ignore")

# 載入 P4-Utils
P4_UTILS_PATH = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
if P4_UTILS_PATH not in sys.path:
    sys.path.append(P4_UTILS_PATH)

from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI
from all_controller import TopologyAnalyzer, install_ecmp_drill_rules
from topo_independent_helper import transform_to_topo_independent

# 設定日誌 (只顯示 ERROR 以上)
logging.basicConfig(level=logging.ERROR)

# ==========================================
# 配置參數
# ==========================================
CONTROL_LEAF = "l1"
TARGET_LEAF = "l2"
TARGET_IP = "10.0.2.2"
SRC_ADD = 1
PORTS = list(range(2, 10))
# 8 個不同頻寬的非對稱拓樸 (p4app bw 0.6~1.3，rate_limiter ×0.8 後的有效容量)
CAPACITY = {2: 0.48, 3: 0.56, 4: 0.64, 5: 0.72,
            6: 0.80, 7: 0.88, 8: 0.96, 9: 1.04}

# ---- 權重控制總開關 ----
# True  -> 完整 W-ECMP+DRILL+ML：啟動裝 anchor，之後 control_step 依 (反應式/ML) 證據動態調權。
# False -> 純靜態 W-ECMP+DRILL：啟動時依各 class 頻寬容量裝好 anchor 權重 [3,4,5,6]，之後
#          「權重永不變動」，完全不參考 ML 預測或任何指標；class 內仍由 dataplane DRILL 逐封包
#          選最短佇列。等同 W-ECMP+DRILL baseline，但仍走本控制器的量測/記錄路徑 (相同 CSV
#          欄位，供 plot_result.py 公平比較)。
ML_WEIGHT_ENABLE = True

# ---- 控制器穩定性參數 (大象/老鼠流量；錨定容量比例 + 有界修正) ----
# 設計：流量是大象 (0.24~0.40M) + 老鼠 (0.06~0.16M) 混合，事先不知誰是大象。
# 大象落到某個容量 class 會把該 class 的佇列/利用率撐高 (自我暴露)，控制器就「降低該
# class 的權重」-> W-ECMP 雜湊較少把新的 (多為老鼠) 流導進大象的 class -> 它們改走較涼的
# class。class 內部由 dataplane DRILL 逐封包把老鼠導到非大象那個埠 (控制器不管)。目標是
# 「把熱 class 的進流量 shed 到各 class load/cap 相近，然後凍結」，不是精確等利用率。
LATENCY_THRESHOLD_MS = 200.0   # 模型示警門檻 (假設模型已重訓、預測可靠)
LOSS_THRESHOLD_PCT   = 2.0
COOLDOWN_SEC         = 6        # 兩次 rehash 的最小間隔 (每次權重變動 = 全流量重雜湊)
SETTLE_SEC           = 2        # rehash 後的量測黑窗：過渡態不可信，不在此期間決策
PERSIST_TICKS        = 2        # 不平衡需連續成立幾拍才動手 (濾掉過渡/雜訊；大象的熱會持續)
RELAX_TICKS          = 6        # 被 shed 的 class 連續轉冷這麼多拍 (=大象已離開的證據) -> 跳回 anchor
IMBALANCE_TOL        = 0.15     # 不動作容忍帶：介於老鼠(~0.13) 與大象(~0.21+) 的利用率隆起之間
WEIGHT_BOUND         = 2        # 權重相對 anchor 的最大偏移 (小整數 anchor 下避免過大修正與 rehash churn)
CORRECTION_GAIN      = 0.5      # 比例修正增益：熱 class 依超出量降權、冷 class 升權
UTIL_SAT             = 0.90     # 「該 class 逼近自身上限」的絕對門檻 (真實頻寬利用率)
QDEPTH_HOT           = 32       # 視為被大象塞住的 max 佇列深度 (rate_limiter 設佇列上限 64 -> 半滿)
RATE_LIMIT_SCALE     = 0.8      # rate_limiter.py 用 set_queue_rate(0.8x p4app bw) -> 有效容量 = 0.8x 連結頻寬
Q_WEIGHT             = 0.5      # 佇列壓力併入 class 熱度訊號的權重 (大象常先以佇列堆積暴露，早於利用率飽和)
WEIGHT_MIN           = 1
WEIGHT_MAX           = 8

# 結果 CSV (與 baseline / plot_1s_metrics 相同欄位，供 plot_result.py 三方比較)
# 動態調權 vs 靜態 (固定容量比例權重) 寫不同檔名，避免互相覆蓋
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

        # 啟動時重裝 4-component W-ECMP+DRILL 轉發 (覆蓋任何 baseline 殘留設定)。
        # 否則若上一個跑的是 baseline_ecmp/wecmp (8 個單埠 component)，本控制器發出的
        # comp_id 1-4 會對應到錯誤的單埠映射，導致流量只走 2-3 個 spine。
        print("[init] 重新安裝 4-component W-ECMP+DRILL 轉發 (覆蓋 baseline 設定)...")
        with open('p4app.json') as _f:
            _p4app = json.load(_f)
        with open(os.devnull, 'w') as _dn, redirect_stdout(_dn), redirect_stderr(_dn):
            install_ecmp_drill_rules(_p4app, self.topo, clear_first=True, verbose=False)
        print("[init] 轉發規則安裝完成")

        self.api_telemetry = SimpleSwitchThriftAPI(self.topo.get_thrift_port(TARGET_LEAF))

        # 載入模型 (安全載入，缺少模型時發出警告)
        self.models = {}
        for k, v in MODELS.items():
            if os.path.exists(v):
                mdl = joblib.load(v)
                # 單列推論用不到平行運算；關掉 joblib worker 可消除 sklearn 平行警告並略為加速
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

        # 預測值平滑緩存
        self.smoothed_latency = 20.0
        self.smoothed_loss    = 0.0

        # 1s 採集器狀態 (對應 realtime_1s_predictor_topo_indep)
        self.raw_history      = collections.deque(maxlen=100)
        self.start_time       = time.time()
        self.last_rehash_time = time.time()
        self.is_rehash_event  = 0
        self.prev_weights_1s  = {p: 1 for p in PORTS}
        self.prev_l1_enq      = {p: 0 for p in PORTS}
        self.prev_l2_ingress  = {p: 0 for p in PORTS}

        self.init_baseline()

        self._api_control   = SimpleSwitchThriftAPI(self.topo.get_thrift_port(CONTROL_LEAF))
        # anchor (容量比例權重) 與每埠真實頻寬都取自 install_ecmp_drill_rules 用的同一個來源
        # (get_ecmp_weights_and_rules，依拓樸圖真實 bw)，故不受 CAPACITY 常數 0.8x bug 影響。
        self._base_weights, self._hw_rules, self._port_cap = self._load_hw_rules()
        self._grp_handle    = None
        self._mbr_handles   = []
        self._last_adj_time = 0.0
        self._settle_until  = 0.0
        self._hot_streak    = 0
        self._relax_streak  = 0
        self._last_state_log = ""   # 被動狀態訊息去重用
        # 不假設 install 成功：讀回 dataplane 實際的 group/member handle 與 per-class 權重，
        # 對齊控制器狀態 (同時讓首次權重變動能正確刪除 install 裝的舊 group，不洩漏)。
        self._sync_from_dataplane()

        # 初始化 l1 enqueue 計數器基準 (在 _api_control 建立後)
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
        self._prev_sample_t = time.time()   # 上次取樣時間，用於計算真實取樣間隔
        self._cum_enq  = 0                   # 自開始累積的 l1 enqueue 封包數
        self._cum_recv = 0                   # 自開始累積的 l2 ingress 封包數 (用於端到端丟包率)

    def _load_hw_rules(self):
        """回傳 (anchor 權重, 硬體規則, 每埠真實頻寬)。

        anchor 權重就是 install_ecmp_drill_rules 實際裝進 dataplane 的容量比例權重 (此拓樸 =
        [3,4,5,6])，同一函式來源 -> 與 dataplane 一致，無需事後讀回核對。

        port_cap = 拓樸圖鏈路 bw × RATE_LIMIT_SCALE：dataplane 被 rate_limiter.py 限速在 0.8x，
        故有效容量是 0.8x 連結頻寬 (0.48/0.64/0.80/0.96)。用有效容量算利用率，UTIL_SAT 等絕對門檻
        才對得上真實飽和點 (否則會低估利用率 1.25x、太晚才判定壅塞)。"""
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
        """讀回 dataplane 實際綁定的 group/member handle 與 per-class 權重，對齊控制器狀態。

        - 抓到 group handle / member handles -> 首次權重變動能刪掉 install 裝的舊 group (不洩漏)。
        - per-class 權重設為 _last_weights；讀不到時退回 anchor。
        - 若與 anchor 不一致 (install 部分失敗等) -> 補裝一次 anchor 對齊真實狀態。
        """
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
            # dataplane 與 anchor 不符 -> 補裝 anchor (會用上面抓到的 handle 清掉舊 group)
            if self._safe_apply(self._base_weights):
                print(f"[init] dataplane 權重 {self._last_weights} != anchor，已補裝 {self._base_weights}")

    def _component_stats(self, feats):
        """每個 class 的 (真實利用率, 佇列熱度)。

        util = Σ原始 mbps / Σ真實埠頻寬 (用 self._port_cap，不經 CAPACITY 常數 -> 不受 0.8x bug)。
        qfrac = class 內最大佇列深度 / QDEPTH_HOT (>=1 視為被大象塞住)。"""
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
        """class 熱度訊號 = 真實利用率 + Q_WEIGHT × (佇列超出熱門門檻的部分)。

        大象常先把佇列堆高 (qfrac>1) 才反映到利用率，故把佇列壓力併入訊號，讓「選最熱 class」與
        「算修正量」都看得到佇列驅動的大象，而非只看 Mbps。"""
        return [u + Q_WEIGHT * max(0.0, q - 1.0) for u, q in zip(utils, qfracs)]

    def is_balanced(self, feats):
        """各 class 真實利用率差距 < IMBALANCE_TOL 且無佇列塞住 -> 已平衡。"""
        utils, qfracs = self._component_stats(feats)
        if not utils:
            return True
        return (max(utils) - min(utils)) < IMBALANCE_TOL and not any(q >= 1.0 for q in qfracs)

    def compute_weights(self, feats):
        """錨定在容量比例 anchor，依各 class 的『熱度訊號』(利用率+佇列壓力) 做有界 shed/boost。

        熱的 class (大象在此) 降權 -> 少導老鼠進去；冷的 class 升權 -> 老鼠有地方去。偏差回 0 時
        desired -> base，自然鬆回 anchor。直接有界四捨五入 (不做權重 EMA)：每次整數翻動 = 一次
        全 rehash，磨損與幅度無關，故反應性交給 PERSIST/SETTLE/COOLDOWN 等閘門控管，不靠 EMA 拖慢。"""
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

            # 1. 建立一個全新的群組並填入新權重對應的成員，避免動態修改正在使用的群組成員導致 BMv2 當機
            new_grp = self._api_control.act_prof_create_group(sel)
            new_mbr_handles = []
            for idx, rule in enumerate(self._hw_rules):
                comp_id = str(rule['comp_id'])
                for _ in range(weights_list[idx]):
                    m = self._api_control.act_prof_create_member(sel, act, [comp_id])
                    self._api_control.act_prof_add_member_to_group(sel, m, new_grp)
                    new_mbr_handles.append(m)

            # 2. 獲取現有轉發表中 TARGET_IP 對應的 Entry Handle
            try:
                self._api_control.load_table_entries_match_to_handle()
                entry_handle = self._api_control.get_handle_from_match("w_ecmp_table", [TARGET_IP])
            except Exception:
                entry_handle = None

            # 3. 如果 entry_handle 存在，原子性地修改表項指向新群組
            if entry_handle is not None:
                try:
                    self._api_control.client.bm_mt_indirect_ws_modify_entry(0, "MyIngress.w_ecmp_table", entry_handle, new_grp)
                except Exception:
                    # 如果 Thrift RPC 修改失敗，退回到刪除並重新添加的方案
                    try:
                        self._api_control.table_delete("MyIngress.w_ecmp_table", entry_handle)
                    except Exception:
                        pass
                    entry_handle = None

            # 4. 如果 entry_handle 不存在（首次執行或之前被刪除），則直接添加表項
            if entry_handle is None:
                subprocess.run(
                    ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
                    input=f"table_indirect_add_with_group w_ecmp_table {TARGET_IP} => {new_grp}\n",
                    text=True, capture_output=True
                )

            # 5. 安全地刪除舊群組與舊成員（因為它們現在已經沒有被任何表項引用了，這在 BMv2 中是完全安全的）
            if self._grp_handle is not None:
                try:
                    self._api_control.act_prof_delete_group(sel, self._grp_handle)
                    for m in self._mbr_handles:
                        self._api_control.act_prof_delete_member(sel, m)
                except Exception:
                    pass

            # 6. 更新實體變數參考
            self._grp_handle = new_grp
            self._mbr_handles = new_mbr_handles

        # rehash 後設量測黑窗：接下來 SETTLE_SEC 內佇列正在重分配，不可拿來決策
        self._settle_until = time.time() + SETTLE_SEC

    def _safe_apply(self, weights):
        """套用權重 + 維護狀態；Thrift 失敗時重建連線並回 False。"""
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
        """被動狀態訊息去重：只在狀態字串改變時印一次，避免每拍洗版干擾單行監控。"""
        if msg != self._last_state_log:
            print(msg, flush=True)
            self._last_state_log = msg

    def control_step(self, feats, preds):
        now = time.time()
        # 1. 黑窗：剛 rehash 完佇列正在重分配，量測不可信 -> 直接跳過 (避免量過渡態又過度修正 -> 擺盪)
        if now < self._settle_until:
            return

        utils, qfracs = self._component_stats(feats)
        if not utils:
            return
        sig      = self._class_pressure(utils, qfracs)        # 利用率 + 佇列壓力
        mean_u   = sum(utils) / len(utils)
        mean_s   = sum(sig) / len(sig)
        hot_i    = max(range(len(sig)), key=lambda i: sig[i])  # 以「熱度訊號」選最熱 class (含佇列)
        spread   = max(utils) - min(utils)
        any_qhot = any(q >= 1.0 for q in qfracs)
        balanced = spread < IMBALANCE_TOL and not any_qhot

        # 硬體安全訊號 (不依賴 RF 模型)：真實丟包 / 佇列塞住 / 真實高利用率
        hw_loss   = feats.get('hw_loss', 0.0)
        hw_bad    = hw_loss > LOSS_THRESHOLD_PCT or any_qhot or mean_u > UTIL_SAT
        model_bad = (preds.get('anomaly', 0) == 1
                     or self.smoothed_latency > LATENCY_THRESHOLD_MS
                     or self.smoothed_loss    > LOSS_THRESHOLD_PCT)

        is_hot = (sig[hot_i] - mean_s > IMBALANCE_TOL) or (qfracs[hot_i] >= 1.0) \
                 or (utils[hot_i] > UTIL_SAT)
        # 證據式鬆回：已偏離 anchor、無熱點、且某個「被 shed 的 class」現在轉冷 (util < 平均-容忍)
        # -> 造成 shed 的大象多半已離開，才鬆回。靜態下大象不離開、其 class 不會轉冷 -> 不會誤鬆回擺盪。
        cold_relax = (self._last_weights != self._base_weights and not is_hot
                      and any(self._last_weights[i] < self._base_weights[i]
                              and utils[i] < mean_u - IMBALANCE_TOL
                              for i in range(len(utils))))

        # persistence：大象的熱/冷會持續，過渡態會散去 -> 連續計數濾雜訊
        self._hot_streak   = self._hot_streak + 1 if is_hot else 0
        self._relax_streak = self._relax_streak + 1 if cold_relax else 0

        def _stat():
            return "  ".join(f"c{i+1}:{u:.2f}/q{q:.2f}"
                             for i, (u, q) in enumerate(zip(utils, qfracs)))

        # A. 熱 class -> shed：硬體證據 (hw_bad) 立刻動；只有模型示警則需持續 PERSIST_TICKS 才動
        if is_hot and (hw_bad or (model_bad and self._hot_streak >= PERSIST_TICKS)):
            if now - self._last_adj_time < COOLDOWN_SEC:
                return
            new = self.compute_weights(feats)
            if new == self._last_weights:
                # 已到修正邊界 / 權重救不了 (多為大象堆疊在最小權重 class) -> 去重記錄，套冷卻
                self._last_adj_time = now
                self._state_log(f"\n[CTRL] at correction bound — cannot improve via weights | {_stat()}")
                return
            if self._safe_apply(new):
                self._hot_streak, self._last_state_log = 0, ""
                self._log_weights("shed/boost", new)
            return

        # B. 證據式鬆回 anchor：被 shed 的 class 持續轉冷 -> 一次跳回容量比例 (一跳/多步都各一次 rehash)
        if cold_relax and self._relax_streak >= RELAX_TICKS \
                and now - self._last_adj_time >= COOLDOWN_SEC:
            if self._safe_apply(self._base_weights):
                self._relax_streak, self._last_state_log = 0, ""
                self._log_weights("relax → anchor", self._base_weights)
            return

        # C. 平衡但整體過載：權重無解 (總量過載動權重也救不了) -> 去重記錄、不 rehash
        if balanced and (hw_loss > LOSS_THRESHOLD_PCT or mean_u > UTIL_SAT):
            self._state_log(f"\n[CTRL] balanced overload — no weight solution | {_stat()}")
            return
        # D. 健康/已收斂 -> 凍結 (不動作)

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
        """用控制器狀態產生 per-port 權重，避免每秒透過 Thrift 走訪 selector members。

        _sync_from_dataplane() 啟動時已讀回一次 dataplane 狀態；之後所有成功套用都會更新
        self._last_weights，因此熱路徑可直接信任本地狀態。
        """
        weights = {p: 1 for p in PORTS}
        for rule, weight in zip(self._hw_rules, self._last_weights):
            for port, _ in rule['ports_and_macs']:
                weights[port] = weight
        return weights

    def collect_1s_data(self, sleep_before=True):
        """採集 1 秒的數據點與硬體真實數據"""
        if sleep_before:
            time.sleep(1.0)
        now_t        = time.time()
        # dt = 與上次取樣的真實間隔 (含處理時間)；不可只用 sleep(1.0)，否則處理較慢時吞吐量會被高估
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

                    # 瞬時 (每秒) 估計 — 受佇列堆積/延遲影響會偏高，僅供時間序列參考
                    if delta_enq > 0:
                        drops            = max(0, delta_enq - delta_ingress)
                        total_delta_enq += delta_enq
                        total_drops     += drops

                    # 累積總量 (不夾值，全埠) — 用於端到端正確丟包率，堆積/排空會自然抵消
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
                    feats[f'src1_port{p}_mbps']       = data_row[f'src1_port{p}_mbps']  # 真實利用率訊號 (不經 CAPACITY)
                feats['qdepth_max_imbalance'] = last_row['QDepth_Imbalance']
                feats['Total_Util_Sum']       = last_row['Total_Util_Sum']
                feats['hw_loss']              = data_row['Real_HW_Loss_Percent']  # 硬體安全觸發 (不依賴 RF 模型)

                status  = "NORMAL" if preds.get('anomaly', 0) == 0 else "\033[91mANOMALY\033[0m"
                hw_lat  = data_row['Real_HW_Latency_ms']
                hw_loss = data_row['Real_HW_Loss_Percent']

                # 記錄一列 (與 baseline / plot_1s_metrics 完全相同欄位 + 各交換機 util)
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
                # 靜態模式：權重永不變動 (啟動時已裝好容量比例 anchor)，不參考任何指標
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
