"""
plot_result.py — 疊圖比較 ECMP / W-ECMP / W-ECMP+DRILL+ML 三種演算法

沿用 plot_1s_metrics.py 的繪圖風格 (matplotlib，2~3 panel，dpi=150)，把各演算法
產生的 CSV (相同欄位：Real_Lat / Real_Loss / Total_Mbps / Util_Sum) 疊在一起比較。

各 CSV 來源：
  * ECMP            : baseline_ecmp.py        -> comparison_ecmp.csv
  * W-ECMP          : baseline_wecmp.py       -> comparison_wecmp.csv
  * W-ECMP+DRILL+ML : realtime_ml_controller.py -> comparison_ml.csv (相同欄位，直接比較)

用法：
    python3 plot_result.py
    python3 plot_result.py --ML        # 只比較 W-ECMP+DRILL 靜態 vs W-ECMP+DRILL+ML 兩條
    python3 plot_result.py ECMP=path1.csv W-ECMP=path2.csv ML=path3.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VAL_DIR = "research_results/data/validation"
OUT_IMG = "research_results/plots/validation/algorithm_comparison.png"
OUT_IMG_UTIL = "research_results/plots/validation/switch_utilization.png"

# 丟棄前 N 秒暖機 (佇列填充/規則安裝過渡)；可用 warmup=N 覆寫
DEFAULT_WARMUP_SEC = 5

# label -> (csv 路徑, 顏色)
DEFAULT_RUNS = {
    "ECMP":            (f"{VAL_DIR}/comparison_ecmp.csv",  "#1f77b4"),
    "W-ECMP":          (f"{VAL_DIR}/comparison_wecmp.csv", "#ff7f0e"),
    "DRILL":           (f"{VAL_DIR}/comparison_drill.csv", "#9467bd"),
    "W-ECMP+DRILL+ML": (f"{VAL_DIR}/comparison_ml.csv",    "#2ca02c"),
}

# --ML 模式：只比較固定容量比例權重 (靜態) vs ML 動態調權，兩者皆 W-ECMP+DRILL，唯一差別是
# 權重會不會被 ML 改動 -> 直接看出 ML 控制本身的增益。
ML_RUNS = {
    "W-ECMP+DRILL":    (f"{VAL_DIR}/comparison_wecmp_drill.csv", "#ff7f0e"),
    "W-ECMP+DRILL+ML": (f"{VAL_DIR}/comparison_ml.csv",          "#2ca02c"),
}


def parse_args():
    """支援 label=path 覆寫與 warmup=N；回傳 (runs, warmup_sec)。"""
    warmup = DEFAULT_WARMUP_SEC
    ml_only = False
    label_args = []
    for arg in sys.argv[1:]:
        if arg.lower() in ("--ml", "-ml"):
            ml_only = True
            continue
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        if k.lower() == "warmup":
            try: warmup = float(v)
            except ValueError: pass
        else:
            label_args.append((k, v))

    # --ML：只比較 W-ECMP+DRILL 靜態 vs ML 兩條 (忽略其餘 label= 覆寫)
    if ml_only:
        return ML_RUNS, warmup

    if not label_args:
        return DEFAULT_RUNS, warmup

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    runs = {label: (path, palette[i % len(palette)])
            for i, (label, path) in enumerate(label_args)}
    return runs, warmup


def plot_switch_utilization(loaded, out_img):
    """各演算法每個 spine 的平均利用率 (load/cap) 長條圖。

    util 已用容量正規化，因此「越平」代表負載越貼近容量比例 (越平衡)；
    圖例附上各演算法跨 spine 的標準差 σ (越小越平衡)。
    """
    # 從第一個 df 找出 util_s* 欄位 (依編號排序)
    first_df = next(iter(loaded.values()))[0]
    util_cols = sorted([c for c in first_df.columns if c.startswith('util_s')],
                       key=lambda c: int(c[len('util_s'):]))
    if not util_cols:
        print("  [略過] CSV 無 per-switch util 欄位 (util_s*)，跳過交換機利用率圖")
        return

    labels = [c.replace('util_', '') for c in util_cols]   # s1..s8
    x = np.arange(len(util_cols))
    n_alg = len(loaded)
    width = 0.8 / n_alg

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (label, (df, color, _)) in enumerate(loaded.items()):
        means = [df[c].mean() if c in df.columns else 0.0 for c in util_cols]
        sigma = float(np.std(means))
        ax.bar(x + i * width - 0.4 + width / 2, means, width,
               label=f"{label}  (σ={sigma:.2f})", color=color, alpha=0.85)

    ax.axhline(1.0, color='red', linestyle=':', alpha=0.6, label='capacity (util=1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Spine switch")
    ax.set_ylabel("Mean utilization (load / capacity)")
    ax.set_title("Per-Switch Utilization")
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    print(f"交換機利用率圖已存至 {out_img}")


def end_to_end_loss(df):
    """從累積計數器算端到端丟包率 (%)：(Σenq - Σrecv) / Σenq，跨整段視窗。

    用累積值而非每秒夾值差，佇列堆積/排空會自然抵消，避免瞬時估計的高估偏差。
    缺欄位時退回瞬時 Real_Loss 平均。
    """
    if 'Cum_Enq' in df.columns and 'Cum_Recv' in df.columns and len(df) >= 1:
        enq  = df['Cum_Enq'].iloc[-1]  - df['Cum_Enq'].iloc[0]
        recv = df['Cum_Recv'].iloc[-1] - df['Cum_Recv'].iloc[0]
        if enq > 0:
            return max(0.0, (enq - recv) / enq * 100.0)
    return float(df['Real_Loss'].mean()) if 'Real_Loss' in df.columns else float('nan')


def main():
    runs, warmup = parse_args()
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)

    loaded = {}
    for label, (path, color) in runs.items():
        if not os.path.exists(path):
            print(f"  [略過] 找不到 {label} 的 CSV: {path}")
            continue
        df = pd.read_csv(path)
        # 以 Timestamp 換算「實際經過秒數」；各演算法每輪耗時不同 (ML 較慢)，用真實時間對齊才公平。
        try:
            ts = pd.to_datetime(df['Timestamp'])
            elapsed = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
        except Exception:
            elapsed = np.arange(len(df), dtype=float)

        # 丟棄前 warmup 秒 (規則安裝過渡 + 佇列填充)，再把時間軸歸零
        mask = elapsed >= warmup
        if mask.sum() < 2:
            print(f"  [略過] {label} 暖機後資料不足 (run < {warmup}s?)")
            continue
        df_w = df[mask].reset_index(drop=True)
        x_w = elapsed[mask] - elapsed[mask][0]
        loaded[label] = (df_w, color, x_w)
        print(f"  [載入] {label}: {len(df)} 列 / {elapsed[-1]:.0f}s "
              f"(丟棄前 {warmup:.0f}s -> 剩 {len(df_w)} 列)  <- {path}")

    if not loaded:
        print("沒有任何可用的 CSV，無法繪圖。")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: 硬體延遲 — 每秒峰值佇列延遲原始值 (不平滑；峰值本身就是該秒的最壞情況)
    for label, (df, color, x) in loaded.items():
        axes[0].plot(x, df['Real_Lat'], 'o-', label=label,
                     color=color, alpha=0.8, markersize=3, linewidth=1.8)
    axes[0].set_ylabel("Latency (ms, per-sec peak)")
    axes[0].set_title(f"Algorithm Comparison — Hardware Ground Truth "
                      f"(8-spine, warmup {warmup:.0f}s dropped)")
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Panel 2: 瞬時丟包 (每秒估計，受佇列堆積影響有雜訊；端到端值見摘要表)
    for label, (df, color, x) in loaded.items():
        axes[1].plot(x, df['Real_Loss'], 'o-', label=label,
                     color=color, alpha=0.8, markersize=3, linewidth=1.8)
    axes[1].set_ylabel("Instantaneous Loss (%)")
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Panel 3: 總吞吐量
    for label, (df, color, x) in loaded.items():
        if 'Total_Mbps' in df.columns:
            axes[2].plot(x, df['Total_Mbps'], 'o-', label=label,
                         color=color, alpha=0.8, markersize=3, linewidth=1.8)
    axes[2].set_ylabel("Total Throughput (Mbps)")
    axes[2].set_xlabel("Time since warmup (seconds)")
    axes[2].legend(loc='upper left')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=150)
    print(f"\n比較圖已存至 {OUT_IMG}")

    # 各交換機利用率 (另存一張圖)
    plot_switch_utilization(loaded, OUT_IMG_UTIL)

    # 摘要統計表 — 延遲以每秒峰值的 p50/p95/max 表示 (尾延遲才是 load balancer 的重點，
    # 不用平均，避免把峰值平均掉)；Loss = 端到端累積丟包率；Util σ = 跨 spine 利用率標準差 (越小越平衡)
    long_run = max((float(x[-1]) if len(x) else 0.0) for _, _, x in loaded.values()) > 90.0
    tail_label = "Lat p99" if long_run else "Lat max"
    print(f"\n=== 指標摘要 (丟棄前 {warmup:.0f}s 暖機；延遲為每秒峰值的分位數) ===")
    print(f"{'演算法':^18} | {'Lat p50':^8} | {'Lat p95':^8} | {tail_label:^8} | "
          f"{'Loss(%)E2E':^11} | {'Mbps':^7} | {'Util σ':^7}")
    print("-" * 88)
    for label, (df, _, _) in loaded.items():
        util_cols = sorted([c for c in df.columns if c.startswith('util_s')],
                           key=lambda c: int(c[len('util_s'):]))
        sigma = float(np.std([df[c].mean() for c in util_cols])) if util_cols else float('nan')
        lat = df['Real_Lat']
        tail_value = lat.quantile(0.99) if long_run else lat.max()
        print(f"{label:^18} | {lat.median():8.2f} | {lat.quantile(0.95):8.2f} | {tail_value:8.2f} | "
              f"{end_to_end_loss(df):9.2f} | {df['Total_Mbps'].mean():6.2f} | {sigma:6.3f}")


if __name__ == "__main__":
    main()
