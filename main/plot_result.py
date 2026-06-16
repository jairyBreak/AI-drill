"""
plot_result.py — overlay-compare ECMP / W-ECMP / DRILL / W-ECMP+DRILL+ML.
Loads each algorithm's CSV (same columns) and overlays them + prints a summary table.
Usage:
    python3 plot_result.py
    python3 plot_result.py --ML        # only W-ECMP+DRILL static vs +ML
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

# drop first N seconds as warmup (queue fill / rule-install transient); override with warmup=N
DEFAULT_WARMUP_SEC = 5

# label -> (csv path, color)
DEFAULT_RUNS = {
    "ECMP":            (f"{VAL_DIR}/comparison_ecmp.csv",  "#1f77b4"),
    "W-ECMP":          (f"{VAL_DIR}/comparison_wecmp.csv", "#ff7f0e"),
    "DRILL":           (f"{VAL_DIR}/comparison_drill.csv", "#9467bd"),
    "W-ECMP+DRILL+ML": (f"{VAL_DIR}/comparison_ml.csv",    "#2ca02c"),
}

# --ML mode: static capacity weights vs ML dynamic weights (both W-ECMP+DRILL) -> isolates the ML gain
ML_RUNS = {
    "W-ECMP+DRILL":    (f"{VAL_DIR}/comparison_wecmp_drill.csv", "#ff7f0e"),
    "W-ECMP+DRILL+ML": (f"{VAL_DIR}/comparison_ml.csv",          "#2ca02c"),
}


def parse_args():
    """Supports label=path overrides and warmup=N; returns (runs, warmup_sec)."""
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

    # --ML: only static vs ML (ignore other label= overrides)
    if ml_only:
        return ML_RUNS, warmup

    if not label_args:
        return DEFAULT_RUNS, warmup

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    runs = {label: (path, palette[i % len(palette)])
            for i, (label, path) in enumerate(label_args)}
    return runs, warmup


def plot_switch_utilization(loaded, out_img):
    """Per-spine mean utilization (load/cap) bar chart; legend shows cross-spine σ (lower = balanced)."""
    # util_s* columns from the first df, sorted by number
    first_df = next(iter(loaded.values()))[0]
    util_cols = sorted([c for c in first_df.columns if c.startswith('util_s')],
                       key=lambda c: int(c[len('util_s'):]))
    if not util_cols:
        print("  [skip] no per-switch util columns (util_s*)")
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
    print(f"switch utilization plot saved -> {out_img}")


def end_to_end_loss(df):
    """E2E loss % from cumulative counters: (Σenq - Σrecv) / Σenq; falls back to mean Real_Loss."""
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
            print(f"  [skip] CSV not found for {label}: {path}")
            continue
        df = pd.read_csv(path)
        # use Timestamp for true elapsed seconds (runs differ in duration; align by real time)
        try:
            ts = pd.to_datetime(df['Timestamp'])
            elapsed = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
        except Exception:
            elapsed = np.arange(len(df), dtype=float)

        # drop the first warmup seconds, then zero the time axis
        mask = elapsed >= warmup
        if mask.sum() < 2:
            print(f"  [skip] {label} too little data after warmup (run < {warmup}s?)")
            continue
        df_w = df[mask].reset_index(drop=True)
        x_w = elapsed[mask] - elapsed[mask][0]
        loaded[label] = (df_w, color, x_w)
        print(f"  [load] {label}: {len(df)} rows / {elapsed[-1]:.0f}s "
              f"(dropped first {warmup:.0f}s -> {len(df_w)} rows)  <- {path}")

    if not loaded:
        print("no usable CSV, nothing to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: hardware latency — per-second peak queue delay (raw; the peak is the worst case)
    for label, (df, color, x) in loaded.items():
        axes[0].plot(x, df['Real_Lat'], 'o-', label=label,
                     color=color, alpha=0.8, markersize=3, linewidth=1.8)
    axes[0].set_ylabel("Latency (ms, per-sec peak)")
    axes[0].set_title(f"Algorithm Comparison — Hardware Ground Truth "
                      f"(8-spine, warmup {warmup:.0f}s dropped)")
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Panel 2: instantaneous loss (per-sec estimate, noisy; E2E value in the summary table)
    for label, (df, color, x) in loaded.items():
        axes[1].plot(x, df['Real_Loss'], 'o-', label=label,
                     color=color, alpha=0.8, markersize=3, linewidth=1.8)
    axes[1].set_ylabel("Instantaneous Loss (%)")
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Panel 3: total throughput
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
    print(f"\ncomparison plot saved -> {OUT_IMG}")

    # per-switch utilization (separate figure)
    plot_switch_utilization(loaded, OUT_IMG_UTIL)

    # summary table — latency as p50/p95/max of per-second peaks (tail is what matters, not mean);
    # Loss = E2E cumulative; Util σ = cross-spine utilization spread (lower = balanced)
    long_run = max((float(x[-1]) if len(x) else 0.0) for _, _, x in loaded.values()) > 90.0
    tail_label = "Lat p99" if long_run else "Lat max"
    print(f"\n=== metric summary (first {warmup:.0f}s warmup dropped; latency = per-sec peak quantiles) ===")
    print(f"{'Algorithm':^18} | {'Lat p50':^8} | {'Lat p95':^8} | {tail_label:^8} | "
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
