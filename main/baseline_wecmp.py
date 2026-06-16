"""
baseline_wecmp.py — static W-ECMP baseline (bandwidth-weighted, no DRILL, no ML).
8 single-port components, member count ∝ uplink bw; weights fixed (never adjust to congestion —
that is the difference from the ML controller).
Usage (run sudo p4run first; this replaces all_controller's config):
    python3 baseline_wecmp.py [duration_s]   (default 60; traffic via traffic.py separately)
"""

import sys
from baseline_common import run_baseline

OUTPUT_CSV = "research_results/data/validation/comparison_wecmp.csv"

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_baseline(mode='bw', label='W-ECMP', duration=duration, csv_path=OUTPUT_CSV)
