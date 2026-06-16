"""
baseline_ecmp.py — pure ECMP baseline (equal weight, no group, no DRILL, no ML).
8 single-port components, equal weight -> 5-tuple hashed uniformly across spines.
Usage (run sudo p4run first; this replaces all_controller's config):
    python3 baseline_ecmp.py [duration_s]   (default 60; traffic via traffic.py separately)
"""

import sys
from baseline_common import run_baseline

OUTPUT_CSV = "research_results/data/validation/comparison_ecmp.csv"

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_baseline(mode='equal', label='ECMP', duration=duration, csv_path=OUTPUT_CSV)
