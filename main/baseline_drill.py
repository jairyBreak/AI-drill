"""
baseline_drill.py — pure DRILL baseline (no group, no weighting, no ML).
All spines in one component (num_nhops=N), single selector member -> every packet enters it and
run_drill picks the shortest of 2 random ports + memory. Capacity-blind per-packet balancing.
Usage (run sudo p4run first; this replaces all_controller's config):
    python3 baseline_drill.py [duration_s]   (default 60; traffic via traffic.py separately)
"""

import sys
from baseline_common import run_baseline

OUTPUT_CSV = "research_results/data/validation/comparison_drill.csv"

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_baseline(mode='drill', label='DRILL', duration=duration, csv_path=OUTPUT_CSV)
