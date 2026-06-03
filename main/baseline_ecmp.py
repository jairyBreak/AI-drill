"""
baseline_ecmp.py — 純 ECMP 基準 (等權，無 group，無 DRILL，無 ML)

每個 spine 視為獨立的單埠 component、權重相同 —— 5-tuple 均勻雜湊到 8 個 spine。
用來與 W-ECMP+DRILL+ML 控制器比較。

用法 (需先 sudo p4run；本腳本取代 all_controller 的設定)：
    python3 baseline_ecmp.py [量測秒數]

預設量測 60 秒，輸出沿用既有 plot_1s_metrics CSV 格式。
流量請另開終端機用 traffic.py 產生 (本腳本不管理流量)。
"""

import sys
from baseline_common import run_baseline

OUTPUT_CSV = "research_results/data/validation/comparison_ecmp.csv"

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_baseline(mode='equal', label='ECMP', duration=duration, csv_path=OUTPUT_CSV)
