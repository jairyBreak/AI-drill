"""
baseline_wecmp.py — 靜態 W-ECMP 基準 (頻寬加權，無 DRILL，無 ML)

每個 spine 視為獨立的單埠 component，member 數正比於上行頻寬：
    0.6/0.6/0.8/0.8/1.0/1.0/1.2/1.2 Mbps  ->  權重 3/3/4/4/5/5/6/6
流量按頻寬比例雜湊分配，但權重固定不會隨壅塞調整 (與 ML 控制器的差異即在此)。

用法 (需先 sudo p4run；本腳本取代 all_controller 的設定)：
    python3 baseline_wecmp.py [量測秒數]

預設量測 60 秒，輸出沿用既有 plot_1s_metrics CSV 格式。
流量請另開終端機用 traffic.py 產生 (本腳本不管理流量)。
"""

import sys
from baseline_common import run_baseline

OUTPUT_CSV = "research_results/data/validation/comparison_wecmp.csv"

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_baseline(mode='bw', label='W-ECMP', duration=duration, csv_path=OUTPUT_CSV)
