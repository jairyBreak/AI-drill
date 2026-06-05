"""
baseline_drill.py — 純 DRILL 基準 (無 group、無加權、無 ML)

把全部 spine 放進「單一 component」(num_nhops=N)、W-ECMP selector 只有一個 member，
因此每個封包都進入這個 component，run_drill 在全部 N 個 spine 中隨機抽 2 個 + 記憶埠，
挑佇列最短者 —— 純 DRILL 逐封包負載平衡，無容量先驗。
用來與 ECMP / W-ECMP / W-ECMP+DRILL+ML 控制器比較。

用法 (需先 sudo p4run；本腳本取代 all_controller 的設定)：
    python3 baseline_drill.py [量測秒數]

預設量測 60 秒，輸出沿用既有 plot_1s_metrics CSV 格式。
流量請另開終端機用 traffic.py 產生 (本腳本不管理流量)。
"""

import sys
from baseline_common import run_baseline

OUTPUT_CSV = "research_results/data/validation/comparison_drill.csv"

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_baseline(mode='drill', label='DRILL', duration=duration, csv_path=OUTPUT_CSV)
