import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", ResourceWarning)

# 導入 1s 拓樸無關控制器與 helper 邏輯
from realtime_1s_predictor_topo_indep import Realtime1sPredictorTopoIndep, FEATURE_NAMES, PORTS, CAPACITY
from topo_independent_helper import transform_to_topo_independent

def run_1s_validation(test_duration=60, output_csv="research_results/data/validation/1s_metrics_validation_8port_0.6*10.csv", output_img="research_results/plots/validation/1s_metrics_comparison_8port_0.6*10.png"):
    print("\n" + "="*100)
    print(f" [1s 尺度拓樸無關預測驗證工具] 開始測試 - 預計時長: {test_duration} 秒")
    print(f" 數據: {output_csv} | 圖表: {output_img}")
    print("="*100 + "\n")
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    
    # 初始化預測器
    predictor = Realtime1sPredictorTopoIndep()

    results = []
    raw_history = collections.deque(maxlen=100)
    start_time = time.time()
    
    print(f"{'時間':^10} | {'Lat (P/R)':^15} | {'Loss (P/R)':^15} | {'Util'}")
    print("-" * 80)
    
    try:
        while time.time() - start_time < test_duration:
            # 1. 採集 1 秒數據與真實標籤
            data_row = predictor.collect_1s_data()
            raw_history.append(data_row)
            
            # 2. 將歷史轉為 DataFrame 進行特徵計算
            df_history = pd.DataFrame(list(raw_history))
            df_transformed = transform_to_topo_independent(df_history, PORTS, CAPACITY, K=3)
            last_row = df_transformed.iloc[-1]
            
            # 3. 模型預測
            X = pd.DataFrame([last_row[FEATURE_NAMES]])
            preds = {}
            for k, m in predictor.models.items():
                p = m.predict(X)[0]
                if k == "latency":
                    p = np.expm1(p)
                preds[k] = p
            
            # 4. 記錄數據
            now_dt = datetime.now()
            entry = {
                'Timestamp': now_dt,
                'Pred_Lat': preds['latency'],
                'Real_Lat': data_row['Real_HW_Latency_ms'],
                'Pred_Loss': preds['loss'],
                'Real_Loss': data_row['Real_HW_Loss_Percent'],
                'Util_Sum': last_row['Total_Util_Sum'],
                'Total_Mbps': last_row['Total_Actual_Mbps']
            }
            results.append(entry)
            
            # 5. 打印狀態
            now_str = now_dt.strftime('%H:%M:%S')
            print(f"{now_str:^10} | "
                  f"{preds['latency']:5.1f}/{data_row['Real_HW_Latency_ms']:4.1f} | "
                  f"{preds['loss']:5.2f}/{data_row['Real_HW_Loss_Percent']:4.1f} | "
                  f"{last_row['Total_Util_Sum']:4.2f}")
                
    except KeyboardInterrupt:
        print("\n測試已提前中止。")
    
    if not results:
        print("沒有數據可供保存。")
        return

    # 保存數據
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"\n數據已寫入 {output_csv}")

    # 繪圖前處理
    plot_df = res_df.copy()
    
    # 設置繪圖
    print("正在生成對比圖表...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    t_axis = range(len(plot_df))

    # Plot 1: Latency
    axes[0].plot(t_axis, plot_df['Real_Lat'], 'o-', label='Real Latency (Hardware)', color='#1f77b4', alpha=0.7, markersize=4)
    axes[0].plot(t_axis, plot_df['Pred_Lat'], 's-', label='Predicted Latency (ML 1s)', color='#d62728', linewidth=2)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_title("Network Performance Validation (1s Scale - 8Port Topo)")
    
    # 動態計算 Y 軸上限
    y_max_lat = max(plot_df['Real_Lat'].max(), plot_df['Pred_Lat'].max()) * 1.2
    if np.isnan(y_max_lat) or y_max_lat < 50: y_max_lat = 100
    axes[0].set_ylim(-5, y_max_lat)

    # Plot 2: Loss
    axes[1].plot(t_axis, plot_df['Real_Loss'], 'o-', label='Real Loss (Hardware)', color='#7f7f7f', alpha=0.7, markersize=4)
    axes[1].plot(t_axis, plot_df['Pred_Loss'], 's-', label='Predicted Loss (ML 1s)', color='#9467bd', linewidth=2)
    axes[1].set_ylabel("Loss Rate (%)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    y_max_loss = max(plot_df['Real_Loss'].max(), plot_df['Pred_Loss'].max()) * 1.2
    if np.isnan(y_max_loss) or y_max_loss < 5: y_max_loss = 10
    axes[1].set_ylim(-1, y_max_loss)

    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"圖表已保存至 {output_img}")

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_1s_validation(duration)
