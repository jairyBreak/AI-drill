import sys
import os
import time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# controller logic
from realtime_ml_controller import MLController, RankECDF

def run_full_validation(test_duration=60, output_csv="research_results/data/validation/full_metrics_validation_good_hash_1.csv", output_img="research_results/plots/validation/full_metrics_comparison_good_hash_1.png"):
    print("\n" + "="*100)
    print(f" [全指標可視化驗證工具 v4.2] 開始測試 - 預計時長: {test_duration} 秒")
    print(f" 模式: 10秒離散採集 (對齊訓練資料集)")
    print(f" 數據: {output_csv} | 圖表: {output_img}")
    print("="*100 + "\n")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_img), exist_ok=True)

    sys.modules['__main__'].RankECDF = RankECDF
    ctrl = MLController()

    # warm up: one 10s collection to sync hardware state
    print(" [system] syncing hardware state (10s)...", end='', flush=True)
    ctrl.collect_window(duration=10.0)
    print(" 完成！\n")

    results = []
    start_time = time.time()
    
    print(f"{'時間':^10} | {'Lat (P/R)':^15} | {'Jit (P/R)':^15} | {'Loss (P/R)':^15} | {'Util'}")
    print("-" * 100)
    
    try:
        while time.time() - start_time < test_duration:
            # 1. collect 10s, extract features + predict
            df = ctrl.collect_window(duration=10.0)
            X, feats = ctrl.extract_features(df)
            
            preds = {}
            for k, m in ctrl.models.items():
                p = m.predict(X)[0]
                if k != "anomaly": p = np.expm1(p)
                preds[k] = p
            
            # 2. read ground truth from the background collector
            real_lat = ctrl.real_latency
            real_jit = ctrl.real_jitter
            real_loss = ctrl.real_loss
            
            # 3. record
            now_dt = datetime.now()
            entry = {
                'Timestamp': now_dt,
                'Pred_Lat': preds['latency'],
                'Real_Lat': real_lat if real_lat > 0 else np.nan,
                'Pred_Jit': preds['jitter'],
                'Real_Jit': real_jit,
                'Pred_Loss': preds['loss'],
                'Real_Loss': real_loss,
                'Util_Sum': feats['Total_Util_Sum']
            }
            results.append(entry)
            
            # 4. print status
            now_str = now_dt.strftime('%H:%M:%S')
            print(f"{now_str:^10} | "
                  f"{preds['latency']:5.1f}/{real_lat:4.1f} | "
                  f"{preds['jitter']:5.1f}/{real_jit:4.2f} | "
                  f"{preds['loss']:5.2f}/{real_loss:4.1f} | "
                  f"{feats['Total_Util_Sum']:4.2f}")
                
    except KeyboardInterrupt:
        print("\n測試已提前中止。")
    
    if not results:
        print("沒有數據可供保存。")
        return

    # save data
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"\ndata written -> {output_csv}")

    plot_df = res_df.copy()

    print("generating comparison plot...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    t_axis = range(len(plot_df))

    # Plot 1: Latency
    axes[0].plot(t_axis, plot_df['Real_Lat'], 'o-', label='Real Latency (Ping)', color='#1f77b4', alpha=0.7, markersize=4)
    axes[0].plot(t_axis, plot_df['Pred_Lat'], 's-', label='Predicted Latency (ML)', color='#d62728', linewidth=2)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_title(f"Network Performance Validation (v4.2 - 10s Discrete Sampling)")
    
    # dynamic Y-axis upper bound
    y_max_lat = max(plot_df['Real_Lat'].max() if not plot_df['Real_Lat'].isna().all() else 0,
                    plot_df['Pred_Lat'].quantile(0.98) if len(plot_df)>1 else plot_df['Pred_Lat'].max()) * 1.2
    if np.isnan(y_max_lat) or y_max_lat < 50: y_max_lat = 100
    axes[0].set_ylim(-10, y_max_lat)

    # Plot 2: Jitter
    axes[1].plot(t_axis, plot_df['Real_Jit'], 'o-', label='Real Jitter (iperf3)', color='#17becf', alpha=0.7, markersize=4)
    axes[1].plot(t_axis, plot_df['Pred_Jit'], 's-', label='Predicted Jitter (ML)', color='#ff7f0e', linewidth=2)
    axes[1].set_ylabel("Jitter (ms)")
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    y_max_jit = max(plot_df['Real_Jit'].max(), 
                    plot_df['Pred_Jit'].quantile(0.98) if len(plot_df)>1 else plot_df['Pred_Jit'].max()) * 1.2
    if np.isnan(y_max_jit) or y_max_jit < 20: y_max_jit = 40
    axes[1].set_ylim(-5, y_max_jit)

    # Plot 3: Loss
    axes[2].plot(t_axis, plot_df['Real_Loss'], 'o-', label='Real Loss (iperf3)', color='#7f7f7f', alpha=0.7, markersize=4)
    axes[2].plot(t_axis, plot_df['Pred_Loss'], 's-', label='Predicted Loss (ML)', color='#9467bd', linewidth=2)
    axes[2].set_ylabel("Loss Rate (%)")
    axes[2].set_xlabel("Sampling Steps (10s per step)")
    axes[2].legend(loc='upper left')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    y_max_loss = max(plot_df['Real_Loss'].max(), plot_df['Pred_Loss'].max()) * 1.2
    if np.isnan(y_max_loss) or y_max_loss < 5: y_max_loss = 10
    axes[2].set_ylim(-1, y_max_loss)

    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"動態自適應圖表已保存至 {output_img}")

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 70
    run_full_validation(duration)
