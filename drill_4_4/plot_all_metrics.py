import sys
import os
import time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# 導入控制器邏輯
from realtime_ml_controller import MLController, RankECDF

def run_full_validation(test_duration=60, output_csv="full_metrics_validation_4.csv", output_img="full_metrics_comparison_4.png"):
    print("\n" + "="*100)
    print(f" [全指標可視化驗證工具] 開始測試 - 預計時長: {test_duration} 秒")
    print(f" 目標指標: Latency, Jitter, Loss")
    print(f" 數據: {output_csv} | 圖表: {output_img}")
    print("="*100 + "\n")
    
    # 設定環境
    sys.modules['__main__'].RankECDF = RankECDF
    ctrl = MLController()
    
    results = []
    start_time = time.time()
    
    print(f"{'時間':^10} | {'Lat (P/R)':^15} | {'Jit (P/R)':^15} | {'Loss (P/R)':^15} | {'Util'}")
    print("-" * 100)
    
    try:
        while time.time() - start_time < test_duration:
            # 1. 採集特徵與預測
            df = ctrl.collect_window(duration=1.0)
            X, feats = ctrl.extract_features(df)
            
            preds = {}
            for k, m in ctrl.models.items():
                p = m.predict(X)[0]
                if k != "anomaly": p = np.expm1(p)
                preds[k] = p
            
            # 2. 獲取背景執行緒採集的真實值
            real_lat = ctrl.real_latency
            real_jit = ctrl.real_jitter
            real_loss = ctrl.real_loss
            
            # 3. 記錄數據
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
            
            # 4. 打印簡要狀態
            now_str = now_dt.strftime('%H:%M:%S')
            print(f"{now_str:^10} | "
                  f"{preds['latency']:5.1f}/{real_lat:4.1f} | "
                  f"{preds['jitter']:5.2f}/{real_jit:4.2f} | "
                  f"{preds['loss']:5.2f}/{real_loss:4.1f} | "
                  f"{feats['Total_Util_Sum']:4.2f}")
                
    except KeyboardInterrupt:
        print("\n測試已提前中止。")
    
    if not results:
        print("沒有數據可供保存。")
        return

    # 保存數據
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"\n數據已寫入 {output_csv}")

    # 繪圖
    print("正在生成三指標對比圖表...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    t_axis = range(len(res_df))

    # Plot 1: Latency
    axes[0].plot(t_axis, res_df['Real_Lat'], 'o-', label='Real Latency (Ping)', color='blue', alpha=0.5, markersize=3)
    axes[0].plot(t_axis, res_df['Pred_Lat'], 's-', label='Predicted Latency (ML)', color='red', linewidth=1.5)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_title("Network Performance Validation: Predicted vs Real")

    # Plot 2: Jitter
    axes[1].plot(t_axis, res_df['Real_Jit'], 'o-', label='Real Jitter (iperf3)', color='cyan', alpha=0.5, markersize=3)
    axes[1].plot(t_axis, res_df['Pred_Jit'], 's-', label='Predicted Jitter (ML)', color='darkorange', linewidth=1.5)
    axes[1].set_ylabel("Jitter (ms)")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Loss
    axes[2].plot(t_axis, res_df['Real_Loss'], 'o-', label='Real Loss (iperf3)', color='black', alpha=0.5, markersize=3)
    axes[2].plot(t_axis, res_df['Pred_Loss'], 's-', label='Predicted Loss (ML)', color='purple', linewidth=1.5)
    axes[2].set_ylabel("Loss Rate (%)")
    axes[2].set_xlabel("Sampling Steps (approx 1.5s per step)")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"對比圖表已保存至 {output_img}")

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_full_validation(duration)
