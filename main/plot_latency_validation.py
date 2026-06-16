import sys
import os
import time
import pandas as pd
import numpy as np
import joblib
import subprocess
import re
import matplotlib.pyplot as plt
from datetime import datetime

# controller logic
from realtime_ml_controller import MLController, RankECDF

def get_single_ping():
    """One synchronous ping -> latency (ms), 2s timeout."""
    try:
        output = subprocess.check_output(
            ["mx", "h1", "ping", "-c", "1", "-W", "2.0", "10.0.2.2"],
            stderr=subprocess.STDOUT, text=True
        )
        match = re.search(r'time=([\d\.]+)\s*ms', output)
        return float(match.group(1)) if match else -1.0
    except:
        return -1.0

def run_visual_validation(test_duration=60, output_csv="latency_validation_4.csv", output_img="latency_comparison_4.png"):
    print("\n" + "="*95)
    print(f" [延遲可視化驗證工具] 開始測試 - 預計時長: {test_duration} 秒")
    print(f" 數據將保存至: {output_csv} | 圖表將保存至: {output_img}")
    print("="*95 + "\n")
    
    ctrl = MLController()

    # warm up
    print(" [system] warming up data cache (10s)...", end='', flush=True)
    for _ in range(10):
        ctrl.collect_window(duration=1.0)
        print(".", end='', flush=True)
    print(" 完成！\n")

    results = []
    start_time = time.time()
    smoothed_lat = 20.0
    
    print(f"{'時間':^10} | {'真實延遲':^12} | {'預測延遲 (EMA)':^15} | {'狀態'}")
    print("-" * 95)
    
    try:
        while time.time() - start_time < test_duration:
            # 1. collect feature window
            df = ctrl.collect_window(duration=1.0)
            X, feats = ctrl.extract_features(df)

            # 2. ground truth from background collector (10s avg)
            real_lat = ctrl.real_latency

            # 3. predict + invert log
            pred_log = ctrl.models['latency'].predict(X)[0]
            raw_pred_lat = np.expm1(pred_log)

            # 4. asymmetric EMA smoothing
            alpha = 0.8 if raw_pred_lat < smoothed_lat else 0.3
            smoothed_lat = (alpha * raw_pred_lat) + ((1 - alpha) * smoothed_lat)

            # 5. record
            now_dt = datetime.now()
            now_str = now_dt.strftime('%H:%M:%S')

            # record prediction even on timeout (real value = NaN)
            entry = {
                'Timestamp': now_dt,
                'Real_Latency': real_lat if real_lat > 0 else np.nan,
                'Predicted_Latency': smoothed_lat,
                'Raw_Prediction': raw_pred_lat,
                'Util_Sum': feats['Total_Util_Sum']
            }
            results.append(entry)
            
            # 6. print
            if real_lat > 0:
                print(f"{now_str:^10} | {real_lat:8.2f} ms | {smoothed_lat:10.2f} ms | OK")
            else:
                print(f"{now_str:^10} | {'TIMEOUT':^12} | {smoothed_lat:10.2f} ms | WARN")
                
    except KeyboardInterrupt:
        print("\n測試已提前中止。")
    
    if not results:
        print("沒有數據可供保存。")
        return

    # save CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"\ndata written -> {output_csv}")

    print("generating comparison plot...")
    plt.figure(figsize=(12, 6))

    t_axis = range(len(res_df))
    plt.plot(t_axis, res_df['Real_Latency'], 'o-', label='Real Latency (Ping)', color='blue', alpha=0.6, markersize=4)
    plt.plot(t_axis, res_df['Predicted_Latency'], 's-', label='Predicted Latency (ML EMA)', color='red', linewidth=2)
    
    # decoration
    plt.title(f"Real vs. Predicted Latency Over Time : 0.4 M * 10 Flows (Duration: {test_duration}s)")
    plt.xlabel("Sampling Steps (approx 1.5s per step)")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # log-scale Y if latency spans too wide
    if res_df['Real_Latency'].max() > 1000:
        plt.yscale('symlog')
        plt.ylabel("Latency (ms) - Log Scale")
        print("提示：由於延遲跨度較大，圖表縱軸已切換為對數比例。")

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"對比圖表已保存至 {output_img}")
    
    # final stats
    valid_df = res_df.dropna(subset=['Real_Latency'])
    if not valid_df.empty:
        mae = np.mean(np.abs(valid_df['Real_Latency'] - valid_df['Predicted_Latency']))
        print(f"測試完成！平均絕對誤差 (MAE): {mae:.2f} ms")

if __name__ == "__main__":
    sys.modules['__main__'].RankECDF = RankECDF
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_visual_validation(duration)
