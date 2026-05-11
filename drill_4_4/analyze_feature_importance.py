import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 設定與訓練腳本一致的特徵清單
SELECTED_FEATURES = [
    "total_qdepth_p99", "total_qdepth_max", "max_qdepth_p99", "qdepth_fft_max_all",
    "src1_port2_mbps_mean", "src1_port3_mbps_mean", "src1_port4_mbps_mean", "src1_port5_mbps_mean",
    "src1_port2_load_util", "src1_port3_load_util", "src1_port4_load_util", "src1_port5_load_util",
    "src1_port2_qdepth_max", "src1_port3_qdepth_max", "src1_port4_qdepth_max", "src1_port5_qdepth_max",
    "src1_port2_mbps_cv", "src1_port3_mbps_cv", "src1_port4_mbps_cv", "src1_port5_mbps_cv",
    "total_qdepth_p99_ecdf", "max_qdepth_p99_ecdf", "mbps_imbalance_ecdf", "qdepth_fft_max_all_ecdf",
    "Total_Util_Sum", "Max_Util_Diff", "Group_Imbalance", "idx_load_balance",
    "qdepth_sq",
    "qdepth_slope"
]

def add_ultimate_features(df):
    df = df.copy()
    df["qdepth_sq"] = df["total_qdepth_p99"] ** 2
    df["qdepth_slope"] = df["total_qdepth_p99"].diff().fillna(0)
    utils = [df[f"src1_port{i}_load_util"] for i in [2,3,4,5] if f"src1_port{i}_load_util" in df.columns]
    if utils:
        df["Total_Util_Sum"] = sum(utils)
        util_stack = np.column_stack(utils)
        df["Max_Util_Diff"] = np.max(util_stack, axis=1) - np.min(util_stack, axis=1)
    means = [df[f"src1_port{i}_mbps_mean"] for i in [2,3,4,5] if f"src1_port{i}_mbps_mean" in df.columns]
    weights = [df[f"Weight_Port{i}"] for i in [2,3,4,5] if f"Weight_Port{i}" in df.columns]
    if len(means) == 4 and len(weights) == 4:
        load_a, w_a = means[0] + means[1], weights[0] + weights[1]
        load_b, w_b = means[2] + means[3], weights[2] + weights[3]
        df["Group_Imbalance"] = np.abs((load_a / w_a.replace(0,1)) - (load_b / w_b.replace(0,1)))
    return df

def run_analysis():
    print("=== 正在進行終極特徵重要性分析 ===\n")
    CSV_PATH = "research_results/data/datasets/training_dataset_ecdf_cleaned.csv"
    if not os.path.exists(CSV_PATH):
        print(f"錯誤：找不到資料集 {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    df = add_ultimate_features(df)
    
    # 使用當前配置訓練一個臨時模型來分析 (100 棵樹即可)
    from sklearn.ensemble import RandomForestRegressor
    
    label_col = "Label_Latency_ms"
    subset = df[df[label_col] != -1.0].copy()
    X = subset[[f for f in SELECTED_FEATURES if f in subset.columns]]
    y = np.log1p(subset[label_col])
    
    print(f"正在訓練分析模型 (特徵數: {X.shape[1]})...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n[結果] 特徵貢獻度排名:")
    report = []
    for f in range(X.shape[1]):
        feat_name = X.columns[indices[f]]
        score = importances[indices[f]]
        res = f"{f + 1:2d}) {feat_name:25}: {score:.4f}"
        print(res)
        report.append(res)
    
    # 繪圖
    plt.figure(figsize=(12, 8))
    plt.title("Ultimate Feature Importance (Latency Model)")
    plt.bar(range(X.shape[1]), importances[indices], color='gold')
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
    plt.tight_layout()
    output_png = "research_results/plots/analysis/ultimate_feature_importance.png"
    plt.savefig(output_png)
    print(f"\n分析圖表已存至 {output_png}")

    report_path = "research_results/plots/analysis/feature_importance_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    run_analysis()
