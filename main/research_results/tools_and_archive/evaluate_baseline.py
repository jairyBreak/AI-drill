import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score

# 原版固定參數
ORIGINAL_PARAMS = dict(
    n_estimators=200, 
    max_depth=15,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)

SELECTED_FEATURES = [
    "Total_Util_Sum", "Max_Util_Diff", "Group_Imbalance",
    "Norm_Load_P2", "Norm_Load_P3", "Norm_Load_P4", "Norm_Load_P5",
    "idx_load_balance", "mbps_imbalance", "max_qdepth_p99",
    "total_qdepth_p99", "total_qdepth_max", "qdepth_max_imbalance",
    "qdepth_fft_max_all", "Weight_Port2", "Weight_Port3", "Weight_Port4", "Weight_Port5",
    "src1_port3_mbps_cv", "src1_port5_mbps_cv", "src1_port4_mbps_cv", "src1_port2_mbps_cv",
    "src1_port5_load_util", "src1_port3_load_util", "src1_port4_load_util", "src1_port2_load_util",
    "src1_port3_qdepth_max", "src1_port5_qdepth_max", "src1_port4_qdepth_max", "src1_port2_qdepth_max"
]

def add_ratio_features(df):
    df = df.copy()
    utils = []
    for i in [2, 3, 4, 5]:
        df[f"Norm_Load_P{i}"] = df[f"src1_port{i}_mbps_mean"] / df[f"Weight_Port{i}"].replace(0, 1)
        u_col = f"src1_port{i}_load_util"
        if u_col in df.columns: utils.append(df[u_col])
    if utils:
        df["Total_Util_Sum"] = sum(utils)
        df["Max_Util_Diff"] = np.max(np.column_stack(utils), axis=1) - np.min(np.column_stack(utils), axis=1)
    load_a = df["src1_port2_mbps_mean"] + df["src1_port3_mbps_mean"]
    weight_a = df["Weight_Port2"] + df["Weight_Port3"]
    load_b = df["src1_port4_mbps_mean"] + df["src1_port5_mbps_mean"]
    weight_b = df["Weight_Port4"] + df["Weight_Port5"]
    df["Group_Imbalance"] = np.abs((load_a / weight_a.replace(0, 1)) - (load_b / weight_b.replace(0, 1)))
    return df

CSV_PATH = "training_dataset_ecdf_cleaned.csv"

def evaluate_baseline():
    print("=== 正在評估原版模型 Baseline (固定參數) ===\n")
    df = pd.read_csv(CSV_PATH)
    df = add_ratio_features(df)
    
    report = []
    report.append(f"Evaluation Date: {pd.Timestamp.now()}")
    report.append(f"Dataset: {CSV_PATH} ({len(df)} rows)")
    report.append("-" * 50)

    # 迴歸模型
    targets = [("Label_Latency_ms", "Latency"), ("Label_Latency_p99_ms", "P99 Latency"), 
               ("Label_Jitter_ms", "Jitter"), ("Label_Loss_Rate", "Loss Rate")]
    
    for col, name in targets:
        subset = df[df[col] != -1.0].copy()
        X = subset[[f for f in SELECTED_FEATURES if f in subset.columns]]
        y = np.log1p(subset[col])
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(RandomForestRegressor(**ORIGINAL_PARAMS), X, y, cv=cv, scoring='r2')
        print(f"Model: {name:15} | R2: {np.mean(scores):.4f}")
        report.append(f"{name:15} | R2: {np.mean(scores):.4f}")

    # 分類模型
    if "Label_Anomaly" not in df.columns:
        df["Label_Anomaly"] = (df["Label_Loss_Rate"] > 0.001).astype(int)
    X = df[[f for f in SELECTED_FEATURES if f in df.columns]]
    y = df["Label_Anomaly"]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(RandomForestClassifier(**ORIGINAL_PARAMS), X, y, cv=cv, scoring='accuracy')
    print(f"Model: Anomaly (All)   | Acc: {np.mean(acc):.4f}")
    report.append(f"{'Anomaly':15} | Acc: {np.mean(acc):.4f}")

    with open("baseline_report.txt", "w") as f:
        f.write("\n".join(report))
    print("\n[成功] 原版表現已記錄至 baseline_report.txt")

if __name__ == "__main__":
    evaluate_baseline()
