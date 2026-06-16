import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

CSV_PATH = "research_results/data/datasets/training_dataset_ecdf.csv"

# RF params for the 10s models (large forest, deeper trees)
BEST_PARAMS = dict(
    n_estimators=800,
    max_depth=20,
    min_samples_leaf=2,
    max_features=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

# feature list (30 base + extras)
SELECTED_FEATURES = [
    "Total_Util_Sum", "Max_Util_Diff", "Group_Imbalance",
    "Norm_Load_P2", "Norm_Load_P3", "Norm_Load_P4", "Norm_Load_P5",
    "idx_load_balance", "mbps_imbalance", "max_qdepth_p99",
    "total_qdepth_p99", "total_qdepth_mean", "total_qdepth_max", "qdepth_max_imbalance",
    "qdepth_fft_max_all", "Weight_Port2", "Weight_Port3", "Weight_Port4", "Weight_Port5",
    "src1_port3_mbps_cv", "src1_port5_mbps_cv", "src1_port4_mbps_cv", "src1_port2_mbps_cv",
    "src1_port5_load_util", "src1_port3_load_util", "src1_port4_load_util", "src1_port2_load_util",
    "src1_port3_qdepth_max", "src1_port5_qdepth_max", "src1_port4_qdepth_max", "src1_port2_qdepth_max",
    "qdepth_danger_ratio_port2", "qdepth_danger_ratio_port3","qdepth_danger_ratio_port4","qdepth_danger_ratio_port5",        
    "qdepth_oscillation"
]

def add_evolved_features(df):
    df = df.copy()
    
    
    # ensure base features present
    for i in [2, 3, 4, 5]:
        df[f"Norm_Load_P{i}"] = df[f"src1_port{i}_mbps_mean"] / df[f"Weight_Port{i}"].replace(0, 1)
    
    utils = [df[f"src1_port{i}_load_util"] for i in [2,3,4,5] if f"src1_port{i}_load_util" in df.columns]
    if utils:
        df["Total_Util_Sum"] = sum(utils)
        df["Max_Util_Diff"] = np.max(np.column_stack(utils), axis=1) - np.min(np.column_stack(utils), axis=1)
    
    load_a, weight_a = df["src1_port2_mbps_mean"]+df["src1_port3_mbps_mean"], df["Weight_Port2"]+df["Weight_Port3"]
    load_b, weight_b = df["src1_port4_mbps_mean"]+df["src1_port5_mbps_mean"], df["Weight_Port4"]+df["Weight_Port5"]
    df["Group_Imbalance"] = np.abs((load_a/weight_a.replace(0,1)) - (load_b/weight_b.replace(0,1)))
    return df


def train_evolved_baseline():
    print("=== 正在訓練經典進化版模型 (致敬最強 Baseline) ===\n")
    df = pd.read_csv(CSV_PATH)

    original_len = len(df)
    df = df[df["Label_Latency_ms"] >= 0]
    print(f"剔除了 {original_len - len(df)} 筆無效樣本 (保留極端高延遲)")

    df = add_evolved_features(df)
    
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    print(f"使用特徵數: {len(available_features)} (30個經典特徵 + 2個新特徵)\n")

    targets = [
        ("Label_Latency_ms", "latency"), 
        ("Label_Jitter_ms", "jitter"), 
        ("Label_Loss_Rate", "loss")
    ]
    
    for label_col, name in targets:
        subset = df[df[label_col] != -1.0].copy()
        X, y_log = subset[available_features], np.log1p(subset[label_col])   # np.log1p(subset[label_col])  
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_list, mae_list = [], []
        
        for train_idx, test_idx in cv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_log_tr, y_te_raw = y_log.iloc[train_idx], subset[label_col].iloc[test_idx]
            
            model = RandomForestRegressor(**BEST_PARAMS)
            model.fit(X_tr, y_log_tr)
            
            preds_orig = np.expm1(model.predict(X_te))
            r2_list.append(r2_score(np.log1p(y_te_raw), model.predict(X_te)))
            mae_list.append(mean_absolute_error(y_te_raw, preds_orig))

            #preds = model.predict(X_te)
            #r2_list.append(r2_score(y_te_raw, preds))  # R2 in real space
            #mae_list.append(mean_absolute_error(y_te_raw, preds))
            
        unit = "%" if name == "loss" else "ms"
        print(f"[{name:15}] R2(log): {np.mean(r2_list):.4f} | MAE: {np.mean(mae_list):.4f} {unit}")

        final_model = RandomForestRegressor(**BEST_PARAMS)
        final_model.fit(X, y_log)
        joblib.dump(final_model, f"rf_model_{name}_simplified.pkl")

        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1] # descending
        
        print(f"\n--- [{name}] Top 10 特徵重要性 ---")
        for i in range(len(available_features)):
            feat_name = available_features[indices[i]]
            feat_weight = importances[indices[i]]
            print(f"{i+1:2d}. {feat_name:<28} ({feat_weight:.4f})")
        print("-" * 40 + "\n")
    # anomaly classifier
    X, y = df[available_features], (df["Label_Loss_Rate"] > 0.001).astype(int)
    clf = RandomForestClassifier(**BEST_PARAMS)
    clf.fit(X, y)
    joblib.dump(clf, "rf_model_anomaly_simplified.pkl")
    print(f"\n[anomaly        ] Accuracy: 訓練完成。")

    print("\n=== 進化完成！這是基於最強經典版的一次物理升級。 ===")

if __name__ == "__main__":
    train_evolved_baseline()
