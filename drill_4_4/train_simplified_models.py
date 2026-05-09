import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, accuracy_score

# ==========================================
# 選定的核心特徵 (包含新加入的 Easy Features)
# ==========================================
SELECTED_FEATURES = [
    "Total_Util_Sum",
    "Max_Util_Diff",
    "Group_Imbalance",
    "Norm_Load_P2", "Norm_Load_P3", "Norm_Load_P4", "Norm_Load_P5",
    "idx_load_balance",
    "mbps_imbalance",
    "max_qdepth_p99",
    "total_qdepth_p99",
    "total_qdepth_max",
    "qdepth_max_imbalance",
    "qdepth_fft_max_all",
    "Weight_Port2", "Weight_Port3", "Weight_Port4", "Weight_Port5",
    "src1_port3_mbps_cv",
    "src1_port5_mbps_cv",
    "src1_port4_mbps_cv",
    "src1_port2_mbps_cv",
    "src1_port5_load_util",
    "src1_port3_load_util",
    "src1_port4_load_util",
    "src1_port2_load_util",
    "src1_port3_qdepth_max",
    "src1_port5_qdepth_max",
    "src1_port4_qdepth_max",
    "src1_port2_qdepth_max"
]

def add_ratio_features(df):
    # --- 新增：特徵工程 (Ratio-based & Saturation) ---
    utils = []
    for i in [2, 3, 4, 5]:
        # 避免除以零
        df[f"Norm_Load_P{i}"] = df[f"src1_port{i}_mbps_mean"] / df[f"Weight_Port{i}"].replace(0, 1)
        
        # 收集利用率列用於計算匯總指標
        u_col = f"src1_port{i}_load_util"
        if u_col in df.columns:
            utils.append(df[u_col])
    
    if utils:
        df["Total_Util_Sum"] = sum(utils)
        # 計算各端口利用率的最大偏差
        util_matrix = np.column_stack(utils)
        df["Max_Util_Diff"] = np.max(util_matrix, axis=1) - np.min(util_matrix, axis=1)
    else:
        df["Total_Util_Sum"] = 0.0
        df["Max_Util_Diff"] = 0.0
    
    # Group A (P2,3) vs Group B (P4,5) 的負載平衡度
    load_a = df["src1_port2_mbps_mean"] + df["src1_port3_mbps_mean"]
    weight_a = df["Weight_Port2"] + df["Weight_Port3"]
    load_b = df["src1_port4_mbps_mean"] + df["src1_port5_mbps_mean"]
    weight_b = df["Weight_Port4"] + df["Weight_Port5"]
    
    df["Group_Imbalance"] = np.abs((load_a / weight_a.replace(0, 1)) - (load_b / weight_b.replace(0, 1)))
    return df

REGRESSION_TARGETS = [
    ("Label_Latency_ms",     "latency_ms"),
    ("Label_Latency_p99_ms", "latency_p99_ms"),
    ("Label_Jitter_ms",      "jitter_ms"),
    ("Label_Loss_Rate",      "loss_rate"),
]

RF_PARAMS = dict(
    n_estimators=200, 
    max_depth=15,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)

# 使用清洗後的數據集
CSV_PATH = "training_dataset_ecdf_cleaned.csv"

def train_regression():
    df = pd.read_csv(CSV_PATH)
    df = add_ratio_features(df)
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    print(f"使用的特徵數量: {len(available_features)} / {len(SELECTED_FEATURES)}")
    
    results = []
    for label_col, label_name in REGRESSION_TARGETS:
        if label_col not in df.columns:
            continue
            
        subset = df[df[label_col] != -1.0].copy()
        X = subset[available_features]
        # 使用對數轉換處理標籤
        y = np.log1p(subset[label_col])
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores = []
        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model = RandomForestRegressor(**RF_PARAMS)
            model.fit(X_tr, y_tr)
            r2_scores.append(r2_score(y_te, model.predict(X_te)))
        
        mean_r2 = np.mean(r2_scores)
        print(f"Model: {label_name:15} | R2: {mean_r2:.4f}")
        
        final_model = RandomForestRegressor(**RF_PARAMS)
        final_model.fit(X, y)
        joblib.dump(final_model, f"rf_regressor_{label_name}_simplified.pkl")
        results.append({"Target": label_name, "R2": mean_r2})
    return results

def train_classifier():
    df = pd.read_csv(CSV_PATH)
    df = add_ratio_features(df)
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    
    # 建立異常標籤 (根據原專案邏輯，若無 Label_Anomaly 則以 Loss Rate > 0.1% 為基準)
    if "Label_Anomaly" not in df.columns:
        df["Label_Anomaly"] = (df["Label_Loss_Rate"] > 0.001).astype(int)
    
    X = df[available_features]
    y = df["Label_Anomaly"]
    
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X, y)
    
    acc = accuracy_score(y, model.predict(X))
    print(f"Model: Anomaly (All)   | Accuracy: {acc:.4f}")
    
    joblib.dump(model, "rf_anomaly_classifier_simplified.pkl")
    return acc

if __name__ == "__main__":
    print("=== 開始訓練簡化版模型 (多維健康指標版) ===\n")
    train_regression()
    train_classifier()
    print("\n=== 訓練完成 ===")
    print("模型已儲存為 *_simplified.pkl")
