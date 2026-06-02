import sys
import os
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error

# ==========================================
# 終極全方位測試配置 (Overnight Exhaustive Search)
# ==========================================
# 涵蓋廣大的超參數空間，旨在榨乾模型每一分潛力
PARAM_DIST = {
    'n_estimators': [100, 200, 400, 600, 800, 1000, 1500, 2000],
    'max_depth': [10, 15, 20, 30, 50, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [0.4, 0.6, 0.8, 1.0, 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

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

CSV_PATH = "training_dataset_ecdf_cleaned.csv"

def run_overnight_tuning():
    print(f"[{datetime.now()}] === 啟動全方位終極調優 (Stability-Focused Mode) ===")
    df = pd.read_csv(CSV_PATH)
    df = add_ratio_features(df)
    
    os.makedirs("tuning_results", exist_ok=True)
    
    full_report = []
    targets = [
        ("Label_Latency_ms", "latency_ms", "regression"),
        ("Label_Latency_p99_ms", "latency_p99_ms", "regression"),
        ("Label_Jitter_ms", "jitter_ms", "regression"),
        ("Label_Loss_Rate", "loss_rate", "regression"),
        ("Label_Anomaly", "anomaly", "classification")
    ]

    if "Label_Anomaly" not in df.columns:
        df["Label_Anomaly"] = (df["Label_Loss_Rate"] > 0.001).astype(int)

    for label_col, name, task_type in targets:
        print(f"\n[{datetime.now()}] 正在處理指標: {name} ({task_type})")
        
        subset = df[df[label_col] != -1.0].copy()
        X = subset[[f for f in SELECTED_FEATURES if f in subset.columns]]
        y = np.log1p(subset[label_col]) if task_type == "regression" else subset[label_col]
        
        base_model = RandomForestRegressor(random_state=42) if task_type == "regression" else RandomForestClassifier(random_state=42)
        scoring = 'r2' if task_type == "regression" else 'accuracy'
        
        # 開啟 return_train_score 來監測泛化能力
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=PARAM_DIST,
            n_iter=100, 
            cv=5,
            scoring=scoring,
            verbose=1,
            n_jobs=-1,
            random_state=42,
            return_train_score=True
        )
        
        start_t = time.time()
        search.fit(X, y)
        end_t = time.time()
        
        # 提取穩定性指標
        res = search.cv_results_
        idx = search.best_index_
        test_score = res['mean_test_score'][idx]
        train_score = res['mean_train_score'][idx]
        std_score = res['std_test_score'][idx]
        gen_gap = abs(train_score - test_score)
        
        best_params = search.best_params_
        duration_min = (end_t - start_t) / 60
        
        # 格式化輸出
        res_str = (f"[{name:15}] Score: {test_score:.4f} | Gap: {gen_gap:.4f} | Std: {std_score:.4f} | Time: {duration_min:.1f}m")
        print(res_str)
        full_report.append(f"{res_str} | Params: {best_params}")
        
        # 儲存詳細結果
        pd.DataFrame(res).to_csv(f"tuning_results/trials_{name}.csv", index=False)
        
        # 儲存最佳模型 (統一標準命名)
        standard_name = f"rf_model_{name}_simplified.pkl"
        joblib.dump(search.best_estimator_, f"tuning_results/rf_best_{name}_overnight.pkl") # 專屬備份
        joblib.dump(search.best_estimator_, standard_name) # 直接覆蓋生產版

    report_path = "tuning_results/final_overnight_report.txt"
    with open(report_path, "w") as f:
        f.write("=== Ultimate Stability-Focused Tuning Report ===\n")
        f.write(f"Completed at: {datetime.now()}\n")
        f.write("-" * 50 + "\n")
        f.write("\n".join(full_report))
    
    print(f"\n[{datetime.now()}] === 調優完成！結果已記錄至 {report_path} ===")

if __name__ == "__main__":
    run_overnight_tuning()
