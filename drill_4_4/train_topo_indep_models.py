import sys
import os
import csv
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

from topo_independent_helper import transform_to_topo_independent

# ==========================================
# 配置與路徑
# ==========================================
CSV_4PORT = "bruh/rolling_training_dataset_1.csv"
CSV_8PORT = "research_results/data/datasets/rolling_training_dataset.csv"

CAPACITY_4PORT = {2: 0.8, 3: 0.8, 4: 1.2, 5: 1.2}
CAPACITY_8PORT = {2: 0.48, 3: 0.48, 4: 0.64, 5: 0.64, 6: 0.8, 7: 0.8, 8: 0.96, 9: 0.96}

BEST_PARAMS = dict(
    n_estimators=100,       # 減少樹的數量以提升即時推論速度 (1s 尺度需要更快的推論)
    max_depth=15,           # 限制深度以防止過擬合高頻雜訊
    min_samples_leaf=4,     # 增加葉節點最小樣本數，提升對 1s 雜訊的魯棒性
    max_features='sqrt',    # 使用 sqrt 能增加樹的多樣性，減少過擬合
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

# 定義特徵清單 (與 helper 中的特徵完全一致)
SELECTED_FEATURES = [
    "Is_Rehash_Event", "Time_Since_Last_Rehash_s", "Rehash_Impact",
    "Total_Util_Sum", "Max_Util_Diff", "Group_Imbalance", 
    "Max_QDepth", "Total_QDepth", "QDepth_Imbalance",
    "Over_Capacity_Sum", "Max_Q_Ratio", "Q_Danger_Flag", "Q_Danger_Count",
    "Total_QDepth_Trend",
    "Total_Actual_Mbps", "Expected_Over_Capacity_Sum",
    "Overflow_Intensity", "Queue_Full_And_Over_Cap"
]
for k in range(3):
    SELECTED_FEATURES.extend([
        f"top{k+1}_qdepth",
        f"top{k+1}_mbps", f"top{k+1}_weight",
        f"top{k+1}_norm_load", f"top{k+1}_expected_util",
        f"top{k+1}_qdepth_trend", f"top{k+1}_mbps_trend"
    ])

def load_and_split_csv(file_path):
    """
    Reads a CSV file line-by-line and dynamically handles mixed row sizes
    (e.g., 32 columns for 4-port, 56 columns for 8-port).
    """
    header_32 = [
        'Timestamp', 'Total_Load_Mbps_Config', 'Flows', 'Time_Since_Traffic_Start_s',
        'Time_Since_Last_Rehash_s', 'Is_Rehash_Event',
        'src1_port2_qdepth', 'src1_port2_max_q_delay_us', 'src1_port2_acc_q_delay_us', 'src1_port2_mbps', 'src1_port2_congestion_drop_rate_percent', 'Weight_Port2',
        'src1_port3_qdepth', 'src1_port3_max_q_delay_us', 'src1_port3_acc_q_delay_us', 'src1_port3_mbps', 'src1_port3_congestion_drop_rate_percent', 'Weight_Port3',
        'src1_port4_qdepth', 'src1_port4_max_q_delay_us', 'src1_port4_acc_q_delay_us', 'src1_port4_mbps', 'src1_port4_congestion_drop_rate_percent', 'Weight_Port4',
        'src1_port5_qdepth', 'src1_port5_max_q_delay_us', 'src1_port5_acc_q_delay_us', 'src1_port5_mbps', 'src1_port5_congestion_drop_rate_percent', 'Weight_Port5',
        'Label_Max_Path_Delay_ms', 'Label_Total_Drop_Rate_Percent'
    ]
    
    header_56 = [
        'Timestamp', 'Total_Load_Mbps_Config', 'Flows', 'Time_Since_Traffic_Start_s',
        'Time_Since_Last_Rehash_s', 'Is_Rehash_Event',
        'src1_port2_qdepth', 'src1_port2_max_q_delay_us', 'src1_port2_acc_q_delay_us', 'src1_port2_mbps', 'src1_port2_congestion_drop_rate_percent', 'Weight_Port2',
        'src1_port3_qdepth', 'src1_port3_max_q_delay_us', 'src1_port3_acc_q_delay_us', 'src1_port3_mbps', 'src1_port3_congestion_drop_rate_percent', 'Weight_Port3',
        'src1_port4_qdepth', 'src1_port4_max_q_delay_us', 'src1_port4_acc_q_delay_us', 'src1_port4_mbps', 'src1_port4_congestion_drop_rate_percent', 'Weight_Port4',
        'src1_port5_qdepth', 'src1_port5_max_q_delay_us', 'src1_port5_acc_q_delay_us', 'src1_port5_mbps', 'src1_port5_congestion_drop_rate_percent', 'Weight_Port5',
        'src1_port6_qdepth', 'src1_port6_max_q_delay_us', 'src1_port6_acc_q_delay_us', 'src1_port6_mbps', 'src1_port6_congestion_drop_rate_percent', 'Weight_Port6',
        'src1_port7_qdepth', 'src1_port7_max_q_delay_us', 'src1_port7_acc_q_delay_us', 'src1_port7_mbps', 'src1_port7_congestion_drop_rate_percent', 'Weight_Port7',
        'src1_port8_qdepth', 'src1_port8_max_q_delay_us', 'src1_port8_acc_q_delay_us', 'src1_port8_mbps', 'src1_port8_congestion_drop_rate_percent', 'Weight_Port8',
        'src1_port9_qdepth', 'src1_port9_max_q_delay_us', 'src1_port9_acc_q_delay_us', 'src1_port9_mbps', 'src1_port9_congestion_drop_rate_percent', 'Weight_Port9',
        'Label_Max_Path_Delay_ms', 'Label_Total_Drop_Rate_Percent'
    ]
    
    rows_32 = []
    rows_56 = []
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=header_32), pd.DataFrame(columns=header_56)
        
        for row in reader:
            if not row:
                continue
            # Trim whitespace or skip header rows repeated in file
            if row[0] == "Timestamp":
                continue
            
            if len(row) == 32:
                converted_row = []
                for idx, val in enumerate(row):
                    if idx == 0:
                        converted_row.append(val)
                    else:
                        try:
                            converted_row.append(float(val))
                        except ValueError:
                            converted_row.append(val)
                rows_32.append(converted_row)
            elif len(row) == 56:
                converted_row = []
                for idx, val in enumerate(row):
                    if idx == 0:
                        converted_row.append(val)
                    else:
                        try:
                            converted_row.append(float(val))
                        except ValueError:
                            converted_row.append(val)
                rows_56.append(converted_row)
                
    df_32 = pd.DataFrame(rows_32, columns=header_32) if rows_32 else pd.DataFrame(columns=header_32)
    df_56 = pd.DataFrame(rows_56, columns=header_56) if rows_56 else pd.DataFrame(columns=header_56)
    
    return df_32, df_56

def train_models():
    print("=== 正在加載與處理資料集 ===")
    
    all_dfs = []
    
    # 1. 跳過 4-Port 資料集 (依據使用者要求只訓練 8-port)
    print("跳過 4-port 資料集，只使用 8-port 資料集...")
        
    # 2. 處理 8-Port 資料集 (當前實驗產生的)
    if os.path.exists(CSV_8PORT):
        print(f"載入 8-port 當前資料集: {CSV_8PORT}")
        df_8_32, df_8_56 = load_and_split_csv(CSV_8PORT)
        print(f"  - 讀取到 {len(df_8_32)} 筆 4-port 格式樣本，{len(df_8_56)} 筆 8-port 格式樣本")
        
        if len(df_8_32) > 0:
            print("  - 正在轉換 8-port 檔案中的 4-port 樣本...")
            df_8_32_transformed = transform_to_topo_independent(df_8_32, [2, 3, 4, 5], CAPACITY_4PORT)
            all_dfs.append(df_8_32_transformed)
        if len(df_8_56) > 0:
            print("  - 正在轉換 8-port 樣本至拓樸無關格式...")
            df_8_56_transformed = transform_to_topo_independent(df_8_56, list(range(2, 10)), CAPACITY_8PORT)
            all_dfs.append(df_8_56_transformed)
    else:
        print(f"提示: 找不到 8-port 當前資料集 {CSV_8PORT}")

    if not all_dfs:
        print("錯誤: 沒有任何有效的資料可以進行訓練。")
        return
        
    # 合併所有拓樸無關轉換後的 DataFrame
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"合併完成！總樣本數: {len(df)}")
    
    # 數據清洗：剔除異常標籤與雜訊
    original_len = len(df)
    df = df[df["Label_Max_Path_Delay_ms"] <= 3000.0]
    df = df[df["Label_Max_Path_Delay_ms"] >= 0]
    print(f"數據清洗: 剔除了 {original_len - len(df)} 筆極端延遲雜訊/無效樣本")
    
    # 確保特徵都在
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]
    print(f"使用特徵數: {len(available_features)}")
    if missing_features:
        print(f"缺失特徵 (將使用預設值填充或排除): {missing_features}")
        
    # 訓練目標
    targets = [
        ("Label_Max_Path_Delay_ms", "latency_1s"),
        ("Label_Total_Drop_Rate_Percent", "loss_1s")
    ]
    
    for label_col, name in targets:
        print(f"\n--- 正在訓練 {name} 拓樸無關模型 ---")
        
        # 過濾無效標籤
        subset = df[df[label_col] != -1.0].copy()
        X = subset[available_features]
        
        # 根據目標決定是否使用 log1p 變換
        use_log = "latency" in name
        if use_log:
            y_target = np.log1p(subset[label_col])
        else:
            y_target = subset[label_col]
            
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_list, mae_list = [], []
        
        for train_idx, test_idx in cv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_target_tr, y_te_raw = y_target.iloc[train_idx], subset[label_col].iloc[test_idx]
            
            model = RandomForestRegressor(**BEST_PARAMS)
            model.fit(X_tr, y_target_tr)
            
            # 預測並還原
            preds = model.predict(X_te)
            if use_log:
                preds_orig = np.expm1(preds)
                r2_val = r2_score(y_target.iloc[test_idx], preds) # 延遲使用log1p空間的R2
            else:
                preds_orig = preds
                r2_val = r2_score(y_te_raw, preds)
                
            r2_list.append(r2_val)
            mae_list.append(mean_absolute_error(y_te_raw, preds_orig))
            
        unit = "%" if "loss" in name else "ms"
        metric_name = "R2(log)" if use_log else "R2(raw)"
        print(f"[{name:12}] {metric_name}: {np.mean(r2_list):.4f} | MAE: {np.mean(mae_list):.4f} {unit}")
        
        # 訓練最終模型
        final_model = RandomForestRegressor(**BEST_PARAMS)
        final_model.fit(X, y_target)
        joblib.dump(final_model, f"rf_model_{name}.pkl")
        print(f"模型已儲存至: rf_model_{name}.pkl")
        
        # 顯示特徵重要性
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"Top 5 特徵重要性:")
        for i in range(5):
            print(f"  {i+1}. {available_features[indices[i]]:<35} ({importances[indices[i]]:.4f})")
            
    # 異常分類模型 (以丟包率 > 0.1% 作為異常)
    print("\n--- 正在訓練 anomaly_1s 拓樸無關分類模型 ---")
    X = df[available_features]
    y = (df["Label_Total_Drop_Rate_Percent"] > 0.1).astype(int)
    
    clf = RandomForestClassifier(**BEST_PARAMS)
    clf.fit(X, y)
    joblib.dump(clf, "rf_model_anomaly_1s.pkl")
    print("分類模型已儲存至: rf_model_anomaly_1s.pkl")
    
    print("\n=== 拓樸無關 1 秒尺度新模型訓練完成！ ===")

if __name__ == "__main__":
    train_models()
