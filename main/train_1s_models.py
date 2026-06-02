import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# 配置與路徑
# ==========================================
CSV_PATH = "research_results/data/datasets/rolling_training_dataset.csv"
CAPACITY = {2: 0.8, 3: 0.8, 4: 0.8, 5: 0.8, 6: 1.2, 7: 1.2, 8: 1.2, 9: 1.2}

# 針對 1 秒尺度與即時推論優化的參數配置
BEST_PARAMS = dict(
    n_estimators=100,       # 減少樹的數量以提升即時推論速度 (1s 尺度需要更快的推論)
    max_depth=15,           # 限制深度以防止過擬合高頻雜訊
    min_samples_leaf=4,     # 增加葉節點最小樣本數，提升對 1s 雜訊的魯棒性
    max_features='sqrt',    # 使用 sqrt 能增加樹的多樣性，減少過擬合
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

def add_1s_features(df):
    """針對 1 秒尺度資料集添加衍生特徵"""
    df = df.copy()
    
    over_capacity_list = []
    
    for p in range(2, 10):
        # 歸一化負載 (Mbps / Weight)
        df[f"Norm_Load_P{p}"] = df[f"src1_port{p}_mbps"] / df[f"Weight_Port{p}"].replace(0, 0.01)
        # 鏈路利用率 (Mbps / Capacity)
        df[f"Util_P{p}"] = df[f"src1_port{p}_mbps"] / CAPACITY[p]
        # 佇列深度比例 (假設最大佇列為 64)
        df[f"Q_Ratio_P{p}"] = df[f"src1_port{p}_qdepth"] / 64.0
        # 超載量 (Mbps大於Capacity的部分)
        over_capacity_list.append(np.maximum(0, df[f"src1_port{p}_mbps"] - CAPACITY[p]))
    
    utils = [df[f"Util_P{p}"] for p in range(2, 10)]
    df["Total_Util_Sum"] = sum(utils)
    df["Max_Util_Diff"] = np.max(np.column_stack(utils), axis=1) - np.min(np.column_stack(utils), axis=1)
    
    # 全網超載總和 (與丟包直接相關)
    df["Over_Capacity_Sum"] = sum(over_capacity_list)
    
    # ==========================================
    # 🆕 預期流量映射 (Traffic Projection for What-If Analysis)
    # 解決痛點：當評估「假想新權重」時，舊的 Mbps 會失效，需要透過總流量與新權重重新分配。
    # ==========================================
    total_mbps = sum(df[f"src1_port{p}_mbps"] for p in range(2, 10))
    total_weight = sum(df[f"Weight_Port{p}"] for p in range(2, 10)).replace(0, 0.01)
    df["Total_Actual_Mbps"] = total_mbps
    
    expected_over_cap_list = []
    for p in range(2, 10):
        weight_ratio = df[f"Weight_Port{p}"] / total_weight
        expected_mbps = total_mbps * weight_ratio
        df[f"Expected_Mbps_P{p}"] = expected_mbps
        df[f"Expected_Util_P{p}"] = expected_mbps / CAPACITY[p]
        expected_over_cap_list.append(np.maximum(0, expected_mbps - CAPACITY[p]))
        
    df["Expected_Over_Capacity_Sum"] = sum(expected_over_cap_list)

    # 群組不平衡度 (Group A: P2..P5 | Group B: P6..P9)
    load_a = sum(df[f"src1_port{p}_mbps"] for p in [2, 3, 4, 5])
    weight_a = sum(df[f"Weight_Port{p}"] for p in [2, 3, 4, 5])
    load_b = sum(df[f"src1_port{p}_mbps"] for p in [6, 7, 8, 9])
    weight_b = sum(df[f"Weight_Port{p}"] for p in [6, 7, 8, 9])
    df["Group_Imbalance"] = np.abs((load_a / weight_a.replace(0, 0.01)) - (load_b / weight_b.replace(0, 0.01)))
    
    # 全域佇列指標
    qdepths = [df[f"src1_port{p}_qdepth"] for p in range(2, 10)]
    df["Max_QDepth"] = np.max(np.column_stack(qdepths), axis=1)
    df["Total_QDepth"] = np.sum(np.column_stack(qdepths), axis=1)
    df["QDepth_Imbalance"] = df["Max_QDepth"] - np.min(np.column_stack(qdepths), axis=1)
    
    # 佇列危險特徵 (丟包通常發生在佇列深度接近 64 時)
    df["Max_Q_Ratio"] = df["Max_QDepth"] / 64.0
    df["Q_Danger_Flag"] = (df["Max_QDepth"] > 40).astype(int)  # 佇列超過40視為危險
    df["Q_Danger_Count"] = sum((df[f"src1_port{p}_qdepth"] > 40).astype(int) for p in range(2, 10))
    
    # 新增互動特徵 (Overflow Intensity) 針對丟包預測推論
    df["Overflow_Intensity"] = df["Over_Capacity_Sum"] * df["Max_Q_Ratio"]
    df["Queue_Full_And_Over_Cap"] = df["Over_Capacity_Sum"] * df["Q_Danger_Flag"]
    
    # ==========================================
    # 時序趨勢特徵 (Time-Series / Temporal Features)
    # ==========================================
    df['Exp_ID'] = (df['Time_Since_Traffic_Start_s'] < df['Time_Since_Traffic_Start_s'].shift(1, fill_value=0)).cumsum()
    for p in range(2, 10):
        df[f'QDepth_Trend_P{p}'] = df.groupby('Exp_ID')[f'src1_port{p}_qdepth'].diff().fillna(0)
        df[f'Mbps_Trend_P{p}'] = df.groupby('Exp_ID')[f'src1_port{p}_mbps'].diff().fillna(0)
    df['Total_QDepth_Trend'] = df.groupby('Exp_ID')['Total_QDepth'].diff().fillna(0)
    df['Rehash_Impact'] = np.exp(-df['Time_Since_Last_Rehash_s'])
    
    return df

# 定義特徵清單 (加入映射特徵)
SELECTED_FEATURES = [
    "Is_Rehash_Event", "Time_Since_Last_Rehash_s", "Rehash_Impact",
    "Total_Util_Sum", "Max_Util_Diff", "Group_Imbalance", 
    "Max_QDepth", "Total_QDepth", "QDepth_Imbalance",
    "Over_Capacity_Sum", "Max_Q_Ratio", "Q_Danger_Flag", "Q_Danger_Count",
    "Total_QDepth_Trend",
    "Total_Actual_Mbps", "Expected_Over_Capacity_Sum",
    "Overflow_Intensity", "Queue_Full_And_Over_Cap"
]
for p in range(2, 10):
    SELECTED_FEATURES.extend([
        f"src1_port{p}_qdepth", f"src1_port{p}_mbps", f"Weight_Port{p}",
        f"Norm_Load_P{p}", f"QDepth_Trend_P{p}", f"Mbps_Trend_P{p}",
        f"Expected_Util_P{p}"
    ])

def train_1s_models():
    print("=== 正在訓練 1 秒尺度新模型 (Rolling V2) ===\n")
    
    if not os.path.exists(CSV_PATH):
        print(f"錯誤: 找不到資料集 {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"載入資料集: {len(df)} 筆樣本")

    # 數據清洗：剔除異常標籤
    original_len = len(df)
    df = df[df["Label_Max_Path_Delay_ms"] <= 3000.0]
    df = df[df["Label_Max_Path_Delay_ms"] >= 0]
    print(f"剔除了 {original_len - len(df)} 筆極端延遲雜訊")

    # 添加衍生特徵
    df = add_1s_features(df)
    
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    print(f"使用特徵數: {len(available_features)}")

    # 訓練目標
    targets = [
        ("Label_Max_Path_Delay_ms", "latency_1s"),
        ("Label_Total_Drop_Rate_Percent", "loss_1s")
    ]
    
    for label_col, name in targets:
        print(f"\n--- 正在訓練 {name} 模型 ---")
        
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
            else:
                preds_orig = preds
            
            r2_list.append(r2_score(y_target.iloc[test_idx], preds))
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
            print(f"  {i+1}. {available_features[indices[i]]:<25} ({importances[indices[i]]:.4f})")

    # 異常分類模型 (以丟包率 > 0.1% 作為異常)
    print("\n--- 正在訓練 anomaly_1s 分類模型 ---")
    X = df[available_features]
    y = (df["Label_Total_Drop_Rate_Percent"] > 0.1).astype(int)
    
    clf = RandomForestClassifier(**BEST_PARAMS)
    clf.fit(X, y)
    joblib.dump(clf, "rf_model_anomaly_1s.pkl")
    print("分類模型已儲存至: rf_model_anomaly_1s.pkl")

    print("\n=== 1 秒尺度新模型訓練完成！ ===")

if __name__ == "__main__":
    train_1s_models()
