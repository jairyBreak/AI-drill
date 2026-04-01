import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib 

csv_path = "training_dataset_master.csv"
df = pd.read_csv(csv_path)

features = [col for col in df.columns if not col.startswith('Label_')]
x = df[features]

targets = ['Label_Loss_Rate', 'Label_Latency_ms', 'Label_Jitter_ms']
models = {}

for target in targets:
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_depth=20,random_state=67, n_jobs=-1)
    rf.fit(x_train, y_train)

    preds = rf.predict(x_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print(f" {target}-> R²: {r2:.4f}, MAE: {mae:.2f}")
    
    # 顯示該指標最關鍵的 3 個物理特徵
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print(f"-> importance ：\n{importance.head(10)}")

    model_name = f"rf_expert_{target.lower()}.pkl"
    joblib.dump(rf, model_name)
    models[target] = rf