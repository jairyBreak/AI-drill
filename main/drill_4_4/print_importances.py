import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from train_1s_models import add_1s_features, SELECTED_FEATURES, CSV_PATH
df = pd.read_csv(CSV_PATH)
df = add_1s_features(df)
available_features = [f for f in SELECTED_FEATURES if f in df.columns]

model_lat = joblib.load('rf_model_latency_1s.pkl')
importances_lat = model_lat.feature_importances_
indices_lat = np.argsort(importances_lat)[::-1]

print("=== Latency 模型特徵重要性 ===")
for i in range(len(available_features)):
    print(f"{i+1:2d}. {available_features[indices_lat[i]]:<30} ({importances_lat[indices_lat[i]]:.4f})")

model_loss = joblib.load('rf_model_loss_1s.pkl')
importances_loss = model_loss.feature_importances_
indices_loss = np.argsort(importances_loss)[::-1]

print("\n=== Loss 模型特徵重要性 ===")
for i in range(len(available_features)):
    print(f"{i+1:2d}. {available_features[indices_loss[i]]:<30} ({importances_loss[indices_loss[i]]:.4f})")
