import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "training_dataset_labeled_v2.csv"
CLASS_NAMES = ["NORMAL", "SUSTAINED_CONGESTION", "BURST_CONGESTION", "NON_CONGESTION_LOSS"]

df = pd.read_csv(CSV_PATH)

# 移除樣本數不足的類別（stratify 需要每類至少 2 筆）
class_counts = df["Label_Class"].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df["Label_Class"].isin(valid_classes)]

features = [col for col in df.columns if not col.startswith("Label_")]
x = df[features]
y = df["Label_Class"]

# stratified split：確保測試集各類別比例與整體一致
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

print(f"訓練集: {len(x_train)} 筆，測試集: {len(x_test)} 筆")
print(f"訓練集標籤分佈:\n{y_train.value_counts().sort_index()}\n")

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=67,
    n_jobs=-1,
)
rf.fit(x_train, y_train)
preds = rf.predict(x_test)

print(f"測試集標籤分佈:\n{y_test.value_counts().sort_index()}\n")

all_labels = list(range(len(CLASS_NAMES)))
print("=== Classification Report ===")
print(classification_report(y_test, preds, target_names=CLASS_NAMES, labels=all_labels, zero_division=0))

cm = confusion_matrix(y_test, preds, labels=all_labels)
print("=== Confusion Matrix ===")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, xticks_rotation=30)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("混淆矩陣圖已存至 confusion_matrix.png")

importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\n=== Top 10 Feature Importance ===")
print(importance.head(10))

joblib.dump(rf, "rf_anomaly_classifier.pkl")
print("\n模型已存至 rf_anomaly_classifier.pkl")
