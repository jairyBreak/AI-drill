import pandas as pd
from topo_independent_helper import transform_to_topo_independent

df = pd.read_csv("research_results/data/datasets/rolling_training_dataset.csv")
ports = [2, 3, 4, 5, 6, 7, 8, 9]
capacity_map = {p: 1.0 for p in ports}

df_transformed = transform_to_topo_independent(df, ports, capacity_map)

old_df = df_transformed.iloc[:63129]
new_df = df_transformed.iloc[-120:]

old_mud = old_df["Max_Util_Diff"].mean()
new_mud = new_df["Max_Util_Diff"].mean()
old_qi = old_df["QDepth_Imbalance"].mean()
new_qi = new_df["QDepth_Imbalance"].mean()
old_gi = old_df["Group_Imbalance"].mean()
new_gi = new_df["Group_Imbalance"].mean()
old_top1 = old_df["top1_mbps"].mean()
new_top1 = new_df["top1_mbps"].mean()
old_top2 = old_df["top2_mbps"].mean()
new_top2 = new_df["top2_mbps"].mean()

print("=== 拓樸無關特徵 (Imbalance Features) 對比 ===")
print(f"Max_Util_Diff      : 舊 {old_mud:.3f} -> 新 {new_mud:.3f}")
print(f"QDepth_Imbalance   : 舊 {old_qi:.3f} -> 新 {new_qi:.3f}")
print(f"Group_Imbalance    : 舊 {old_gi:.3f} -> 新 {new_gi:.3f}")

print("\n=== Top-K 特徵 (前兩塞的 Port) 對比 ===")
print(f"top1_mbps          : 舊 {old_top1:.3f} Mbps -> 新 {new_top1:.3f} Mbps")
print(f"top2_mbps          : 舊 {old_top2:.3f} Mbps -> 新 {new_top2:.3f} Mbps")
