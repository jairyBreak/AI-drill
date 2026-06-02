import pandas as pd
import numpy as np

# Load the dataset
dataset_path = '/home/p4/drill/drill_4_4/research_results/data/datasets/rolling_training_dataset.csv'
df = pd.read_csv(dataset_path)

print(f"### Dataset Overview ###")
print(f"Total records: {len(df)}")
print(f"Missing values: {df.isnull().sum().sum()}")

print("\n### Feature Distribution (Environmental) ###")
for col in ['Total_Load_Mbps_Config', 'Flows']:
    print(f"\n{col} Statistics:")
    print(df[col].describe())
    print(f"Unique values: {df[col].nunique()}")

print("\n### Label Distribution (Ground Truth) ###")
for col in ['Label_Max_Path_Delay_ms', 'Label_Total_Drop_Rate_Percent']:
    print(f"\n{col} Statistics:")
    print(df[col].describe())
    # Count how many zero values
    zeros = (df[col] == 0).sum()
    print(f"Zero values: {zeros} ({zeros/len(df)*100:.2f}%)")

print("\n### Port-Specific Metrics Distribution (Summary) ###")
# Port 2 as a proxy for all ports to check general range
port_cols = ['src1_port2_qdepth', 'src1_port2_mbps', 'src1_port2_congestion_drop_rate_percent', 'src1_port2_acc_q_delay_us']
print(df[port_cols].describe())

print("\n### Time Context Check ###")
print(df[['Time_Since_Traffic_Start_s', 'Time_Since_Last_Rehash_s']].describe())

# Check for data imbalance in labels (e.g., if most data is low delay)
print("\n### Delay Buckets ###")
bins = [0, 10, 50, 100, 500, 1000, np.inf]
labels = ['0-10ms', '10-50ms', '50-100ms', '100-500ms', '500-1000ms', '1000ms+']
delay_dist = pd.cut(df['Label_Max_Path_Delay_ms'], bins=bins, labels=labels).value_counts().sort_index()
print(delay_dist)
