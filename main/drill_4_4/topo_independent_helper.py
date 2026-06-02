import pandas as pd
import numpy as np

def transform_to_topo_independent(df, ports, capacity_map, K=3):
    """
    Transforms raw per-port features of an arbitrary topology (e.g. 4-port, 8-port)
    into a fixed set of topology-independent features (global aggregates + Top-K outliers).
    """
    df = df.copy()
    
    # Pre-calculate per-port trends if they exist or we can calculate them
    df['Exp_ID'] = (df['Time_Since_Traffic_Start_s'] < df['Time_Since_Traffic_Start_s'].shift(1, fill_value=0)).cumsum()
    
    # Calculate per-port derived metrics
    for p in ports:
        df[f"src1_port{p}_norm_load"] = df[f"src1_port{p}_mbps"] / df[f"Weight_Port{p}"].replace(0, 0.01)
        df[f"src1_port{p}_util"] = df[f"src1_port{p}_mbps"] / capacity_map[p]
        df[f"src1_port{p}_q_ratio"] = df[f"src1_port{p}_qdepth"] / 64.0
        df[f"src1_port{p}_qdepth_trend"] = df.groupby('Exp_ID')[f"src1_port{p}_qdepth"].diff().fillna(0)
        df[f"src1_port{p}_mbps_trend"] = df.groupby('Exp_ID')[f"src1_port{p}_mbps"].diff().fillna(0)
        
        # Expected metrics
        total_mbps = sum(df[f"src1_port{p_inner}_mbps"] for p_inner in ports)
        total_weight = sum(df[f"Weight_Port{p_inner}"] for p_inner in ports).replace(0, 0.01)
        weight_ratio = df[f"Weight_Port{p}"] / total_weight
        expected_mbps = total_mbps * weight_ratio
        df[f"src1_port{p}_expected_util"] = expected_mbps / capacity_map[p]

    # Global features (topology independent)
    total_mbps = sum(df[f"src1_port{p}_mbps"] for p in ports)
    total_weight = sum(df[f"Weight_Port{p}"] for p in ports).replace(0, 0.01)
    
    df["Total_Actual_Mbps"] = total_mbps
    
    utils = [df[f"src1_port{p}_util"] for p in ports]
    df["Total_Util_Sum"] = sum(utils)
    df["Max_Util_Diff"] = np.max(np.column_stack(utils), axis=1) - np.min(np.column_stack(utils), axis=1)
    
    qdepths = [df[f"src1_port{p}_qdepth"] for p in ports]
    df["Max_QDepth"] = np.max(np.column_stack(qdepths), axis=1)
    df["Total_QDepth"] = np.sum(np.column_stack(qdepths), axis=1)
    df["QDepth_Imbalance"] = df["Max_QDepth"] - np.min(np.column_stack(qdepths), axis=1)
    df["Max_Q_Ratio"] = df["Max_QDepth"] / 64.0
    df["Q_Danger_Flag"] = (df["Max_QDepth"] > 40).astype(int)
    df["Q_Danger_Count"] = sum((df[f"src1_port{p}_qdepth"] > 40).astype(int) for p in ports)
    
    over_capacity_list = [np.maximum(0, df[f"src1_port{p}_mbps"] - capacity_map[p]) for p in ports]
    df["Over_Capacity_Sum"] = sum(over_capacity_list)
    
    expected_over_cap_list = []
    for p in ports:
        weight_ratio = df[f"Weight_Port{p}"] / total_weight
        expected_mbps = total_mbps * weight_ratio
        expected_over_cap_list.append(np.maximum(0, expected_mbps - capacity_map[p]))
    df["Expected_Over_Capacity_Sum"] = sum(expected_over_cap_list)
    
    df["Overflow_Intensity"] = df["Over_Capacity_Sum"] * df["Max_Q_Ratio"]
    df["Queue_Full_And_Over_Cap"] = df["Over_Capacity_Sum"] * df["Q_Danger_Flag"]
    
    # Calculate group imbalance (using the half-split of ports)
    half = len(ports) // 2
    ports_a = ports[:half]
    ports_b = ports[half:]
    load_a = sum(df[f"src1_port{p}_mbps"] for p in ports_a)
    weight_a = sum(df[f"Weight_Port{p}"] for p in ports_a)
    load_b = sum(df[f"src1_port{p}_mbps"] for p in ports_b)
    weight_b = sum(df[f"Weight_Port{p}"] for p in ports_b)
    df["Group_Imbalance"] = np.abs((load_a / weight_a.replace(0, 0.01)) - (load_b / weight_b.replace(0, 0.01)))
    
    df['Total_QDepth_Trend'] = df.groupby('Exp_ID')['Total_QDepth'].diff().fillna(0)
    df['Rehash_Impact'] = np.exp(-df['Time_Since_Last_Rehash_s'])

    # Now, for each row, extract the Top K ports based on qdepth
    qdepth_matrix = df[[f"src1_port{p}_qdepth" for p in ports]].values
    sorted_col_indices = np.argsort(-qdepth_matrix, axis=1)
    
    # We will build arrays for the Top K features
    top_k_features = {
        'qdepth': [],
        'mbps': [],
        'weight': [],
        'norm_load': [],
        'expected_util': [],
        'qdepth_trend': [],
        'mbps_trend': []
    }
    
    metrics = {}
    for metric_name in ['qdepth', 'mbps', 'weight', 'norm_load', 'expected_util', 'qdepth_trend', 'mbps_trend']:
        if metric_name == 'weight':
            cols = [f"Weight_Port{p}" for p in ports]
        elif metric_name == 'qdepth':
            cols = [f"src1_port{p}_qdepth" for p in ports]
        else:
            cols = [f"src1_port{p}_{metric_name}" for p in ports]
        metrics[metric_name] = df[cols].values

    num_rows = len(df)
    
    # Extract top K
    for k in range(K):
        if k < len(ports):
            col_idx = sorted_col_indices[:, k]
            for m_name, val_matrix in metrics.items():
                top_k_features[m_name].append(val_matrix[np.arange(num_rows), col_idx])
        else:
            # If K > num_ports, pad with zeros
            for m_name in metrics:
                top_k_features[m_name].append(np.zeros(num_rows))

    # Add the top K features to the dataframe
    for m_name, arrays in top_k_features.items():
        for k in range(K):
            df[f"top{k+1}_{m_name}"] = arrays[k]

    base_cols = [
        "Timestamp", "Total_Load_Mbps_Config", "Flows", 
        "Time_Since_Traffic_Start_s", "Time_Since_Last_Rehash_s", "Is_Rehash_Event"
    ]
    
    target_cols = [col for col in df.columns if col.startswith("Label_")]
    
    selected_features = [
        "Is_Rehash_Event", "Time_Since_Last_Rehash_s", "Rehash_Impact",
        "Total_Util_Sum", "Max_Util_Diff", "Group_Imbalance", 
        "Max_QDepth", "Total_QDepth", "QDepth_Imbalance",
        "Over_Capacity_Sum", "Max_Q_Ratio", "Q_Danger_Flag", "Q_Danger_Count",
        "Total_QDepth_Trend",
        "Total_Actual_Mbps", "Expected_Over_Capacity_Sum",
        "Overflow_Intensity", "Queue_Full_And_Over_Cap"
    ]
    
    for k in range(K):
        selected_features.extend([
            f"top{k+1}_qdepth",
            f"top{k+1}_mbps", f"top{k+1}_weight",
            f"top{k+1}_norm_load", f"top{k+1}_expected_util",
            f"top{k+1}_qdepth_trend", f"top{k+1}_mbps_trend"
        ])
        
    final_cols = base_cols + selected_features + target_cols
    
    # Deduplicate columns in final_cols while preserving order
    seen = set()
    final_cols_unique = []
    for col in final_cols:
        if col in df.columns and col not in seen:
            final_cols_unique.append(col)
            seen.add(col)
            
    return df[final_cols_unique]
