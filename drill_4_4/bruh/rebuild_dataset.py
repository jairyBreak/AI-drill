import pandas as pd

MASTER_CSV  = "training_dataset_master.csv"
OUT_CSV     = "training_dataset_v2.csv"
VALID_ROWS  = 1068   # 後 1500 筆是 iperf3 掉線廢資料，只保留前 1068 筆
PORTS       = [2, 3, 4, 5]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    port_qdepth_maxes = [f"src1_port{n}_qdepth_max" for n in PORTS]
    port_mbps_means   = [f"src1_port{n}_mbps_mean"  for n in PORTS]

    # Per-port: coefficient of variation (std/mean) — relative burstiness
    for n in PORTS:
        mean_col = f"src1_port{n}_mbps_mean"
        std_col  = f"src1_port{n}_mbps_std"
        out[f"src1_port{n}_mbps_cv"] = (
            out[std_col] / out[mean_col].replace(0, float("nan"))
        ).fillna(0).round(4)

    # Per-port: utilization ratio = mean throughput / capacity
    capacity = {2: 0.8, 3: 0.8, 4: 1.2, 5: 1.2}
    for n in PORTS:
        out[f"src1_port{n}_load_util"] = (
            out[f"src1_port{n}_mbps_mean"] / capacity[n]
        ).round(4)

    # Cross-port: qdepth asymmetry
    out["qdepth_max_imbalance"] = (
        out[port_qdepth_maxes].max(axis=1) - out[port_qdepth_maxes].min(axis=1)
    ).round(4)

    # Cross-port: load distribution unevenness
    out["mbps_imbalance"] = out[port_mbps_means].std(axis=1, ddof=0).round(4)

    # Cross-port: total queue depth
    out["total_qdepth_max"] = out[port_qdepth_maxes].sum(axis=1)

    return out


def main():
    master = pd.read_csv(MASTER_CSV)
    print(f"master CSV: {len(master)} 筆，取前 {VALID_ROWS} 筆有效資料")

    valid = master.iloc[:VALID_ROWS].copy()
    v2    = add_derived_features(valid)

    v2.to_csv(OUT_CSV, index=False)
    new_cols = [c for c in v2.columns if c not in master.columns]
    print(f"完成：{len(v2)} 筆 → {OUT_CSV}  ({len(v2.columns)} 欄，新增 {len(new_cols)} 欄)")
    print(f"新欄位：{new_cols}")


if __name__ == "__main__":
    main()
