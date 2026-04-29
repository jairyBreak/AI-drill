"""
Extract temporal features from raw_telemetry/experiment_N.csv files.

Each experiment file has 100 rows sampled at 10 Hz. Row i (0-indexed) in the
base dataset corresponds to experiment_{i+1}.csv.

Output: training_dataset_temporal.csv — base dataset with 19 extra temporal columns.
"""
import os
import sys
import numpy as np
import pandas as pd

PORTS = [2, 3, 4, 5]
TELEMETRY_DIR = "raw_telemetry"
BASE_CSV = sys.argv[1] if len(sys.argv) > 1 else "training_dataset_master.csv"
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else "training_dataset_temporal.csv"


def temporal_features_for_experiment(exp_path: str) -> dict:
    df = pd.read_csv(exp_path)
    t = np.arange(len(df))
    feats = {}

    slopes_qdepth = []
    cvs_qdepth = []

    for p in PORTS:
        qdepth_col = f"src1_port{p}_qdepth"
        mbps_col = f"src1_port{p}_mbps"

        if qdepth_col not in df.columns or mbps_col not in df.columns:
            for key in [
                f"qdepth_p99_port{p}", f"qdepth_slope_port{p}",
                f"qdepth_cv_port{p}", f"mbps_slope_port{p}",
            ]:
                feats[key] = np.nan
            continue

        q = df[qdepth_col].values.astype(float)
        m = df[mbps_col].values.astype(float)

        # p99 queue depth
        feats[f"qdepth_p99_port{p}"] = float(np.percentile(q, 99))

        # linear slope of queue depth over time
        q_slope = float(np.polyfit(t, q, 1)[0]) if len(q) >= 2 else 0.0
        feats[f"qdepth_slope_port{p}"] = q_slope
        slopes_qdepth.append(q_slope)

        # coefficient of variation of queue depth
        q_mean = float(np.mean(q))
        q_cv = float(np.std(q) / q_mean) if q_mean > 0 else 0.0
        feats[f"qdepth_cv_port{p}"] = q_cv
        cvs_qdepth.append(q_cv)

        # linear slope of throughput over time
        feats[f"mbps_slope_port{p}"] = float(np.polyfit(t, m, 1)[0]) if len(m) >= 2 else 0.0

    # summary columns
    p99_vals = [feats.get(f"qdepth_p99_port{p}", np.nan) for p in PORTS]
    feats["max_qdepth_p99"] = float(np.nanmax(p99_vals)) if p99_vals else np.nan
    feats["total_qdepth_p99"] = float(np.nansum(p99_vals))
    feats["qdepth_oscillation"] = float(np.mean(cvs_qdepth)) if cvs_qdepth else np.nan

    return feats


def main():
    base_df = pd.read_csv(BASE_CSV)
    n_rows = len(base_df)
    print(f"Base dataset: {BASE_CSV}  ({n_rows} rows)")

    temporal_rows = []
    missing = 0

    for i in range(n_rows):
        exp_path = os.path.join(TELEMETRY_DIR, f"experiment_{i + 1}.csv")
        if not os.path.exists(exp_path):
            print(f"  WARNING: missing {exp_path} — row {i} will have NaN temporal features")
            missing += 1
            temporal_rows.append({})
        else:
            temporal_rows.append(temporal_features_for_experiment(exp_path))

    temporal_df = pd.DataFrame(temporal_rows)
    out_df = pd.concat([base_df.reset_index(drop=True), temporal_df.reset_index(drop=True)], axis=1)

    out_df.to_csv(OUT_CSV, index=False)
    n_new = len(temporal_df.columns)
    print(f"Wrote {OUT_CSV}  ({len(out_df)} rows, {n_new} new temporal columns, {missing} missing files)")


if __name__ == "__main__":
    main()
