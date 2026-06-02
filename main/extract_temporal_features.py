"""
Extract temporal features from raw_telemetry/experiment_N.csv files.
Added: FFT Spectral features to capture oscillatory jitter.
"""
import os
import sys
import numpy as np
import pandas as pd

PORTS = [2, 3, 4, 5]
TELEMETRY_DIR = "raw_telemetry"
BASE_CSV = sys.argv[1] if len(sys.argv) > 1 else "training_dataset_master.csv"
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else "research_results/data/datasets/training_dataset_temporal.csv"


def temporal_features_for_experiment(exp_path: str) -> dict:
    df = pd.read_csv(exp_path)
    t = np.arange(len(df))
    feats = {}

    slopes_qdepth = []
    cvs_qdepth = []
    fft_max_q = []
    means_qdepth = []

    for p in PORTS:
        qdepth_col = f"src1_port{p}_qdepth"
        mbps_col = f"src1_port{p}_mbps"

        if qdepth_col not in df.columns or mbps_col not in df.columns:
            continue

        q = df[qdepth_col].values.astype(float)
        m = df[mbps_col].values.astype(float)

        # p99 queue depth
        feats[f"qdepth_p99_port{p}"] = float(np.percentile(q, 99))
        
        # 新增: 平均隊列深度 (更穩定)
        q_mean = float(np.mean(q))
        feats[f"qdepth_mean_port{p}"] = q_mean
        means_qdepth.append(q_mean)

        # linear slope
        feats[f"qdepth_slope_port{p}"] = float(np.polyfit(t, q, 1)[0]) if len(q) >= 2 else 0.0
        
        # coefficient of variation
        q_mean_val = float(np.mean(q))
        q_cv = float(np.std(q) / q_mean_val) if q_mean_val > 0 else 0.0
        feats[f"qdepth_cv_port{p}"] = q_cv
        cvs_qdepth.append(q_cv)

        # Max Change
        q_diff = np.max(np.abs(np.diff(q))) if len(q) >= 2 else 0.0
        feats[f"qdepth_max_diff_port{p}"] = float(q_diff)

        danger_threshold = 40.0 
        feats[f"qdepth_danger_ratio_port{p}"] = float(np.sum(q > danger_threshold) / len(q)) if len(q) > 0 else 0.0

        # --- FFT Features (Frequency Domain) ---
        if len(q) > 10:
            # Center the signal (remove DC component)
            q_centered = q - q_mean_val
            # Real FFT
            q_fft = np.abs(np.fft.rfft(q_centered))
            # Max magnitude in spectrum (captures dominant oscillation)
            f_max = float(np.max(q_fft))
            feats[f"qdepth_fft_max_port{p}"] = f_max
            fft_max_q.append(f_max)
            # Spectral Centroid (mean frequency)
            freqs = np.fft.rfftfreq(len(q))
            if np.sum(q_fft) > 0:
                feats[f"qdepth_fft_centroid_port{p}"] = float(np.sum(freqs * q_fft) / np.sum(q_fft))
            else:
                feats[f"qdepth_fft_centroid_port{p}"] = 0.0
        else:
            feats[f"qdepth_fft_max_port{p}"] = 0.0
            feats[f"qdepth_fft_centroid_port{p}"] = 0.0

        # mbps slope
        feats[f"mbps_slope_port{p}"] = float(np.polyfit(t, m, 1)[0]) if len(m) >= 2 else 0.0

    # summary columns
    p99_vals = [feats.get(f"qdepth_p99_port{p}", np.nan) for p in PORTS]
    feats["max_qdepth_p99"] = float(np.nanmax(p99_vals)) if p99_vals else np.nan
    feats["total_qdepth_p99"] = float(np.nansum(p99_vals))
    feats["total_qdepth_mean"] = float(np.nansum(means_qdepth)) # 新增總平均
    feats["qdepth_oscillation"] = float(np.mean(cvs_qdepth)) if cvs_qdepth else np.nan
    feats["qdepth_fft_max_all"] = float(np.max(fft_max_q)) if fft_max_q else 0.0

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
            missing += 1
            temporal_rows.append({})
        else:
            temporal_rows.append(temporal_features_for_experiment(exp_path))

    temporal_df = pd.DataFrame(temporal_rows)
    out_df = pd.concat([base_df.reset_index(drop=True), temporal_df.reset_index(drop=True)], axis=1)

    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with FFT features. Missing files: {missing}")


if __name__ == "__main__":
    main()
