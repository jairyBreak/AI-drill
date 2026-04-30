"""
ECDF-based feature engineering following the statistical framework from the project document.
Added: ECDF for FFT features.
"""
import sys
import numpy as np
import pandas as pd
import joblib

PORTS = [2, 3, 4, 5]
ALPHA = 1.1  # scales ECDF above 1 for positive threshold at ~P90.9

BASE_CSV = sys.argv[1] if len(sys.argv) > 1 else "training_dataset_temporal.csv"
OUT_CSV  = sys.argv[2] if len(sys.argv) > 2 else "training_dataset_ecdf.csv"
ECDF_PKL = "ecdf_objects.pkl"

CAPACITY = {2: 0.8, 3: 0.8, 4: 1.2, 5: 1.2}


class RankECDF:
    """Simple rank-based empirical CDF; no scipy version dependency."""

    def __init__(self):
        self._sorted = None

    def fit(self, values: np.ndarray) -> "RankECDF":
        clean = values[~np.isnan(values)]
        self._sorted = np.sort(clean)
        return self

    def transform(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._sorted is None or len(self._sorted) == 0:
            return np.zeros_like(x)
        return np.searchsorted(self._sorted, x, side="right") / len(self._sorted)


def _ecdf_col(df: pd.DataFrame, src_col: str, ecdf_objects: dict) -> np.ndarray:
    ecdf = RankECDF().fit(df[src_col].values)
    ecdf_objects[src_col] = ecdf
    return ecdf.transform(df[src_col].values)


def main():
    df = pd.read_csv(BASE_CSV)
    print(f"Loaded {BASE_CSV}  ({len(df)} rows, {len(df.columns)} columns)")

    ecdf_objects: dict = {}
    ecdf_cols: dict = {}  # new_col_name → np.ndarray

    # ── per-port features ──────────────────────────────────────────────────────
    for p in PORTS:
        cap = CAPACITY[p]

        # queue depth features
        for raw in [f"src1_port{p}_qdepth_max", f"qdepth_p99_port{p}", f"qdepth_max_diff_port{p}", f"qdepth_fft_max_port{p}"]:
            if raw in df.columns:
                ecdf_cols[f"{raw}_ecdf"] = _ecdf_col(df, raw, ecdf_objects)

        # throughput features
        mbps_col = f"src1_port{p}_mbps_mean"
        if mbps_col in df.columns:
            drop_col = f"src1_port{p}_mbps_drop"
            df[drop_col] = (cap - df[mbps_col]).clip(lower=0.0)
            ecdf_cols[f"{drop_col}_ecdf"] = _ecdf_col(df, drop_col, ecdf_objects)

        # instability indicators
        for raw in [
            f"src1_port{p}_mbps_std",
            f"src1_port{p}_mbps_cv",
            f"src1_port{p}_load_util",
            f"qdepth_cv_port{p}",
            f"qdepth_fft_centroid_port{p}",
        ]:
            if raw in df.columns:
                ecdf_cols[f"{raw}_ecdf"] = _ecdf_col(df, raw, ecdf_objects)

        # slopes
        if f"qdepth_slope_port{p}" in df.columns:
            ecdf_cols[f"qdepth_slope_port{p}_ecdf"] = _ecdf_col(df, f"qdepth_slope_port{p}", ecdf_objects)
        
        if f"mbps_slope_port{p}" in df.columns:
            ms_col = f"mbps_slope_port{p}"
            neg_drop_col = f"mbps_slope_neg_port{p}"
            df[neg_drop_col] = (-df[ms_col]).clip(lower=0.0)
            ecdf_cols[f"{neg_drop_col}_ecdf"] = _ecdf_col(df, neg_drop_col, ecdf_objects)

    # ── cross-port summary features ────────────────────────────────────────────
    for raw in ["qdepth_max_imbalance", "mbps_imbalance", "total_qdepth_max",
                "max_qdepth_p99", "total_qdepth_p99", "qdepth_oscillation",
                "qdepth_fft_max_all"]:
        if raw in df.columns:
            ecdf_cols[f"{raw}_ecdf"] = _ecdf_col(df, raw, ecdf_objects)

    # attach all ECDF columns
    for col_name, arr in ecdf_cols.items():
        df[col_name] = arr

    # ── combination indices ────────────────────────────────────────────────────
    def product_index(*col_names, n: int = 2) -> np.ndarray:
        available = [c for c in col_names if c in df.columns]
        if not available:
            return np.zeros(len(df))
        product = np.ones(len(df))
        for c in available:
            product *= ALPHA * df[c].values
        return product ** n

    # congestion: queue build-up + throughput degradation
    q_ecdf_cols = [f"src1_port{p}_qdepth_max_ecdf" for p in PORTS if f"src1_port{p}_qdepth_max_ecdf" in df.columns]
    drop_ecdf_cols = [f"src1_port{p}_mbps_drop_ecdf" for p in PORTS if f"src1_port{p}_mbps_drop_ecdf" in df.columns]
    if q_ecdf_cols and drop_ecdf_cols:
        q_max_ecdf = np.max(np.column_stack([df[c].values for c in q_ecdf_cols]), axis=1)
        drop_max_ecdf = np.max(np.column_stack([df[c].values for c in drop_ecdf_cols]), axis=1)
        df["idx_congestion"] = ((ALPHA * q_max_ecdf) * (ALPHA * drop_max_ecdf)) ** 2
    else:
        df["idx_congestion"] = 0.0

    # instability: queue oscillation + FFT spectral energy
    cv_ecdf_cols = [f"qdepth_cv_port{p}_ecdf" for p in PORTS if f"qdepth_cv_port{p}_ecdf" in df.columns]
    fft_ecdf_cols = [f"qdepth_fft_max_port{p}_ecdf" for p in PORTS if f"qdepth_fft_max_port{p}_ecdf" in df.columns]
    if cv_ecdf_cols and fft_ecdf_cols:
        cv_max = np.max(np.column_stack([df[c].values for c in cv_ecdf_cols]), axis=1)
        fft_max = np.max(np.column_stack([df[c].values for c in fft_ecdf_cols]), axis=1)
        df["idx_instability"] = ((ALPHA * cv_max) * (ALPHA * fft_max)) ** 2
    else:
        df["idx_instability"] = 0.0

    # load balance
    df["idx_load_balance"] = product_index(
        "src1_port2_load_util_ecdf",
        "src1_port3_load_util_ecdf",
        "src1_port4_load_util_ecdf",
        "src1_port5_load_util_ecdf",
        "mbps_imbalance_ecdf",
    )

    df.to_csv(OUT_CSV, index=False)
    joblib.dump(ecdf_objects, ECDF_PKL)
    print(f"Wrote {OUT_CSV} with ECDF-transformed FFT features.")


if __name__ == "__main__":
    main()
