"""
Train 4 Random Forest regression models on ECDF-enriched features.

For each target (avg latency, p99 latency, jitter, loss rate):
  - 5-fold cross-validation: per-fold and mean R², RMSE, MAE
  - Comparison table: R² with ECDF features vs. raw features only
  - Feature importance bar chart saved as PNG
  - Model saved as rf_regressor_<label>.pkl

Input : training_dataset_ecdf.csv   (or first CLI arg)
"""
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "training_dataset_ecdf.csv"

TARGETS = [
    ("Label_Latency_ms",     "latency_ms"),
    ("Label_Latency_p99_ms", "latency_p99_ms"),
    ("Label_Jitter_ms",      "jitter_ms"),
    ("Label_Loss_Rate",      "loss_rate"),
]

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)

ECDF_SUFFIXES = ("_ecdf", "idx_congestion", "idx_instability", "idx_load_balance")

INVALID_LABEL = -1.0


def make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(**RF_PARAMS)


def cv_metrics(model, X: pd.DataFrame, y: pd.Series, k: int = 5) -> dict:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores, rmse_scores, mae_scores = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        m = RandomForestRegressor(**RF_PARAMS)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        r2_scores.append(r2_score(y_te, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_te, preds)))
        mae_scores.append(mean_absolute_error(y_te, preds))

    return {
        "r2_folds": r2_scores,
        "r2_mean": np.mean(r2_scores),
        "rmse_mean": np.mean(rmse_scores),
        "mae_mean": np.mean(mae_scores),
    }


def save_importance_plot(model: RandomForestRegressor, feature_names, label: str, top_n: int = 20):
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    top = imp.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    top[::-1].plot.barh(ax=ax)
    ax.set_title(f"Feature Importance — {label} (top {top_n})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = f"rf_importance_{label}.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {CSV_PATH}  ({len(df)} rows, {len(df.columns)} columns)\n")

    all_label_cols = [t[0] for t in TARGETS]
    feature_cols = [c for c in df.columns if not c.startswith("Label_")]

    # feature sets for comparison
    raw_feature_cols = [c for c in feature_cols
                        if not any(c.endswith(sfx) or c == sfx for sfx in ECDF_SUFFIXES)]
    ecdf_feature_cols = feature_cols  # all features including ECDF cols

    comparison_rows = []

    for label_col, label_name in TARGETS:
        if label_col not in df.columns:
            print(f"Skipping {label_col} — column not found in dataset.\n")
            continue

        # drop invalid measurements
        mask = df[label_col] != INVALID_LABEL
        subset = df[mask].copy()
        if len(subset) < 50:
            print(f"Skipping {label_col} — only {len(subset)} valid rows.\n")
            continue

        y = subset[label_col]

        print(f"═══ {label_col} ({len(subset)} valid rows) ═══")

        # ── CV with raw features only ──────────────────────────────────────
        X_raw = subset[[c for c in raw_feature_cols if c in subset.columns]]
        raw_metrics = cv_metrics(make_rf(), X_raw, y)
        print(f"  Raw features only   — R²: {raw_metrics['r2_mean']:.4f}  "
              f"RMSE: {raw_metrics['rmse_mean']:.4f}  MAE: {raw_metrics['mae_mean']:.4f}")

        # ── CV with all features (raw + temporal + ECDF + indices) ────────
        X_all = subset[[c for c in ecdf_feature_cols if c in subset.columns]]
        all_metrics = cv_metrics(make_rf(), X_all, y)
        print(f"  All features (ECDF) — R²: {all_metrics['r2_mean']:.4f}  "
              f"RMSE: {all_metrics['rmse_mean']:.4f}  MAE: {all_metrics['mae_mean']:.4f}")

        per_fold_str = "  ".join(f"{v:.3f}" for v in all_metrics["r2_folds"])
        print(f"  Per-fold R²: {per_fold_str}\n")

        comparison_rows.append({
            "Target": label_col,
            "R²_raw": round(raw_metrics["r2_mean"], 4),
            "R²_ecdf": round(all_metrics["r2_mean"], 4),
            "Δ_R²": round(all_metrics["r2_mean"] - raw_metrics["r2_mean"], 4),
            "RMSE_ecdf": round(all_metrics["rmse_mean"], 4),
            "MAE_ecdf": round(all_metrics["mae_mean"], 4),
        })

        # ── final model trained on all data ───────────────────────────────
        final_model = make_rf()
        final_model.fit(X_all, y)

        png_path = save_importance_plot(final_model, X_all.columns, label_name)
        pkl_path = f"rf_regressor_{label_name}.pkl"
        joblib.dump(final_model, pkl_path)
        print(f"  Saved: {pkl_path}  |  {png_path}\n")

    # ── comparison summary ─────────────────────────────────────────────────
    print("═══ Comparison: ECDF features vs. raw features ═══")
    cmp_df = pd.DataFrame(comparison_rows)
    print(cmp_df.to_string(index=False))


if __name__ == "__main__":
    main()
