import sys
import pandas as pd

CSV_IN  = sys.argv[1] if len(sys.argv) > 1 else "training_dataset_master.csv"
CSV_OUT = sys.argv[2] if len(sys.argv) > 2 else "training_dataset_labeled.csv"

NORMAL               = 0
SUSTAINED_CONGESTION = 1
BURST_CONGESTION     = 2
NON_CONGESTION_LOSS  = 3
HIGH_JITTER          = 4
UNBALANCED_LOAD      = 5

CLASS_NAMES = {
    NORMAL:               "NORMAL",
    SUSTAINED_CONGESTION: "SUSTAINED_CONGESTION",
    BURST_CONGESTION:     "BURST_CONGESTION",
    NON_CONGESTION_LOSS:  "NON_CONGESTION_LOSS",
    HIGH_JITTER:          "HIGH_JITTER",
    UNBALANCED_LOAD:      "UNBALANCED_LOAD",
}

QDEPTH_COLS = [
    "src1_port2_qdepth_max",
    "src1_port3_qdepth_max",
    "src1_port4_qdepth_max",
    "src1_port5_qdepth_max",
]

def classify_row(row):
    max_q   = max(row[c] for c in QDEPTH_COLS)
    loss    = row["Label_Loss_Rate"]
    latency = row["Label_Latency_ms"]
    jitter  = row["Label_Jitter_ms"]
    imbalance = row.get("idx_load_balance", row.get("mbps_imbalance", 0))

    if loss > 1.0 and max_q < 10 and latency < 50:
        return NON_CONGESTION_LOSS
    if max_q == 64 and latency > 200 and loss > 5.0:
        return SUSTAINED_CONGESTION
    if imbalance > 0.08:
        return UNBALANCED_LOAD
    if max_q >= 10:
        return BURST_CONGESTION
    if jitter > 15.0:
        return HIGH_JITTER
    return NORMAL

if __name__ == "__main__":
    df = pd.read_csv(CSV_IN)
    df = df.copy()
    df["Label_Class"] = df.apply(classify_row, axis=1)
    counts = df["Label_Class"].value_counts().sort_index()
    print("=== 標籤分佈 (Sweet Spot: 15ms / 0.08) ===")
    for cls_id, cnt in counts.items():
        print(f"  {cls_id} {CLASS_NAMES[cls_id]:<22}: {cnt:>5} 筆  ({cnt/len(df)*100:.1f}%)")
    df.to_csv(CSV_OUT, index=False)
    print(f"\n已儲存 → {CSV_OUT}  (共 {len(df)} 筆)")
