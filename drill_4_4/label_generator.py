import pandas as pd

CSV_IN  = "training_dataset_master.csv"
CSV_OUT = "training_dataset_labeled.csv"

NORMAL               = 0
SUSTAINED_CONGESTION = 1
BURST_CONGESTION     = 2
NON_CONGESTION_LOSS  = 3

CLASS_NAMES = {
    NORMAL:               "NORMAL",
    SUSTAINED_CONGESTION: "SUSTAINED_CONGESTION",
    BURST_CONGESTION:     "BURST_CONGESTION",
    NON_CONGESTION_LOSS:  "NON_CONGESTION_LOSS",
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

    # 規則 1: 有丟包但佇列空 → 非壅塞型丟包 (鏈路問題)
    if loss > 1.0 and max_q < 10 and latency < 50:
        return NON_CONGESTION_LOSS

    # 規則 2: 佇列打滿 + 高延遲 + 明顯丟包 → 持續性壅塞
    if max_q == 64 and latency > 200 and loss > 5.0:
        return SUSTAINED_CONGESTION

    # 規則 3: 有佇列積累 或 丟包+高 jitter 組合 → 突發性壅塞
    if max_q >= 10 or (loss > 2.0 and jitter > 15.0):
        return BURST_CONGESTION

    return NORMAL


if __name__ == "__main__":
    df = pd.read_csv(CSV_IN)
    df["Label_Class"] = df.apply(classify_row, axis=1)

    counts = df["Label_Class"].value_counts().sort_index()
    print("=== 標籤分佈 ===")
    for cls_id, cnt in counts.items():
        print(f"  {cls_id} {CLASS_NAMES[cls_id]:<22}: {cnt:>5} 筆  ({cnt/len(df)*100:.1f}%)")

    df.to_csv(CSV_OUT, index=False)
    print(f"\n已儲存 → {CSV_OUT}  (共 {len(df)} 筆，新增 Label_Class 欄)")
