# 2026/04/21 石石專題進度報告

---

## 修正 `start_env.sh` — p4run 路徑

`sudo` 不繼承 venv 的 PATH，所以直接跑 `sudo p4run` 找不到指令。把 `P4RUN_CMD` 改成 p4run 的完整絕對路徑就解決了。

```bash
# 改前
P4RUN_CMD="sudo p4run"

# 改後
P4RUN_CMD="sudo /home/linuxey/Capstone/p4dev-python-venv/bin/p4run"
```

---

## 修正 `p4-utils/thrift_API.py` — KeyError: 'act_prof_name'

`dataset_builder.py` 一跑就噴 `無法連線至 l1 的 Thrift API: 'act_prof_name'`，所有實驗全部跳過。

根本原因在 `thrift_API.py` 第 316 行：這裡在處理 Action Profile 的時候，用了一個 if/else 來相容新舊兩種 JSON 格式。新格式（我們 P4 編譯器產出的）用的是 `"action_profile"` 這個 key，所以走 if 那條，但 if/else 結束之後，第 316 行仍然無條件去讀 `j_table["act_prof_name"]`，而新格式的 JSON 根本沒有這個 key，直接 KeyError。

```python
# 問題所在（第 316 行）
self.action_profs[j_table["act_prof_name"]] = action_prof  # 新格式沒有這個 key

# 修改後：兩種格式都能處理
prof_key = j_table.get("act_prof_name") or j_table.get("action_profile")
self.action_profs[prof_key] = action_prof
```

---

## 新增 `label_generator.py` — 異常四類別標注

原本的訓練資料只有三個 regression label（`Label_Latency_ms`、`Label_Jitter_ms`、`Label_Loss_Rate`），三個值彼此高度相關，本質上都是同一個網路狀態的不同側面。我們決定把它們合併成一個四類別的分類標籤 `Label_Class`，讓模型直接學「這組路由策略對應哪種網路狀態」。

四個類別的判斷邏輯按優先順序如下：

| 類別 | 值 | 規則 |
|---|---|---|
| `NON_CONGESTION_LOSS` | 3 | Loss > 1% 但 max_qdepth < 10 且 Latency < 50ms |
| `SUSTAINED_CONGESTION` | 1 | max_qdepth == 64 且 Latency > 200ms 且 Loss > 5% |
| `BURST_CONGESTION` | 2 | max_qdepth ≥ 10 或（Loss > 2% 且 Jitter > 15ms） |
| `NORMAL` | 0 | 其餘 |

NON_CONGESTION_LOSS 優先判斷是因為它和壅塞型異常的本質不同（佇列沒滿卻有丟包，代表是鏈路問題），如果放到最後才判斷很容易被 BURST 誤判。max_qdepth 的上限是 64 而不是硬體的 256，是因為 `telemetry_collector.py` 裡有做 `min(64, q_depth)` 的 cap，所以 64 就代表「達到程式定義的壅塞飽和點」。

這支程式對現有的 `training_dataset_master.csv` 做後處理，不需要 Mininet，直接跑就好，輸出 `training_dataset_labeled.csv`。

```bash
python label_generator.py
```

---

## 修改 `dataset_builder.py` — 保留原始遙測時序

原本每次實驗的 100 筆原始遙測資料（`temp_x_{i}.csv`）用完就直接刪掉。我們把 `os.remove()` 改成 `shutil.move()`，把每次實驗的原始時序存進 `raw_telemetry/` 資料夾。

```python
# 改前
os.remove(temp_x_csv)

# 改後
import shutil
os.makedirs("raw_telemetry", exist_ok=True)
shutil.move(temp_x_csv, f"raw_telemetry/experiment_{iteration_id}.csv")
```

保留的原因：現有的 `qdepth_max` 特徵只是 10 秒內的最大值，沒辦法分辨「整個 10 秒都堵」和「最後一瞬間才衝高」。有了完整時序之後，未來可以計算「qdepth 超過某個門檻的時間比例」，讓 SUSTAINED 和 BURST 兩個類別的標注更精確。

---

## 目前進度

四個改動都還沒 commit。`dataset_builder.py` 正在背景跑，預計需要 5–6 小時跑完 1500 次實驗。等資料收集完成之後，下一步是執行 `label_generator.py` 產出分類資料集，再把 `random_forest_test.py` 從 regression 改成 classification，補上 precision/recall/F1 等正確的評估指標。
