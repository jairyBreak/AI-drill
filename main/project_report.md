# AI-Drill 專題進度報告

---

## 本週改動（2026-04-22）

### 1. 修正 `start_env.sh` — p4run 路徑

`sudo` 不繼承 venv 的 PATH，導致直接跑 `sudo p4run` 找不到指令。把 `P4RUN_CMD` 改成 p4run 的完整絕對路徑解決。

```bash
# 改前
P4RUN_CMD="sudo p4run"

# 改後
P4RUN_CMD="sudo /home/linuxey/Capstone/p4dev-python-venv/bin/p4run"
```

---

### 2. 修正 `p4-utils/thrift_API.py` — KeyError: 'act_prof_name'

`dataset_builder.py` 一跑就噴 `無法連線至 l1 的 Thrift API: 'act_prof_name'`，所有實驗全部跳過。

根本原因：`thrift_API.py` 第 316 行在解析 Action Profile 時，if/else 結束後仍無條件讀取 `j_table["act_prof_name"]`，但新格式 JSON（我們的 P4 編譯器產出）使用的是 `"action_profile"` 這個 key，導致 KeyError。

```python
# 修改前（第 316 行）
self.action_profs[j_table["act_prof_name"]] = action_prof

# 修改後：相容新舊格式
prof_key = j_table.get("act_prof_name") or j_table.get("action_profile")
self.action_profs[prof_key] = action_prof
```

---

### 3. 新增 `label_generator.py` — 四類別標注

原始訓練資料的三個 regression label（`Label_Latency_ms`、`Label_Jitter_ms`、`Label_Loss_Rate`）高度相關，本質上都是同一網路狀態的不同側面。改為合併成一個四類別分類標籤 `Label_Class`，讓模型直接學「這組路由策略對應哪種網路狀態」。

判斷邏輯（按優先順序）：

| 類別 | 值 | 規則 |
|------|----|------|
| `NON_CONGESTION_LOSS` | 3 | Loss > 1% 且 max_qdepth < 10 且 Latency < 50ms |
| `SUSTAINED_CONGESTION` | 1 | max_qdepth == 64 且 Latency > 200ms 且 Loss > 5% |
| `BURST_CONGESTION` | 2 | max_qdepth ≥ 10 或（Loss > 2% 且 Jitter > 15ms） |
| `NORMAL` | 0 | 其餘 |

`NON_CONGESTION_LOSS` 優先判斷，因為它與壅塞型異常本質不同（佇列未滿卻有丟包，代表鏈路問題）。`max_qdepth` 上限為 64 而非硬體的 256，因為 `telemetry_collector.py` 有 `min(64, q_depth)` 的 cap，64 代表程式定義的壅塞飽和點。

支援 CLI args，可對不同資料集執行：

```bash
python label_generator.py                                          # 預設: master.csv → labeled.csv
python label_generator.py training_dataset_v2.csv training_dataset_labeled_v2.csv
```

---

### 4. 修改 `dataset_builder.py` — 保留原始遙測時序

原本每次實驗的 100 筆原始遙測資料（`temp_x_{i}.csv`）用完即刪。改為保存至 `raw_telemetry/` 資料夾。

```python
# 改前
os.remove(temp_x_csv)

# 改後
import shutil
os.makedirs("raw_telemetry", exist_ok=True)
shutil.move(temp_x_csv, f"raw_telemetry/experiment_{iteration_id}.csv")
```

保留原因：現有 `qdepth_max` 只是 10 秒內的最大值，無法區分「整個 10 秒都堵」與「最後一瞬間才衝高」。完整時序可計算 `qdepth_persistence`（超過門檻的時間比例），未來用於更精確區分 SUSTAINED 與 BURST。

---

### 5. 清除廢資料、新增推導特徵、重新訓練分類器

**問題發現**：檢查 `training_dataset_master.csv` 後發現後 1500 筆全是廢資料——`Label_Loss_Rate == 100%`、所有 `qdepth_max == 0`。原因是那批資料收集期間 iperf3 server 掉線，客戶端無法連線，封包全部「送出但無回應」被統計為 100% loss，而佇列因為沒有真實流量所以始終為空。

**處理方式**：

新增 `rebuild_dataset.py`，執行以下流程：
1. 從 `training_dataset_master.csv` 取前 1068 筆有效資料
2. 從現有聚合欄位計算 11 個推導特徵（不需要 Mininet，純離線）
3. 輸出 `training_dataset_v2.csv`（35 欄）

新增的 11 個推導特徵：

| 特徵 | 計算方式 | 物理意義 |
|------|----------|----------|
| `src1_port{N}_mbps_cv` | std / mean | 相對突發性（越高越不穩定） |
| `src1_port{N}_load_util` | mbps_mean / capacity | 鏈路使用率 |
| `qdepth_max_imbalance` | max(qdepth_maxes) − min(qdepth_maxes) | 各 port 壅塞不對稱程度 |
| `mbps_imbalance` | std([port_means]) | 負載分配不均程度 |
| `total_qdepth_max` | sum(qdepth_maxes) | 全域壅塞嚴重度 |

同步更新 `dataset_builder.py`，未來新實驗也會直接計算這些推導特徵。

**訓練結果對比**：

| 指標 | Baseline（24 欄） | V2（35 欄） |
|------|-------------------|------------|
| 資料量 | 1068 筆 | 1068 筆 |
| 整體 accuracy | 96% | **97%** |
| NORMAL recall | 100% | 100% |
| SUSTAINED recall | 96% | **98%** |
| BURST recall | 86% | 86% |

Feature importance top-2 全為新增特徵：`total_qdepth_max`（#1）、`qdepth_max_imbalance`（#2）。

BURST recall 持平於 86%，原因是最關鍵的時序特徵 `qdepth_persistence` 需要 raw telemetry，而現有 1068 筆有效資料無對應的原始時序檔案。此項改善留待下次重新收集資料時補入。

---

## 下一步

- 重新收集 1500+ 筆資料（確保 iperf3 server 持續運行）
- 從 `raw_telemetry/` 加入 `qdepth_persistence`（queue 飽和比例）特徵，預期可提升 BURST recall
- 考慮方向 A：閉迴路控制（telemetry → 分類器預測 → 自動調整 W-ECMP 權重）
