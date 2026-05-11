# DRILL + W-ECMP: P4 網路監控與效能預測系統:

本專案實作了一套結合 **DRILL (Deep Reinforcement Learning-based Load balancing)** 核心思想與 **W-ECMP (Weighted Equal-Cost Multi-Path)** 的智慧化流量工程架構。

在本架構中，**DRILL** 被定義為一種 **Per-packet (逐包)** 的負載均衡機制，它在數據平面實時監控各輸出端口的 **Queue Depth (佇列深度)**，並針對每個抵達的封包，從候選路徑中挑選出佇列壓力最小的端口進行轉發。這種機制保證了對微秒級流量波動的極速反應。

我們透過 P4 帶內網路遙測 (INT) 將這些底層物理特徵（如佇列斜率、飽和度、ECDF 分佈等）上報給控制器，利用機器學習 (Random Forest) 模型，實現對網路延遲 (Latency)、抖動 (Jitter) 與丟包率 (Loss) 的實時高精度預測，進而為 W-ECMP 的權重調整提供智能化的決策支持。

---

## 🚀 快速開始步驟

### 1. 啟動實驗環境
開啟 Mininet 與 P4 開關拓撲：
```bash
./start_env.sh
```

### 2. 數據採集與資料集構建
如果需要增加新的訓練樣本（自動化打流量並紀錄特徵）：
```bash
python3 dataset_builder.py
```
*產出：`research_results/data/datasets/` 下的原始數據與標籤。*

### 3. 生產與訓練最佳化模型（Pull 後先做）
使用當前最佳的「經典進化版」參數進行訓練，並自動顯示 R2 與 MAE 指標：
```bash
python3 train_simplified_models.py
```
*產出：`rf_model_*_simplified.pkl` 模型檔案。*

### 4. 啟動實時監控預測
啟動控制器，實時觀察預測值與真實流量（iperf3/Ping）的對比：
```bash
python3 realtime_ml_controller.py
```

### 5. 執行視覺化驗證圖表
在有流量背景的情況下，錄製並產生專業的效能對比圖：
```bash
python3 plot_all_metrics.py [持續秒數]
```
*圖表儲存於：`research_results/plots/validation/full_metrics_comparison.png`*

---

## 🛠️ 進階分析工具

- **特徵貢獻分析**：查看哪些網路指標對預測最重要。
  ```bash
  python3 analyze_feature_importance.py
  ```
- **魯棒性驗證**：在 `research_results/tools_and_archive/` 中可找到交叉驗證與穩定性測試腳本。

---

## 📁 專案結構說明

- **`research_results/`**：核心產出資料夾。
    - `plots/`：包含驗證對比圖與特徵分析圖。
    - `data/`：包含訓練資料集與實驗數據 CSV。
    - `tools_and_archive/`：存放歷史實驗腳本與超參數調優記錄。
- **`p4src/`**：P4 原始碼，定義了 DRILL (Queue-aware) 與 W-ECMP 的數據平面邏輯。
- **`rf_model_*.pkl`**：當前線上運作的機器學習模型。

---

## 📈 目前工作進度
- [x] **架構整合**：成功結合 DRILL (Per-packet Queue Monitoring) 遙測機制與 W-ECMP 轉發架構。
- [x] **滑動視窗機制**：實作 10s 滑動視窗，對齊模型訓練尺度。
- [x] **特徵進化**：加入 `qdepth_sq` (非線性) 與 `qdepth_slope` (趨勢) 特徵。
- [x] **自動調優**：完成大規模超參數搜尋，確立了經典進化版 (Guardian Mode) 為最佳配置。
- [x] **視覺優化**：圖表自動限幅與自適應 Y 軸縮放。

---

## 🔜 下一階段重點：閉環動態權重控制

目前的系統已完成「遙測感知」與「智能預測」，下一步的核心目標是實現 **「閉環控制 (Closed-loop Control)」**。由於 ML 模型目前預測的是端對端 (E2E) 指標，我們將採用以下三種方案將全局預測轉化為 Per-port-group 的權重決策：


### 方案 A：虛擬權重模擬 (What-if Simulation) —— **優先實作**
利用已訓練的模型作為「網路數位孿生」，在控制器端模擬權重變動的影響：
- **模擬執行**：針對 Port 2, 3, 4, 5 分別建構虛擬特徵向量（假設流量完全偏向該埠）。
- **預測對比**：詢問模型：「如果流量走這條路，預期延遲是多少？」。
- **決策**：選取預測延遲最低的埠，增加其在 W-ECMP 中的權重。

### 方案 B：特徵貢獻歸因 (Feature Attribution)
利用隨機森林的特徵貢獻度，實時找出擁塞元兇：
- **故障定位**：當預測到 `ANOMALY` 時，立刻分析哪個埠的 `qdepth` 或 `util` 貢獻了最高的決策權重。
- **動態懲罰**：對貢獻擁塞特徵最多的埠進行「扣分」，主動調降其分擔比例。

### 方案 C：多路徑指標分解 (Multi-path Decomposition)
升級資料集與模型架構，實現更精細的控制：
- **精細化標籤**：修改 `dataset_builder` 以採集各路徑專屬的延遲數據。
- **多輸出預測**：訓練一個能同時輸出 $Y = [Lat_2, Lat_3, Lat_4, Lat_5]$ 的多目標模型。

接下來實做在 P4 上：

2.  **動態權重計算 (Dynamic Weight Assignment)**：
      - 根據健康分數的反比關係，動態計算 W-ECMP 的流量分擔比例（如：健康路徑佔 70%，擁塞路徑佔 10%）。
3.  **數據平面強制執行 (Enforcement)**：
      - 透過 Thrift API 定期（如每秒）將計算出的權重下發至 P4 開關的 `w_ecmp_table` 或 `port_map_reg` 中。
      - **最終目標**：在網路發生大規模丟包或延遲劇增前，系統能主動完成流量遷移，實現網路自癒。
---

## 🔮 未來優化方向 (TODO)

- [ ] **多階段混合預測 (Multi-stage Prediction)**：
    - 實作「正常/擁塞」分類器與「高精度迴歸器」的結合，針對低延遲區間進行誤差專化壓制。
- [ ] **頻譜特徵深度挖掘 (Advanced Spectral Analysis)**：
    - 引入多尺度 FFT 特徵，特別是「低頻能量佔比」，以增強對長週期網路抖動 (Jitter) 的預測能力。
- [ ] **閉環控制集成 (Closed-loop Control)**：
    - 將 ML 預測結果直接回饋給 P4 控制平面，實現自動化路徑切換 (Path Switching) 或動態權重調整。
- [ ] **實時模型在線學習 (Online Learning)**：
    - 開發自動化的模型漂移偵測，並實作實時增量學習，以適應動態變化的網路拓撲與流量模式。
- [ ] **MAE 極致優化**：
    - 探索更先進的損失函數（如 Huber Loss 或 Quantile Regression），針對長尾分佈數據進一步降低物理誤差。
