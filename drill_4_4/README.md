# DRILL + W-ECMP: P4 網路監控與效能預測系統

本專案實作了一套結合 **DRILL (Deep Reinforcement Learning-based Load balancing)** 核心思想與 **W-ECMP (Weighted Equal-Cost Multi-Path)** 的智慧化流量工程與效能預測架構。

在數據平面（Data Plane）中，我們實作了 **Per-packet (逐包)** 的負載均衡機制，實時監控輸出端口的佇列深度（Queue Depth），並從候選路徑中挑選出佇列壓力最小的端口進行轉發（極速反應微秒級流量波動）。同時，藉由帶內網路遙測 (INT) 將佇列斜率、飽和度、累積延遲等底層特徵上報給控制平面，結合機器學習（Random Forest）預測網路延遲 (Latency)、抖動 (Jitter) 與丟包率 (Loss)，最終為動態權重決策提供網路狀態感知與 What-If 權重兵棋推演的支援。

---

## 🚀 雙軌工作流指引

本專案將監控與預測劃分為兩種時間尺度的模型工作流：

### 方案一：10秒尺度（離散/歷史分析工作流）—— 專注宏觀分佈
用於分析長週期流量趨勢，並在穩定狀態下進行高精度的全局效能評估。

1. **啟動環境**：
   ```bash
   ./start_env.sh
   ```
   *此腳本會啟動 Mininet、[all_controller.py](file:///home/p4/drill/drill_4_4/all_controller.py)（Thrift/CLI 規則配置）以及 [rate_limiter.py](file:///home/p4/drill/drill_4_4/rate_limiter.py)（自動依據 `p4app.json` 設定 BMv2 限速）。*

2. **資料集採集**：
   ```bash
   python3 dataset_builder.py
   ```
   *每 10 秒進行一次離散流量實驗，自動調節負載與權重，並將採集的 telemetry 特徵與 iperf3 真實量測結果存入 `training_dataset_master.csv`。*

3. **特徵工程（時序與 ECDF 轉換）**：
   ```bash
   # 提取時序特徵（如 FFT 頻譜特徵、變異係數等）
   python3 extract_temporal_features.py
   
   # 計算經驗累積分布函數 (ECDF) 特徵，並生成 ECDF 物件 pickle 檔
   python3 build_ecdf_features.py
   ```
   *產出：`research_results/data/datasets/training_dataset_ecdf.csv` 與 `ecdf_objects.pkl`。*

4. **訓練 10s 離散模型**：
   ```bash
   python3 train_simplified_models.py
   ```
   *產出：`rf_model_*_simplified.pkl` 系列模型。*

5. **實時監控與驗證**：
   ```bash
   # 啟動實時 ML 監控控制器（搭配 ping 與 iperf 探針）
   python3 realtime_ml_controller.py
   
   # 錄製並繪製 10s 尺度的全指標對比圖
   python3 plot_all_metrics.py [持續秒數]
   ```

---

### 方案二：1秒尺度（滾動/瞬態預測工作流）—— 專注 Rehash 響應與 What-If 模擬
適用於需要對網路拓撲變更、流量偏斜或 W-ECMP 重新雜湊（Rehash）進行秒級快速反應的場景。包含預期流量映射（Traffic Projection）特徵，以消除 Data Leakage。

1. **啟動環境**：
   ```bash
   ./start_env.sh
   ```

2. **資料集採集（滾動長連線）**：
   ```bash
   python3 rolling_dataset_builder.py
   ```
   *運行長達 120 秒的並發 UDP 流量實驗，並在實驗中途隨機觸發 Rehash 事件以捕獲瞬態特徵。數據以 1 秒為單位滾動寫入至 `research_results/data/datasets/rolling_training_dataset.csv`。*

3. **訓練 1s 瞬態模型**：
   ```bash
   python3 train_1s_models.py
   ```
   *引入 `Is_Rehash_Event`, `Rehash_Impact`（指數衰減處理）以及 What-If 決策專用的 `Expected_Over_Capacity_Sum` 與 `Expected_Util_*`。*
   *產出：`rf_model_*_1s.pkl` 系列模型。*

4. **實時預測與驗證**：
   ```bash
   # 啟動 1s 實時預測器（從 L1 讀取真實權重，從 L2 讀取遙測資料與丟包統計）
   python3 realtime_1s_predictor.py
   
   # 錄製並繪製 1s 尺度的預測與真實指標對比圖
   python3 plot_1s_metrics.py [持續秒數]
   ```

---

## 🛠️ 進階分析與輔助工具

- **特徵貢獻度分析**：
  - `python3 analyze_feature_importance.py`：針對 10s 離散模型進行特徵貢獻度排序，並將結果圖表存至 `research_results/plots/analysis/ultimate_feature_importance.png`。
  - `python3 print_importances.py`：讀取並列出當前 1s 延遲與丟包模型的特徵重要性排名。
- **網路限速控制器 (Rate Limiter)**：
  - `python3 rate_limiter.py`：解析 `p4app.json` 的拓撲鏈路頻寬，並自動透過 Thrift API 將 BMv2 實體端口的 Queue Rate (PPS) 與 Max Queue Depth 限制寫入交換機硬體。
- **雜湊分佈測試**：
  - `python3 test_best_hash.py`：啟動多個 concurrent iperf3 伺服器與客戶端，測試並評估 P4 數據平面在特定 5-tuple 雜湊下的轉發均衡度。
- **UDP 流量產生器**：
  - `python3 traffic_gen.py <目標IP> <頻寬Mbps> [持續秒數]`：一個輕量級的 socket 流量產生器，用於測試特定路徑頻寬。

---

## 📁 專案檔案結構說明

### 核心網路與 P4 程式碼
*   [p4src/ecmp.p4](file:///home/p4/drill/drill_4_4/p4src/ecmp.p4)：定義 DRILL (Queue-aware Per-packet 轉發) 與 W-ECMP (帶內遙測 INT 與計數器) 的數據平面行為。
*   [network.py](file:///home/p4/drill/drill_4_4/network.py)：使用 P4-Utils API 宣告並建立實驗網路由 (4 Leaf + 4 Spine 拓樸)。
*   [p4app.json](file:///home/p4/drill/drill_4_4/p4app.json)：Mininet 拓撲、鏈路頻寬、最大佇列長度以及 CLI 指令檔的設定檔。
*   `*-commands.txt`：各 P4 交換機（l1~l4, s1~s4）初始載入的流表與轉發規則。

### 數據採集與特徵工程
*   [dataset_builder.py](file:///home/p4/drill/drill_4_4/dataset_builder.py)：10s 離散實驗資料集構建器。
*   [rolling_dataset_builder.py](file:///home/p4/drill/drill_4_4/rolling_dataset_builder.py)：1s 滾動長連線實驗（含 Rehash 事件觸發）資料集構建器。
*   [telemetry_collector.py](file:///home/p4/drill/drill_4_4/telemetry_collector.py)：以 10Hz (0.1s) 頻率讀取交換機遙測暫存器與計數器的背景程式。
*   [extract_temporal_features.py](file:///home/p4/drill/drill_4_4/extract_temporal_features.py)：時域特徵提取器（SLOPE、CV、FFT 頻譜轉換）。
*   [build_ecdf_features.py](file:///home/p4/drill/drill_4_4/build_ecdf_features.py)：對提取的特徵進行累積經驗分佈轉換，計算擁塞、不穩定度及負載均衡指數。
*   [iperf_parser.py](file:///home/p4/drill/drill_4_4/iperf_parser.py)：用於精確解析 iperf3 JSON 報告與 Ping 個別樣本的統計模組。
*   [label_generator.py](file:///home/p4/drill/drill_4_4/label_generator.py)：依據佇列深度、延遲與丟包閾值為樣本標記網路狀態類別。

### 模型訓練與實時預測
*   [train_simplified_models.py](file:///home/p4/drill/drill_4_4/train_simplified_models.py)：10s 模型隨機森林訓練腳本。
*   [train_1s_models.py](file:///home/p4/drill/drill_4_4/train_1s_models.py)：1s 模型隨機森林訓練腳本。
*   [realtime_ml_controller.py](file:///home/p4/drill/drill_4_4/realtime_ml_controller.py)：10s 尺度實時預測與監控服務。
*   [realtime_1s_predictor.py](file:///home/p4/drill/drill_4_4/realtime_1s_predictor.py)：1s 尺度實時預測與硬體 Ground Truth 對比服務。

### 效能驗證與繪圖
*   [plot_all_metrics.py](file:///home/p4/drill/drill_4_4/plot_all_metrics.py)：錄製並繪製 10s 全指標預測對比折線圖。
*   [plot_1s_metrics.py](file:///home/p4/drill/drill_4_4/plot_1s_metrics.py)：錄製並繪製 1s 預測與真實硬體指標對比折線圖。
*   [plot_latency_validation.py](file:///home/p4/drill/drill_4_4/plot_latency_validation.py)：延遲預測特化版驗證與繪圖工具（含 EMA 非對稱平滑處理）。

---

## 📈 目前工作進度

- [x] **架構整合**：成功結合 DRILL (Per-packet Queue Monitoring) 遙測機制與 W-ECMP 轉發架構。
- [x] **雙尺度模型並行 (Dual-scale Models)**：建立 10 秒尺度歷史分析模型 (`_simplified.pkl`) 與 1 秒尺度瞬態響應模型 (`_1s.pkl`) 的雙軌並行機制。
- [x] **時間與映射特徵 (Time-Series & Projection Features)**：在 1 秒模型中引入 `QDepth_Trend`, `Rehash_Impact` 衰減，以及用於權重兵棋推演的 `Expected_Util_PX` 特徵，徹底消除 Data Leakage，使得模型具備 "What-If" 預測能力。
- [x] **Thrift API 批次寫入優化**：在 [all_controller.py](file:///home/p4/drill/drill_4_4/all_controller.py) 中引入 `SimpleSwitchThriftAPI` 批次 CLI 指令緩衝機制 (`commit_cli_cmds`)，大幅減少子行程 (subprocess) 開銷。
- [x] **硬體限速自動化**：開發 [rate_limiter.py](file:///home/p4/drill/drill_4_4/rate_limiter.py) 自適應讀取鏈路設定並套用 BMv2 隊列頻寬 PPS 限制，保證實驗與資料集數據的一致性。
- [x] **動態 Y 軸限幅與視覺優化**：圖表驗證工具可自適應不同流量負載下的延遲與丟包範圍。

---

## 🔜 下一階段重點：閉環動態權重控制

目前系統已完成「遙測感知」與「What-If 智能預測」，下一步的核心目標是實現 **「閉環控制 (Closed-loop Control)」**。
利用預期流量映射特徵，我們可以在控制器端進行 **虛擬權重模擬 (What-if Simulation)**：
1. 當 1s 預測器偵測到潛在擁塞或異常時，針對各候選埠（如 Port 2, 3, 4, 5）虛擬生成新權重配置下的 expected 特徵向量。
2. 詢問 1s 延遲與丟包模型：「如果我將流量權重切換為此配置，下一秒預計的延遲與丟包是多少？」。
3. 選擇預估效能最優的權重，並透過 [all_controller.py](file:///home/p4/drill/drill_4_4/all_controller.py) 實作的 Thrift API 批次下發至 P4 數據平面的 `w_ecmp_table` 與 `port_map_reg` 中，實現主動式自癒網路。
