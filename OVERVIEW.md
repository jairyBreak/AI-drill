# AI-Drill: ML-Augmented W-ECMP Load Balancer

## Project Goal

Build a **closed-loop, adaptive load balancer** for a simulated P4 data center network, in two stages:

**Stage 1 — Prediction (complete):** Train Random Forest models to predict next-second QoS metrics (latency, jitter, packet loss) from real-time P4 switch telemetry. The models learn from ~4000 collected experiments where random ECMP weights were applied and the resulting network performance measured.

**Stage 2 — Control (not yet implemented):** Train a second model (or policy) that takes the Stage 1 predictions as input and outputs optimal ECMP weight adjustments. This closes the loop: telemetry → predict QoS → adjust weights → better performance.

The underlying network uses two complementary routing mechanisms:
- **W-ECMP**: probabilistic traffic splitting across path groups, with weights set by the control plane.
- **DRILL** (Distributed Randomized In-network Load balancing): inside each group, the P4 dataplane itself picks the least-queued port every packet via power-of-two-choices with memory.

---

## Repository Layout

```
AI-drill/
├── jsq_2_2/              # Original 2-switch ECMP reference (educational baseline)
└── drill_4_4/            # Active project: 4-leaf/4-spine asymmetric topology
    ├── p4src/            # P4 program (dataplane)
    ├── research_results/ # Curated datasets, validation runs, plots
    │   ├── data/datasets/      # Processed feature datasets (master → temporal → ECDF → cleaned)
    │   ├── data/validation/    # Per-run validation CSVs (4 validation experiments)
    │   ├── plots/validation/   # Predicted vs real comparison charts
    │   ├── plots/analysis/     # Feature importance plots
    │   └── tools_and_archive/  # Older experimental scripts
    └── bruh/             # Early algorithm prototypes (not production)
```

---

## Network Topology

```
  h1 ── l1 ──┬─── s1 ───┬── l2 ── h2
             ├─── s2 ───┤
             ├─── s3 ───┤         (l3, l4 connect symmetrically to all spines)
             └─── s4 ───┘
  h3 ── l3               h4 ── l4
```

- **4 hosts**: h1–h4 at 10.0.1.1 / 10.0.2.2 / 10.0.3.3 / 10.0.4.4
- **4 leaf switches**: l1–l4 (Thrift ports 9090–9093)
- **4 spine switches**: s1–s4 (Thrift ports 9094–9097)

**Asymmetric uplinks** (this is the key design choice — creates two distinct W-ECMP components):

| Port at l1/l2 | Spine | Physical BW | Soft-capped (×0.8 via rate_limiter) |
|---|---|---|---|
| 2 | s1 | 1.0 Mbps | **0.8 Mbps** |
| 3 | s2 | 1.0 Mbps | **0.8 Mbps** |
| 4 | s3 | 1.5 Mbps | **1.2 Mbps** |
| 5 | s4 | 1.5 Mbps | **1.2 Mbps** |

l3/l4 are symmetric (all 1.0 Mbps). The asymmetry on l1/l2 makes weight optimization non-trivial and interesting.

---

## Starting the Environment
After activating p4dev's environment...
```bash
cd drill_4_4 && ./start_env.sh
```
Cleans Mininet state, starts p4run in the foreground, then schedules `all_controller.py` (after 10 s) and `rate_limiter.py` (after 13 s) in a background subshell. When it prints "背景設施全部就緒," open a second terminal.

### Or Manually

```bash
sudo p4run
python all_controller.py      # install W-ECMP routing rules
python rate_limiter.py        # apply port rate/queue limits
```

---

## P4 Dataplane (`p4src/ecmp.p4`)

### State (Registers & Counters)

| Name | Size | Purpose |
|---|---|---|
| `q_depth_reg` | 512 | Live queue depth per port — written by every departing packet |
| `last_best_p_reg` | 512 | DRILL memory: last winning port per ECMP group |
| `port_map_reg` | 1024 | `(group_id × 16 + logical_idx)` → physical port |
| `path_max_queue_depth_reg` | 1024 | Max queue depth per (src_id, port) since last reset — read and zeroed by telemetry collector |
| `port_bytes_counter` | 256 | Cumulative bytes per port — delta = throughput |
| `cnt_ingress / cnt_egress` | 512 | Per-port packet counters |

### Forwarding Pipeline

```
ipv4_lpm
 └─ set_w_ecmp → w_ecmp_table     (CRC16 action_selector on 5-tuple → component_id)
                  └─ drill_params_table  (component_id → run_drill action)
                       └─ ecmp_group_to_nhop  (component_id + best_port → MAC/egress)
```

- **`w_ecmp_table`** uses an `action_selector` where group member count = weight. A flow is hashed to a component probabilistically proportional to its weight.
- **`run_drill`**: picks two distinct random ports in the component, reads their queue depths plus the last-best port's depth, selects the minimum, and writes back the winner. This is DRILL (power-of-two-choices with memory) running entirely in hardware, per-packet.
- **Egress** writes the real dequeue depth back into `q_depth_reg`, keeping DRILL's register current.

### In-Band Network Telemetry (INT)
Packets originating at port 1 (host-facing) get a custom header (`etherType = 0x9999`) with `path_queue_depth` (accumulated max along the path) and `src_id` (low 8 bits of source IP). At the destination leaf (arriving at port 1) the accumulated depth is written into `path_max_queue_depth_reg[src_id * 16 + port]` and the INT header is stripped. The telemetry collector polls this register at 10 Hz and resets it after each read.

---

## Control Plane

### `all_controller.py` — W-ECMP Rule Installer

**`TopologyAnalyzer`** reads `p4app.json` + `topology.json`, builds a NetworkX graph, annotates each edge with capacity-factor labels, then groups all shortest paths between each leaf pair into **components** by their edge-label signature. For l1→l2, this yields:
- Component 1 (via s1, s2): bottleneck sum = 2.0 Mbps → weight 2
- Component 2 (via s3, s4): bottleneck sum = 3.0 Mbps → weight 3

**`LeafController`** buffers all `simple_switch_CLI` commands and fires one subprocess per switch (not one per command), then writes `port_map_reg` entries directly via Thrift RPC.

### `rate_limiter.py`
Sets hardware `queue_rate` and `queue_depth` (64 packets) per port on every switch. Applies 0.8× to configured bandwidth, translating the topology's logical Mbps into the actual soft caps the traffic sees.

---

## Data Collection (`dataset_builder.py`)

Each of 1500 iterations:
1. Randomly assign weights [1–8] per component → apply to l1 via Thrift.
2. Run in parallel:
   - **Telemetry** (`telemetry_collector.py`): poll `path_max_queue_depth_reg` + `port_bytes_counter` at 10 Hz for 10 s → 100 rows, reset register after each read. Throughput uses EMA (α=0.3).
   - **QoS labels** (`iperf_parser.py`): 12 parallel UDP flows (random per-flow BW: 0.1/0.2/0.3/0.4 Mbps, 1400-byte packets) + 100 concurrent pings. Extracts `avg_latency`, `p99_latency`, `avg_jitter`, `loss_rate` from iperf3 JSON.
3. Aggregate 100 telemetry rows → 1 feature row: queue depth → max; throughput → mean + std.
4. Append to `training_dataset_master.csv`. Sleep 3 s for queue cooldown.

**Dataset note**: The cleaned training set in `research_results/data/datasets/training_dataset_ecdf_cleaned.csv` has **~4887 rows** (from multiple collection runs merged and cleaned).

---

## Feature Engineering Pipeline

```
training_dataset_master.csv
    │
    ├── rebuild_dataset.py           → training_dataset_v2.csv
    │    trim to valid rows; add per-port CV, load_util, imbalance columns
    │
    ├── extract_temporal_features.py → training_dataset_temporal.csv
    │    re-read each raw_telemetry/experiment_N.csv (100 rows @ 10 Hz):
    │    add per-port qdepth_p99, qdepth_slope, qdepth_cv, mbps_slope
    │
    ├── build_ecdf_features.py       → training_dataset_ecdf.csv
    │    rank-based ECDF transform of all features → [0,1]
    │    compute 3 composite indices:
    │      idx_congestion    = (α·max_q_ecdf × α·max_drop_ecdf)²
    │      idx_instability   = (α·max_cv_ecdf × α·neg_slope_ecdf)²
    │      idx_load_balance  = product of util ECDFs × imbalance ECDF
    │
    └── [manual cleaning]            → training_dataset_ecdf_cleaned.csv
         (research_results/data/datasets/ — the active training dataset)
```

`train_simplified_models.py` adds two more features at training time:
- `qdepth_sq` = `total_qdepth_p99²` (quadratic amplification of severe congestion)
- `qdepth_slope` = row-to-row difference of `total_qdepth_p99` (congestion trend)

---

## Stage 1: QoS Prediction Models

### Training: `train_simplified_models.py`

**Input**: `research_results/data/datasets/training_dataset_ecdf_cleaned.csv` (~4887 rows, 134 columns)

**Feature set** (32 selected features):

| Category | Features |
|---|---|
| Per-port queue depth | `src1_port{2-5}_qdepth_max` |
| Per-port throughput stability | `src1_port{2-5}_mbps_cv`, `src1_port{2-5}_load_util` |
| Per-port normalized load | `Norm_Load_P{2-5}` = mean_mbps / weight |
| Cross-port aggregates | `Total_Util_Sum`, `Max_Util_Diff`, `Group_Imbalance` |
| ECDF / indices | `idx_load_balance`, `mbps_imbalance` |
| Temporal/statistical | `max_qdepth_p99`, `total_qdepth_p99`, `total_qdepth_max`, `qdepth_max_imbalance` |
| Frequency domain | `qdepth_fft_max_all` (max FFT magnitude across ports — detects oscillation) |
| Control action | `Weight_Port{2-5}` (what weights were set when data was collected) |
| Engineered | `qdepth_sq`, `qdepth_slope` |

**RF configuration**: 500 trees, max_depth=20, min_samples_leaf=1, max_features=0.8, log1p label transform.

**Targets** (3 regressors + 1 classifier):
- `Label_Latency_ms` → `rf_model_latency_simplified.pkl`
- `Label_Jitter_ms` → `rf_model_jitter_simplified.pkl`
- `Label_Loss_Rate` → `rf_model_loss_simplified.pkl`
- Anomaly (loss > 0.001) → `rf_model_anomaly_simplified.pkl`

**Evaluation** (5-fold CV on log-space, reported in original units):

| Target | R² | MAE |
|---|---|---|
| Latency | 0.91 | ~369 ms |
| Latency p99 | 0.92 | ~745 ms |
| Jitter | 0.85 | ~15 ms |
| Loss rate | 0.85 | ~4.9% |
| Anomaly accuracy | — | 95.4% |

---

## Stage 1: Real-Time Inference (`realtime_ml_controller.py`)

`MLController` runs a live prediction loop using the trained models:

### Background threads (always running)
| Thread | What it does |
|---|---|
| `bg_iperf_client` | Single 0.1 Mbps UDP probe flow h1→h2, 3600 s |
| `bg_log_tail` | Tails the iperf3 server log file, parses per-second jitter and loss |
| `bg_ping` | Pings h2 every 0.5 s, appends RTT to `lat_buffer` |

### Per-second control loop
1. **`collect_window(1.0 s)`**: polls `path_max_queue_depth_reg` and `port_bytes_counter` at 10 Hz for 1 second (10 samples) → appended to a 100-sample sliding window (deque). The window maintains 10 s of history, matching the training data scale.
2. **`extract_features(df)`**: computes the 32-feature vector from the sliding window, including ECDF lookups from `ecdf_objects.pkl` for the `idx_load_balance` index.
3. **Run all 5 models** → apply exponential smoothing (40% new, 60% old) to predicted latency/jitter/loss.
4. **Print dashboard** showing predicted vs real values side by side:
   ```
   [14:32:01] NORMAL  | Lat:  23.1/ 21.0ms | Jit:  9.2/ 8.7ms | Loss:  0.1/ 0.0% | Util: 0.38
   ```

The controller currently **only observes and predicts** — it does not yet adjust weights.

---

## Stage 2: Weight Optimization (Planned)

The next step is to close the control loop. The Stage 1 predictions give a 1-second lookahead on QoS. A second model or policy will:
1. Take the current predictions (predicted latency, jitter, loss) as input.
2. Output an adjustment to the W-ECMP component weights.
3. Apply the new weights via the Thrift API.

This turns the system into a proper feedback controller: predict → act → observe → repeat.

---

## Supporting Tools

| File | Purpose |
|---|---|
| `analyze_feature_importance.py` | Trains a lightweight 100-tree model and plots feature importances; output in `research_results/plots/analysis/` |
| `traffic_gen.py` | Simple Python UDP traffic generator (raw sockets, alternative to iperf3) |
| `plot_all_metrics.py` | Validation comparison plots (predicted vs real) |
| `plot_latency_validation.py` | Latency-specific validation line charts |
| `rate_limiter.py` | Hardware queue rate/depth enforcement via Thrift |
| `network.py` | Python NetworkAPI alternative to `p4app.json` (partial — 2-leaf only) |

---

## Key Files Reference

| File | Role |
|---|---|
| `p4src/ecmp.p4` | P4 dataplane: DRILL + W-ECMP forwarding logic |
| `p4app.json` | Canonical topology (authoritative for bandwidths + switch config) |
| `start_env.sh` | One-command environment launcher |
| `all_controller.py` | Topology analysis + batched W-ECMP rule installation |
| `rate_limiter.py` | Applies soft bandwidth caps on all switch ports |
| `dataset_builder.py` | Automated experiment loop: random weights → measure → record |
| `telemetry_collector.py` | 10 Hz P4 register poller (queue depth + EMA throughput) |
| `iperf_parser.py` | iperf3 UDP + concurrent ping → latency/jitter/loss labels |
| `rebuild_dataset.py` | Trim master CSV to valid rows + add derived features |
| `extract_temporal_features.py` | p99, slope, CV features from raw 10 Hz time-series |
| `build_ecdf_features.py` | ECDF transform + 3 composite congestion indices |
| `label_generator.py` | Rule-based 4-class congestion labeling |
| `train_simplified_models.py` | **Active training script** — 32-feature RF on ~4887 rows |
| `realtime_ml_controller.py` | **Active inference** — sliding-window prediction loop |
| `analyze_feature_importance.py` | Feature importance analysis and plotting |
| `research_results/data/datasets/training_dataset_ecdf_cleaned.csv` | Active training dataset (~4887 rows, 134 columns) |
| `research_results/data/validation/optimized_report.txt` | Model performance summary |
| `rf_model_latency_simplified.pkl` | Trained latency predictor |
| `rf_model_jitter_simplified.pkl` | Trained jitter predictor |
| `rf_model_loss_simplified.pkl` | Trained loss rate predictor |
| `rf_model_anomaly_simplified.pkl` | Trained congestion anomaly classifier |
| `ecdf_objects.pkl` | Fitted ECDF lookup objects for inference |

---

## Development History

The `bruh/` directory holds early prototypes, ignore it:
- **`grouping.py`**: proved the component-signature grouping concept that became `TopologyAnalyzer`.
- **`l1_controller.py`**: hardcoded 2-component controller that preceded `LeafController`.

The `research_results/tools_and_archive/` directory holds older scripts used during model development (hyperparameter search, early regression attempts) that have been superseded by `train_simplified_models.py`.
