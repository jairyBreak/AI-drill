# AI-Drill: ML-Augmented W-ECMP Load Balancer

## Project Goal

Build a **closed-loop, adaptive load balancer** for a simulated P4 data center network, in two stages:

**Stage 1 — Prediction (complete):** Train Random Forest models to predict next-second QoS metrics (latency, packet loss) from real-time P4 switch telemetry collected via In-Band Network Telemetry (INT) and hardware counters.

**Stage 2 — Control (implemented):** At runtime the controller reads P4 hardware registers every second, transforms telemetry into topology-independent features, predicts congestion, and rewrites ECMP weights on the ingress leaf switch. This closes the loop: telemetry → predict QoS → adjust weights → observe → repeat.

The underlying network uses two complementary routing mechanisms:
- **W-ECMP**: probabilistic traffic splitting across spine groups, with weights set by the control plane.
- **DRILL** (Distributed Randomized In-network Load balancing): inside each group, the P4 dataplane itself picks the least-queued port every packet via power-of-two-choices with memory. Runs fully in hardware, no control plane involved.

---

## Repository Layout

```
AI-drill/
├── jsq_2_2/              # 2-switch ECMP reference (educational baseline)
└── main/                 # Active project: 8-leaf/8-spine asymmetric topology
    ├── p4src/            # P4 program (dataplane)
    ├── research_results/ # Curated datasets, validation runs, plots
    │   ├── data/datasets/      # Processed feature datasets (master → temporal → ECDF → cleaned)
    │   ├── data/validation/    # Per-run validation CSVs
    │   └── plots/             # Predicted vs real comparison charts
    └── bruh/             # Early algorithm prototypes (not production)
```

---

## Network Topology

```
  h1 ── l1 ──┬── s1 ──┬── l2 ── h2
             ├── s2 ──┤
             ├── s3 ──┤
             ├── s4 ──┤
             ├── s5 ──┤
             ├── s6 ──┤
             ├── s7 ──┤
             └── s8 ──┘
  (l3–l8 connect identically to all 8 spines)
```

- **8 hosts**: h1–h8 at `10.0.{i}.{i}/24`
- **8 leaf switches**: l1–l8 (Thrift ports 9090–9097)
- **8 spine switches**: s1–s8 (Thrift ports 9098–9105)

**Asymmetric uplinks** (the key design choice — creates 4 distinct W-ECMP components):

| Port at lN | Spine | Link BW | Effective cap (rate_limiter ×0.8) |
|---|---|---|---|
| 2 | s1 | 0.6 Mbps | **0.48 Mbps** |
| 3 | s2 | 0.6 Mbps | **0.48 Mbps** |
| 4 | s3 | 0.8 Mbps | **0.64 Mbps** |
| 5 | s4 | 0.8 Mbps | **0.64 Mbps** |
| 6 | s5 | 1.0 Mbps | **0.80 Mbps** |
| 7 | s6 | 1.0 Mbps | **0.80 Mbps** |
| 8 | s7 | 1.2 Mbps | **0.96 Mbps** |
| 9 | s8 | 1.2 Mbps | **0.96 Mbps** |

Same-BW spine pairs form the 4 ECMP components: s1/s2 (0.48M), s3/s4 (0.64M), s5/s6 (0.80M), s7/s8 (0.96M). l3–l8 are symmetric to all spines.

---

## Starting the Environment

After activating p4dev's environment:
```bash
cd main && ./start_env.sh
```
Cleans Mininet state, compiles and starts p4run (16 BMv2 switches, foreground), polls all 16 Thrift ports (9090–9105) until ready, then launches `all_controller.py` (programs ECMP rules) and `rate_limiter.py` (sets queue caps) in a background subshell. When it prints "背景設施全部就緒", open a second terminal.

### Or manually
```bash
sudo p4run
python3 all_controller.py      # install W-ECMP routing rules
sudo python3 rate_limiter.py   # apply port rate/queue limits
```

---

## P4 Dataplane (`p4src/ecmp.p4`)

### State (Registers & Counters)

| Name | Size | Purpose |
|---|---|---|
| `q_depth_reg` | 512 | Live queue depth per port — written by every departing packet |
| `last_best_p_reg` | 512 | DRILL memory: last winning port per ECMP group |
| `port_map_reg` | 1024 | `(comp_id × 16 + logical_idx)` → physical port |
| `path_max_queue_depth_reg` | 1024 | Max queue depth per (src_id, port) since last reset — read and zeroed by controller |
| `path_max_q_delay_reg` | 1024 | Max queue delay in µs per path since last reset |
| `path_acc_q_delay_reg` | 1024 | Accumulated queue delay in µs per path — hardware ground-truth latency |
| `port_bytes_counter` | 256 | Cumulative bytes per port — delta = throughput |
| `cnt_enq / cnt_ingress` | 512 | Per-port enqueue (l1) / ingress (l2) packet counters — difference = drops |

### Forwarding Pipeline

```
ipv4_lpm
 └─ set_w_ecmp → w_ecmp_table     (CRC16 action_selector on 5-tuple → component_id)
                  └─ drill_params_table  (component_id → run_drill, num_nhops)
                       └─ ecmp_group_to_nhop  (component_id + best_port → MAC/egress)
```

- **`w_ecmp_table`** uses an `action_selector` where group member count = weight. A flow's 5-tuple is hashed to a component proportionally to its weight.
- **`run_drill`**: picks two distinct random ports in the component, reads their `q_depth_reg` values plus the last-best port's depth, selects the minimum-queue port, writes back the winner to `q_depth_reg` and `last_best_p_reg`. Runs fully in hardware per-packet.
- **Egress** writes the real dequeue depth back into `q_depth_reg`.

### In-Band Network Telemetry (INT)
Packets from port 1 (host-facing) get a custom header (`etherType=0x9999`) carrying accumulated `path_queue_depth`, `max_q_delay`, `acc_q_delay`, and `src_id` (low 8 bits of source IP). At the destination leaf (arriving at port 1) the accumulated values are written into the respective registers at index `src_id * 16 + ingress_port`, and the INT header is stripped. The controller reads and zeros these registers every second.

---

## Control Plane

### `all_controller.py` — W-ECMP Rule Installer

**`TopologyAnalyzer`** reads `p4app.json` + `topology.json`, builds a NetworkX graph, annotates each edge with capacity-factor labels, then groups all shortest paths between each leaf pair into **components** by their bottleneck-BW signature. For l1→l2 this yields 4 components (s1/s2, s3/s4, s5/s6, s7/s8).

**`LeafController`** buffers all `simple_switch_CLI` commands and fires one subprocess per switch (not one per command), then writes `port_map_reg` entries directly via Thrift RPC.

### `rate_limiter.py`
Sets hardware `queue_rate` (PPS) and `queue_depth` (64 packets) per port on all 16 switches via Thrift. Applies 0.8× to the configured link BW from `p4app.json`.

---

## Data Collection (`dataset_builder.py` and `rolling_dataset_builder.py`)

### 10-second discrete pipeline (original, `dataset_builder.py`)

Each of 1500 iterations:
1. Randomly assign 2-element weight list → apply to l1 via Thrift
2. Run in parallel:
   - **Telemetry** (`telemetry_collector.py`): poll `path_max_queue_depth_reg` + `port_bytes_counter` at 10 Hz for 10 s → 100 rows. Reads **l2 ports 2–5 only** (s1–s4; s5–s8 not monitored).
   - **QoS labels** (`iperf_parser.py`): 12 parallel UDP flows + 100 concurrent pings → `avg_latency`, `avg_jitter`, `loss_rate`
3. Aggregate 100 telemetry rows → 1 feature row → append to `training_dataset_master.csv`

### 1-second rolling pipeline (current, `rolling_dataset_builder.py`)

300 experiments × 120s each. Reads **all 8 spine ports** (2–9). Every second:
- Reads `path_max_queue_depth_reg`, `path_acc_q_delay_reg` from l2; resets after read
- Reads `port_bytes_counter` delta → Mbps
- Reads `cnt_enq` (l1) and `cnt_ingress` (l2) delta → per-port and total drop rate
- Tracks `Is_Rehash_Event` and `Time_Since_Last_Rehash_s`
- Labels: `Label_Max_Path_Delay_ms` (INT accumulated delay), `Label_Total_Drop_Rate_Percent`
- Weights change every 30s within each experiment (covers diverse load scenarios)

Output: `research_results/data/datasets/rolling_training_dataset.csv` (~63k rows)

---

## Feature Engineering Pipeline

### 10-second pipeline
```
training_dataset_master.csv
    ├── rebuild_dataset.py              → training_dataset_v2.csv
    │    trim to valid rows; add per-port CV, load_util, imbalance columns
    ├── extract_temporal_features.py   → training_dataset_temporal.csv
    │    re-read each raw_telemetry/experiment_N.csv (100 rows @ 10 Hz):
    │    add per-port qdepth_p99, qdepth_slope, qdepth_cv, mbps_slope, FFT max
    ├── build_ecdf_features.py         → training_dataset_ecdf.csv
    │    rank-based ECDF transform of all features → [0,1]
    │    compute 3 composite indices:
    │      idx_congestion    = (α·max_q_ecdf × α·max_drop_ecdf)²
    │      idx_instability   = (α·max_cv_ecdf × α·neg_slope_ecdf)²
    │      idx_load_balance  = product of util ECDFs × imbalance ECDF
    └── [manual cleaning]              → training_dataset_ecdf_cleaned.csv
         (research_results/data/datasets/ — active 10s training set, ~4887 rows, 134 cols)
```

### 1-second topo-independent transform (`topo_independent_helper.py`)

Converts raw per-port measurements into a **fixed 39-feature vector** regardless of topology size or number of ports. Used by `realtime_ml_controller.py` at inference time.

- **18 global aggregate features**: `Total_Util_Sum`, `Max_Util_Diff`, `Group_Imbalance`, `Max_QDepth`, `Total_QDepth`, `QDepth_Imbalance`, `Max_Q_Ratio`, `Q_Danger_Flag`, `Q_Danger_Count`, `Over_Capacity_Sum`, `Expected_Over_Capacity_Sum`, `Overflow_Intensity`, `Queue_Full_And_Over_Cap`, `Total_Actual_Mbps`, `Total_QDepth_Trend`, `Is_Rehash_Event`, `Time_Since_Last_Rehash_s`, `Rehash_Impact`
- **7 features × Top-3 most-congested ports**: `qdepth`, `mbps`, `weight`, `norm_load`, `expected_util`, `qdepth_trend`, `mbps_trend`

---

## Stage 1: QoS Prediction Models

### Training: `train_simplified_models.py` (10-second models)

**Input**: `research_results/data/datasets/training_dataset_ecdf_cleaned.csv` (~4887 rows, 134 columns)

**Feature set** (32 selected features): per-port queue depth, throughput stability, normalized load, cross-port aggregates, ECDF indices, temporal statistics, FFT max, control weights, plus two engineered features (`qdepth_sq`, `qdepth_slope`).

**RF configuration**: 500 trees, max_depth=20, min_samples_leaf=1, max_features=0.8, log1p label transform.

**Targets** (3 regressors + 1 classifier):
- `Label_Latency_ms` → `rf_model_latency_simplified.pkl`
- `Label_Jitter_ms` → `rf_model_jitter_simplified.pkl` *(trained but **excluded** from controller — inaccurate)*
- `Label_Loss_Rate` → `rf_model_loss_simplified.pkl`
- Anomaly (loss > 0.001) → `rf_model_anomaly_simplified.pkl`

**Evaluation** (5-fold CV):

| Target | R² | MAE |
|---|---|---|
| Latency | 0.91 | ~369 ms |
| Latency p99 | 0.92 | ~745 ms |
| Loss rate | 0.85 | ~4.9% |
| Anomaly accuracy | — | 95.4% |

### Training: `train_1s_models.py` (1-second models)

**Input**: `rolling_training_dataset.csv`

**RF configuration**: 100 trees, max_depth=15, min_samples_leaf=4, max_features='sqrt'

| File | Target | Status |
|---|---|---|
| `rf_model_latency_1s.pkl` | `Label_Max_Path_Delay_ms`, log1p | **Not yet trained** |
| `rf_model_loss_1s.pkl` | `Label_Total_Drop_Rate_Percent` | **Not yet trained** |
| `rf_model_anomaly_1s.pkl` | Anomaly classifier (loss > 0.1%) | **Not yet trained** |

Run: `python3 train_1s_models.py`

**⚠ Known bug**: `CAPACITY` dict in `train_1s_models.py` uses `{2:0.8, 3:0.8, 4:0.8, 5:0.8, 6:1.2, ...}` instead of actual rate-limited values `{2:0.48, 3:0.48, 4:0.64, 5:0.64, 6:0.80, 7:0.80, 8:0.96, 9:0.96}`. Fix before training.

---

## Stage 1 + 2: Real-Time Controller (`realtime_ml_controller.py`)

`MLController` runs passively (injects no traffic). It controls **only the l1→l2 W-ECMP class
weights**; DRILL inside each class and the rest of the fabric are untouched. The design target is
**elephant/mice** traffic: the controller does not know which flow is an elephant — an elephant simply
reveals itself as sustained queue/util on whichever class it hashes to, and the controller responds by
**shedding that class's weight** so fewer (mostly mouse) flows hash into it. DRILL inside the class
already steers packets to the non-elephant port.

### Per-second loop
1. **`collect_1s_data()`** — reads l2 hardware registers for all 8 spine ports:
   - `path_max_queue_depth_reg`, `path_acc_q_delay_reg` (reset after read)
   - `port_bytes_counter` delta → Mbps
   - l1 `cnt_enq` vs l2 `cnt_ingress` delta → hardware drop rate
   - Reads current ECMP weights from l1's action profile; flags rehash events
2. **Feature transform** — appends row to a 100-entry rolling deque, calls `transform_to_topo_independent(K=3)`, takes the latest row as a 39-feature vector
3. **Prediction** — runs all loaded RF models; applies exponential smoothing to latency/loss. The control path also works **without** the models via hardware-evidence triggers (below).
4. **`control_step()`** — decide whether/how to rebalance (see Control logic).
5. **`compute_weights()`** — anchored bounded correction (see below).
6. **`apply_weights()`** — build a fresh action-profile group, **atomically repoint** the `w_ecmp_table` entry to it (hitless — no forwarding gap), then free the old group.

### Control logic (anchor-and-correct)

The earlier controller oscillated: every weight change rebuilds the selector group and — because BMv2's
`action_selector` is **not** consistent-hashing — reshuffles *all* flows; measuring during that
transient caused over-correction. Key fix: rehash cost is magnitude-independent, so stability comes
from changing **rarely**, not by small steps.

**Tuning constants** (top of file):

| Const | Value | Role |
|---|---|---|
| `RATE_LIMIT_SCALE` | 0.8 | effective capacity = link bw × this (matches `rate_limiter.py`) |
| `IMBALANCE_TOL` | 0.15 | no-action band, sized *between* a mouse (~0.13) and elephant (~0.21+) util bump |
| `WEIGHT_BOUND` | 2 | max deviation of any weight from its anchor |
| `CORRECTION_GAIN` | 0.5 | proportional gain of the shed/boost |
| `Q_WEIGHT` | 0.5 | how much queue pressure adds to a class's hotness |
| `QDEPTH_HOT` | 32 | queue depth (of the 64-deep cap) marking a class congested |
| `UTIL_SAT` | 0.90 | absolute "class near its own limit" threshold (effective util) |
| `COOLDOWN_SEC` | 6 | min spacing between rehashes |
| `SETTLE_SEC` | 2 | post-rehash measurement blackout |
| `PERSIST_TICKS` / `RELAX_TICKS` | 2 / 6 | ticks an imbalance / cold-class must persist before acting |

**Signals** (`_component_stats` → `_class_pressure`):
- `util_i = Σ raw mbps / Σ (link bw × RATE_LIMIT_SCALE)` — uses **effective** capacity from the topology graph, so it is independent of the `CAPACITY` constant.
- `qfrac_i = max port queue depth in class / QDEPTH_HOT`.
- `pressure_i = util_i + Q_WEIGHT · max(0, qfrac_i − 1)` — drives *both* hot-class selection and the correction, so a queue-building elephant is caught before util saturates.

**Anchor + correction** (`compute_weights`): anchor = capacity-proportional weights `[3,4,5,6]` taken
from `get_ecmp_weights_and_rules` (the same source the dataplane was installed with). For each class:
`desired = base · (1 − CORRECTION_GAIN · (pressure_i − mean) / mean)`, clamped to `base ± WEIGHT_BOUND`
then to `[1,8]` and rounded. Hot classes shed, cold classes boost; as imbalance → 0, `desired → base`,
so it relaxes toward the anchor by construction.

**Decision order** (`control_step`):
1. **Settle blackout** — if within `SETTLE_SEC` of the last rehash, return (don't measure transients).
2. **Shed (hot class)** — if the hottest class is genuinely hot *and* there's harm evidence:
   `hw_bad` (real hardware loss > threshold, any `qfrac ≥ 1`, or `mean util > UTIL_SAT`) fires
   immediately; the RF model can also fire it but only after `PERSIST_TICKS`. Respects `COOLDOWN_SEC`.
   If the computed weights equal the current ones (already at the bound), it logs *"cannot improve via
   weights"* instead of churning.
3. **Relax (evidence-based)** — if a class that was shed below its anchor goes **cold**
   (`util < mean − IMBALANCE_TOL`) for `RELAX_TICKS`, the elephant has left → jump back to the anchor in
   one rehash. On static traffic the elephant keeps its class warm, so this never fires → controller
   **freezes**; on dynamic traffic it returns to `[3,4,5,6]` once load drops.
4. **Balanced overload** — balanced but high real loss / util → log *"no weight solution"* (weights
   can't fix aggregate overload), no rehash.
5. Otherwise freeze. Passive-state logs are deduped (`_state_log`) so they don't spam the monitor line.

**Robustness:** `_sync_from_dataplane()` at startup reads the live group/member handles and per-class
weights instead of assuming the install succeeded (and so the first weight change frees the install's
group cleanly). `_safe_apply` rebuilds the Thrift connection and reverts on RPC failure.

### Display
Overwrites one terminal line per second:
```
[16:07:25] NORMAL  | Lat:  20.0/  9.1ms | Loss:  0.0/ 0.0% | Util: 0.10
```
Format: `predicted / hardware_ground_truth`. Weight changes (`shed/boost`, `relax → anchor`) print on a new line.

---

## Traffic Generator (`traffic.py`)

18 UDP flows h1→h2, ports 5100–5117. BWs from `{0.06, 0.08, 0.16, 0.24, 0.40}` Mbps (total = 3.12 Mbps). Modes:
- `--default`: 1 flow at 0.3 Mbps
- `--static`: all 18 flows with a shuffled fixed BW assignment
- `--dynamic`: flows reshuffle to a new random BW assignment every ~10s
- `--elmice`: **elephant/mice** — 2 persistent elephants (0.46M, rotate every 15s) + 16 mice (0.14M, rotate every 5s)

Live spine monitor overlay (in-place overwrite, 8 lines): reads `q_depth_reg` from l1 and `port_bytes_counter` from l2.

---

## Evaluation (`test.sh` + `plot_result.py`)

`baseline_ecmp.py`, `baseline_wecmp.py`, and `baseline_drill.py` run the **same** `p4src/ecmp.p4` dataplane with only the control-plane config changed, so all four algorithms are directly comparable. `test.sh` sweeps them for a given duration; `plot_result.py` overlays the per-second CSVs and prints a summary (first 5s dropped as warmup). Latency is the per-second **peak** queue delay; Util σ is the standard deviation of mean per-spine utilization (lower = better balanced).

All four are the same pipeline with a different W-ECMP/DRILL component layout (`baseline_common.py`):

| Algorithm | Components | Weights | DRILL |
|---|---|---|---|
| ECMP | 8 single-port | all equal | off (`num_nhops=1`) |
| W-ECMP | 8 single-port | ∝ bandwidth (3/3/4/4/5/5/6/6) | off |
| DRILL | 1 component, all 8 ports | trivial | on across all 8 (`num_nhops=8`) |
| W-ECMP+DRILL+ML | 4 capacity-pair components | ML-controlled `[3,4,5,6]±` | on within each pair (`num_nhops=2`) |

**Elephant/mice (`--elmice`), 60s, 5s warmup dropped:**

| Algorithm | Lat p50 | Lat p95 | Lat max | Loss E2E | Mbps | Util σ |
|---|--:|--:|--:|--:|--:|--:|
| ECMP | 101.35 | 251.54 | 564.74 | 0.00 | 2.65 | 0.160 |
| W-ECMP | 37.66 | 334.10 | 457.67 | 0.00 | 2.62 | 0.161 |
| DRILL | **22.57** | 170.42 | 340.47 | 0.00 | 2.86 | 0.148 |
| **W-ECMP+DRILL+ML** | 32.04 | **160.17** | **176.28** | 0.00 | 2.83 | **0.062** |

**Takeaway:** on elephant/mice traffic the full **W-ECMP+DRILL+ML** controller delivers the best tail latency — lowest p95 and a dramatically lower max (176ms vs 340–565ms) — and by far the most balanced spine utilization (σ 0.062, ~half of every other approach), while staying within a few ms of DRILL on median. Pure DRILL has the lowest median but is capacity-blind (worse tail and σ); static ECMP/W-ECMP are worst on the tail. (`traffic.py` currently reshuffles assignments per run, so seed it for a strictly fair head-to-head.)

---

## Known Gaps

1. **1s models missing** — `rf_model_*_1s.pkl` don't exist. The controller still operates via its **hardware-evidence triggers** (real loss / queue / util); RF predictions are an optional secondary path. To train: `rolling_dataset_builder.py` (data) → `train_1s_models.py`.
2. **`train_1s_models.py` / `realtime_1s_predictor.py` CAPACITY bug** — these use raw `{2:0.8, …}` instead of the effective rate-limited `{2:0.48, …}`, so models trained from them get miscalibrated utilization features. The **live controller is unaffected** — it derives effective capacity (link bw × 0.8) itself.
3. **10s models trained on s1–s4 only** — `telemetry_collector.py` reads ports 2–5. s5–s8 were invisible during collection. Simplified models have no knowledge of the higher-BW spine pair.
4. **10s dataset used 2 ECMP components** — `dataset_builder.py` applied 2-element weight lists. The network has 4 components. Old models never saw s5–s8 weighted.
5. ~~No evaluation baseline~~ **(resolved)** — `baseline_{ecmp,wecmp,drill}.py` + `test.sh` + `plot_result.py` compare all four algorithms on the same dataplane. See **Evaluation** above.
6. **Comparison runs not seeded** — `traffic.py` reshuffles flow→port assignments per invocation, so the four algorithms don't see byte-identical traffic. Seed it before publishing head-to-head numbers.

---

## Supporting Tools

| File | Purpose |
|---|---|
| `analyze_feature_importance.py` | Trains a lightweight 100-tree model and plots feature importances |
| `plot_all_metrics.py` | Validation comparison plots (predicted vs real) |
| `plot_1s_metrics.py` | Validation plots for 1s topo-independent predictions |
| `traffic_gen.py` | Raw UDP socket traffic generator (alternative to iperf3) |
| `network.py` | Python NetworkAPI network definition (alternative to p4app.json) |
| `realtime_1s_predictor.py` | Passive 1s monitor using port-specific features |
| `realtime_1s_predictor_topo_indep.py` | Passive 1s monitor using topo-independent features |

---

## Key Files Reference

| File | Role |
|---|---|
| `p4src/ecmp.p4` | P4 dataplane: DRILL + W-ECMP forwarding logic |
| `p4app.json` | Canonical topology (authoritative for bandwidths + switch config) |
| `start_env.sh` | One-command environment launcher |
| `all_controller.py` | Topology analysis + batched W-ECMP rule installation |
| `rate_limiter.py` | Applies soft bandwidth caps on all 16 switch ports |
| `dataset_builder.py` | Old 10s experiment loop: random 2-component weights → measure → record |
| `rolling_dataset_builder.py` | Current 1s data collection: all 8 spines, rehash tracking, INT labels |
| `telemetry_collector.py` | Old 10 Hz P4 register poller (ports 2–5 only) |
| `topo_independent_helper.py` | Converts raw 8-port measurements to 39 topology-agnostic features |
| `train_simplified_models.py` | **Active 10s training script** — 32-feature RF on ECDF-cleaned dataset |
| `train_1s_models.py` | **Active 1s training script** — 39-feature RF on rolling dataset |
| `realtime_ml_controller.py` | **Active live controller** — 1s anchor-and-correct loop (W-ECMP+DRILL+ML) |
| `baseline_common.py` | Shared baseline install/measure logic (per-algorithm component layouts) |
| `baseline_ecmp.py` / `baseline_wecmp.py` / `baseline_drill.py` | Static baselines on the same dataplane |
| `test.sh` | Sweeps ML / DRILL / ECMP / W-ECMP for a given duration |
| `plot_result.py` | Overlays the four comparison CSVs + prints the summary table |
| `traffic.py` | iperf3 traffic generator (`--static/--dynamic/--elmice`) with live spine monitor |
| `iperf_parser.py` | iperf3 UDP + concurrent ping → latency/jitter/loss labels |
| `build_ecdf_features.py` | ECDF rank-transform + 3 composite congestion indices |
| `research_results/data/datasets/training_dataset_ecdf_cleaned.csv` | Active 10s training dataset (~4887 rows, 134 cols) |
| `research_results/data/datasets/rolling_training_dataset.csv` | Active 1s training dataset (~63k rows) |
| `rf_model_latency_simplified.pkl` | Trained 10s latency predictor |
| `rf_model_loss_simplified.pkl` | Trained 10s loss rate predictor |
| `rf_model_anomaly_simplified.pkl` | Trained 10s congestion anomaly classifier |
