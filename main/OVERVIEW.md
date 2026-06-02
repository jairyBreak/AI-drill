# AI-Drill: Technical Overview

## What this project is

An adaptive load balancer for a simulated P4 data center network. Random Forest models are trained on telemetry collected from the hardware registers of P4 switches. At runtime the controller reads those same registers every second, applies a topology-independent feature transform, predicts congestion, and rewrites ECMP weights on the ingress leaf switch to keep traffic balanced across all spine paths.

---

## Network topology

8 hosts, 8 leaf switches, 8 spine switches. Full mesh between every leaf and every spine. All traffic in experiments runs **h1 → l1 → spines → l2 → h2**.

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

**Hosts**: h1–h8 at `10.0.{i}.{i}/24`  
**Leaf Thrift ports**: l1–l8 → 9090–9097  
**Spine Thrift ports**: s1–s8 → 9098–9105

### Link bandwidths (asymmetric by design)

Rate-limiter applies 0.8× scale to link BW:

| Spines | Link BW | Effective cap | Leaf port |
|--------|---------|---------------|-----------|
| s1, s2 | 0.6 Mbps | **0.48 Mbps** | 2, 3 |
| s3, s4 | 0.8 Mbps | **0.64 Mbps** | 4, 5 |
| s5, s6 | 1.0 Mbps | **0.80 Mbps** | 6, 7 |
| s7, s8 | 1.2 Mbps | **0.96 Mbps** | 8, 9 |

The asymmetry creates 4 distinct **ECMP components** (spine pairs with identical BW), making weight optimization non-trivial.

---

## P4 dataplane (`p4src/ecmp.p4`)

Two routing mechanisms run simultaneously:

- **W-ECMP**: a 5-tuple hash selects a component (spine pair) probabilistically proportional to its action profile member count. Member count = weight — the control plane sets this.
- **DRILL**: within a component, the switch picks two random ports, reads their live queue depths from `q_depth_reg`, selects the least-queued one, and updates `last_best_p_reg`. Runs fully in hardware per-packet.

### Forwarding pipeline

```
ipv4_lpm → set_w_ecmp
           w_ecmp_table (action_selector, 5-tuple hash → comp_id)
           drill_params_table (comp_id → run_drill, num_nhops)
           ecmp_group_to_nhop (comp_id + best_port → egress MAC/port)
```

### Registers

| Register | Size | Purpose |
|----------|------|---------|
| `q_depth_reg` | 512 | Live queue depth per port — updated by every egress packet |
| `last_best_p_reg` | 512 | DRILL memory: last winning port per group |
| `port_map_reg` | 1024 | `comp_id × 16 + logical_idx` → physical egress port |
| `path_max_queue_depth_reg` | 1024 | Peak queue depth per `(src_id × 16 + port)` since last reset |
| `path_acc_q_delay_reg` | 1024 | Accumulated queue delay in µs per path — hardware ground-truth latency |

### Counters

| Counter | Purpose |
|---------|---------|
| `port_bytes_counter` | Cumulative bytes per port — delta gives throughput |
| `cnt_enq` | Enqueue count per port (l1 side) |
| `cnt_ingress` | Ingress count per port (l2 side) — `cnt_enq - cnt_ingress` = drops |

### In-band telemetry (INT)
Packets from h1 (port 1) receive a custom `etherType=0x9999` header carrying accumulated `path_queue_depth` and `src_id`. At l2 the accumulated depth is written into `path_max_queue_depth_reg[src_id × 16 + port]` and the INT header is stripped.

---

## Environment startup (`start_env.sh`)

```
start_env.sh
 ├── mn -c                      # clean Mininet state
 ├── p4run                      # compile + launch 16 BMv2 switches (foreground)
 └── [background subshell]
      ├── nc poll all 16 Thrift ports (9090–9105)  # wait for switches to be ready
      ├── python3 all_controller.py                 # program ECMP rules on all leaves
      └── python3 rate_limiter.py                   # set queue rates/depths
```

---

## Control plane

### `all_controller.py`
Uses `TopologyAnalyzer` to build a NetworkX graph from `p4app.json`, compute all shortest paths between each leaf pair, and group paths into components by bottleneck BW signature. For l1→l2 this yields 4 components (s1/s2, s3/s4, s5/s6, s7/s8).

`LeafController` buffers all CLI commands and fires one subprocess per switch (not per command), then writes `port_map_reg` entries via Thrift.

### `rate_limiter.py`
Sets `set_queue_rate` and `set_queue_depth` (64 packets) on every switch-to-switch port via Thrift. Applies 0.8× to the configured link BW.

---

## Data collection

### 10-second discrete pipeline (original)

**`dataset_builder.py`** — 1500 iterations:
1. Randomly pick 2-element weight list → apply to l1's ECMP group
2. Run `telemetry_collector.py` in parallel (10 Hz polling, 10 s)
3. Run `iperf_parser.py` (12 concurrent UDP flows + 100 pings → latency/jitter/loss labels)
4. Aggregate 100 telemetry rows → 1 feature row → append to master CSV

Telemetry reads **l2 ports 2–5 only** (s1–s4). s5–s8 are not monitored in this pipeline.  
Output: `research_results/data/datasets/training_dataset_master.csv` (~4887 rows)

### 1-second rolling pipeline (current)

**`rolling_dataset_builder.py`** — continuous 1s snapshots with weight changes:
- Reads all 8 spine ports (2–9) from both l1 and l2
- Tracks `Is_Rehash_Event` (when weights change) and `Time_Since_Last_Rehash_s`
- Records hardware ground-truth latency (`path_acc_q_delay_reg`) and loss (`cnt_enq - cnt_ingress`)

Output: `research_results/data/datasets/rolling_training_dataset.csv` (~63k rows)

---

## Feature engineering

### 10s pipeline
```
training_dataset_master.csv
  → extract_temporal_features.py   adds p99, slope, CV, FFT per port (from raw_telemetry/)
  → build_ecdf_features.py         ECDF rank-transform + 3 composite indices
  → training_dataset_ecdf_cleaned.csv   (active training set, 4887 rows, 134 cols)
```

### 1s topo-independent transform (`topo_independent_helper.py`)
Converts raw 8-port measurements into a **fixed 39-feature vector** regardless of topology size:
- 18 global aggregate features (total util, queue imbalance, overflow indicators, etc.)
- 7 features × top-3 most-congested ports (qdepth, mbps, weight, norm_load, expected_util, trends)

This transform is used by both `rolling_dataset_builder.py` during training and `realtime_ml_controller.py` at inference time.

---

## Models

### 10-second models (exist, trained on ports 2–5 only)

| File | Target | Notes |
|------|--------|-------|
| `rf_model_latency_simplified.pkl` | Latency (ms) | 800 trees, log1p label, 32 features |
| `rf_model_jitter_simplified.pkl` | Jitter (ms) | Same config — **not used by controller** |
| `rf_model_loss_simplified.pkl` | Loss rate | Same config |
| `rf_model_anomaly_simplified.pkl` | Anomaly flag | Classifier, loss > 0.001 threshold |
| `ecdf_objects.pkl` | ECDF transformers | Required for 10s inference only |

### 1-second models (missing — need training)

| File | Status |
|------|--------|
| `rf_model_latency_1s.pkl` | **Not trained yet** |
| `rf_model_loss_1s.pkl` | **Not trained yet** |
| `rf_model_anomaly_1s.pkl` | **Not trained yet** |

Train with: `python3 train_1s_models.py` (requires `rolling_training_dataset.csv`)

---

## Realtime ML controller (`realtime_ml_controller.py`)

The active controller. Runs fully passively — injects no traffic.

### Per-second loop
1. **`collect_1s_data()`** — reads from l2 hardware registers for all 8 spine ports:
   - Queue depth and accumulated delay per path
   - Byte counter delta → Mbps
   - `cnt_enq` (l1) vs `cnt_ingress` (l2) delta → hardware loss
   - Current ECMP weights from l1's action profile
2. **Feature transform** — appends row to rolling 100-entry history, runs `transform_to_topo_independent(K=3)`, takes the latest row as a 39-feature vector
3. **Prediction** — runs loaded models (warns and skips if missing); `latency` output is `expm1`-transformed
4. **`control_step()`** — decides whether to rebalance:
   - Skip if within 4s cooldown or `max_util < 0.1`
   - Trigger if: anomaly==1, max_util > 0.6, queue imbalance > 15, predicted latency > 200ms, predicted loss > 2%
5. **`compute_weights()`** — scores each ECMP component by `free_bw × queue_headroom`, normalises to integers in [1, 8]
6. **`apply_weights()`** — rewrites l1's action profile members in-place (same group handle, so table entry is never cleared and other l1 routes are undisturbed)

### Display
Overwrites a single terminal line every second:
```
[16:07:25] NORMAL  | Lat:  20.0/  9.1ms | Loss:  0.0/ 0.0% | Util: 0.10
```
Format: `predicted / hardware_ground_truth`. Weight changes print on a new line.

---

## Traffic generator (`traffic.py`)

Generates iperf3 UDP traffic h1→h2 across 18 ports (5100–5117).

| Mode | Description |
|------|-------------|
| `--default` | 1 flow at 0.3 Mbps (for baseline/testing) |
| `--static` | 18 flows, fixed random BW assignment |
| `--dynamic` | 18 flows, BW reshuffled every ~10s; live spine monitor overlay |

Flows use BWs from `{0.06, 0.08, 0.16, 0.24, 0.40}` Mbps. Total = 3.12 Mbps peak.

Live monitor shows per-spine queue depth and throughput with an in-place overwrite display.

---

## Other scripts

| Script | Purpose |
|--------|---------|
| `realtime_1s_predictor.py` | Passive 1s monitor using port-specific features (needs 1s models) |
| `realtime_1s_predictor_topo_indep.py` | Passive 1s monitor using topo-independent features (needs 1s models) |
| `train_simplified_models.py` | Trains the 10s simplified models from `training_dataset_ecdf_cleaned.csv` |
| `train_1s_models.py` | Trains 1s models from `rolling_training_dataset.csv` |
| `train_topo_indep_models.py` | Alternative trainer using topo-independent features |
| `plot_all_metrics.py` | Validation plots: predicted vs real for all metrics |
| `plot_1s_metrics.py` | Validation plots for 1s topo-independent predictions |
| `extract_temporal_features.py` | Adds p99/slope/CV/FFT features from raw 10Hz telemetry |
| `build_ecdf_features.py` | ECDF rank-transform + composite congestion indices |
| `label_generator.py` | Rule-based 6-class congestion labeler |
| `analyze_feature_importance.py` | RF feature importance plots |
| `iperf_parser.py` | Runs iperf3 UDP + pings, returns latency/jitter/loss |
| `traffic_gen.py` | Raw UDP socket traffic generator (alternative to iperf3) |
| `network.py` | Python NetworkAPI network definition (alternative to p4app.json) |

---

## Known gaps

1. **1s models don't exist** — `rf_model_*_1s.pkl` are missing. Controller falls back to reactive thresholds only. Run `rolling_dataset_builder.py` then `train_1s_models.py`.

2. **10s models trained on s1–s4 only** — `telemetry_collector.py` reads ports 2–5. s5–s8 were invisible during 10s dataset collection. These models have no knowledge of the higher-BW spines.

3. **10s dataset used 2 ECMP components** — `dataset_builder.py` applied 2-element weight lists. The network has 4 components. Old models never saw the s5–s8 group weighted.

4. **Capacity constants inconsistent** — `train_1s_models.py` and `realtime_1s_predictor.py` use wrong CAPACITY values `{2:0.8, 3:0.8, ...}` instead of actual rate-limited values `{2:0.48, 3:0.48, 4:0.64, 5:0.64, 6:0.80, 7:0.80, 8:0.96, 9:0.96}`.

5. **No evaluation baseline** — no script compares controller performance against equal-weight ECMP under identical traffic.
