# AI-Drill: ML-Augmented Adaptive Load Balancer

A closed-loop, P4-based data center load balancer. Random Forest models trained on P4 switch telemetry predict congestion one second ahead; the controller uses those predictions to rewrite ECMP weights in real time.

## What it does

Traffic flows between hosts through a simulated **8-leaf / 8-spine fabric** running on BMv2/Mininet, where **every spine has a distinct uplink bandwidth** (0.6 → 1.3 Mbps). Each packet is forwarded using three layers:

- **W-ECMP** (across groups): probabilistic splitting across capacity-clustered spine groups, weighted by aggregate group capacity.
- **DRILL** (within a group): the P4 dataplane picks the least-queued port per-packet via power-of-two-choices with memory. No control plane involvement — but it only balances *load* when ports in the group drain at similar rates, so it is only correct inside near-symmetric groups.
- **ML control** (1-second loop): reads P4 hardware registers, builds topology-independent features, predicts latency/loss, and reweights W-ECMP on l1 to relocate elephant flow-mass off hot groups — the one thing DRILL's queue-equalization cannot do.

### Grouping rule

Strict-symmetry DRILL would shatter 8 distinct capacities into 8 single-port groups (= plain W-ECMP, no micro-balancing). Instead the control plane clusters **adjacent-capacity** spines into pairs `(s1,s2) (s3,s4) (s5,s6) (s7,s8)` (`all_controller.py: _cluster_by_capacity`), keeping ≥2 ports per group for a real DRILL second choice, and weights by aggregate capacity → `[3,4,5,6]`. Weights are reduced to a smallest-integer ratio so the BMv2 selector stays ~18 members (coprime capacity sums would otherwise explode to 76, stalling the control loop). See [`OVERVIEW.md`](OVERVIEW.md) for where this holds and where it degrades.

### What the ML adds

W-ECMP+DRILL is capacity-correct but **elephant-blind**: an elephant flow hashes onto one group for its lifetime and DRILL can only move its *packets* between that group's two ports, not the elephant *off* the group. The controller detects the resulting sustained queue/util pressure and **sheds that group's weight** (no flow classification needed); RF predictions let it act a second early. It trades a little median latency for much better tail latency and ~half the utilization spread.

### Low-overhead control

A weight change rehashes *all* flows (BMv2 `action_selector` isn't consistent-hashing), so the loop changes weights rarely and cheaply: it trusts local weight state instead of re-reading the selector each tick; repoints the forwarding entry to a freshly-built group **atomically** (hitless, no BMv2 mutation of the in-use group); keeps weights small-integer; and gates rehashes behind cooldown / settle / persist timers so it freezes on steady traffic.

## Structure

```
AI-drill/
├── jsq_2_2/       # 2-switch ECMP reference topology (educational baseline)
└── main/          # Active project — P4 sources, controllers, traffic, models
```

See [`OVERVIEW.md`](OVERVIEW.md) for the full technical reference.

## Quick start

```bash
cd main
./start_env.sh                        # boot P4 network, program routes, set rate limits
# in a second terminal:
python3 traffic.py --elmice           # elephant/mice traffic (also: --static, --dynamic)
python3 realtime_ml_controller.py     # run the ML controller
```

Compare all four algorithms on the same dataplane and plot the result:

```bash
./test.sh 60                          # sweep ML / DRILL / ECMP / W-ECMP (with traffic running)
python3 plot_result.py                # overlay CSVs + print summary (first 5s dropped)
```

## Results

Elephant/mice traffic (`--elmice`), 60s, 5s warmup dropped — latency is per-second peak queue delay, Util σ is per-spine utilization spread (lower = more balanced). *(Numbers are from the earlier symmetric-pair topology; the current all-distinct config reduces to the same `[3,4,5,6]` weights, re-run pending.)*

| Algorithm | Lat p50 | Lat p95 | Lat max | Loss E2E | Mbps | Util σ |
|---|--:|--:|--:|--:|--:|--:|
| ECMP | 101.35 | 251.54 | 564.74 | 0.00 | 2.65 | 0.160 |
| W-ECMP | 37.66 | 334.10 | 457.67 | 0.00 | 2.62 | 0.161 |
| DRILL | **22.57** | 170.42 | 340.47 | 0.00 | 2.86 | 0.148 |
| **W-ECMP+DRILL+ML** | 32.04 | **160.17** | **176.28** | 0.00 | 2.83 | **0.062** |

**W-ECMP+DRILL+ML** wins the tail (lowest p95, and max latency 176ms vs 340–565ms) and is by far the most balanced (σ 0.062, ~half the others), staying within a few ms of DRILL on median. DRILL alone has the lowest median but is capacity-blind; static ECMP/W-ECMP are worst on the tail.

See [`OVERVIEW.md`](OVERVIEW.md) for the full technical reference.
