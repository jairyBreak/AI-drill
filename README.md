# AI-Drill: ML-Augmented Adaptive Load Balancer

A closed-loop, P4-based data center load balancer. Random Forest models trained on P4 switch telemetry predict congestion one second ahead; the controller uses those predictions to rewrite ECMP weights in real time.

## What it does

Traffic flows between hosts through a simulated **8-leaf / 8-spine fabric** running on BMv2/Mininet. Each packet is forwarded using two mechanisms:

- **W-ECMP**: probabilistic splitting across 4 spine groups, with weights set by the control plane.
- **DRILL**: within each group, the P4 dataplane picks the least-queued port per-packet using power-of-two-choices with memory. No control plane involvement.

A control plane reads P4 hardware registers every second, transforms the telemetry into topology-independent features, predicts latency and loss, and rewrites ECMP weights on l1 to keep load balanced across all 8 asymmetric spines.

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

Elephant/mice traffic (`--elmice`), 60s, 5s warmup dropped — latency is per-second peak queue delay, Util σ is per-spine utilization spread (lower = more balanced):

| Algorithm | Lat p50 | Lat p95 | Lat max | Loss E2E | Mbps | Util σ |
|---|--:|--:|--:|--:|--:|--:|
| ECMP | 101.35 | 251.54 | 564.74 | 0.00 | 2.65 | 0.160 |
| W-ECMP | 37.66 | 334.10 | 457.67 | 0.00 | 2.62 | 0.161 |
| DRILL | **22.57** | 170.42 | 340.47 | 0.00 | 2.86 | 0.148 |
| **W-ECMP+DRILL+ML** | 32.04 | **160.17** | **176.28** | 0.00 | 2.83 | **0.062** |

**W-ECMP+DRILL+ML** wins the tail (lowest p95, and max latency 176ms vs 340–565ms) and is by far the most balanced (σ 0.062, ~half the others), staying within a few ms of DRILL on median. DRILL alone has the lowest median but is capacity-blind; static ECMP/W-ECMP are worst on the tail.

See [`OVERVIEW.md`](OVERVIEW.md) for the full technical reference.
