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
└── main/          # Active project — see main/OVERVIEW.md for full technical reference
```

## Quick start

```bash
cd main
./start_env.sh                        # boot P4 network, program routes, set rate limits
# in a second terminal:
python3 traffic.py --dynamic          # generate 18-flow rotating traffic
python3 realtime_ml_controller.py     # run the ML controller
```

See `OVERVIEW.md` (this directory) or `main/OVERVIEW.md` for the full technical reference.
