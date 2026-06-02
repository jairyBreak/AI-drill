# A Two-Timescale Load Balancer Guided by ML
> 大象、老鼠、森林：基於機器學習的雙時間尺度負載平衡器

A closed-loop, P4-based data center network load balancer that uses Random Forest models to predict congestion and actively adjust ECMP weights in real time.

## What it does

Traffic flows between hosts through a simulated 8-leaf / 8-spine fabric running on BMv2. Each packet is forwarded using **W-ECMP** (weighted probabilistic splitting across spine groups) combined with **DRILL** (per-packet power-of-two-choices in hardware). A control plane reads P4 hardware registers every second, predicts network QoS, and rewrites ECMP weights on l1 to keep load balanced across all 8 spines.

## Structure

```
AI-drill/
├── jsq_2_2/       # 2-switch reference topology (educational baseline)
└── drill_4_4/     # Active project — see OVERVIEW.md for full documentation
```

## Quick start

```bash
cd drill_4_4
./start_env.sh          # boots P4 network, programs routes, sets rate limits
# in a second terminal:
python3 traffic.py --dynamic        # generate 18-flow rotating traffic
python3 realtime_ml_controller.py   # run the ML controller
```

See `drill_4_4/OVERVIEW.md` for the full technical reference.
