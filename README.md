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

## Topology

8 hosts, 8 leaf switches (l1–l8), 8 spine switches (s1–s8) on BMv2/Mininet. Every leaf connects to every spine, and **each spine uplink has a distinct bandwidth** — the asymmetry the load balancer must handle.

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

- **Hosts:** h1–h8 at `10.0.{i}.{i}/24` → h1 = `10.0.1.1`, h2 = `10.0.2.2`, …
- **Thrift ports:** leaves l1–l8 → 9090–9097, spines s1–s8 → 9098–9105
- **Interfaces:** on each leaf, `eth1` is host-facing; `eth2…eth9` are the uplinks to s1…s8
- **Asymmetric uplinks** (rate-limited to ×0.8 of link BW), clustered into 4 DRILL pairs weighted by aggregate capacity → `[3,4,5,6]`:

| l1 port | iface | Spine | Link BW | Eff. cap | DRILL group | Weight |
|--:|:--|:--|--:|--:|:-:|--:|
| 2 | l1-eth2 | s1 | 0.6 | 0.48 | (s1,s2) | 3 |
| 3 | l1-eth3 | s2 | 0.7 | 0.56 | | |
| 4 | l1-eth4 | s3 | 0.8 | 0.64 | (s3,s4) | 4 |
| 5 | l1-eth5 | s4 | 0.9 | 0.72 | | |
| 6 | l1-eth6 | s5 | 1.0 | 0.80 | (s5,s6) | 5 |
| 7 | l1-eth7 | s6 | 1.1 | 0.88 | | |
| 8 | l1-eth8 | s7 | 1.2 | 0.96 | (s7,s8) | 6 |
| 9 | l1-eth9 | s8 | 1.3 | 1.04 | | |

## Structure

```
AI-drill/
├── jsq_2_2/       # 2-switch ECMP reference topology (educational baseline)
└── main/          # Active project — P4 sources, controllers, traffic, models
```

See [`OVERVIEW.md`](OVERVIEW.md) for the full technical reference.

## Running the system

```bash
cd main
./start_env.sh        # boot the P4 net (16 BMv2 switches), program W-ECMP routes, set rate limits
```

`start_env.sh` runs `p4run` in the foreground (Mininet CLI). When it prints `背景設施全部就緒` ("background facilities ready"), the fabric is up — open a **second terminal** for everything below.

```bash
# generate traffic, then run the controller (separate shells)
python3 traffic.py --elmice           # elephant/mice (also: --static, --dynamic, --default)
python3 realtime_ml_controller.py     # the ML controller
                                      #   set ML_WEIGHT_ENABLE=False for static W-ECMP+DRILL
```

## Testing & monitoring

Run a command inside a host's network namespace with `mx <host>` (h1 = `10.0.1.1`, h2 = `10.0.2.2`).

**Single iperf3 flow, h1 → h2** (UDP, 0.1 Mbps, 20 s):

```bash
# terminal A — server on h2
mx h2
iperf3 -s

# terminal B — client on h1
mx h1
iperf3 -c 10.0.2.2 -u -b 0.1M -t 20
```

**Live per-interface bandwidth** (e.g. all of l1's uplinks `l1-eth2…l1-eth9`):

```bash
bmon -p "l1-eth*"        # also: l2-eth*, s5-eth*, … one pane per interface
```

**Ping latency**, h1 → h2:

```bash
mx h1
ping 10.0.2.2
```

## Comparing algorithms

With traffic running, sweep all four algorithms on the same dataplane and plot:

```bash
./test.sh 60             # sweep ML / DRILL / ECMP / W-ECMP
python3 plot_result.py   # overlay the four CSVs + print the summary table (first 5 s dropped)
```

## Results

Elephant/mice traffic (`--elmice`), 60s, 5s warmup dropped — latency is per-second peak queue delay, Util σ is per-spine utilization spread (lower = more balanced). Measured on the current all-distinct (0.6→1.3 Mbps) topology.

| Algorithm | Lat p50 | Lat p95 | Lat p99 | Throughput | Util σ |
|---|--:|--:|--:|--:|--:|
| ECMP | 108.66 | 443.84 | 1185.99 | 2.53 | 0.118 |
| W-ECMP | 79.64 | 1025.95 | 2485.76 | 2.41 | 0.087 |
| DRILL (naive) | **18.97** | 159.50 | 187.78 | 2.87 | 0.122 |
| **W-ECMP+DRILL+ML** | 20.27 | **141.16** | **172.24** | 2.86 | **0.043** |

**W-ECMP+DRILL+ML** wins the tail outright — lowest p95 (141ms) and p99 (172ms), beating even naive DRILL — and is by far the most balanced (σ 0.043, ~2.8× better than DRILL), for the price of ~1ms of median. Naive DRILL has the lowest median but is capacity-blind (worse tail and σ); static ECMP/W-ECMP are worst on the tail (p99 6.9× and 14.4× higher than ML).

See [`OVERVIEW.md`](OVERVIEW.md) for the full technical reference.
