# AI-Drill Dashboard Demo Guide

## 1. Start The P4 Environment

From the project root:

```bash
cd /home/p4/drill/main
./start_env.sh
```

Wait until the terminal prints:

```text
Dashboard: http://127.0.0.1:8080
```

Then open:

```text
http://127.0.0.1:8080
```

`start_env.sh` starts Mininet/P4, applies queue rate limits, starts `realtime_ml_controller.py`, and serves the dashboard.

## 2. Allow Other Machines To Connect

By default the dashboard only listens on localhost. To allow other machines on the same network:

```bash
cd /home/p4/drill/main
DASHBOARD_HOST=0.0.0.0 DASHBOARD_PORT=8080 ./start_env.sh
```

Find your machine IP:

```bash
hostname -I
```

Other users can open:

```text
http://YOUR_MACHINE_IP:8080
```

Do not expose this dashboard to the public internet. It has no authentication.

## 3. Generate Demo Traffic

Open a second terminal:

```bash
cd /home/p4/drill/main
sudo python3 traffic.py --elmice --no-monitor 120
```

Recommended traffic modes:

```bash
sudo python3 traffic.py --elmice --no-monitor 120
sudo python3 traffic.py --static --no-monitor 120
sudo python3 traffic.py --dynamic --no-monitor 120
```

Heavier traffic:

```bash
sudo python3 traffic.py --elmice --no-monitor --scale 1.3 120
```

## 4. Dashboard Controls

- The main topology shows the demo path:
  ```text
  h1 -> l1 -> s1..s8 -> l2 -> h2
  ```
- Link line width and badge show current Mbps.
- Click a spine or leaf switch to focus related links.
- Click the same switch again to clear the focus.
- The right-side port table is fixed to `l2 Ports`.
- The `Util` bar is calculated as:
  ```text
  Mbps / effective capacity
  ```
- Queue is shown as a number because queue depth is usually 0 or 1 in this demo.

## 5. What The Metrics Mean

Top status cards:

- `Pred Lat`: Random Forest predicted latency.
- `HW Lat`: hardware/INT observed latency.
- `Pred Loss`: Random Forest predicted loss.
- `HW Loss`: hardware counter estimated loss.
- `Total`: total measured Mbps.
- `Rehash`: seconds since last W-ECMP selector weight change.

Right panel:

- `W-ECMP Components`: current weight / anchor weight.
- `l2 Ports`: per-port utilization, Mbps, queue, ingress packet rate, egress packet rate.

## 6. Useful Options

Change dashboard port:

```bash
DASHBOARD_PORT=8081 ./start_env.sh
```

Disable dashboard and use the old controller installer path:

```bash
DASHBOARD_ENABLE=0 ./start_env.sh
```

Run the controller manually:

```bash
python3 realtime_ml_controller.py
python3 realtime_ml_controller.py 120
python3 realtime_ml_controller.py --web-host 0.0.0.0 --web-port 8080
python3 realtime_ml_controller.py --no-web
```

## 7. Shutdown

In the Mininet terminal:

```text
exit
```

or press `Ctrl+C`.

`start_env.sh` will clean up the background controller and Mininet state.

## 8. Troubleshooting

Dashboard does not open:

```bash
cat controller_output.log
```

Rate limiter failed:

```bash
cat rate_limiter_output.log
```

Port is already in use:

```bash
DASHBOARD_PORT=8081 ./start_env.sh
```

Other machines cannot connect:

- Start with `DASHBOARD_HOST=0.0.0.0`.
- Check the correct machine IP with `hostname -I`.
- Make sure firewall rules allow the dashboard port.
- If running inside a VM, use bridged networking or configure port forwarding.
