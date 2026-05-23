import os
import re
import sys
import time
import random
import tempfile
import argparse
import threading
import subprocess
from contextlib import redirect_stdout, redirect_stderr

SOURCE_HOST = "h1"
TARGET_IP   = "10.0.2.2"
TARGET_HOST = "h2"
INTERVAL_S  = 10

PORTS    = [5100, 5101, 5102, 5103, 5104, 5105]
FLOW_BWS = ["0.60M", "0.24M", "0.20M", "0.16M", "0.12M", "0.08M"]   # total = 1.40M
STAGGER  = 0.5   # seconds between starting each client (desync TCP timers)

SPINE_CAP   = {2: 0.64, 3: 0.80, 4: 0.96, 5: 1.12}  # Mbps soft-caps (matches rate_limiter ×0.8)
SPINE_NAMES = {2: "s1",  3: "s2",  4: "s3",  5: "s4"}
L1_THRIFT   = 9090  # queue depth lives here (egress congestion point)
L2_THRIFT   = 9091  # byte counters live here (ML controller reads same source)

_monitor_header = ""  # set by run_dynamic; included in the in-place monitor block


def _make_thrift(port):
    try:
        p4path = os.environ.get('P4_UTILS_PATH', '/home/p4/p4-utils')
        if p4path not in sys.path:
            sys.path.insert(0, p4path)
        from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI
        return SimpleSwitchThriftAPI(port)
    except Exception:
        return None


def _monitor_loop(stop_event):
    api_l1 = _make_thrift(L1_THRIFT)
    api_l2 = _make_thrift(L2_THRIFT)
    if api_l1 is None or api_l2 is None:
        return

    prev_bytes = {}
    devnull = open(os.devnull, 'w')
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            for p in SPINE_CAP:
                prev_bytes[p] = api_l2.counter_read('port_bytes_counter', p)[0]
    except Exception:
        devnull.close()
        return
    prev_t = time.time()
    first  = True

    while not stop_event.is_set():
        time.sleep(1.0)
        now = time.time()
        dt  = max(now - prev_t, 0.001)
        parts = []
        try:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                for p in sorted(SPINE_CAP):
                    q   = api_l1.register_read('q_depth_reg', p)
                    cnt = api_l2.counter_read('port_bytes_counter', p)[0]
                    db  = max(0, cnt - prev_bytes[p])
                    mbps = (db * 8) / (dt * 1e6)
                    cap  = SPINE_CAP[p]
                    pct  = int(mbps / cap * 100) if cap > 0 else 0
                    bar  = "#" * min(pct // 10, 10)
                    parts.append(f"{SPINE_NAMES[p]}  q={int(q):2d}/64  {mbps:.2f}/{cap:.2f} Mbps  [{bar:<10}] {pct:3d}%")
                    prev_bytes[p] = cnt
        except Exception:
            break
        prev_t = now
        if parts:
            hdr = _monitor_header
            all_lines = ([hdr] if hdr else []) + parts
            if not first:
                sys.stdout.write(f"\033[{len(all_lines)}A")
            for line in all_lines:
                sys.stdout.write(f"\r\033[K  {line}\n")
            sys.stdout.flush()
            first = False

    devnull.close()


def _start_monitor():
    ev = threading.Event()
    threading.Thread(target=_monitor_loop, args=(ev,), daemon=True).start()
    return ev


def _start_servers():
    procs = []
    for port in PORTS:
        p = subprocess.Popen(
            ["mx", TARGET_HOST, "iperf3", "-s", "-p", str(port)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp,
        )
        procs.append(p)
    time.sleep(1.0)
    return procs


def _start_clients(assignment, duration, plot_dir=None):
    """
    assignment: [(port, bw_str), ...]
    Clients are started with STAGGER seconds between each to desync timers.
    Returns (procs, log_paths).
    """
    procs, logs = [], []
    ivl = ["-i", "1"] if plot_dir else ["-i", "0"]
    for i, (port, bw) in enumerate(assignment):
        if i > 0:
            time.sleep(STAGGER)
        lp = os.path.join(plot_dir, f"flow{i}_p{port}.log") if plot_dir else None
        out = open(lp, "w") if lp else subprocess.DEVNULL
        procs.append(subprocess.Popen(
            ["mx", SOURCE_HOST, "iperf3", "-u", "-c", TARGET_IP,
             "-p", str(port), "-b", bw, "-t", str(duration)] + ivl,
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp,
        ))
        logs.append(lp)
    return procs, logs


def _kill(_procs):
    for host in (SOURCE_HOST, TARGET_HOST):
        subprocess.run(
            ["mx", host, "pkill", "-9", "iperf3"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(0.5)


def _parse_single(path):
    """(t_end, mbps) from single-stream iperf3 interval lines."""
    if not path or not os.path.exists(path):
        return []
    pat = re.compile(r'(\d+\.\d+)-(\d+\.\d+)\s+sec.*?([\d\.]+)\s+([KMG])bits/sec')
    out = []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            if float(m.group(2)) - float(m.group(1)) > 1.5:
                continue  # skip summary line
            mbps = float(m.group(3)) * {'K': 1e-3, 'M': 1.0, 'G': 1e3}[m.group(4)]
            out.append((float(m.group(2)), mbps))
    return out


def _plot(series, rotation_times):
    import matplotlib.pyplot as plt
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (label, ts, mbps) in enumerate(series):
        if ts:
            ax.plot(ts, mbps, label=label, color=colors[i % 10], linewidth=1.2)
    for rt in rotation_times:
        ax.axvline(rt, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("iperf3 flow bitrates")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "traffic_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    plt.show()


# ---------------------------------------------------------------------------

def run_default(plot):
    _kill([])
    srvs = _start_servers()
    lp = None
    if plot:
        lp = os.path.join(tempfile.mkdtemp(), "default.log")
        out = open(lp, "w")
        ivl = ["-i", "1"]
    else:
        out = subprocess.DEVNULL
        ivl = ["-i", "0"]
    cli = subprocess.Popen(
        ["mx", SOURCE_HOST, "iperf3", "-u", "-c", TARGET_IP,
         "-p", str(PORTS[0]), "-b", "0.3M", "-t", "3600"] + ivl,
        stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp,
    )
    print(f"[DEFAULT] 1 flow → {TARGET_IP}:{PORTS[0]} @ 0.3M  (Ctrl+C to stop)")
    _mon = _start_monitor()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    _kill([cli])
    _kill(srvs)
    if plot:
        data = _parse_single(lp)
        _plot([("0.3M", [d[0] for d in data], [d[1] for d in data])], [])


def run_static(plot):
    _kill([])
    srvs = _start_servers()
    bws = FLOW_BWS[:]
    random.shuffle(bws)
    assignment = list(zip(PORTS, bws))
    log_dir = tempfile.mkdtemp() if plot else None
    clients, logs = _start_clients(assignment, duration=3600, plot_dir=log_dir)
    print("[STATIC] Flow assignment (Ctrl+C to stop):")
    for port, bw in assignment:
        print(f"  port {port}: {bw}")
    _mon = _start_monitor()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    _kill(clients)
    _kill(srvs)
    if plot:
        series = []
        for (port, bw), lp in zip(assignment, logs):
            data = _parse_single(lp)
            series.append((f"port {port} ({bw})", [d[0] for d in data], [d[1] for d in data]))
        _plot(series, [])


def run_dynamic(plot):
    global _monitor_header
    _kill([])
    _mon = _start_monitor()
    log_dir = tempfile.mkdtemp() if plot else None

    rotation_times = []
    # per-port accumulated series for plot
    port_series = {p: ([], []) for p in PORTS}

    current_clients = []
    current_assignment = []
    current_logs = []
    srvs = []
    start_t = time.time()
    rotation = 0

    def _collect(offset):
        for (port, _), lp in zip(current_assignment, current_logs):
            for t, m in _parse_single(lp):
                port_series[port][0].append(offset + t)
                port_series[port][1].append(m)

    try:
        while True:
            elapsed = time.time() - start_t
            rotation_times.append(elapsed)

            _kill(current_clients)  # pkill is system-wide; kills servers too
            time.sleep(0.3)

            if plot and current_assignment:
                _collect(rotation_times[-2] if len(rotation_times) > 1 else 0.0)

            current_clients.clear()
            srvs = _start_servers()  # restart servers (pkill killed them too)
            rot_dir = None
            if plot:
                rot_dir = os.path.join(log_dir, f"r{rotation}")
                os.makedirs(rot_dir)

            bws = FLOW_BWS[:]
            random.shuffle(bws)
            assignment = list(zip(PORTS, bws))
            clients, logs = _start_clients(
                assignment, duration=INTERVAL_S + 5 + len(PORTS) * STAGGER,
                plot_dir=rot_dir,
            )
            current_clients.extend(clients)
            current_assignment = assignment
            current_logs = logs

            _monitor_header = (f"[t={elapsed:.0f}s]  " +
                               "  ".join(f"{p}:{bw}" for p, bw in assignment))

            rotation += 1
            time.sleep(INTERVAL_S)

    except KeyboardInterrupt:
        pass

    _kill(current_clients)
    time.sleep(0.3)
    if plot and current_assignment:
        _collect(rotation_times[-1] if rotation_times else 0.0)

    _kill(srvs)

    if plot:
        series = [
            (f"port {p}", port_series[p][0], port_series[p][1])
            for p in PORTS
        ]
        _plot(series, rotation_times)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="iperf3 traffic generator h1→h2")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--static",  action="store_true",
                     help="4 flows (1 heavy @ 0.12M + 3 light @ 0.06M), fixed, ~0.3M total")
    grp.add_argument("--dynamic", action="store_true",
                     help="Same 4 flows but heavy rotates randomly every 10s")
    parser.add_argument("--plot", action="store_true",
                        help="Capture and plot per-flow throughput at exit")
    args = parser.parse_args()

    if args.static:
        run_static(args.plot)
    elif args.dynamic:
        run_dynamic(args.plot)
    else:
        run_default(args.plot)


if __name__ == "__main__":
    main()
