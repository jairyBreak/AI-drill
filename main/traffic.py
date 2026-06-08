import os
import re
import sys
import time
import random
import tempfile
import argparse
import threading
import subprocess
import signal
import shutil
from contextlib import redirect_stdout, redirect_stderr

SOURCE_HOST = "h1"
TARGET_IP   = "10.0.2.2"
TARGET_HOST = "h2"
INTERVAL_S  = 10

PORTS    = [5100, 5101, 5102, 5103, 5104, 5105,
            5106, 5107, 5108, 5109, 5110, 5111,
            5112, 5113, 5114, 5115, 5116, 5117]
FLOW_BWS = [
    "0.40M", "0.40M", "0.40M",
    "0.24M", "0.24M", "0.24M",
    "0.16M", "0.16M", "0.16M", "0.16M",
    "0.08M", "0.08M", "0.08M", "0.08M",
    "0.06M", "0.06M", "0.06M", "0.06M",
]  # total = 3.12M
MICE_BWS = ["0.14M", "0.14M", "0.14M", "0.14M",
            "0.14M", "0.14M", "0.14M", "0.14M",
            "0.135M", "0.135M", "0.135M", "0.135M",
            "0.135M", "0.135M", "0.135M", "0.135M"]  # total = 2.20M
ELEPHANT_BWS = ["0.46M", "0.46M"]  # total = 0.92M
MICE_COUNT = 16
MICE_INTERVAL_S = 5
ELEPHANT_INTERVAL_S = 15
STAGGER  = 0.3   # seconds between starting each client (desync TCP timers)
ELMICE_STAGGER = 0.03

SPINE_CAP   = {2: 0.48, 3: 0.48, 4: 0.64, 5: 0.64,
               6: 0.80, 7: 0.80, 8: 0.96, 9: 0.96}
SPINE_NAMES = {2: "s1", 3: "s2", 4: "s3", 5: "s4",
               6: "s5", 7: "s6", 8: "s7", 9: "s8"}
L1_THRIFT   = 9090  # queue depth lives here (egress congestion point)
L2_THRIFT   = 9091  # byte counters live here (ML controller reads same source)

_monitor_header = ""  # set by run_dynamic; included in the in-place monitor block
DISABLE_MONITOR = False


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
    prev_line_count = 0

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
        if not parts:
            continue

        width = max(40, shutil.get_terminal_size((120, 20)).columns - 4)
        hdr = _monitor_header
        display_lines = ([hdr] if hdr else []) + parts
        display_lines = [
            line if len(line) <= width else line[:max(0, width - 3)] + "..."
            for line in display_lines
        ]

        if not first and prev_line_count > 0:
            sys.stdout.write(f"\033[{prev_line_count}A")
        for line in display_lines:
            sys.stdout.write(f"\r\033[K  {line}\n")
        sys.stdout.flush()
        prev_line_count = len(display_lines)
        first = False

    devnull.close()


def _start_monitor():
    ev = threading.Event()
    if not DISABLE_MONITOR:
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


def _start_clients(assignment, duration, plot_dir=None, stagger=STAGGER):
    """
    assignment: [(port, bw_str), ...]
    Clients are started with stagger seconds between each to desync timers.
    Returns (procs, log_paths).
    """
    procs, logs = [], []
    ivl = ["-i", "1"] if plot_dir else ["-i", "0"]
    for i, (port, bw) in enumerate(assignment):
        if i > 0:
            time.sleep(stagger)
        lp = os.path.join(plot_dir, f"flow{i}_p{port}.log") if plot_dir else None
        out = open(lp, "w") if lp else subprocess.DEVNULL
        procs.append(subprocess.Popen(
            ["mx", SOURCE_HOST, "iperf3", "-u", "-c", TARGET_IP,
             "-p", str(port), "-b", bw, "-t", f"{duration:g}"] + ivl,
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


def _stop_procs(procs, grace=0.2):
    """Stop only the client processes we started, preserving servers/elephants."""
    for p in procs:
        if p.poll() is not None:
            continue
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass
    if grace > 0:
        time.sleep(grace)
    for p in procs:
        if p.poll() is not None:
            continue
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


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
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (label, ts, mbps) in enumerate(series):
        if ts:
            ax.plot(ts, mbps, label=label, color=colors[i % 20], linewidth=1.2)
    for rt in rotation_times:
        ax.axvline(rt, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("iperf3 flow bitrates")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "traffic_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    plt.show()


# ---------------------------------------------------------------------------

def run_default(plot, duration):
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
         "-p", str(PORTS[0]), "-b", "0.3M",
         "-t", f"{(duration if duration is not None else 3600):g}"] + ivl,
        stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp,
    )
    dur_label = f"{duration:g}s" if duration is not None else "Ctrl+C to stop"
    print(f"[DEFAULT] 1 flow → {TARGET_IP}:{PORTS[0]} @ 0.3M  ({dur_label})")
    _mon = _start_monitor()
    start = time.time()
    try:
        while duration is None or time.time() - start < duration:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    _kill([cli])
    _kill(srvs)
    if plot:
        data = _parse_single(lp)
        _plot([("0.3M", [d[0] for d in data], [d[1] for d in data])], [])


def run_static(plot, duration):
    _kill([])
    srvs = _start_servers()
    bws = FLOW_BWS[:]
    random.shuffle(bws)
    assignment = list(zip(PORTS, bws))
    log_dir = tempfile.mkdtemp() if plot else None
    client_duration = duration if duration is not None else 3600
    clients, logs = _start_clients(assignment, duration=client_duration, plot_dir=log_dir)
    dur_label = f"{duration:g}s" if duration is not None else "Ctrl+C to stop"
    print(f"[STATIC] Flow assignment ({dur_label}):")
    for port, bw in assignment:
        print(f"  port {port}: {bw}")
    _mon = _start_monitor()
    start = time.time()
    try:
        while duration is None or time.time() - start < duration:
            time.sleep(0.5)
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


def run_dynamic(plot, duration):
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
        while duration is None or time.time() - start_t < duration:
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
            if duration is None:
                time.sleep(INTERVAL_S)
            else:
                time.sleep(min(INTERVAL_S, max(0.0, duration - (time.time() - start_t))))

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


def run_elmice(duration, plot):
    global _monitor_header
    _kill([])
    _mon = _start_monitor()
    srvs = _start_servers()
    log_dir = tempfile.mkdtemp() if plot else None

    elephant_clients, elephant_logs = [], []
    mice_clients, mice_logs = [], []
    elephant_assignment, mice_assignment = [], []
    rotation_times = []
    port_series = {p: ([], []) for p in PORTS}
    start_t = time.time()
    next_elephant = 0.0
    next_mice = 0.0
    elephant_round = 0
    mice_round = 0
    elephant_count = len(ELEPHANT_BWS)

    def _collect(assignment, logs, offset):
        for (port, _), lp in zip(assignment, logs):
            for t, m in _parse_single(lp):
                port_series[port][0].append(offset + t)
                port_series[port][1].append(m)

    def _new_elephants(elapsed):
        nonlocal elephant_clients, elephant_logs, elephant_assignment, elephant_round
        if plot and elephant_assignment:
            _collect(elephant_assignment, elephant_logs, max(0.0, elapsed - ELEPHANT_INTERVAL_S))
        _stop_procs(elephant_clients)

        ports = PORTS[:]
        random.shuffle(ports)
        selected = sorted(ports[:elephant_count])
        bws = ELEPHANT_BWS[:]
        random.shuffle(bws)
        elephant_assignment = list(zip(selected, bws))

        rot_dir = None
        if plot:
            rot_dir = os.path.join(log_dir, f"elephant_r{elephant_round}")
            os.makedirs(rot_dir)
        client_duration = ELEPHANT_INTERVAL_S + MICE_INTERVAL_S + len(elephant_assignment) * ELMICE_STAGGER + 5
        elephant_clients, elephant_logs = _start_clients(
            elephant_assignment, client_duration, rot_dir, stagger=ELMICE_STAGGER)
        elephant_round += 1

    def _new_mice(elapsed):
        nonlocal mice_clients, mice_logs, mice_assignment, mice_round
        if plot and mice_assignment:
            _collect(mice_assignment, mice_logs, max(0.0, elapsed - MICE_INTERVAL_S))
        _stop_procs(mice_clients)

        elephant_ports = {p for p, _ in elephant_assignment}
        mouse_ports = [p for p in PORTS if p not in elephant_ports]
        random.shuffle(mouse_ports)
        mouse_ports = sorted(mouse_ports[:min(MICE_COUNT, len(mouse_ports))])
        bws = MICE_BWS[:len(mouse_ports)]
        random.shuffle(bws)
        mice_assignment = list(zip(mouse_ports, bws))

        rot_dir = None
        if plot:
            rot_dir = os.path.join(log_dir, f"mice_r{mice_round}")
            os.makedirs(rot_dir)
        client_duration = MICE_INTERVAL_S + len(mice_assignment) * ELMICE_STAGGER + 3
        mice_clients, mice_logs = _start_clients(
            mice_assignment, client_duration, rot_dir, stagger=ELMICE_STAGGER)
        mice_round += 1

    print("[ELMICE] Persistent elephant + fast-changing mice")
    dur_label = f"{duration:g}s" if duration is not None else "Ctrl+C to stop"
    print(f"  duration: {dur_label}")
    print(f"  elephants: {len(ELEPHANT_BWS)} flows x {ELEPHANT_BWS}, rotate every {ELEPHANT_INTERVAL_S}s")
    print(f"  mice: {MICE_COUNT} flows x {MICE_BWS}, rotate every {MICE_INTERVAL_S}s")
    print(f"  elmice client stagger: {ELMICE_STAGGER}s")

    try:
        while duration is None or time.time() - start_t < duration:
            elapsed = time.time() - start_t
            if elapsed >= next_elephant:
                rotation_times.append(elapsed)
                if plot and mice_assignment:
                    _collect(mice_assignment, mice_logs, max(0.0, elapsed - MICE_INTERVAL_S))
                _stop_procs(mice_clients)
                mice_clients, mice_logs, mice_assignment = [], [], []
                _new_elephants(elapsed)
                next_elephant = elapsed + ELEPHANT_INTERVAL_S
                next_mice = elapsed

            if elapsed >= next_mice:
                rotation_times.append(elapsed)
                _new_mice(elapsed)
                next_mice = elapsed + MICE_INTERVAL_S

            _monitor_header = (
                f"[t={elapsed:.0f}s] elephant "
                + " ".join(f"{p}:{bw}" for p, bw in elephant_assignment)
                + f" | mice {len(mice_assignment)} flows, total 2.20M"
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start_t
    _stop_procs(mice_clients)
    _stop_procs(elephant_clients)
    if plot:
        if mice_assignment:
            _collect(mice_assignment, mice_logs, max(0.0, elapsed - MICE_INTERVAL_S))
        if elephant_assignment:
            _collect(elephant_assignment, elephant_logs, max(0.0, elapsed - ELEPHANT_INTERVAL_S))
    _kill(srvs)

    if plot:
        series = [
            (f"port {p}", port_series[p][0], port_series[p][1])
            for p in PORTS
        ]
        _plot(series, sorted(set(rotation_times)))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="iperf3 traffic generator h1→h2")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--static",  action="store_true",
                     help="18 flows (fixed BW assignment)")
    grp.add_argument("--dynamic", action="store_true",
                     help="18 flows, BW rotates randomly every 10s")
    grp.add_argument("--elmice", action="store_true",
                     help="2 persistent elephants plus 16 fast-changing mice flows")
    parser.add_argument("duration", nargs="?", type=float,
                        help="duration in seconds; omitted means run until Ctrl+C")
    parser.add_argument("--plot", action="store_true",
                        help="Capture and plot per-flow throughput at exit")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Disable Thrift monitor (useful when called by other scripts)")
    args = parser.parse_args()

    if args.no_monitor:
        global DISABLE_MONITOR
        DISABLE_MONITOR = True

    if args.static:
        run_static(args.plot, args.duration)
    elif args.dynamic:
        run_dynamic(args.plot, args.duration)
    elif args.elmice:
        run_elmice(args.duration, args.plot)
    else:
        run_default(args.plot, args.duration)


if __name__ == "__main__":
    main()
