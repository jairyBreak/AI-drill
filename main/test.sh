#!/bin/bash
set -euo pipefail

DURATION="${1:-65}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

cd "$(dirname "$0")"

echo "Starting background traffic generator (--elmice)..."
sudo pkill -9 -f "traffic.py" || true
sudo pkill -9 -f "iperf3" || true
sudo python3 traffic.py --elmice --no-monitor > /dev/null 2>&1 &
sleep 2  # wait for traffic to start

run_model() {
    local label="$1"
    shift

    echo
    echo "============================================================"
    echo "Running ${label} for ${DURATION}s"
    echo "============================================================"
    "$PYTHON_BIN" "$@" "$DURATION"
}

run_model "ML"    realtime_ml_controller.py
run_model "DRILL" baseline_drill.py
run_model "ECMP"  baseline_ecmp.py
run_model "WECMP" baseline_wecmp.py

echo "Killing background traffic generator..."
sudo pkill -f "traffic.py" || true
sudo pkill -f "iperf3" || true
echo
echo "All runs complete."
