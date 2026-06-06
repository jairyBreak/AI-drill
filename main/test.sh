#!/bin/bash
set -euo pipefail

DURATION="${1:-65}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

cd "$(dirname "$0")"

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

echo
echo "All runs complete."
