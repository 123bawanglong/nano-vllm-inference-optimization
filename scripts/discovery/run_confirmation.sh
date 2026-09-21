#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../profiling/environment.sh"
cd "$ROOT"
NSYS=/home/xietaibo/tools/nsight-systems-cli-2026.5.1/opt/nvidia/nsight-systems-cli/2026.5.1/bin/nsys
PARENT="$(cat "$ROOT/scripts/discovery/latest_result.txt")"
OUT="$PARENT/confirmation"
"$PYTHON" -m scripts.discovery.capture prepare --out "$OUT"
for run in 1 2 3; do
    "$PYTHON" -m scripts.discovery.idle_check --output "$OUT/idle_timing_${run}.json"
    "$PYTHON" -m scripts.discovery.capture timing --out "$OUT" --run-id "$run" > "$OUT/timing_${run}.log" 2>&1
    tail -n 2 "$OUT/timing_${run}.log"
    "$PYTHON" -m scripts.discovery.idle_check --output "$OUT/idle_profile_${run}.json"
    "$NSYS" profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false --discard-environment=true --output="$OUT/native_${run}" "$PYTHON" -m scripts.discovery.capture profile --out "$OUT" --run-id "$run" > "$OUT/nsys_${run}.log" 2>&1
    "$NSYS" export --type=sqlite --output="$OUT/native_${run}.sqlite" "$OUT/native_${run}.nsys-rep" > "$OUT/export_${run}.log" 2>&1
    printf 'Confirmation run %s exported\n' "$run"
done
