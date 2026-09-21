#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../profiling/environment.sh"
cd "$ROOT"
OUT="$ROOT/results/discovery_$(date +%Y%m%d_%H%M%S)"
"$PYTHON" -m scripts.discovery.capture prepare --out "$OUT"
printf '%s\n' "$OUT" > "$ROOT/scripts/discovery/latest_result.txt"
for run in 1 2 3; do
    "$PYTHON" -m scripts.discovery.capture timing --out "$OUT" --run-id "$run" 2>&1 | tee "$OUT/timing_${run}.log"
    "$NSYS" profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false --discard-environment=true --output="$OUT/native_${run}" "$PYTHON" -m scripts.discovery.capture profile --out "$OUT" --run-id "$run" 2>&1 | tee "$OUT/nsys_${run}.log"
    "$NSYS" export --type=sqlite --output="$OUT/native_${run}.sqlite" "$OUT/native_${run}.nsys-rep" 2>&1 | tee "$OUT/export_${run}.log"
done
