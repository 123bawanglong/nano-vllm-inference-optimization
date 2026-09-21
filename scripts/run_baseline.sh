#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/profiling/environment.sh"
cd "$ROOT"
OUT="${1:-$ROOT/results/baseline_$(date +%Y%m%d_%H%M%S)}"
"$PYTHON" -m unittest discover -s "$ROOT/tests" -v
"$PYTHON" "$ROOT/benchmarks/baseline.py" prepare --out "$OUT"
"$PYTHON" "$ROOT/benchmarks/baseline.py" inspect --out "$OUT" 2>&1 | tee "$OUT/layout_smoke.log"
for run in 1 2 3; do
    "$PYTHON" "$ROOT/benchmarks/baseline.py" sample --out "$OUT" --run-id "$run" 2>&1 | tee "$OUT/process_$run.log"
done
echo "Baseline raw evidence: $OUT"
