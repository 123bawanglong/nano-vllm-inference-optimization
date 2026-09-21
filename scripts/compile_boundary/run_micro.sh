#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../profiling/environment.sh"
cd "$ROOT"
"$PYTHON" -m unittest discover -s tests -p test_compile_boundary.py -v
OUT="$ROOT/results/compile_boundary_$(date +%Y%m%d_%H%M%S)"
export EXPERIMENT_OUT="$OUT"
"$PYTHON" -c 'import os; from pathlib import Path; from scripts.compile_boundary.common import prepare; prepare(Path(os.environ["EXPERIMENT_OUT"]))'
printf '%s\n' "$OUT" > "$ROOT/scripts/compile_boundary/latest_result.txt"
"$PYTHON" -m scripts.compile_boundary.micro --out "$OUT" 2>&1 | tee "$OUT/micro.log"
