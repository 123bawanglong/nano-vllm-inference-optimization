#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
OUT="$ROOT/results/profiling_20260918"
"$PYTHON" "$ROOT/scripts/profiling/capture_decode.py" --tool torch \
    --baseline "$BASELINE" --output "$OUT/torch_capture.json" --capture-steps 16 \
    2>&1 | tee "$OUT/torch_capture.log"
