#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
OUT="$ROOT/results/profiling_20260918"
mkdir -p "$OUT"
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output "$OUT/native_decode_nsys" \
    "$PYTHON" "$ROOT/scripts/profiling/capture_decode.py" --tool cuda \
    --baseline "$BASELINE" --output "$OUT/nsys_capture.json" --capture-steps 16 \
    2>&1 | tee "$OUT/nsys_capture.log"
