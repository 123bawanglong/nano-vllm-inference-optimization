#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../profiling/environment.sh"
cd "$ROOT"
OUT="$(cat "$ROOT/scripts/compile_boundary/latest_model_result.txt")"
"$PYTHON" -m scripts.compile_boundary.reporting.summarize --out "$OUT"
"$PYTHON" -m unittest discover -s tests -v
git status --short --branch
