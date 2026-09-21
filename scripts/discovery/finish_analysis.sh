#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../profiling/environment.sh"
cd "$ROOT"
OUT="$(cat "$ROOT/scripts/discovery/latest_result.txt")"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m scripts.discovery.analyze --out "$OUT"
"$PYTHON" -m scripts.discovery.analyze --out "$OUT/confirmation"
"$PYTHON" -m scripts.discovery.verify --out "$OUT"
