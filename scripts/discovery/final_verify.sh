#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../profiling/environment.sh"
cd "$ROOT"
OUT="$(cat "$ROOT/scripts/discovery/latest_result.txt")"
"$PYTHON" -m scripts.discovery.verify --out "$OUT"
"$PYTHON" -m unittest discover -s tests -v
git status --short --branch
