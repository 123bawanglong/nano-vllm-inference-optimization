#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../profiling/environment.sh"
cd "$ROOT"
OUT="$(cat "$ROOT/scripts/discovery/latest_result.txt")"
"$PYTHON" -m scripts.discovery.analyze --out "$OUT" "$@"
