#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
source scripts/profiling/environment.sh
test -f REPRODUCTION_INPUTS.json
test ! -f results/full_project_20260919/manifest.json
"$PYTHON" scripts/full_project/kernel_validation.py --label final
bash scripts/full_project/collect.sh discovery
bash scripts/full_project/run.sh prepare
bash scripts/full_project/validate.sh
bash scripts/full_project/collect.sh ncu
bash scripts/full_project/collect.sh ab
"$PYTHON" scripts/full_project/analyze_ab.py
