#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
source scripts/profiling/environment.sh
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4
"$PYTHON" -m scripts.full_project.study "$@"
