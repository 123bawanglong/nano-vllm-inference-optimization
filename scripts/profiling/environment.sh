#!/usr/bin/env bash
# Shared environment for the optional low-level shell tools.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/nano-vllm-fusion/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/nano-vllm-fusion/triton}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
export NSYS="${NSYS:-nsys}"
BASELINE="${QK_BASELINE_DIR:-$ROOT/results/baseline_20260918_170614}"
