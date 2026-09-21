#!/usr/bin/env bash
# Source this only from profiling scripts; do not alter the frozen baseline runner.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCHINDUCTOR_CACHE_DIR=/home/xietaibo/.cache/nano-vllm-fusion-learning/inductor
export TRITON_CACHE_DIR=/home/xietaibo/.cache/nano-vllm-fusion-learning/triton
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
PYTHON=/home/xietaibo/nano-vllm/.venv/bin/python
BASELINE="$ROOT/results/baseline_20260918_170614"
