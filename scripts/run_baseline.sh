#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCHINDUCTOR_CACHE_DIR=/home/xietaibo/.cache/nano-vllm-fusion-learning/inductor
export TRITON_CACHE_DIR=/home/xietaibo/.cache/nano-vllm-fusion-learning/triton
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=/home/xietaibo/nano-vllm/.venv/bin/python
OUT="${1:-$ROOT/results/baseline_$(date +%Y%m%d_%H%M%S)}"
"$PYTHON" -m unittest discover -s "$ROOT/tests" -v
"$PYTHON" "$ROOT/benchmarks/baseline.py" prepare --out "$OUT"
"$PYTHON" "$ROOT/benchmarks/baseline.py" inspect --out "$OUT" 2>&1 | tee "$OUT/layout_smoke.log"
for run in 1 2 3; do
    "$PYTHON" "$ROOT/benchmarks/baseline.py" sample --out "$OUT" --run-id "$run" 2>&1 | tee "$OUT/process_$run.log"
done
echo "Baseline raw evidence: $OUT"
