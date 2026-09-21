#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
OUT="$ROOT/results/profiling_20260918"
# The first matching node is layer0 input RMSNorm. Skip it, then collect
# layer0 Q norm, K norm, Q RoPE, K RoPE and the native KV cache store.
# Attribution must be verified against the actual timeline and launch sizes.
ncu --target-processes application-only --profile-from-start off \
    --graph-profiling node --replay-mode kernel --cache-control all --clock-control none \
    --kernel-name-base function \
    --kernel-name 'regex:triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0|triton_poi_fused__to_copy_add_cat_index_mul_split_sub_[01]|store_kvcache_kernel' \
    --launch-skip 1 --launch-count 5 \
    --section SpeedOfLight --section LaunchStats --section Occupancy \
    --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
    --section SchedulerStats --section WarpStateStats \
    --export "$OUT/native_qk_rope_cache_graph" \
    "$PYTHON" "$ROOT/scripts/profiling/capture_decode.py" --tool cuda \
    --baseline "$BASELINE" --output "$OUT/ncu_capture.json" --capture-steps 1 \
    2>&1 | tee "$OUT/ncu_capture.log"
ncu --import "$OUT/native_qk_rope_cache_graph.ncu-rep" --csv --page raw > "$OUT/ncu_raw.csv"
ncu --import "$OUT/native_qk_rope_cache_graph.ncu-rep" --csv --page details > "$OUT/ncu_details.csv"
