#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
source scripts/profiling/environment.sh
OUT="$ROOT/results/roofline_supplement_20260920"
mkdir -p "$OUT"
REPLAY="$OUT/replay_source"
if [[ ! -d "$REPLAY" ]]; then
  "$PYTHON" scripts/full_project/create_reproduction.py "$REPLAY"
  cp /home/xietaibo/nano-vllm-reproduction-layout-check-20260919/src/qk_norm_rope/kernel.cu "$REPLAY/src/qk_norm_rope/kernel.cu"
  cp results/full_project_20260919/{manifest,workloads}.json "$REPLAY/results/full_project_20260919/"
fi
cd "$REPLAY"
source scripts/profiling/environment.sh
cp '/home/xietaibo/Documents/NVIDIA Nsight Compute/2025.1.1/Sections/SpeedOfLight_HierarchicalSingleRooflineChart.section' "$OUT/SingleRoofline.section"
for variant in native fused; do
  "$PYTHON" -m scripts.discovery.idle_check --output "$OUT/idle_${variant}.json"
  "$CUDA_HOME/bin/ncu" --target-processes application-only --profile-from-start off --graph-profiling node --replay-mode kernel --cache-control all --clock-control none --kernel-name-base function \
    --section SpeedOfLight --section SpeedOfLight_HierarchicalSingleRooflineChart --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
    --export "$OUT/${variant}_roofline" "$PYTHON" -m scripts.full_project.study ncu --variant "$variant" > "$OUT/${variant}.log" 2>&1
  "$CUDA_HOME/bin/ncu" --import "$OUT/${variant}_roofline.ncu-rep" --page raw --csv > "$OUT/${variant}_raw.csv"
  "$CUDA_HOME/bin/ncu" --import "$OUT/${variant}_roofline.ncu-rep" --page details --csv > "$OUT/${variant}_details.csv"
  echo "Completed $variant Roofline"
done
