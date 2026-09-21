#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
source scripts/profiling/environment.sh
OUT="$ROOT/results/full_project_20260919"
NSYS=/home/xietaibo/tools/nsight-systems-cli-2026.5.1/opt/nvidia/nsight-systems-cli/2026.5.1/bin/nsys
MODE="${1:?discovery|ncu|ab}"
if [[ "$MODE" == discovery ]]; then
  FRESH="$OUT/discovery_recheck"
  "$PYTHON" -m scripts.discovery.capture prepare --out "$FRESH"
  "$PYTHON" -m scripts.discovery.idle_check --output "$FRESH/idle.json"
  "$NSYS" profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false --discard-environment=true --output="$FRESH/native_1" "$PYTHON" -m scripts.discovery.capture profile --out "$FRESH" --run-id 1 > "$FRESH/nsys_1.log" 2>&1
  "$NSYS" export --type=sqlite --output="$FRESH/native_1.sqlite" "$FRESH/native_1.nsys-rep" > "$FRESH/export_1.log" 2>&1
  "$PYTHON" -m scripts.discovery.analyze --out "$FRESH" --run-id 1
elif [[ "$MODE" == ncu ]]; then
  for variant in native fused; do
    "$CUDA_HOME/bin/ncu" --target-processes application-only --profile-from-start off --graph-profiling node --replay-mode kernel --cache-control all --clock-control none --kernel-name-base function \
      --section SpeedOfLight --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
      --export "$OUT/${variant}_qk_rope" "$PYTHON" -m scripts.full_project.study ncu --variant "$variant" > "$OUT/ncu_${variant}.log" 2>&1
    "$CUDA_HOME/bin/ncu" --import "$OUT/${variant}_qk_rope.ncu-rep" --page raw --csv > "$OUT/ncu_${variant}_raw.csv"
    "$CUDA_HOME/bin/ncu" --import "$OUT/${variant}_qk_rope.ncu-rep" --page details --csv > "$OUT/ncu_${variant}_details.csv"
  done
elif [[ "$MODE" == ab ]]; then
  for pair in $(seq 1 10); do
    if (( pair % 2 )); then variants="native fused"; else variants="fused native"; fi
    for variant in $variants; do
      "$PYTHON" -m scripts.discovery.idle_check --output "$OUT/idle_ab_${pair}_${variant}.json"
      bash scripts/full_project/run.sh timing --variant "$variant" --pair "$pair" > "$OUT/ab_${pair}_${variant}.log" 2>&1
      tail -n 1 "$OUT/ab_${pair}_${variant}.log"
    done
  done
else
  echo "Unknown mode: $MODE" >&2
  exit 2
fi
