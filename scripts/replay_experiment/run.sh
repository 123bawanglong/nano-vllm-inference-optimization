#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
PROJECT="$PWD"
cd "$PROJECT/results/closed_loop_20260921/replay"
source scripts/profiling/environment.sh
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4
case "${1:?mode}" in
  discovery) bash scripts/full_project/collect.sh discovery ;;
  prepare) bash scripts/full_project/run.sh prepare ;;
  numerical-native) bash scripts/full_project/run.sh numerical --variant native > "$PROJECT/results/closed_loop_20260921/numerical_native.log" 2>&1 ;;
  v1|v2)
    if [[ "$1" == v1 ]]; then tag=v1_fp32_intermediate; else tag=v2_bf16_boundary; fi
    "$PYTHON" "$PROJECT/scripts/replay_experiment/variant_numerical.py" "$tag" > "$PROJECT/results/closed_loop_20260921/${tag}.log" 2>&1
    tail -n 10 "$PROJECT/results/closed_loop_20260921/${tag}.log"
    ;;
  validate-final)
    "$PYTHON" scripts/full_project/kernel_validation.py --label final > "$PROJECT/results/closed_loop_20260921/kernel_validation.log" 2>&1
    bash scripts/full_project/run.sh micro > results/full_project_20260919/micro.log 2>&1
    bash scripts/full_project/run.sh numerical --variant fused > results/full_project_20260919/numerical_fused.log 2>&1
    bash scripts/full_project/run.sh trace --variant native > results/full_project_20260919/integration_native.log 2>&1
    bash scripts/full_project/run.sh trace --variant fused > results/full_project_20260919/integration_fused.log 2>&1
    tail -n 8 results/full_project_20260919/numerical_fused.log
    ;;
  diagnostic)
    "$PYTHON" scripts/full_project/kernel_model_diagnostic.py --label replay_v2 --mode standalone > "$PROJECT/results/closed_loop_20260921/diagnostic_v2.log" 2>&1
    "$PYTHON" scripts/full_project/kernel_model_diagnostic.py --label replay_v3 --mode native_graph > "$PROJECT/results/closed_loop_20260921/diagnostic_v3.log" 2>&1
    tail -n 1 "$PROJECT/results/closed_loop_20260921/diagnostic_v2.log"
    tail -n 1 "$PROJECT/results/closed_loop_20260921/diagnostic_v3.log"
    ;;
  ab)
    bash scripts/full_project/collect.sh ab
    "$PYTHON" scripts/full_project/analyze_ab.py
    ;;
  ncu-native|ncu-final)
    OUT="$PROJECT/results/closed_loop_20260921"
    tag="${1#ncu-}"
    "$PYTHON" -m scripts.discovery.idle_check --output "$OUT/idle_ncu_${tag}.json"
    "$CUDA_HOME/bin/ncu" --target-processes application-only --profile-from-start off --graph-profiling node --replay-mode kernel --cache-control all --clock-control none --kernel-name-base function --import-source yes \
      --section SpeedOfLight --section SpeedOfLight_HierarchicalSingleRooflineChart --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section SourceCounters \
      --export "$OUT/${tag}_full" "$PYTHON" "$PROJECT/scripts/replay_experiment/native_ncu.py" "$tag" > "$OUT/ncu_${tag}.log" 2>&1
    "$CUDA_HOME/bin/ncu" --import "$OUT/${tag}_full.ncu-rep" --page raw --csv > "$OUT/${tag}_raw.csv"
    "$CUDA_HOME/bin/ncu" --import "$OUT/${tag}_full.ncu-rep" --page details --print-details all --csv > "$OUT/${tag}_details.csv"
    "$CUDA_HOME/bin/ncu" --import "$OUT/${tag}_full.ncu-rep" --page source --print-source sass > "$OUT/${tag}_sass.txt"
    echo "NCU $tag complete"
    ;;
  *) echo "Unknown mode: $1" >&2; exit 2 ;;
esac
