#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
OUT="$PWD/results/full_project_20260919"
run_checked() {
  local label="$1"
  shift
  if ! bash scripts/full_project/run.sh "$@" > "$OUT/$label.log" 2>&1; then
    tail -n 40 "$OUT/$label.log"
    exit 1
  fi
  tail -n 8 "$OUT/$label.log"
}
run_checked micro micro
run_checked numerical_native numerical --variant native
run_checked numerical_fused numerical --variant fused
run_checked integration_native trace --variant native
run_checked integration_fused trace --variant fused
