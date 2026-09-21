#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../profiling/environment.sh"
cd "$ROOT"
MICRO_OUT="$(cat "$ROOT/scripts/compile_boundary/latest_result.txt")"
OUT="$ROOT/results/compile_boundary_model_$(date +%Y%m%d_%H%M%S)"
export EXPERIMENT_OUT="$OUT" MICRO_OUT
"$PYTHON" - <<'PY'
import os,json,shutil
from pathlib import Path
from scripts.compile_boundary.common import baseline,prepare
out=Path(os.environ['EXPERIMENT_OUT'])
micro=Path(os.environ['MICRO_OUT'])
prepare(out)
assert json.loads((micro/'micro.json').read_text())['complete']
shutil.copy2(micro/'micro.json',out/'micro.json')
shutil.copy2(Path('scripts/compile_boundary/PROTOCOL.md'),out/'PROTOCOL.md')
shutil.copy2(Path('scripts/compile_boundary/DIAGNOSTIC_EXTENSION.md'),out/'DIAGNOSTIC_EXTENSION.md')
baseline.dump(out/'micro_provenance.json',dict(directory=str(micro),
    manifest_sha256=baseline.sha(micro/'manifest.json'),result_sha256=baseline.sha(micro/'micro.json'),
    extension_reason='Exact gate failed: numerical and hotspot diagnostics only; skip formal A/B performance'))
PY
printf '%s\n' "$OUT" > "$ROOT/scripts/compile_boundary/latest_model_result.txt"
for mode in numerical profile; do
    for variant in native joint; do
        "$PYTHON" -m scripts.compile_boundary.model_study "$mode" --variant "$variant" --out "$OUT" 2>&1 | tee "$OUT/${mode}_${variant}.log"
    done
done
