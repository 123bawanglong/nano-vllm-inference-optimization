"""Paths shared by experiment modules; the public CLI sets QK_* per run."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('QK_RUN_DIR', ROOT / 'results/local')).expanduser().resolve()
BASELINE = Path(os.environ.get('QK_BASELINE_DIR', ROOT / 'results/baseline_20260918_170614')).expanduser().resolve()
MODEL = Path(os.environ.get('QK_MODEL', ROOT / 'models/Qwen3-0.6B')).expanduser().resolve()
# Immutable captured inputs are separate from fresh measurements.
FIXTURE = ROOT / 'results/compile_boundary_20260918_233238/real_first_layer_fixtures.pt'
MODEL_FIXTURE = ROOT / 'results/full_project_20260919/kernel_model_fixtures_before.pt'
