"""Paths shared by experiment modules; the public CLI sets QK_* per run."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('QK_RUN_DIR', ROOT / 'results/local')).expanduser().resolve()
BASELINE = Path(os.environ.get('QK_BASELINE_DIR', OUT / 'baseline')).expanduser().resolve()
MODEL = Path(os.environ.get('QK_MODEL', ROOT / 'models/Qwen3-0.6B')).expanduser().resolve()
# Immutable captured inputs are separate from fresh measurements.
FIXTURE = ROOT / 'tests/fixtures/real_first_layer_fixtures.pt'
MODEL_FIXTURE = ROOT / 'tests/fixtures/kernel_model_fixtures_before.pt'
