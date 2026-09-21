"""Experiment adapter; never changes the frozen model source or writes a kernel."""
import importlib.util
import inspect
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'results/baseline_20260918_170614'
spec = importlib.util.spec_from_file_location('native_baseline', ROOT / 'benchmarks/baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def make_workloads():
    rows = json.loads((BASELINE / 'workloads.json').read_text())
    for i, prompt in enumerate((64, 256)):
        for repeat in range(3):
            seed = 20260918 + (7 + i) * 100 + repeat
            rng = random.Random(seed)
            rows.append(dict(name=f'b2_p{prompt}_o256', batch=2, prompt=prompt,
                output=256, repeat=repeat, seed=seed,
                prompts=[[rng.randrange(1000, 100000) for _ in range(prompt)] for _ in range(2)]))
    return sorted(rows, key=lambda x: (x['batch'], x['prompt'], x['repeat']))


def experiment_sources():
    return {str(p.relative_to(ROOT)): baseline.sha(p)
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def prepare(out):
    import torch
    baseline.check_manifest(BASELINE)
    out.mkdir(parents=True, exist_ok=False)
    baseline.dump(out / 'workloads.json', make_workloads())
    baseline.dump(out / 'manifest.json', dict(
        baseline_manifest_sha256=baseline.sha(BASELINE / 'baseline_manifest.json'),
        baseline_sources=baseline.sources(), experiment_sources=experiment_sources(),
        config=baseline.CONFIG, workload_sha256=baseline.sha(out / 'workloads.json'),
        protocol_sha256=baseline.sha(Path(__file__).with_name('PROTOCOL.md')),
        torch=torch.__version__, gpu=baseline.gpu(),
        variants=['native', 'joint_decode'], handwritten_kernel=False,
        joint_prefill=False, kv_store_changed=False))


def check(out):
    baseline.check_manifest(BASELINE)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['experiment_sources'] == experiment_sources(), 'Experiment code changed'
    assert manifest['workload_sha256'] == baseline.sha(out / 'workloads.json')
    return manifest


def make_joint():
    import torch
    from nanovllm.layers.layernorm import RMSNorm
    from nanovllm.layers.rotary_embedding import RotaryEmbedding
    raw_norm = inspect.unwrap(RMSNorm.rms_forward)
    raw_rope = inspect.unwrap(RotaryEmbedding.forward)
    assert raw_norm is not RMSNorm.rms_forward and raw_rope is not RotaryEmbedding.forward

    def joint(q, k, positions, q_norm, k_norm, rope):
        return raw_rope(rope, positions, raw_norm(q_norm, q), raw_norm(k_norm, k))

    return torch.compile(joint, fullgraph=True, dynamic=False)


def install_joint():
    from nanovllm.models.qwen3 import Qwen3Attention
    from nanovllm.utils.context import get_context
    original = Qwen3Attention.forward
    joint = make_joint()

    def forward(self, positions, hidden_states):
        if get_context().is_prefill or self.qkv_bias:
            return original(self, positions, hidden_states)
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = joint(q, k, positions, self.q_norm, self.k_norm, self.rotary_emb)
        return self.o_proj(self.attn(q, k, v).flatten(1, -1))

    Qwen3Attention.forward = forward


def trace_kernels(path):
    import gzip
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        trace = json.load(f)
    events = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']
    return [dict(name=e['name'], us=e['dur'], ts=e['ts'],
                 grid=e.get('args', {}).get('grid'), block=e.get('args', {}).get('block'),
                 stream=e.get('args', {}).get('stream')) for e in events]


def save_trace(profiler, path):
    import gzip
    import shutil
    raw = path.with_suffix('')
    profiler.export_chrome_trace(str(raw))
    with raw.open('rb') as source, gzip.open(path, 'wb') as dest:
        shutil.copyfileobj(source, dest)
    raw.unlink()


def error_metrics(actual, expected):
    import torch
    diff = (actual.float() - expected.float()).abs()
    return dict(exact=torch.equal(actual, expected), max_abs=diff.max().item(),
                mean_abs=diff.mean().item(), different=int((actual != expected).sum().item()),
                elements=actual.numel(), finite=bool(torch.isfinite(actual).all().item()))
