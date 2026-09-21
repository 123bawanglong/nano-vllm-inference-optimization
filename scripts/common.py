"""Shared workload, trace and numerical helpers for the fusion experiment."""
import hashlib
import json
import random
from scripts.runtime import ROOT, BASELINE
from scripts import baseline


def make_workloads():
    rows = json.loads((ROOT / 'docs/results/workloads.json').read_text())
    for i, prompt in enumerate((64, 256)):
        for repeat in range(3):
            seed = 20260918 + (7 + i) * 100 + repeat
            rng = random.Random(seed)
            rows.append(dict(name=f'b2_p{prompt}_o256', batch=2, prompt=prompt,
                output=256, repeat=repeat, seed=seed,
                prompts=[[rng.randrange(1000, 100000) for _ in range(prompt)] for _ in range(2)]))
    return sorted(rows, key=lambda x: (x['batch'], x['prompt'], x['repeat']))


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


def digest(tensor):
    import torch
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def selected_steps(item):
    selected = {0, 1, 17, item['output'] - 1}
    selected.update(i for i in range(1, item['output']) if (item['prompt'] + i - 1) % 256 in (0, 255))
    return selected


def hooks(llm):
    audit = dict(cache_hits=0, preemptions=0)
    allocate = llm.scheduler.block_manager.allocate
    def checked_allocate(seq, cached):
        audit['cache_hits'] += cached
        return allocate(seq, cached)
    llm.scheduler.block_manager.allocate = checked_allocate
    preempt = llm.scheduler.preempt
    def checked_preempt(seq):
        audit['preemptions'] += 1
        return preempt(seq)
    llm.scheduler.preempt = checked_preempt
    return audit
