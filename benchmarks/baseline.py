"""Stage-0 native baseline. No replacement kernel is implemented here."""
import argparse
import atexit
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
from scripts.runtime import MODEL
CONFIG = dict(max_num_seqs=8, max_num_batched_tokens=4096,
              max_model_len=4096, gpu_memory_utilization=0.75,
              tensor_parallel_size=1, enforce_eager=False,
              kvcache_block_size=256, num_kvcache_blocks=64)
CASES = [dict(name=f'b{b}_p{p}_o256', batch=b, prompt=p, output=256)
         for b in (1, 4, 8) for p in (64, 256)]
CASES.append(dict(name='b1_p2048_o32', batch=1, prompt=2048, output=32))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True, cwd=ROOT).strip()


def sources():
    paths = list((ROOT / 'nanovllm').rglob('*.py'))
    paths += list((ROOT / 'benchmarks').glob('*.py'))
    paths += list((ROOT / 'scripts').glob('*.sh'))
    paths += [ROOT / 'scripts/runtime.py', ROOT / 'scripts/experiment.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def gpu():
    return command('nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw', '--format=csv')


def prepare(out):
    import torch
    import nanovllm
    import flash_attn
    assert Path(nanovllm.__file__).resolve().is_relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    fixtures = []
    for i, case in enumerate(CASES):
        for repeat in range(3):  # one warmup, two measured requests per case
            seed = 20260918 + i * 100 + repeat
            rng = random.Random(seed)
            fixtures.append(dict(**case, repeat=repeat, seed=seed,
                                 prompts=[[rng.randrange(1000, 100000) for _ in range(case['prompt'])]
                                          for _ in range(case['batch'])]))
    dump(out / 'workloads.json', fixtures)
    model_files = {p.name: sha(p) for p in sorted(MODEL.iterdir())
                   if p.is_file() and p.suffix in ('.json', '.safetensors', '.jinja')}
    manifest = dict(upstream=command('git', 'rev-parse', 'HEAD'),
                    branch=command('git', 'branch', '--show-current'),
                    source_sha256=sources(), model=str(MODEL), model_sha256=model_files,
                    config=CONFIG, workloads_sha256=sha(out / 'workloads.json'),
                    sampling=dict(temperature=1.0, ignore_eos=True),
                    attention_backend='flash_attn', dtype='torch.bfloat16',
                    process_count=3, repeats_per_case=2)
    dump(out / 'baseline_manifest.json', manifest)
    dump(out / 'environment.json', dict(python=platform.python_version(), platform=platform.platform(),
         torch_cuda=torch.version.cuda, gpu=gpu(), capability=torch.cuda.get_device_capability(),
         packages={p: importlib.metadata.version(p) for p in ('torch','triton','transformers','flash-attn')},
         nanovllm_path=nanovllm.__file__, flash_attn_path=flash_attn.__file__,
         compiler_cache=os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
         nvcc=command(str(Path(os.environ['CUDA_HOME'])/'bin/nvcc') if os.environ.get('CUDA_HOME') else 'nvcc', '--version')))
    (out / 'environment_pip_freeze.txt').write_text(command(str(Path(os.sys.executable)), '-m', 'pip', 'freeze') + '\n')
    (out / 'baseline.patch').write_text(command('git', 'diff', '--', 'nanovllm/engine/model_runner.py') + '\n')
    print(f'Prepared {out}', flush=True)


def check_manifest(out):
    manifest = json.loads((out / 'baseline_manifest.json').read_text())
    assert manifest['source_sha256'] == sources(), 'Source changed after baseline was frozen'
    assert manifest['workloads_sha256'] == sha(out / 'workloads.json')
    assert manifest['config'] == CONFIG
    assert Path(manifest['model']).resolve() == MODEL, 'Model path changed after baseline was frozen'
    return manifest


def engine(eager=False):
    import torch
    import nanovllm
    from nanovllm import LLM
    assert Path(nanovllm.__file__).resolve().is_relative_to(ROOT)
    config = dict(CONFIG, enforce_eager=eager)
    llm = LLM(str(MODEL), **config)
    assert llm.model_runner.config.num_kvcache_blocks == 64
    assert next(llm.model_runner.model.parameters()).dtype == torch.bfloat16
    if not eager:
        assert sorted(llm.model_runner.graphs) == [1, 2, 4, 8]
    return llm


def close(llm):
    atexit.unregister(llm.exit)
    llm.exit()


def request(llm, item):
    import torch
    from nanovllm import SamplingParams
    torch.manual_seed(item['seed'])
    torch.cuda.synchronize()
    start = time.perf_counter()
    for prompt in item['prompts']:
        llm.add_request(prompt, SamplingParams(temperature=1.0, max_tokens=item['output'], ignore_eos=True))
    phases = {'prefill': [], 'decode': []}
    counts = {'prefill': 0, 'decode': 0}
    tokens = {}
    first_token_ms = None
    while not llm.is_finished():
        t = time.perf_counter()
        outputs, n = llm.step()  # native sampler's .tolist() synchronizes each step
        elapsed = time.perf_counter() - t
        phase = 'prefill' if n > 0 else 'decode'
        phases[phase].append(elapsed)
        counts[phase] += abs(n)
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - start) * 1000
        tokens.update(outputs)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    ordered = [tokens[k] for k in sorted(tokens)]
    assert len(ordered) == item['batch'] and all(len(x) == item['output'] for x in ordered)
    assert counts == dict(prefill=item['batch'] * item['prompt'], decode=item['batch'] * (item['output'] - 1))
    assert len(phases['prefill']) == 1 and len(phases['decode']) == item['output'] - 1
    assert not llm.scheduler.block_manager.used_block_ids
    return dict(case=item['name'], repeat=item['repeat'], seed=item['seed'],
                batch=item['batch'], input_length=item['prompt'], output_length=item['output'],
                e2e_ms=seconds * 1000, engine_first_token_ms=first_token_ms,
                output_tokens_per_s=item['batch'] * item['output'] / seconds,
                decode_step_ms=1000 * sum(phases['decode']) / len(phases['decode']),
                prefill_step_ms=1000 * sum(phases['prefill']), counts=counts,
                step_ms={k: [1000*x for x in v] for k,v in phases.items()},
                output_token_ids=ordered)


def sample(out, run_id):
    check_manifest(out)
    path = out / f'process_{run_id}.json'
    assert not path.exists(), f'Refusing to overwrite {path}'
    before = gpu()
    start = time.perf_counter()
    llm = engine()
    init_seconds = time.perf_counter() - start
    # Infrequent host-side guards; identical for every future comparison arm.
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
    results = []
    try:
        for item in json.loads((out / 'workloads.json').read_text()):
            row = request(llm, item)
            row['measured'] = item['repeat'] > 0
            results.append(row)
            assert audit == dict(cache_hits=0, preemptions=0), audit
            print(f"process={run_id} {row['case']} repeat={row['repeat']} e2e={row['e2e_ms']:.2f}ms decode_step={row['decode_step_ms']:.3f}ms", flush=True)
        dump(path, dict(run_id=run_id, init_seconds=init_seconds, gpu_before=before,
                       gpu_after=gpu(), audit=audit, config=CONFIG,
                       manifest_sha256=sha(out / 'baseline_manifest.json'),
                       source_sha256=sources(), rows=results))
    finally:
        close(llm)


def inspect(out):
    """Untimed eager layout audit and native numerical fixtures; separate process."""
    import torch
    from nanovllm.utils.context import get_context
    check_manifest(out)
    llm = engine(eager=True)
    attn = llm.model_runner.model.model.layers[0].self_attn
    observations = {}
    def tensor(t):
        return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                    contiguous=t.is_contiguous(), storage_offset=t.storage_offset())
    def hook(name):
        def capture(module, inputs):
            ctx = get_context()
            key = f'{name}/{"prefill" if ctx.is_prefill else "decode"}/tokens{inputs[-1].shape[0]}'
            if key not in observations:
                observations[key] = dict(inputs=[tensor(t) for t in inputs if isinstance(t, torch.Tensor)],
                    slots=tensor(ctx.slot_mapping), slot_values=ctx.slot_mapping.cpu().tolist())
        return capture
    handles = [m.register_forward_pre_hook(hook(name)) for name,m in
               [('q_norm',attn.q_norm),('k_norm',attn.k_norm),('rope',attn.rotary_emb),('attention',attn.attn)]]
    original_run = llm.model_runner.run_model
    logits = []
    def capture_logits(input_ids, positions, is_prefill):
        y = original_run(input_ids, positions, is_prefill)
        assert torch.isfinite(y).all().item()
        logits.append(dict(input_ids=input_ids.cpu(), positions=positions.cpu(),
                           is_prefill=is_prefill, logits=y.cpu()))
        return y
    llm.model_runner.run_model = capture_logits
    try:
        fixtures = json.loads((out / 'workloads.json').read_text())
        records = []
        for batch in (1,4,8):
            item = dict(next(x for x in fixtures if x['batch'] == batch and x['prompt'] == 64 and x['repeat'] == 0))
            item['output'] = 3
            item['name'] += '_layout_smoke'
            records.append(request(llm,item))
        torch.save(logits, out / 'native_eager_logits.pt')
        dump(out / 'layout_smoke.json', dict(eager_audit_only=True, timed_benchmark=False,
             layer=0, tensors=observations, k_cache=tensor(attn.attn.k_cache),
             v_cache=tensor(attn.attn.v_cache), q_heads=attn.num_heads,
             kv_heads=attn.num_kv_heads, head_dim=attn.head_dim,
             logits_all_finite=True, logits_fixture_sha256=sha(out / 'native_eager_logits.pt'),
             records=records))
    finally:
        for h in handles:
            h.remove()
        close(llm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','inspect','sample'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run-id', type=int, default=1)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.out)
    elif args.mode == 'inspect':
        inspect(args.out)
    else:
        sample(args.out, args.run_id)
