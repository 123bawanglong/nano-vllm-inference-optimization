"""Sequential runs prevent competing benchmarks on the same GPU."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'src/nano-vllm-cuda'
OUT = REPO / 'results/fusion'


def run(name, argv):
    print('START', name, flush=True)
    with (OUT / f'{name}.log').open('w') as log:
        process = subprocess.Popen([sys.executable, *map(str, argv)], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith(('IMPORT', 'baseline_', 'cuda_', '{"op"', 'FAILED', 'OK', 'Traceback')):
                print(line.rstrip(), flush=True)
        status = process.wait()
    if status:
        print((OUT / f'{name}.log').read_text()[-12000:], flush=True)
        raise SystemExit(f'{name} failed: {status}')
    print('DONE', name, flush=True)


def verify_source():
    manifest = json.loads((REPO / 'docs/upstream_manifest.json').read_text())
    source = REPO / manifest['source']
    changed = [name for name, checksum in manifest['sha256'].items()
               if hashlib.sha256((source / name).read_bytes()).hexdigest() != checksum]
    assert not changed, f'Original source changed: {changed}'
    return {'original_unchanged': True, 'checked_files': len(manifest['sha256']), 'commit': manifest['commit']}


def compare_logits():
    import torch
    comparisons = []
    for graph in (0, 1):
        base = torch.load(OUT / f'baseline_all_graph{graph}_logits.pt', weights_only=True)
        for ops in (('all',) if graph == 0 else ('all', 'add_rms', 'silu')):
            custom = torch.load(OUT / f'cuda_{ops}_graph{graph}_logits.pt', weights_only=True)
            assert len(base) == len(custom) == 4
            for step, (a, b) in enumerate(zip(base, custom)):
                assert torch.isfinite(b).all() and a.shape == b.shape
                cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
                relative_l2 = ((a - b).norm() / a.norm()).item()
                item = {'graph': graph, 'ops': ops, 'step': step,
                        'max_abs': (a-b).abs().max().item(), 'relative_l2': relative_l2,
                        'min_cosine': cosine, 'top1_match_fraction': (a.argmax(-1) == b.argmax(-1)).float().mean().item()}
                comparisons.append(item)
                # BF16 full-model equivalence, not bitwise identity. Fixed teacher inputs.
                assert cosine > 0.999 and relative_l2 < 0.02, item
        # Counterbalanced independent processes should satisfy the same checks.
        reverse_base = OUT / f'baseline_all_graph{graph}_reverse_logits.pt'
        reverse_custom = OUT / f'cuda_all_graph{graph}_reverse_logits.pt'
        if reverse_base.exists() and reverse_custom.exists():
            reverse_a = torch.load(reverse_base, weights_only=True)
            reverse_b = torch.load(reverse_custom, weights_only=True)
            assert len(reverse_a) == len(reverse_b) == 4
            for step, (a, b) in enumerate(zip(reverse_a, reverse_b)):
                assert torch.isfinite(b).all() and a.shape == b.shape
                cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
                relative_l2 = ((a-b).norm()/a.norm()).item()
                item = {'graph': graph, 'ops': 'all', 'label': 'reverse', 'step': step,
                        'max_abs': (a-b).abs().max().item(), 'relative_l2': relative_l2,
                        'min_cosine': cosine, 'top1_match_fraction': (a.argmax(-1)==b.argmax(-1)).float().mean().item()}
                comparisons.append(item)
                assert cosine > 0.999 and relative_l2 < 0.02, item
    (OUT / 'verification.json').write_text(json.dumps({**verify_source(), 'logit_comparisons': comparisons}, indent=2))
    print('Logits and original-source verification passed', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'micro', 'engine', 'verify'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    verify_source()
    if args.stage in ('all', 'micro'):
        run('correctness', [REPO / 'tests/test_cuda_fused.py', '-f'])
        run('microbench', [REPO / 'scripts/fusion/bench_fused.py'])
    if args.stage in ('all', 'engine'):
        # Reverse main A/B order between graph modes to reduce simple ordering bias.
        variants = [('baseline', 'all', 1), ('cuda', 'all', 1),
                    ('cuda', 'all', 0), ('baseline', 'all', 0),
                    ('cuda', 'add_rms', 1), ('cuda', 'silu', 1)]
        for backend, ops, graph in variants:
            tag = f'{backend}_{ops}_graph{graph}'
            run(tag, [REPO / 'scripts/fusion/bench_engine.py', '--backend', backend,
                      '--ops', ops, '--graph', graph])
        # Independent fresh processes in reversed A/B order, same workloads.
        for backend, graph in [('cuda', 1), ('baseline', 1), ('baseline', 0), ('cuda', 0)]:
            tag = f'{backend}_all_graph{graph}_reverse'
            run(tag, [REPO / 'scripts/fusion/bench_engine.py', '--backend', backend,
                      '--ops', 'all', '--graph', graph, '--label', 'reverse'])
    if args.stage in ('all', 'engine', 'verify'):
        compare_logits()
    if args.stage in ('all', 'engine'):
        print('Measured results saved in', OUT, flush=True)


if __name__ == '__main__':
    main()
