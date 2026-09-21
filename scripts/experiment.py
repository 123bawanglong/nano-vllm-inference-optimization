"""Run the Q/K fusion experiment in isolated, reproducible stages.

python -m scripts.experiment --help
Each GPU task runs in a fresh process; profiler results are never timing inputs.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def configuration(stage, out, model):
    if stage == 'prepare':
        if out.exists():
            raise FileExistsError(f'Use a new output directory: {out}')
        if model is None or not model.expanduser().is_dir():
            raise ValueError('--model must point to a local Qwen3-0.6B model directory')
        return {'model': str(model.expanduser().resolve()), 'schema': 1}
    if model is not None:
        raise ValueError('--model is set once by prepare; use a new run to change it')
    if stage == 'kernel':
        if out.exists():
            raise FileExistsError(f'Use a new output directory: {out}')
        return {}
    return json.loads((out / 'run.json').read_text())


def environment(out, cfg):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), QK_RUN_DIR=str(out),
               QK_BASELINE_DIR=str(out / 'baseline'),
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    if 'model' in cfg:
        env['QK_MODEL'] = cfg['model']
    cache = Path(env.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'nano-vllm-fusion'
    env.setdefault('TORCHINDUCTOR_CACHE_DIR', str(cache / 'inductor'))
    env.setdefault('TRITON_CACHE_DIR', str(cache / 'triton'))
    env.setdefault('TORCH_CUDA_ARCH_LIST', '12.0')
    env.setdefault('MAX_JOBS', '2')
    if env.get('CUDA_HOME'):
        env['PATH'] = str(Path(env['CUDA_HOME']) / 'bin') + os.pathsep + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = str(Path(env['CUDA_HOME']) / 'lib64') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    return env


def commands(stage, out, tool='systems', nsys='nsys', ncu='ncu'):
    def py(module, *args):
        return [sys.executable, '-m', module, *map(str, args)]

    def study(*args):
        return py('scripts.study', *args)

    def idle(name):
        return py('scripts.discovery.idle_check', '--output', out / f'idle_{name}.json')

    if stage == 'prepare':
        jobs = [('freeze_baseline', py('scripts.baseline', 'prepare', '--out', out / 'baseline'))]
        for run in range(1, 4):
            jobs += [(f'idle_baseline_{run}', idle(f'baseline_{run}')),
                     (f'baseline_{run}', py('scripts.baseline', 'sample', '--out', out / 'baseline', '--run-id', run))]
        return jobs + [('freeze_experiment', study('prepare'))]
    gate = ('kernel_validation', py('scripts.kernel_validation', '--label', 'final'))
    if stage == 'kernel':
        return [gate]
    if stage == 'validate':
        return [gate] + [(f'numerical_{v}', study('numerical', '--variant', v)) for v in ('native', 'fused')] + [
            (f'integration_{v}', study('trace', '--variant', v)) for v in ('native', 'fused')]
    if stage == 'micro':
        return [('idle_micro', idle('micro')), ('micro', study('micro'))]
    if stage == 'ab':
        jobs = []
        for pair in range(1, 11):
            for variant in (('native', 'fused') if pair % 2 else ('fused', 'native')):
                tag = f'ab_{pair:02d}_{variant}'
                jobs += [(f'idle_{tag}', idle(tag)), (tag, study('timing', '--variant', variant, '--pair', pair))]
        return jobs
    if stage == 'analyze':
        return [('paired_statistics', py('scripts.analyze_ab'))]
    if tool == 'systems':
        dest = out / 'discovery'
        jobs = [('discovery_prepare', py('scripts.discovery.capture', 'prepare', '--out', dest))]
        for run in range(1, 4):
            stem = dest / f'native_{run}'
            jobs += [(f'idle_discovery_{run}', idle(f'discovery_{run}')),
                     (f'systems_{run}', [nsys, 'profile', '--trace=cuda,nvtx', '--cuda-graph-trace=node',
                      '--sample=none', '--cpuctxsw=none', '--capture-range=cudaProfilerApi',
                      '--capture-range-end=stop', '--force-overwrite=false', '--discard-environment=true',
                      f'--output={stem}', *py('scripts.discovery.capture', 'profile', '--out', dest, '--run-id', run)]),
                     (f'export_{run}', [nsys, 'export', '--type=sqlite', f'--output={stem}.sqlite', f'{stem}.nsys-rep']),
                     (f'rankings_{run}', py('scripts.discovery.analyze', '--out', dest, '--run-id', run))]
        return jobs + [('rankings_summary', py('scripts.discovery.analyze', '--out', dest))]
    jobs = []
    for variant in ('native', 'fused'):
        sections = ['SpeedOfLight', 'SpeedOfLight_HierarchicalSingleRooflineChart', 'LaunchStats', 'Occupancy',
                    'MemoryWorkloadAnalysis', 'ComputeWorkloadAnalysis', 'SchedulerStats', 'WarpStateStats', 'SourceCounters']
        cmd = [ncu, '--target-processes', 'application-only', '--profile-from-start', 'off',
               '--graph-profiling', 'node', '--replay-mode', 'kernel', '--cache-control', 'all',
               '--clock-control', 'none', '--kernel-name-base', 'function', '--import-source', 'yes']
        for section in sections:
            cmd += ['--section', section]
        cmd += ['--export', str(out / f'{variant}_qk_rope'), *study('ncu', '--variant', variant)]
        jobs += [(f'idle_ncu_{variant}', idle(f'ncu_{variant}')), (f'ncu_{variant}', cmd)]
    return jobs


def run_tasks(tasks, out, env):
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    for label, cmd in tasks:
        print(f'[{label}] {shlex.join(cmd)}', flush=True)
        # Exclusive logs prevent accidental repeats from replacing evidence.
        path = logs / f'{label}.log'
        with path.open('x') as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print('\n'.join(path.read_text(errors='replace').splitlines()[-8:]), flush=True)
        if result.returncode:
            raise RuntimeError(f'{label} failed ({result.returncode}); see {path}. Results retained; use a new run to retry.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['kernel', 'prepare', 'profile', 'validate', 'micro', 'ab', 'analyze'])
    parser.add_argument('--out', type=Path, required=True, help='fresh measurements, never a published evidence directory')
    parser.add_argument('--model', type=Path, help='local Qwen3-0.6B weights; prepare only')
    parser.add_argument('--tool', choices=['systems', 'compute'], default='systems')
    parser.add_argument('--nsys', default=os.environ.get('NSYS', 'nsys'))
    parser.add_argument('--ncu', default=os.environ.get('NCU', 'ncu'))
    parser.add_argument('--dry-run', action='store_true', help='show commands without creating files or launching GPU work')
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    try:
        cfg = configuration(args.stage, out, args.model)
        env = environment(out, cfg)
        env['NSYS'] = args.nsys
        tasks = commands(args.stage, out, args.tool, args.nsys, args.ncu)
        if args.dry_run:
            for label, cmd in tasks:
                print(f'[{label}] {shlex.join(cmd)}')
            return
        if args.stage == 'profile':
            profiler = args.nsys if args.tool == 'systems' else args.ncu
            if not shutil.which(profiler, path=env['PATH']):
                raise FileNotFoundError(f'Profiler not found: {profiler}; set --nsys or --ncu')
        if args.stage in ('prepare', 'kernel'):
            out.mkdir(parents=True, exist_ok=False)
            (out / 'run.json').write_text(json.dumps(cfg, indent=2) + '\n')
        elif args.stage != 'analyze':
            # Reject changed source or workload before generating new evidence.
            subprocess.run([sys.executable, '-m', 'scripts.study', 'check'], cwd=ROOT, env=env, check=True)
        run_tasks(tasks, out, env)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
