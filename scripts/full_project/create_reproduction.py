"""Create a fresh local experiment snapshot without copying measured outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]

def main(destination):
    destination=destination.resolve()
    if destination.exists():raise FileExistsError('Use a new directory; existing work is never overwritten')
    destination.mkdir(parents=True)
    for name in ('nanovllm','benchmarks','scripts','src','tests'):
        shutil.copytree(ROOT/name,destination/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    inputs=[
        'docs/experiment_protocol.md',
        'results/baseline_20260918_170614/baseline_manifest.json',
        'results/baseline_20260918_170614/workloads.json',
        'results/baseline_20260918_170614/process_1.json',
        'results/compile_boundary_20260918_233238/real_first_layer_fixtures.pt',
        'results/full_project_20260919/kernel_model_fixtures_before.pt',
    ]
    for name in inputs:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    baseline=json.loads((destination/inputs[1]).read_text(encoding='utf-8'))
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    for name,digest in baseline['source_sha256'].items():assert sha(destination/name)==digest,name
    manifest=dict(parent_project=str(ROOT),copied_inputs={name:sha(destination/name) for name in inputs},
        measured_outputs_copied=False,baseline_source_files_verified=len(baseline['source_sha256']),
        model_and_python='Uses the same absolute WSL model/toolchain paths as the original experiment')
    (destination/'REPRODUCTION_INPUTS.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(destination)
    print('Ready: cd into this new directory, then bash scripts/full_project/reproduce.sh')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path)
    main(p.parse_args().destination)
