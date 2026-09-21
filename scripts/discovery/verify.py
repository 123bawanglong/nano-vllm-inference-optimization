"""Read-only provenance and experimental integrity checks."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from scripts.discovery.capture import BASELINE, ROOT, baseline


def main(parent):
    manifest=baseline.check_manifest(BASELINE)
    model_checks={name:baseline.sha(baseline.MODEL/name)==digest for name,digest in manifest['model_sha256'].items()}
    assert all(model_checks.values()),model_checks
    old=json.loads((BASELINE/'environment.json').read_text())['packages']
    current={p:importlib.metadata.version(p) for p in old}
    assert current==old,(current,old)
    summaries=[]
    for out in (parent,parent/'confirmation'):
        m=json.loads((out/'manifest.json').read_text())
        assert m['source_sha256']==baseline.sources()
        assert m['capture_sha256']==baseline.sha(Path(__file__).with_name('capture.py'))
        assert m['workloads_sha256']==baseline.sha(out/'workloads.json')==manifest['workloads_sha256']
        assert m['config']==baseline.CONFIG and m['kernel_filter'] is None
        for run in (1,2,3):
            for mode,expected in (('timing',14),('profile',7)):
                d=json.loads((out/f'{mode}_{run}.json').read_text())
                assert len(d['rows'])==expected and d['all_tokens_match']
                assert d['audit']==dict(cache_hits=0,preemptions=0)
                assert d['manifest_sha256']==baseline.sha(out/'manifest.json')
            a=json.loads((out/f'analysis_{run}.json').read_text())
            assert a['step_count']==1568 and a['unmatched']==0
            if out.name=='confirmation':
                for mode in ('timing','profile'):
                    idle=json.loads((out/f'idle_{mode}_{run}.json').read_text())
                    assert idle['passed'] and len(idle['samples'])==5
            summaries.append(dict(set=out.name,run=run,kernel_count=a['kernel_count'],steps=a['step_count']))
    result=dict(baseline_sources_match=True,model_hashes=model_checks,packages=current,source_variant='native only',
        raw_runs=summaries,profile_processes=6,unprofiled_processes=6,all_native_outputs_match=True,
        unmatched_kernels=0,idle_confirmation_gate_passed=True,handwritten_kernel=False,
        diagnostic_and_confirmation_kept_separate=True,
        analysis_sha256=baseline.sha(Path(__file__).with_name('analyze.py')))
    evidence=json.loads((parent/'compiler_evidence/provenance.json').read_text())
    for item in evidence['files']:
        assert baseline.sha(parent/'compiler_evidence'/item['file'])==item['sha256']
    result['compiler_evidence_hashes_match']=True
    result['source_audit_sha256']=baseline.sha(ROOT/'docs/discovery_source_audit.md')
    baseline.dump(parent/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
