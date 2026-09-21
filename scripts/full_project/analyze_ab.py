"""Predeclared descriptive paired analysis. No run exclusion or cherry picking."""
import json
import random
import statistics
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
from scripts.runtime import OUT

def percentile(values,q):
    x=sorted(values); index=(len(x)-1)*q; lo=int(index); hi=min(lo+1,len(x)-1)
    return x[lo]+(x[hi]-x[lo])*(index-lo)

def main():
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest_sha=sha(OUT/'manifest.json')
    workloads=json.loads((OUT/'workloads.json').read_text())
    expected={(r['name'],r['repeat']):r for r in workloads if r['repeat'] in (1,2)}
    ordered_cases=list(dict.fromkeys(r['name'] for r in workloads))
    samples={}
    hashes={}
    for pair in range(1,11):
        for variant in ('native','fused'):
            path=OUT/f'ab_{pair:02d}_{variant}.json'
            data=json.loads(path.read_text())
            hashes[path.name]=sha(path)
            assert data['pair']==pair and data['variant']==variant
            assert data['manifest_sha256']==manifest_sha
            assert data['all_tokens_exact'] and data['audit']==dict(cache_hits=0,preemptions=0)
            assert len(data['rows'])==14
            shift=(pair-1)%len(ordered_cases)
            prescribed_order=ordered_cases[shift:]+ordered_cases[:shift]
            assert data['case_order']==prescribed_order, 'Missing/duplicate/reordered workload'
            assert [(r['case'],r['repeat']) for r in data['rows']]==[(c,r) for c in prescribed_order for r in (1,2)]
            assert {(r['case'],r['repeat']) for r in data['rows']}==set(expected)
            for row in data['rows']:
                fixture=expected[row['case'],row['repeat']]
                assert (row['seed'],row['batch'],row['input_length'],row['output_length'])==(fixture['seed'],fixture['batch'],fixture['prompt'],fixture['output'])
            samples[pair,variant]=data
        assert samples[pair,'native']['case_order']==samples[pair,'fused']['case_order']
    cases=samples[1,'native']['case_order']
    results=[]
    for case in cases:
        pairs=[]
        for pair in range(1,11):
            values={}
            for variant in ('native','fused'):
                rows=[r for r in samples[pair,variant]['rows'] if r['case']==case]
                assert len(rows)==2
                values[variant]={key:statistics.mean(r[key] for r in rows) for key in
                    ('e2e_ms','decode_step_ms','prefill_step_ms','engine_first_token_ms','output_tokens_per_s')}
                for phase in ('early','late'):
                    values[variant][phase]=statistics.mean(statistics.mean(
                        r['step_ms']['decode'][:32 if r['output_length']==256 else 15] if phase=='early'
                        else r['step_ms']['decode'][-32 if r['output_length']==256 else -16:]) for r in rows)
            reductions={key:100*(1-values['fused'][key]/values['native'][key]) for key in values['native'] if key!='output_tokens_per_s'}
            pairs.append(dict(pair=pair,native=values['native'],fused=values['fused'],reduction_pct=reductions,
                throughput_gain_pct=100*(values['fused']['output_tokens_per_s']/values['native']['output_tokens_per_s']-1)))
        changes=[p['reduction_pct']['e2e_ms'] for p in pairs]
        rng=random.Random(20260919)
        bootstrap=[statistics.mean(rng.choices(changes,k=10)) for _ in range(20000)]
        ci=[percentile(bootstrap,.025),percentile(bootstrap,.975)]
        results.append(dict(case=case,native_e2e_ms=statistics.mean(p['native']['e2e_ms'] for p in pairs),
            fused_e2e_ms=statistics.mean(p['fused']['e2e_ms'] for p in pairs),
            e2e_reduction_pct=statistics.mean(changes),ci95=ci,paired_e2e_reduction_pct=changes,
            decode_reduction_pct=statistics.mean(p['reduction_pct']['decode_step_ms'] for p in pairs),
            prefill_reduction_pct=statistics.mean(p['reduction_pct']['prefill_step_ms'] for p in pairs),
            first_token_reduction_pct=statistics.mean(p['reduction_pct']['engine_first_token_ms'] for p in pairs),
            early_decode_reduction_pct=statistics.mean(p['reduction_pct']['early'] for p in pairs),
            late_decode_reduction_pct=statistics.mean(p['reduction_pct']['late'] for p in pairs),
            throughput_gain_pct=statistics.mean(p['throughput_gain_pct'] for p in pairs),pairs=pairs))
    memory={}
    for variant in ('native','fused'):
        memory[variant]={key.replace('_bytes','_MiB'):[min(samples[pair,variant][key] for pair in range(1,11))/2**20,
            max(samples[pair,variant][key] for pair in range(1,11))/2**20]
            for key in ('peak_allocated_bytes','peak_reserved_bytes','steady_allocated_bytes')}
    summary=dict(rows=results,memory=memory,all_tokens_exact=True,processes=20,measured_requests=280,
        manifest_sha256=manifest_sha,source_sha256=sha(Path(__file__)),input_hashes=hashes,
        method='Per process per case mean of 2 measured requests; mean of 10 paired relative reductions;20,000 percentile-bootstrap samples of pairs;no exclusions',
        ci_scope='descriptive per-configuration 95% CI;not simultaneous over 7 cases;desktop GPU not locked',
        source_files=[f'ab_{p:02d}_{v}.json' for p in range(1,11) for v in ('native','fused')])
    (OUT/'ab_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for row in results:print(row['case'],round(row['e2e_reduction_pct'],3),row['ci95'])

if __name__=='__main__':main()
