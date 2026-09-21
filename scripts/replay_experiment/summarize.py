"""Print a compact result log from the new measurements; never render screenshots."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.fusion_report import build_report as metrics

OUT = ROOT / 'results/closed_loop_20260921'
DATA = OUT / 'replay/results/full_project_20260919'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    inputs = []
    def get(path):
        inputs.append(path)
        return read(path)
    discovery = get(OUT / 'discovery_summary.json')
    variants = []
    paths = [OUT / 'numerical_v1_fp32_intermediate.json',
             OUT / 'numerical_v2_bf16_boundary.json', DATA / 'numerical_fused.json']
    for tag, path in zip(('V1 FP32 intermediate', 'V2 restore BF16 boundary', 'V3 match native Graph reduction'), paths):
        data = get(path)
        assert data['complete'] and len(data['rows']) == 7
        selected = [c for r in data['rows'] for c in r['comparisons'] if 'logits' in c]
        variants.append(dict(tag=tag, exact_steps=sum(r['exact_steps'] for r in data['rows']),
            total_steps=sum(r['step_count'] for r in data['rows']), exact_gate=data['exact_gate'],
            selected_logits_max_abs=max(c['logits']['max_abs'] for c in selected),
            selected_cache_exact=all(c['cache']['exact'] for c in selected),
            cases=[{k:r[k] for k in ('case','step_count','exact_steps')} for r in data['rows']]))
    assert variants[-1]['exact_steps'] == variants[-1]['total_steps'] == 1568
    diagnostics=[]
    for tag in ('replay_v2', 'replay_v3'):
        d=get(DATA / f'kernel_model_diagnostic_{tag}.json')
        diagnostics.append(dict(tag=tag,rows=len(d['rows']),exact_gate=d['exact_gate'],
            mismatch_rows=sum(not all(e['exact'] for e in r['errors'].values()) for r in d['rows'])))
    gates=[]
    for name in ('kernel_validation_final.json','kernel_model_fixture_validation_final.json'):
        d=get(DATA/name)
        assert d['exact_gate']
        gates.append(dict(file=name,rows=len(d['rows']),exact_gate=d['exact_gate']))
    micro=get(DATA/'micro.json')
    assert micro['complete'] and micro['exact_gate'] and len(micro['rows']) == 9
    micro_rows=[]
    for r in micro['rows']:
        assert len(r['kernels']['native'])==4 and len(r['kernels']['fused'])==1
        n,f=r['median_us']['native'],r['median_us']['fused']
        micro_rows.append(dict(case=r['case'],native_us=n,fused_us=f,latency_reduction_pct=100*(1-f/n),speedup=n/f))
    integration={}
    for v in ('native','fused'):
        d=get(DATA/f'integration_{v}.json')
        assert d['replacement_verified'] and d['all_tokens_exact']
        integration[v]=d['counts']
    assert integration['native']['total']-integration['fused']['total']==84
    metrics.DATA=OUT
    ncu={v:metrics.load_ncu(v) for v in ('native','final')}
    for v in ncu:
        inputs.extend([OUT/f'{v}_raw.csv',OUT/f'{v}_details.csv',OUT/f'{v}_full.ncu-rep'])
    result=dict(scope='Fresh controlled replay; prior knowledge retained; no historical optimization results reused.',
        discovery=discovery,variants=variants,model_intermediate_diagnostics=diagnostics,
        fixture_gates=gates,micro=micro_rows,integration=integration,ncu=ncu)
    if (DATA/'ab_summary.json').exists():
        result['ab']=get(DATA/'ab_summary.json')
    result['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUT/'experiment_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['FRESH CONTROLLED REPLAY / 2026-09-21',
           'Source: results/closed_loop_20260921 (no old optimization measurements)',
           '',f"NSYS kernels={discovery['kernel_events']} steps={discovery['model_steps']} unmatched={discovery['unmatched']}",
           '', 'SAME-HISTORY FULL-MODEL LOGITS / selected KV slots in all 28 layers',
           'Version                              exact steps    selected logits max_abs']
    for r in variants:
        lines.append(f"{r['tag']:<37}{r['exact_steps']:>4}/{r['total_steps']:<7}{r['selected_logits_max_abs']:.8f}")
    lines+=['','MODEL INTERMEDIATES (Q/K outputs after Norm + RoPE)']
    for r in diagnostics:lines.append(f"{r['tag']}: {r['rows']-r['mismatch_rows']}/{r['rows']} exact, gate={r['exact_gate']}")
    lines+=['','FIXTURE GATES']+[f"{r['file']}: PASS {r['rows']} cases" for r in gates]
    lines+=['','INTEGRATION / one real Decode step / batch=1',
            f"native={integration['native']}\nfused ={integration['fused']}",
            '', 'NOTE: all variants specialize Q16/K8/D128; V2 -> V3 changes reduction path.',
            'BF16 rounding is retained in registers between Norm and RoPE; KV store stays separate.']
    (OUT/'correctness_results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    micro_lines=['GRAPH MICROBENCHMARK / model-context real first-layer fixtures',
        '1000 chains/replay, 10 alternating measurements; median us/chain',
        'case                 native us    fused us    reduction    speedup']
    for r in micro_rows:
        micro_lines.append(f"{r['case']:<21}{r['native_us']:>9.3f}{r['fused_us']:>12.3f}{r['latency_reduction_pct']:>11.2f}%{r['speedup']:>9.2f}x")
    micro_lines+=['','All 9 cases exact; four kernels -> one kernel.','NCU cold replay times are not used as benchmark times.']
    (OUT/'micro_results.txt').write_text('\n'.join(micro_lines)+'\n',encoding='utf-8')
    if 'ab' in result:
        a=result['ab']
        ab_lines=['END-TO-END / 10 PAIRED PROCESSES / 280 REQUESTS',
            'Alternating native/fused order; 2 requests/case/process; all samples retained.',
            'case                 native ms    fused ms   paired reduction      95% CI',
            '                                                 (positive = faster)']
        for r in a['rows']:
            ab_lines.append(f"{r['case']:<20}{r['native_e2e_ms']:>10.2f}{r['fused_e2e_ms']:>12.2f}{r['e2e_reduction_pct']:>13.2f}%   [{r['ci95'][0]:6.2f}, {r['ci95'][1]:6.2f}]%")
        ab_lines+=['',f"All output tokens match frozen baseline: {a['all_tokens_exact']}",
            'CI: pair-level 20,000-resample percentile bootstrap; descriptive per case.',
            'Desktop GPU / clocks not locked; no timing profiler.']
        (OUT/'ab_results.txt').write_text('\n'.join(ab_lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print('\n'.join(micro_lines))


if __name__=='__main__':
    main()
