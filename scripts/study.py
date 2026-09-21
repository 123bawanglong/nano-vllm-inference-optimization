"""Reproducible CUDA prototype experiment. Profiling is never used as timing."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import statistics
from pathlib import Path
import torch
from scripts.common import baseline, BASELINE, ROOT, error_metrics, save_trace, trace_kernels
from scripts.common import hooks, selected_steps, digest

from scripts.runtime import OUT, FIXTURE, MODEL_FIXTURE


def operational_sources():
    paths = [Path(__file__)]
    paths += [Path(__file__).with_name(name) for name in ('kernel_validation.py','analyze_ab.py')]
    paths += [ROOT/'scripts/common.py', ROOT/'scripts/discovery/idle_check.py']
    paths += [ROOT/'scripts/experiment.py', ROOT/'scripts/runtime.py', ROOT/'scripts/discovery/capture.py', ROOT/'scripts/discovery/analyze.py']
    paths += sorted((ROOT / 'src/qk_norm_rope').glob('*.py'))
    paths += sorted((ROOT / 'src/qk_norm_rope').glob('*.cpp'))
    paths += sorted((ROOT / 'src/qk_norm_rope').glob('*.cu'))
    return {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'manifest.json').exists(), 'Do not overwrite frozen experiment'
    frozen = baseline.check_manifest(BASELINE)
    for name, value in frozen['model_sha256'].items():
        assert baseline.sha(baseline.MODEL/name) == value
    (OUT/'workloads.json').write_bytes((BASELINE/'workloads.json').read_bytes())
    baseline.dump(OUT/'manifest.json', dict(baseline_sha256=baseline.sha(BASELINE/'baseline_manifest.json'),
        model_hashes_verified=True, sources=operational_sources(), config=baseline.CONFIG,
        workloads_sha256=baseline.sha(OUT/'workloads.json'), fixture_sha256=baseline.sha(FIXTURE),
        protocol_sha256=baseline.sha(ROOT/'docs/reproduce.md'), model_fixture_sha256=baseline.sha(MODEL_FIXTURE),
        paired_processes=10, measured_requests_per_case_per_process=2, gpu=baseline.gpu()))


def check():
    baseline.check_manifest(BASELINE)
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['sources'] == operational_sources(), 'Experiment code changed'
    assert manifest['workloads_sha256'] == baseline.sha(OUT/'workloads.json')
    assert manifest['fixture_sha256'] == baseline.sha(FIXTURE)
    assert manifest['model_fixture_sha256'] == baseline.sha(MODEL_FIXTURE)
    assert manifest['baseline_sha256'] == baseline.sha(BASELINE/'baseline_manifest.json')
    assert manifest['protocol_sha256'] == baseline.sha(ROOT/'docs/reproduce.md')
    return json.loads((OUT/'workloads.json').read_text())


def kernel_gate():
    result=json.loads((OUT/'kernel_validation_final.json').read_text())
    assert result['exact_gate'] and len(result['rows'])==138
    assert result['fixture_sha256']==baseline.sha(FIXTURE)
    assert result['validation_sha256']==baseline.sha(Path(__file__).with_name('kernel_validation.py'))
    for path,digest_value in result['source_sha256'].items():
        assert baseline.sha(ROOT/path)==digest_value, ('Stale kernel validation',path)
    captured=json.loads((OUT/'kernel_model_fixture_validation_final.json').read_text())
    assert captured['exact_gate'] and len(captured['rows'])==294
    assert baseline.sha(ROOT/captured['fixture_path'])==captured['fixture_sha256']
    assert captured['validation_sha256']==baseline.sha(Path(__file__).with_name('kernel_validation.py'))
    for path,digest_value in captured['source_sha256'].items():
        assert baseline.sha(ROOT/path)==digest_value, ('Stale model fixture validation',path)


@torch.inference_mode()
def numerical(variant):
    rows = check()
    assert not (OUT/f'numerical_{variant}.json').exists(), 'Refuse overwriting numerical evidence'
    llm = baseline.engine()
    audit = hooks(llm)
    runner = llm.model_runner
    original_model, original_sampler = runner.run_model, runner.sampler.forward
    results = []
    try:
        for item in [r for r in rows if r['repeat']==1]:
            baseline.request(llm, next(r for r in rows if r['name']==item['name'] and r['repeat']==0))
            steps, snapshots, sampled, comparisons = [], {}, [], []
            state = dict(index=0)
            ref_path = OUT/f"reference_{item['name']}.pt"
            reference = torch.load(ref_path, map_location='cpu', weights_only=False) if variant=='fused' else None
            selected = selected_steps(item)
            def capture(input_ids, positions, is_prefill):
                from nanovllm.utils.context import get_context
                y = original_model(input_ids, positions, is_prefill)
                value = y.detach().cpu()
                idx = state['index']
                record = dict(index=idx, prefill=is_prefill, logits_hash=digest(value),
                    input_ids_hash=digest(input_ids.cpu()), positions_hash=digest(positions.cpu()),
                    finite=bool(torch.isfinite(value).all()))
                assert record['finite']
                steps.append(record)
                if idx in selected:
                    slots = get_context().slot_mapping
                    if is_prefill: slots = slots[item['prompt']-1::item['prompt']]
                    cache = runner.kv_cache.flatten(2,3)[:,:,slots.long()].cpu()
                    snapshots[idx] = dict(logits=value, cache=cache)
                if reference is not None:
                    expected = reference['steps'][idx]
                    assert record['input_ids_hash']==expected['input_ids_hash']
                    assert record['positions_hash']==expected['positions_hash']
                    comp = dict(index=idx,prefill=is_prefill,exact=record['logits_hash']==expected['logits_hash'])
                    if idx in selected:
                        comp.update(logits=error_metrics(value,reference['snapshots'][idx]['logits']),
                            cache=error_metrics(cache,reference['snapshots'][idx]['cache']))
                    comparisons.append(comp)
                return y
            def sampler(logits, temperatures):
                idx = state['index']
                result = (torch.tensor(reference['sampled'][idx],dtype=torch.int64,device=logits.device)
                          if reference is not None else original_sampler(logits, temperatures))
                sampled.append(result.cpu().tolist())
                state['index'] += 1
                return result
            runner.run_model, runner.sampler.forward = capture, sampler
            measured = baseline.request(llm,item)
            runner.run_model, runner.sampler.forward = original_model, original_sampler
            assert state['index']==item['output'] and audit==dict(cache_hits=0,preemptions=0)
            if reference is None:
                torch.save(dict(steps=steps,snapshots=snapshots,sampled=sampled),ref_path)
            row = dict(case=item['name'],step_count=len(steps),selected_steps=sorted(selected),
                all_finite=True,exact_steps=sum(c['exact'] for c in comparisons) if reference else len(steps),
                comparisons=comparisons,output_token_ids=measured['output_token_ids'],
                reference_sha256=baseline.sha(ref_path),teacher_forced=variant=='fused')
            results.append(row)
            complete = len(results)==7
            gate = all(r['exact_steps']==r['step_count'] and
                all(c.get('cache',{'exact':True})['exact'] for c in r['comparisons']) for r in results)
            baseline.dump(OUT/f'numerical_{variant}.json',dict(variant=variant,rows=results,
                complete=complete,exact_gate=gate,audit=audit,timing_is_diagnostic=True,
                manifest_sha256=baseline.sha(OUT/'manifest.json')))
            print(f"NUMERICAL {variant} {item['name']} exact={row['exact_steps']}/{len(steps)}",flush=True)
        assert gate, 'Full-model exact gate failed; diagnostic JSON retained'
    finally:
        baseline.close(llm)


@torch.inference_mode()
def timing(variant, pair):
    rows = check()
    kernel_gate()
    for v in ('native','fused'):
        numerical_result=json.loads((OUT/f'numerical_{v}.json').read_text())
        assert numerical_result['complete'] and numerical_result['exact_gate'], 'Numerical gate failed'
        assert numerical_result['manifest_sha256']==baseline.sha(OUT/'manifest.json')
    micro_result=json.loads((OUT/'micro.json').read_text())
    assert micro_result['complete'] and micro_result['exact_gate']
    assert micro_result['manifest_sha256']==baseline.sha(OUT/'manifest.json')
    integrated=json.loads((OUT/'integration_fused.json').read_text())
    assert integrated['replacement_verified'] and integrated['manifest_sha256']==baseline.sha(OUT/'manifest.json')
    cases=list(dict.fromkeys(r['name'] for r in rows))
    shift=(pair-1)%len(cases)
    cases=cases[shift:]+cases[:shift]
    dest=OUT/f'ab_{pair:02d}_{variant}.json'
    assert not dest.exists(), 'Refuse overwriting a sample'
    gpu_before=baseline.gpu()
    llm=baseline.engine()
    audit=hooks(llm)
    results=[]
    try:
        for name in cases:
            baseline.request(llm,next(r for r in rows if r['name']==name and r['repeat']==0))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        steady_allocated=torch.cuda.memory_allocated()
        for name in cases:
            for repeat in (1,2):
                item=next(r for r in rows if r['name']==name and r['repeat']==repeat)
                row=baseline.request(llm,item)
                reference=json.loads((BASELINE/'process_1.json').read_text())['rows']
                expected=next(r for r in reference if r['case']==name and r['repeat']==repeat)
                assert row['output_token_ids']==expected['output_token_ids'], (name,'output mismatch')
                assert audit==dict(cache_hits=0,preemptions=0)
                results.append(row)
                print(f"AB pair={pair} variant={variant} {name} r={repeat} e2e={row['e2e_ms']:.3f} ms",flush=True)
        baseline.dump(dest,dict(variant=variant,pair=pair,rows=results,case_order=cases,audit=audit,
            gpu_before=gpu_before,gpu_after=baseline.gpu(),all_tokens_exact=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            steady_allocated_bytes=steady_allocated,manifest_sha256=baseline.sha(OUT/'manifest.json')))
    finally:
        baseline.close(llm)


@contextmanager
def fixture_functions():
    from src.qk_norm_rope import fused, build
    build()
    # The standalone compiler can choose another reduction tree for B1.
    # Warm the real native engine first, then use those exact norm/rope modules.
    llm=baseline.engine()
    attn=llm.model_runner.model.model.layers[0].self_attn
    data=torch.load(FIXTURE,map_location='cuda',weights_only=False)
    qn,kn,rope=attn.q_norm,attn.k_norm,attn.rotary_emb
    assert torch.equal(qn.weight,data['q_weight']) and torch.equal(kn.weight,data['k_weight'])
    assert torch.equal(rope.cos_sin_cache[:4096],data['cos_sin_cache'])
    functions={}
    for name,fixture in data['fixtures'].items():
        packed,positions=fixture['packed'],fixture['positions']
        q,k,_=packed.split([2048,1024,1024],-1)
        q,k=q.view(-1,16,128),k.view(-1,8,128)
        native=lambda q=q,k=k,p=positions:rope(p,qn(q),kn(k))
        custom=lambda q=q,k=k,p=positions:fused(q,k,qn.weight,kn.weight,p,rope.cos_sin_cache,data['eps'],reduction_mode='native_graph')
        functions[name]=(native,custom,packed)
    try:
        yield functions
    finally:
        baseline.close(llm)


@torch.inference_mode()
def micro():
    check()
    kernel_gate()
    with fixture_functions() as functions:
        micro_functions(functions)


def micro_functions(fixture_fns):
    results=[]
    for name,(native,fused,packed) in fixture_fns.items():
        before=packed.clone()
        functions=dict(native=native,fused=fused)
        graphs={};outputs={};times={v:[] for v in functions};timing_graphs={}
        for v,fn in functions.items():
            for _ in range(30):fn()
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):outputs[v]=fn()
            graphs[v]=graph
            graph.replay()
        torch.cuda.synchronize()
        errors={label:error_metrics(outputs['fused'][i],outputs['native'][i]) for i,label in enumerate(('q','k'))}
        assert all(e['exact'] and e['finite'] for e in errors.values()), errors
        assert torch.equal(packed,before)
        for v,fn in functions.items():
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(1000):fn()
            timing_graphs[v]=g
            g.replay()
        torch.cuda.synchronize()
        for repeat in range(10):
            for v in (('native','fused') if repeat%2==0 else ('fused','native')):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();timing_graphs[v].replay();end.record();end.synchronize()
                times[v].append(start.elapsed_time(end))
        kernels={}
        for v,g in graphs.items():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
                g.replay();torch.cuda.synchronize()
            trace=OUT/f'micro_{name}_{v}.json.gz'
            save_trace(p,trace);kernels[v]=trace_kernels(trace)
        assert len(kernels['native'])==4 and len(kernels['fused'])==1, kernels
        results.append(dict(case=name,errors=errors,times_us=times,
            median_us={v:statistics.median(t) for v,t in times.items()},kernels=kernels,input_unchanged=True))
        baseline.dump(OUT/'micro.json',dict(rows=results,complete=len(results)==9,exact_gate=True,
            manifest_sha256=baseline.sha(OUT/'manifest.json'),
            method='1000 independent chain calls per CUDA Graph,10 alternating paired measurements'))
        print('MICRO',name,results[-1]['median_us'],flush=True)


@torch.inference_mode()
def ncu(variant):
    check()
    with fixture_functions() as functions:
        capture_ncu(functions,variant)


def capture_ncu(functions,variant):
    native,fused,_=functions['b1_p64_o256']
    fn=native if variant=='native' else fused
    for _ in range(100):fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    g.replay();torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    g.replay();torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print('NCU captured',variant,flush=True)


@torch.inference_mode()
def integration_trace(variant):
    rows=check()
    llm=baseline.engine()
    audit=hooks(llm)
    item=next(r for r in rows if r['name']=='b1_p64_o256' and r['repeat']==1)
    state=dict(index=0)
    original_step=llm.step
    profiler=None
    try:
        baseline.request(llm,next(r for r in rows if r['name']==item['name'] and r['repeat']==0))
        def step():
            nonlocal profiler
            if state['index']==17:
                torch.cuda.synchronize()
                profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
                profiler.start()
            output=original_step()
            if state['index']==17:
                torch.cuda.synchronize();profiler.stop()
            state['index']+=1
            return output
        llm.step=step
        result=baseline.request(llm,item)
        expected=next(r for r in json.loads((BASELINE/'process_1.json').read_text())['rows'] if r['case']==item['name'] and r['repeat']==1)
        assert result['output_token_ids']==expected['output_token_ids']
        assert audit==dict(cache_hits=0,preemptions=0)
        path=OUT/f'integration_{variant}.json.gz'
        save_trace(profiler,path)
        kernels=trace_kernels(path)
        counts=dict(total=len(kernels),fused=sum('qk_kernel' in k['name'] for k in kernels),
            rope=sum('cat_index_mul_split_sub' in k['name'] for k in kernels),
            kv_store=sum('store_kvcache_kernel' in k['name'] for k in kernels))
        assert counts['kv_store']==28
        if variant=='native':assert counts['fused']==0 and counts['rope']==56
        else:
            native=json.loads((OUT/'integration_native.json').read_text())
            assert native['manifest_sha256']==baseline.sha(OUT/'manifest.json')
            assert counts['fused']==28 and counts['rope']==0
            assert native['counts']['total']-counts['total']==84,(native['counts'],counts)
        baseline.dump(OUT/f'integration_{variant}.json',dict(variant=variant,counts=counts,kernels=kernels,
            replacement_verified=True,all_tokens_exact=True,audit=audit,step=17,
            manifest_sha256=baseline.sha(OUT/'manifest.json')))
        print('INTEGRATION',variant,counts,flush=True)
    finally:
        baseline.close(llm)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','check','numerical','timing','micro','ncu','trace'])
    p.add_argument('--variant',choices=['native','fused'],default='native')
    p.add_argument('--pair',type=int,default=1)
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='check':check()
    elif a.mode=='micro':micro()
    elif a.mode=='ncu':ncu(a.variant)
    else:
        if a.variant=='fused':
            from src.qk_norm_rope.adapter import install
            install()
        if a.mode=='numerical':numerical(a.variant)
        elif a.mode=='trace':integration_trace(a.variant)
        else:timing(a.variant,a.pair)
