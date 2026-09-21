"""Unmodified-engine hotspot traces and same-history diagnostic model checks."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from scripts.compile_boundary.common import (baseline, check, install_joint,
                                            error_metrics, save_trace, trace_kernels)


def digest(tensor):
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


@torch.inference_mode()
def numerical(out, variant):
    from nanovllm.utils.context import get_context
    llm = baseline.engine()
    audit = hooks(llm)
    runner = llm.model_runner
    original_model = runner.run_model
    original_sampler = runner.sampler.forward
    results = []
    try:
        rows = json.loads((out / 'workloads.json').read_text())
        for item in [r for r in rows if r['repeat'] == 1]:
            # Model warmup used different prompts, no prefix reuse across measurements.
            baseline.request(llm, next(r for r in rows if r['name']==item['name'] and r['repeat']==0))
            steps, snapshots, sampled, comparisons = [], {}, [], []
            state = dict(index=0)
            ref_path = out / f"reference_{item['name']}.pt"
            reference = torch.load(ref_path, map_location='cpu', weights_only=False) if variant=='joint' else None
            selected = selected_steps(item)

            def capture_model(input_ids, positions, is_prefill):
                y = original_model(input_ids, positions, is_prefill)
                value = y.detach().cpu()
                idx = state['index']
                record = dict(index=idx, prefill=is_prefill, logits_hash=digest(value),
                    input_ids_hash=digest(input_ids.cpu()), positions_hash=digest(positions.cpu()),
                    finite=bool(torch.isfinite(value).all().item()))
                assert record['finite']
                steps.append(record)
                if idx in selected:
                    slots = get_context().slot_mapping
                    if is_prefill:
                        slots = slots[item['prompt']-1::item['prompt']]
                    cache = runner.kv_cache.flatten(2, 3)[:, :, slots.long()].cpu()
                    snapshots[idx] = dict(logits=value, cache=cache)
                if reference:
                    expected = reference['steps'][idx]
                    assert record['input_ids_hash']==expected['input_ids_hash']
                    assert record['positions_hash']==expected['positions_hash']
                    comp = dict(index=idx, prefill=is_prefill, exact=record['logits_hash']==expected['logits_hash'])
                    if idx in selected:
                        comp['logits'] = error_metrics(value, reference['snapshots'][idx]['logits'])
                        comp['written_cache_all_layers'] = error_metrics(cache, reference['snapshots'][idx]['cache'])
                        comp['argmax_equal'] = bool(torch.equal(value.argmax(-1), reference['snapshots'][idx]['logits'].argmax(-1)))
                    comparisons.append(comp)
                return y

            def sampler(logits, temperatures):
                idx = state['index']
                if reference:
                    value = torch.tensor(reference['sampled'][idx], dtype=torch.int64, device=logits.device)
                else:
                    value = original_sampler(logits, temperatures)
                sampled.append(value.cpu().tolist())
                state['index'] += 1
                return value

            runner.run_model, runner.sampler.forward = capture_model, sampler
            measured = baseline.request(llm, item)
            runner.run_model, runner.sampler.forward = original_model, original_sampler
            assert state['index']==item['output']
            assert audit==dict(cache_hits=0,preemptions=0), audit
            if variant=='native':
                torch.save(dict(steps=steps, snapshots=snapshots, sampled=sampled), ref_path)
                row = dict(case=item['name'], step_count=len(steps), all_finite=True,
                           reference_sha256=baseline.sha(ref_path), selected_steps=sorted(selected))
            else:
                assert sampled==reference['sampled']
                row = dict(case=item['name'], step_count=len(steps), all_finite=True,
                           exact_steps=sum(x['exact'] for x in comparisons), comparisons=comparisons,
                           teacher_forced=True, sampled_history_equal=True)
            row['output_token_ids'] = measured['output_token_ids']
            results.append(row)
            baseline.dump(out/f'numerical_{variant}.json', dict(variant=variant,rows=results,audit=audit,
                complete=len(results)==9, numerical_gate='exact equality, no tolerance relaxation',
                timings_are_not_benchmark=True))
            print(f"NUMERICAL {variant} {item['name']} steps={len(steps)} exact={row.get('exact_steps', 'reference')}", flush=True)
    finally:
        baseline.close(llm)


@torch.inference_mode()
def profile(out, variant):
    from nanovllm import SamplingParams
    llm = baseline.engine()
    audit = hooks(llm)
    summaries = []
    try:
        fixtures = json.loads((out/'workloads.json').read_text())
        for item in [r for r in fixtures if r['repeat']==1]:
            baseline.request(llm, next(r for r in fixtures if r['name']==item['name'] and r['repeat']==0))
            reference = torch.load(out/f"reference_{item['name']}.pt", weights_only=False)
            state = dict(index=0)
            # Retain native sampler GPU work; correct the returned IDs on host for matched histories.
            original_run = llm.model_runner.run
            def forced_run(seqs, is_prefill):
                values = original_run(seqs, is_prefill)
                expected = reference['sampled'][state['index']]
                state['index'] += 1
                if variant=='native':
                    assert values==expected
                return expected
            llm.model_runner.run = forced_run
            torch.manual_seed(item['seed'])
            for prompt in item['prompts']:
                llm.add_request(prompt, SamplingParams(temperature=1.0, max_tokens=item['output'], ignore_eos=True))
            windows = {'prefill': (0, 1)}
            if item['output']==256:
                windows.update(early=(17,33), late=(225,241))
            else:
                windows['short_decode']=(16,32)
            active, prof = None, None
            idx = 0
            while not llm.is_finished():
                for label,(start,stop) in windows.items():
                    if idx==start:
                        torch.cuda.synchronize()
                        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                   torch.profiler.ProfilerActivity.CUDA])
                        prof.start()
                        active=label
                if active:
                    with torch.profiler.record_function(f"{variant}_{active}_step{idx}"):
                        llm.step()
                else:
                    llm.step()
                idx += 1
                if active and idx==windows[active][1]:
                    torch.cuda.synchronize()
                    prof.stop()
                    path = out/f"trace_{variant}_{item['name']}_{active}.json.gz"
                    save_trace(prof,path)
                    kernels=trace_kernels(path)
                    assert kernels
                    start,stop=windows[active]
                    record = dict(variant=variant,case=item['name'],phase=active,steps=stop-start,
                        first_step=start,last_step=stop-1,trace=path.name,kernels=kernels,
                        layers=len(llm.model_runner.model.model.layers),graph=active!='prefill',
                        input_history='native reference',timing='diagnostic profiled kernel duration')
                    summaries.append(record)
                    baseline.dump(out/f'profile_{variant}.json',dict(rows=summaries,complete=False))
                    print(f"PROFILE {variant} {item['name']} {active} kernels={len(kernels)}",flush=True)
                    active=None
            assert idx==item['output'] and state['index']==item['output']
            llm.model_runner.run = original_run
            assert audit==dict(cache_hits=0,preemptions=0), audit
        baseline.dump(out/f'profile_{variant}.json',dict(rows=summaries,complete=True,audit=audit))
    finally:
        baseline.close(llm)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['numerical','profile'])
    parser.add_argument('--variant',choices=['native','joint'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    check(args.out)
    if args.variant=='joint':
        install_joint()
    (numerical if args.mode=='numerical' else profile)(args.out,args.variant)
