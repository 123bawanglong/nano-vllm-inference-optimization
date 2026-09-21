"""Profile a warmed window of the unchanged native CUDA-Graph decode path."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('native_baseline', ROOT / 'benchmarks/baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tool', choices=['cuda','torch'], required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture-steps', type=int, default=16)
    args = parser.parse_args()
    assert 1 <= args.capture_steps <= 239
    assert not args.output.exists(), f'Refusing to overwrite {args.output}'
    baseline.check_manifest(args.baseline)
    import torch
    from nanovllm import SamplingParams
    from nanovllm.utils.context import get_context
    fixtures = json.loads((args.baseline / 'workloads.json').read_text())
    warmup = next(x for x in fixtures if x['name']=='b1_p64_o256' and x['repeat']==0)
    item = next(x for x in fixtures if x['name']=='b1_p64_o256' and x['repeat']==1)
    llm = baseline.engine()
    profile = None
    active = False
    try:
        baseline.request(llm, warmup)
        torch.manual_seed(item['seed'])
        llm.add_request(item['prompts'][0], SamplingParams(temperature=1.0, max_tokens=256, ignore_eos=True))
        if args.tool == 'torch':
            profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                        profile_memory=False, with_stack=False)
        count = 0
        tokens = {}
        starts_at = 17  # prefill + 16 decode steps are outside capture
        stops_at = starts_at + args.capture_steps
        while not llm.is_finished():
            if count == starts_at:
                torch.cuda.synchronize()
                if profile:
                    profile.start()
                else:
                    torch.cuda.cudart().cudaProfilerStart()
                active = True
            if active:
                torch.cuda.nvtx.range_push(f'native_decode_step_{count}')
            output, n = llm.step()
            if active:
                assert n == -1
                torch.cuda.nvtx.range_pop()
            tokens.update(output)
            count += 1
            if count == stops_at:
                torch.cuda.synchronize()
                if profile:
                    profile.stop()
                else:
                    torch.cuda.cudart().cudaProfilerStop()
                active = False
        assert count == 256 and len(tokens)==1
        token_ids = next(iter(tokens.values()))
        assert len(token_ids)==256 and not llm.scheduler.block_manager.used_block_ids
        expected = json.loads((args.baseline/'process_1.json').read_text())
        reference = next(r for r in expected['rows'] if r['case']==item['name'] and r['repeat']==1)
        matches = token_ids == reference['output_token_ids'][0]
        assert matches, 'Profiler execution changed generated tokens relative to saved native baseline'
        trace = None
        if profile:
            trace = args.output.with_suffix('.trace.json')
            profile.export_chrome_trace(str(trace))
        baseline.dump(args.output, dict(tool=args.tool, config=baseline.CONFIG,
            workload=dict(batch=1,input_tokens=64,output_tokens=256,seed=item['seed']),
            capture_decode_steps=args.capture_steps,first_decode_step=starts_at,
            last_decode_step=stops_at-1,first_position=64+starts_at-1,
            graph=True,model_layers=len(llm.model_runner.model.model.layers),
            baseline_manifest_sha256=baseline.sha(args.baseline/'baseline_manifest.json'),
            driver_sha256=baseline.sha(Path(__file__)),source_sha256=baseline.sources(),
            output_token_ids=token_ids,baseline_output_matches=matches,
            chrome_trace=str(trace) if trace else None,gpu_after=baseline.gpu()))
        print(f'CAPTURE COMPLETE: {args.output}; {args.capture_steps} decode steps; baseline tokens match',flush=True)
    finally:
        if active:
            if profile:
                profile.stop()
            else:
                torch.cuda.cudart().cudaProfilerStop()
        baseline.close(llm)


if __name__ == '__main__':
    main()
