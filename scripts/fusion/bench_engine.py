"""Run each backend in a fresh process to avoid import/capture contamination."""
import argparse
import atexit
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'src/nano-vllm-cuda'


def prompts(batch, length, seed):
    rng = random.Random(seed)
    return [[rng.randint(100, 10000) for _ in range(length)] for _ in range(batch)]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['baseline', 'cuda'], required=True)
    parser.add_argument('--ops', default='all')
    parser.add_argument('--graph', type=int, default=1)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    os.environ['NANOVLLM_FUSED_OPS'] = args.ops
    sys.path.insert(0, str(REPO/'src/nano-vllm-upstream') if args.backend == 'baseline' else str(ROOT))
    import nanovllm
    from nanovllm import LLM, SamplingParams
    print('IMPORT', nanovllm.__file__, flush=True)
    torch.manual_seed(123)
    torch._dynamo.config.recompile_limit = 64
    llm = LLM(os.environ['NANOVLLM_MODEL'], enforce_eager=not args.graph,
              tensor_parallel_size=1, max_num_seqs=16, max_num_batched_tokens=4096,
              max_model_len=1024, gpu_memory_utilization=0.6)
    result_dir = REPO / 'results/fusion'
    result_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{args.backend}_{args.ops}_graph{args.graph}'
    if args.label:
        tag += '_' + args.label

    # Teacher forcing: both backends receive identical tokens at every step.
    logits = []
    def capture_logits(module, inputs, output):
        logits.append(output.detach().float().cpu())
    handle = llm.model_runner.model.lm_head.register_forward_hook(capture_logits)
    for prompt in prompts(2, 32, 101):
        llm.add_request(prompt, SamplingParams(max_tokens=4, ignore_eos=True))
    for step in range(4):
        seqs, prefill = llm.scheduler.schedule()
        llm.model_runner.call('run', seqs, prefill)
        llm.scheduler.postprocess(seqs, [200 + step] * len(seqs), prefill)
    assert llm.is_finished()
    handle.remove()
    torch.save(logits, result_dir / f'{tag}_logits.pt')

    # Warm up the complete generate path and all relevant batch sizes.
    for b, p in ((1, 128), (8, 256)):
        for warm in range(2):
            llm.generate(prompts(b, p, 700 + warm), SamplingParams(max_tokens=16, ignore_eos=True), use_tqdm=False)
    cases = []
    for batch, prompt_len, output_len in ((1, 128, 64), (8, 256, 64)):
        repetitions = []
        for rep in range(5):
            torch.manual_seed(500 + rep)
            ids = prompts(batch, prompt_len, 9000 + rep)
            params = SamplingParams(max_tokens=output_len, ignore_eos=True)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for prompt in ids:
                llm.add_request(prompt, params)
            prefill_ms, decode_ms, completed = [], [], []
            while not llm.is_finished():
                start = time.perf_counter()
                output, n = llm.step()
                torch.cuda.synchronize()
                duration = (time.perf_counter() - start) * 1000
                (prefill_ms if n > 0 else decode_ms).append(duration)
                completed.extend(output)
            # Include detokenization as generate() does.
            texts = [llm.tokenizer.decode(tokens) for _, tokens in completed]
            elapsed = (time.perf_counter() - begin) * 1000
            assert len(completed) == batch and all(len(tokens) == output_len for _, tokens in completed)
            assert all(isinstance(text, str) for text in texts)
            repetitions.append({'wall_ms': elapsed, 'output_tok_s': batch * output_len * 1000 / elapsed,
                                'prefill_step_ms': prefill_ms, 'decode_step_ms': decode_ms})
        cases.append({'batch': batch, 'prompt_len': prompt_len, 'output_len': output_len,
                      'repetitions': repetitions,
                      'median_wall_ms': statistics.median(r['wall_ms'] for r in repetitions),
                      'median_output_tok_s': statistics.median(r['output_tok_s'] for r in repetitions),
                      'median_decode_step_ms': statistics.median(v for r in repetitions for v in r['decode_step_ms'])})
        print(tag, json.dumps({k: v for k, v in cases[-1].items() if k != 'repetitions'}), flush=True)
    (result_dir / f'{tag}.json').write_text(json.dumps({
        'backend': args.backend, 'ops': args.ops, 'graph': args.graph, 'import': nanovllm.__file__,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'dtype': str(llm.model_runner.config.hf_config.dtype),
        'num_kvcache_blocks': llm.model_runner.config.num_kvcache_blocks,
        'memory_allocated_bytes': torch.cuda.memory_allocated(), 'cases': cases,
    }, indent=2))
    atexit.unregister(llm.exit)
    llm.exit()


if __name__ == '__main__':
    main()
