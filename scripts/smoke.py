"""Run one request using the installed mixed-scheduling package."""
import argparse
import atexit
import os

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.environ.get('NANOVLLM_MODEL'))
    parser.add_argument('--prompt', default='Explain KV cache in one sentence.')
    parser.add_argument('--eager', action='store_true')
    args = parser.parse_args()
    if not args.model:
        parser.error('provide --model or set NANOVLLM_MODEL to a local model directory')
    engine = LLM(args.model, enforce_eager=args.eager, max_num_seqs=16,
                 max_num_batched_tokens=512, max_model_len=4096,
                 gpu_memory_utilization=0.6)
    try:
        print(engine.generate([args.prompt], SamplingParams(max_tokens=32), use_tqdm=False)[0]['text'])
    finally:
        atexit.unregister(engine.exit)
        engine.exit()


if __name__ == '__main__':
    main()
