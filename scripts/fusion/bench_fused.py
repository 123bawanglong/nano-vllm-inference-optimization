"""Original compiled functions vs CUDA. All compilation is outside timing."""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'src/nano-vllm-cuda'
sys.path.insert(0, str(ROOT))
from nanovllm.layers.cuda_fused import add_rms_norm, silu_and_mul


def original_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(REPO/'src/nano-vllm-upstream/nanovllm/layers'/filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_time(fn, iterations=100):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / iterations


def make_graph(fn, calls=64):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            output = fn()
    return g, output


@torch.inference_mode()
def main():
    torch.manual_seed(321)
    torch._dynamo.config.recompile_limit = 64
    Norm = original_module('baseline_norm', 'layernorm.py').RMSNorm
    Activation = original_module('baseline_activation', 'activation.py').SiluAndMul
    results = []
    profiles = []
    # Original FP32 add_rms_forward mutates x through x.float().add_().
    # Integration preserves that fallback; benchmark actual model dtypes here.
    for dtype in (torch.bfloat16, torch.float16):
        norm = Norm(1024).cuda().to(dtype)
        activation = Activation().cuda()
        for rows in (1, 8, 32, 128, 512, 2048):
            x = torch.randn(rows, 1024, device='cuda', dtype=dtype)
            r = torch.randn_like(x)
            gates = torch.randn(rows, 6144, device='cuda', dtype=dtype)
            cases = {
                'add_rms': {
                    'compiled': lambda: norm.add_rms_forward(x, r),
                    'cuda': lambda: add_rms_norm(x, r, norm.weight, norm.eps),
                    'eager': lambda: Norm.add_rms_forward.__wrapped__(norm, x, r),
                },
                'silu': {
                    'compiled': lambda: activation(gates),
                    'cuda': lambda: silu_and_mul(gates),
                    'eager': lambda: Activation.forward.__wrapped__(activation, gates),
                },
            }
            for op, funcs in cases.items():
                for fn in funcs.values():
                    for _ in range(10):
                        fn()
                torch.cuda.synchronize()
                orig, custom = funcs['compiled'](), funcs['cuda']()
                pairs = list(zip(orig, custom)) if op == 'add_rms' else [(orig, custom)]
                err = []
                for a, b in pairs:
                    tol = 2e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 2e-5
                    torch.testing.assert_close(a, b, rtol=tol, atol=tol)
                    err.append(float((a.float() - b.float()).abs().max()))
                graphs = {name: make_graph(fn) for name, fn in funcs.items()}
                samples = {name: {'graph_us': [], 'dispatch_us': []} for name in funcs}
                for rep in range(5):
                    order = list(funcs) if rep % 2 == 0 else list(reversed(funcs))
                    for name in order:
                        samples[name]['graph_us'].append(event_time(graphs[name][0].replay, 10) / 64)
                        samples[name]['dispatch_us'].append(event_time(funcs[name], 50))
                item = {'op': op, 'dtype': str(dtype), 'rows': rows, 'hidden': 1024 if op == 'add_rms' else 3072,
                        'max_abs_error_vs_compiled': err, 'samples': samples,
                        'median': {name: {metric: statistics.median(values) for metric, values in data.items()}
                                   for name, data in samples.items()}}
                results.append(item)
                if dtype == torch.bfloat16 and rows in (1, 512):
                    for name, fn in funcs.items():
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                            fn()
                            torch.cuda.synchronize()
                        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                        profiles.append({'op': op, 'rows': rows, 'backend': name,
                                         'gpu_events': [e.name for e in events], 'gpu_event_count': len(events)})
                print(json.dumps({k: item[k] for k in ('op', 'dtype', 'rows', 'median')}), flush=True)
                del graphs
    output = REPO / 'results/fusion'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'microbench.json').write_text(json.dumps({
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'graph_calls': 64, 'graph_replays': 10, 'rounds': 5,
        'notes': 'graph_us: amortized CUDA Graph device timing, hot buffers; dispatch_us includes GPU idle gaps from Python submission. Not HBM bandwidth.',
        'results': results, 'profiles': profiles,
    }, indent=2))


if __name__ == '__main__':
    main()
