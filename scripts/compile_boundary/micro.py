"""Real first-layer fixtures and Graph benchmark of original versus joint compile."""
import argparse
import json
import statistics
from pathlib import Path
import torch
from scripts.compile_boundary.common import (baseline, check, make_joint, error_metrics,
                                            save_trace, trace_kernels)


@torch.inference_mode()
def main(out):
    check(out)
    from nanovllm.utils.context import get_context
    llm = baseline.engine(eager=True)
    attn = llm.model_runner.model.model.layers[0].self_attn
    captures = {}
    current = {}
    handles = []

    def position_hook(module, inputs):
        if not get_context().is_prefill:
            current['positions'] = inputs[0].detach().clone()

    def qkv_hook(module, inputs, output):
        if not get_context().is_prefill and current['case'] not in captures:
            captures[current['case']] = dict(packed=output.detach().clone(),
                                              positions=current['positions'])

    handles.extend([attn.register_forward_pre_hook(position_hook),
                    attn.qkv_proj.register_forward_hook(qkv_hook)])
    results = []
    try:
        items = [x for x in json.loads((out / 'workloads.json').read_text()) if x['repeat'] == 0]
        for item in items:
            current['case'] = item['name']
            baseline.request(llm, dict(item, output=2))
        for h in handles:
            h.remove()
        handles.clear()
        torch.save(dict(fixtures={name:{k:v.cpu() for k,v in data.items()} for name,data in captures.items()},
             q_weight=attn.q_norm.weight.cpu(), k_weight=attn.k_norm.weight.cpu(),
             eps=attn.q_norm.eps, cos_sin_cache=attn.rotary_emb.cos_sin_cache[:4096].cpu()),
             out / 'real_first_layer_fixtures.pt')
        joint = make_joint()
        for item in items:
            name = item['name']
            data = captures[name]
            packed, positions = data['packed'], data['positions']
            before = packed.clone()
            q, k, _ = packed.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q, k = q.view(-1, attn.num_heads, attn.head_dim), k.view(-1, attn.num_kv_heads, attn.head_dim)
            def native():
                return attn.rotary_emb(positions, attn.q_norm(q), attn.k_norm(k))
            def fused():
                return joint(q, k, positions, attn.q_norm, attn.k_norm, attn.rotary_emb)
            functions = dict(native=native, joint=fused)
            graphs, outputs = {}, {}
            for variant, fn in functions.items():
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[variant] = fn()
                graphs[variant] = graph
                graph.replay()
            torch.cuda.synchronize()
            errors = {label:error_metrics(outputs['joint'][i], outputs['native'][i]) for i,label in enumerate(('q','k'))}
            assert torch.equal(packed, before), 'Operator destroyed input'
            times = {'native': [], 'joint': []}
            # A repeated graph makes submission overhead negligible for these tiny chains.
            timing_graphs = {}
            for variant, fn in functions.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(1000):
                        fn()
                timing_graphs[variant] = graph
                graph.replay()
            torch.cuda.synchronize()
            for repeat in range(10):
                for variant in (('native','joint') if repeat % 2 == 0 else ('joint','native')):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    timing_graphs[variant].replay()
                    end.record()
                    end.synchronize()
                    times[variant].append(start.elapsed_time(end))  # ms/1000*1000 = us per chain
            kernel_data = {}
            for variant, graph in graphs.items():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                    graph.replay()
                    torch.cuda.synchronize()
                path = out / f'micro_{name}_{variant}.json.gz'
                save_trace(prof, path)
                kernel_data[variant] = trace_kernels(path)
                assert kernel_data[variant]
            row = dict(case=name, q_shape=list(q.shape), q_stride=list(q.stride()),
                       k_shape=list(k.shape), k_stride=list(k.stride()),
                       k_offset=k.storage_offset(), positions=positions.cpu().tolist(),
                       errors=errors, input_unchanged=True, times_us=times,
                       median_us={v:statistics.median(t) for v,t in times.items()}, kernels=kernel_data,
                       timing_method='1000 chain invocations in a graph, 10 paired graph timings')
            results.append(row)
            baseline.dump(out / 'micro.json', dict(rows=results, complete=len(results)==len(items),
                exact_gate=all(e['exact'] and e['finite'] for r in results for e in r['errors'].values())))
            print(f"MICRO {name} errors={errors} medians={row['median_us']} kernel_counts="
                  f"{ {v:len(k) for v,k in kernel_data.items()} }", flush=True)
            del timing_graphs, graphs, outputs
    finally:
        for h in handles:
            h.remove()
        baseline.close(llm)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
