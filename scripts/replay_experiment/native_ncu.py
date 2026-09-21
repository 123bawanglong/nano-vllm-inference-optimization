"""Capture one warmed real-model-context chain; no timing claims from NCU."""
import sys
import torch
from scripts.compile_boundary.common import baseline, ROOT

@torch.inference_mode()
def main(tag):
    llm = baseline.engine()
    try:
        attn = llm.model_runner.model.model.layers[0].self_attn
        data = torch.load(ROOT/'results/compile_boundary_20260918_233238/real_first_layer_fixtures.pt', map_location='cuda', weights_only=True)
        fixture = data['fixtures']['b1_p64_o256']
        q, k, _ = fixture['packed'].split([2048, 1024, 1024], -1)
        q, k = q.view(-1, 16, 128), k.view(-1, 8, 128)
        pos = fixture['positions']
        assert torch.equal(data['q_weight'], attn.q_norm.weight)
        assert torch.equal(data['k_weight'], attn.k_norm.weight)
        assert torch.equal(data['cos_sin_cache'], attn.rotary_emb.cos_sin_cache[:4096])
        if tag == 'native':
            fn = lambda: attn.rotary_emb(pos, attn.q_norm(q), attn.k_norm(k))
        else:
            from src.qk_norm_rope import fused, build
            build()
            fn = lambda: fused(q,k,attn.q_norm.weight,attn.k_norm.weight,pos,attn.rotary_emb.cos_sin_cache,attn.q_norm.eps,reduction_mode='native_graph')
        for _ in range(100): fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): fn()
        graph.replay(); torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        graph.replay(); torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print('NCU CAPTURED', tag, 'B1 Q16 K8 D128', flush=True)
    finally:
        baseline.close(llm)

if __name__ == '__main__': main(sys.argv[1])
