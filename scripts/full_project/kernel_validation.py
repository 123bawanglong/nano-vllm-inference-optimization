"""Exact native compiled RMSNorm + RoPE gate; records every numerical result."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
from scripts.runtime import OUT, FIXTURE, MODEL_FIXTURE

def metrics(a, b):
    d = (a.float()-b.float()).abs()
    return dict(exact=torch.equal(a,b), different=(a!=b).sum().item(),
                max_abs=d.max().item(), mean_abs=d.mean().item(), elements=a.numel(),
                finite=torch.isfinite(a).all().item())

@torch.inference_mode()
def main(label):
    spec = importlib.util.find_spec('src.qk_norm_rope') if (ROOT/'src').exists() else None
    assert spec is not None, 'Missing fused Q/K RMSNorm+RoPE implementation'
    from src.qk_norm_rope import fused, load_extension
    from nanovllm.layers.layernorm import RMSNorm
    from nanovllm.layers.rotary_embedding import RotaryEmbedding
    load_extension()
    fixture_path=FIXTURE
    data=torch.load(fixture_path,weights_only=True,map_location='cuda')
    qw,kw,cache=data['q_weight'],data['k_weight'],data['cos_sin_cache']
    qn,kn=RMSNorm(128,data['eps']).cuda(),RMSNorm(128,data['eps']).cuda()
    qn.weight=torch.nn.Parameter(qw);kn.weight=torch.nn.Parameter(kw)
    rope=RotaryEmbedding(128,128,4096,1000000.).cuda();rope.cos_sin_cache=cache
    cases=[(name,d['packed'],d['positions']) for name,d in data['fixtures'].items()]
    torch.manual_seed(20260919)
    for b in (1,2,4,8):
        for seed in range(10):
            generator=torch.Generator(device='cuda').manual_seed(20260919+seed)
            packed=torch.randn((b,4096),generator=generator,device='cuda',dtype=torch.bfloat16)
            pos=torch.tensor(([0,255,256,4095]*2)[:b],device='cuda')
            cases.append((f'packed_random_b{b}_seed{seed}',packed,pos))
        for kind,scale in [('zeros',0),('random',1),('small',1e-10),('large',1e10),('extreme',1e18)]:
            # Strided packed views include padding and a nonzero storage offset.
            backing=torch.randn((b,8192),device='cuda',dtype=torch.bfloat16)*scale
            packed=backing[:,1:8193:2][:,:4096]
            pos=torch.tensor(([0,255,256,4095]*2)[:b],device='cuda')
            cases.append((f'{kind}_b{b}',packed,pos))
    rows=[]
    for name,packed,pos in cases:
        q,k,_=packed.split([2048,1024,1024],-1)
        q=q.view(-1,16,128);k=k.view(-1,8,128)
        args=(q,k,qw,kw,pos,cache,data['eps'])
        before=packed.clone()
        expected=rope(pos,qn(q),kn(k))
        for warps in (1,4):
            actual=fused(*args,warps=warps)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gout=fused(*args,warps=warps)
            graph.replay();torch.cuda.synchronize()
            row=dict(case=name,warps=warps,q_stride=q.stride(),k_stride=k.stride(),
                errors={s:metrics(a,e) for s,a,e in zip(('q','k'),actual,expected)},
                graph_exact=all(torch.equal(a,g) for a,g in zip(actual,gout)),
                input_unchanged=torch.equal(before,packed))
            rows.append(row)
            print(name,warps,row['errors'],flush=True)
    # Metadata rejection must precede launching kernels.
    invalid=[('dtype',(q.float(),*args[1:])),('heads',(q[:,:15],*args[1:])),
             ('positions',(q,k,qw,kw,pos.int(),cache,data['eps'])),
             ('weight',(q,k,qw[:-1],kw,pos,cache,data['eps'])),
             ('cache_dtype',(q,k,qw,kw,pos,cache.bfloat16(),data['eps'])),
             ('epsilon',(q,k,qw,kw,pos,cache,float('nan'))),
             ('cpu',(q,k,qw.cpu(),kw,pos,cache,data['eps'])),
             ('zero_stride',(q,k,qw[:1].expand(128),kw,pos,cache,data['eps']))]
    guards=[]
    for name,bad in invalid:
        try: fused(*bad)
        except (RuntimeError,ValueError,TypeError): guards.append(name)
        else: raise AssertionError(f'Missing input guard: {name}')
    model_path=MODEL_FIXTURE
    model_data=torch.load(model_path,weights_only=True,map_location='cuda')
    model_rows=[]
    identity_cache=torch.cat((torch.ones((1,1,64),device='cuda'),torch.zeros((1,1,64),device='cuda')),-1)
    for sample in model_data['samples']:
        args=(sample['q'],sample['k'],sample['qw'],sample['kw'],sample['p'],model_data['cache'],sample['eps'])
        zero_positions=torch.zeros_like(sample['p'])
        for warps in (1,4):
            actual=fused(*args,warps=warps,reduction_mode='native_graph')
            norms=fused(*args[:4],zero_positions,identity_cache,args[-1],warps=warps,reduction_mode='native_graph')
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): gout=fused(*args,warps=warps,reduction_mode='native_graph')
            graph.replay();torch.cuda.synchronize()
            model_rows.append(dict(case=sample['case'],step=sample['step'],layer=sample['layer'],warps=warps,
                errors={s:metrics(a,sample[r]) for s,a,r in zip(('q','k','q_norm','k_norm'),(*actual,*norms),('rq','rk','nq','nk'))},
                graph_exact=all(torch.equal(a,g) for a,g in zip(actual,gout))))
    result=dict(label=label,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'src/qk_norm_rope').glob('*')) if p.is_file()},
        rows=rows,guards=guards,
        exact_gate=all(all(e['exact'] and e['finite'] for e in r['errors'].values())
                       and r['graph_exact'] and r['input_unchanged'] for r in rows))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'kernel_validation_{label}.json').write_text(json.dumps(result,indent=2))
    model_result=dict(label=label,rows=model_rows,fixture_path=str(model_path.relative_to(ROOT)),fixture_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        source_sha256=result['source_sha256'],validation_sha256=result['validation_sha256'],
        exact_gate=all(all(e['exact'] and e['finite'] for e in r['errors'].values()) and r['graph_exact'] for r in model_rows))
    (OUT/f'kernel_model_fixture_validation_{label}.json').write_text(json.dumps(model_result,indent=2))
    assert result['exact_gate'], 'Exact native compiled contract failed; see diagnostic JSON'
    assert model_result['exact_gate'], 'Actual captured model intermediate contract failed'
    print('PASS exact native compiled contract',len(rows),'cases',flush=True)
    print('PASS exact model captured norm/rope/graph contract',len(model_rows),'cases',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',default='current');main(p.parse_args().label)
