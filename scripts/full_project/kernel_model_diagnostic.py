"""Hold actual native CUDA Graph intermediates; compare after each graph replay.

Instrumentation changes tensor lifetimes only, never native arithmetic or source.
"""
import argparse
import json
from pathlib import Path
import torch
from scripts.compile_boundary.common import baseline, ROOT, error_metrics
from src.qk_norm_rope import build,fused

@torch.inference_mode()
def main(label, mode):
    from nanovllm.models.qwen3 import Qwen3Attention
    from nanovllm.utils.context import get_context
    original=Qwen3Attention.forward
    captures={};samples=[];rows=[]
    def instrument(self,positions,hidden):
        if not torch.cuda.is_current_stream_capturing():return original(self,positions,hidden)
        packed=self.qkv_proj(hidden)
        q,k,v=packed.split((self.q_size,self.kv_size,self.kv_size),-1)
        q=q.view(-1,self.num_heads,self.head_dim);k=k.view(-1,self.num_kv_heads,self.head_dim)
        v=v.view(-1,self.num_kv_heads,self.head_dim)
        nq,nk=self.q_norm(q),self.k_norm(k)
        rq,rk=self.rotary_emb(positions,nq,nk)
        captures.setdefault(hidden.shape[0],[]).append(dict(q=q,k=k,nq=nq,nk=nk,rq=rq,rk=rk,
          qw=self.q_norm.weight,kw=self.k_norm.weight,p=positions,cache=self.rotary_emb.cos_sin_cache,eps=self.q_norm.eps))
        return self.o_proj(self.attn(rq,rk,v).flatten(1,-1))
    Qwen3Attention.forward=instrument;build();llm=baseline.engine()
    runner=llm.model_runner;run_model=runner.run_model
    state=dict(step=0,case='')
    def inspect(ids,positions,is_prefill):
        output=run_model(ids,positions,is_prefill)
        if not is_prefill:
            for layer,c in enumerate(captures[ids.numel()]):
                args=(c['q'],c['k'],c['qw'],c['kw'],c['p'],c['cache'],c['eps'])
                actual=fused(*args,**({'reduction_mode':mode} if mode else {}))
                errors={s:error_metrics(a,c[r]) for s,a,r in zip(('q','k'),actual,('rq','rk'))}
                row=dict(case=state['case'],step=state['step'],layer=layer,batch=ids.numel(),
                    q_stride=c['q'].stride(),k_stride=c['k'].stride(),errors=errors)
                rows.append(row)
                mismatch=not all(e['exact'] for e in errors.values())
                if state['step']==1 or (mismatch and sum(s.get('mismatch',False) for s in samples)<8):
                    samples.append(dict(case=state['case'],step=state['step'],layer=layer,mismatch=mismatch,
                      **{name:c[name].cpu() for name in ('q','k','qw','kw','p','nq','nk','rq','rk')},eps=c['eps']))
                if mismatch:print('MISMATCH',row,flush=True)
        state['step']+=1
        return output
    runner.run_model=inspect
    workloads=json.loads((ROOT/'results/baseline_20260918_170614/workloads.json').read_text())
    try:
        for item in [x for x in workloads if x['repeat']==1 and x['name'] in ('b1_p64_o256','b1_p256_o256','b1_p2048_o32','b4_p64_o256','b8_p64_o256')]:
            state.update(step=0,case=item['name'])
            baseline.request(llm,dict(item,output=18))
        out=ROOT/'results/full_project_20260919'
        result=dict(label=label,reduction_mode=mode,rows=rows,exact_gate=all(all(e['exact'] for e in r['errors'].values()) for r in rows),
                    captured_layers={b:len(v) for b,v in captures.items()},torch=torch.__version__,
                    source_sha256={str(p.relative_to(ROOT)):baseline.sha(p) for p in (ROOT/'src/qk_norm_rope').glob('*') if p.is_file()})
        baseline.dump(out/f'kernel_model_diagnostic_{label}.json',result)
        torch.save(dict(samples=samples,cache=captures[1][0]['cache'][:4096].cpu()),out/f'kernel_model_fixtures_{label}.pt')
        print('MODEL INTERMEDIATE GATE',result['exact_gate'],'rows',len(rows),flush=True)
    finally: baseline.close(llm)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--mode',default=None)
    a=p.parse_args();main(a.label,a.mode)
