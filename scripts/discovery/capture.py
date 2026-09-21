"""Unfiltered native full-request discovery; no model/kernel substitution."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
from scripts.runtime import BASELINE
NSYS=os.environ.get('NSYS', 'nsys')
from scripts import baseline


def prepare(out):
    baseline.check_manifest(BASELINE)
    out.mkdir(parents=True,exist_ok=False)
    (out/'workloads.json').write_bytes((BASELINE/'workloads.json').read_bytes())
    (out/'PROTOCOL.md').write_bytes((ROOT/'docs/reproduce.md').read_bytes())
    baseline.dump(out/'manifest.json',dict(baseline_sha256=baseline.sha(BASELINE/'baseline_manifest.json'),
        workloads_sha256=baseline.sha(out/'workloads.json'),source_sha256=baseline.sources(),
        capture_sha256=baseline.sha(Path(__file__)),protocol_sha256=baseline.sha(out/'PROTOCOL.md'),
        config=baseline.CONFIG,nsys=NSYS,nsys_version=baseline.command(NSYS,'--version'),
        gpu=baseline.gpu(),runs=3,kernel_filter=None,variant='native',case_count=7))


def run(out,run_id,mode):
    import torch
    baseline.check_manifest(BASELINE)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['capture_sha256']==baseline.sha(Path(__file__))
    assert manifest['workloads_sha256']==baseline.sha(out/'workloads.json')
    fixtures=json.loads((out/'workloads.json').read_text())
    reference=json.loads((BASELINE/'process_1.json').read_text())['rows']
    cases=list(dict.fromkeys(x['name'] for x in fixtures))
    shift=(run_id-1)*2
    cases=cases[shift:]+cases[:shift]
    before=baseline.gpu()
    llm=baseline.engine()
    audit=dict(cache_hits=0,preemptions=0)
    allocate=llm.scheduler.block_manager.allocate
    def checked_allocate(seq,cached):
        audit['cache_hits']+=cached
        return allocate(seq,cached)
    llm.scheduler.block_manager.allocate=checked_allocate
    preempt=llm.scheduler.preempt
    def checked_preempt(seq):
        audit['preemptions']+=1
        return preempt(seq)
    llm.scheduler.preempt=checked_preempt
    original_step=llm.step
    records=[]
    active=False
    try:
        for name in cases:
            baseline.request(llm,next(x for x in fixtures if x['name']==name and x['repeat']==0))
        torch.cuda.synchronize()
        if mode=='profile':
            torch.cuda.cudart().cudaProfilerStart()
            active=True
        for name in cases:
            repeats=(1,) if mode=='profile' else (1,2)
            for repeat in repeats:
                item=next(x for x in fixtures if x['name']==name and x['repeat']==repeat)
                state=dict(index=0)
                def step():
                    idx=state['index']
                    torch.cuda.nvtx.range_push(f'DISCOVERY|{name}|{idx}')
                    try:
                        return original_step()
                    finally:
                        torch.cuda.nvtx.range_pop()
                        state['index']+=1
                if mode=='profile':llm.step=step
                row=baseline.request(llm,item)
                llm.step=original_step
                ref=next(r for r in reference if r['case']==name and r['repeat']==repeat)
                assert row['output_token_ids']==ref['output_token_ids'], (name,'baseline token mismatch')
                assert audit==dict(cache_hits=0,preemptions=0),audit
                if mode=='profile':assert state['index']==item['output']
                records.append(row)
                print(f"{mode} run={run_id} {name} repeat={repeat} ms={row['e2e_ms']:.2f} tokens_match=True",flush=True)
        if active:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            active=False
        baseline.dump(out/f'{mode}_{run_id}.json',dict(mode=mode,run_id=run_id,rows=records,
            cases_in_order=cases,audit=audit,all_tokens_match=True,gpu_before=before,gpu_after=baseline.gpu(),
            manifest_sha256=baseline.sha(out/'manifest.json')))
    finally:
        if active:torch.cuda.cudart().cudaProfilerStop()
        baseline.close(llm)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','timing','profile'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run-id',type=int,default=1)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.out)
    else:run(args.out,args.run_id,args.mode)
