"""Rank every captured kernel; semantic attribution validates complete model structure."""
import argparse
import bisect
import collections
import csv
import json
import sqlite3
import statistics
from pathlib import Path
from scripts.discovery.capture import ROOT, BASELINE, baseline


def quantile(values,p):
    a=sorted(values)
    x=(len(a)-1)*p
    lo=int(x);hi=min(lo+1,len(a)-1)
    return a[lo]+(a[hi]-a[lo])*(x-lo)


def phase_names(step,output):
    if step==0:return ['full_request','prefill']
    phases=['full_request','decode_all']
    if output==256:
        if step<=32:phases.append('decode_early')
        if step>=224:phases.append('decode_late')
    else:
        phases.append('decode_early' if step<=15 else 'decode_late')
    return phases


def is_norm(name):
    return 'mean_mul_pow_rsqrt' in name or 'mean_pow_rsqrt' in name


def categories(events):
    """Return one mutually exclusive semantic category per event; no double counting."""
    names=[e['name'] for e in events]
    cats=['other']*len(events)
    stores=[i for i,n in enumerate(names) if n=='store_kvcache_kernel']
    assert len(stores)==28, ('store count',len(stores))
    for i in stores:
        assert i>=4 and all(is_norm(n) for n in names[i-4:i-2]), names[i-4:i+1]
        assert all('cat_index_mul_split_sub' in n for n in names[i-2:i]), names[i-4:i+1]
        cats[i-4:i+1]=['qk_rmsnorm','qk_rmsnorm','rope','rope','kv_write']
    for i,n in enumerate(names):
        if cats[i]!='other':continue
        if is_norm(n):cats[i]='residual_add_rms_or_input_final_norm'
        elif 'mul_silu_split' in n:cats[i]='silu_mul'
        elif 'flash::' in n or 'flash_fwd' in n or 'flash_fwd' in n.lower():cats[i]='attention'
        elif 'gemm' in n.lower() or 'gemv' in n.lower() or 'cutlass::' in n or 'cublasLt::' in n:cats[i]='matrix_products'
    norms=[i for i,c in enumerate(cats) if c=='residual_add_rms_or_input_final_norm']
    assert len(norms)==57, ('other norm count',len(norms))
    cats[norms[0]]='input_rmsnorm'
    cats[norms[-1]]='final_add_rmsnorm'
    for i in norms[1:-1]:cats[i]='residual_add_rmsnorm'
    acts=[i for i,c in enumerate(cats) if c=='silu_mul']
    assert len(acts)==28, ('activation count',len(acts))
    # Model ends at final norm; then LM head, native sampler and possible index/copy kernels.
    after=norms[-1]+1
    for i in range(after,len(events)):
        n=names[i]
        if n.startswith('triton_') and any(s in n for s in ('softmax','argmax','exponential')):
            cats[i]='sampling'
    head=[i for i in range(after,len(events)) if cats[i]=='matrix_products']
    assert head, ('LM head missing',names[after:])
    # The model has ended; compute_logits finishes at LM head, then run calls Sampler.
    # Include softmax's amax preparation kernels whose names lack 'softmax'.
    assert any(c=='sampling' for c in cats[max(head)+1:]), ('sampler missing',names[after:])
    for i in range(max(head)+1,len(events)):
        assert cats[i] in ('sampling','other') and (names[i].startswith('triton_') or 'distribution_elementwise_grid_stride_kernel' in names[i]), names[i]
        cats[i]='sampling'
    for i in head:cats[i]='matrix_lm_head'
    for store in stores:
        previous=max(i for i in norms if i<store-4)
        following=min(i for i in norms if i>store)
        assert any(cats[i]=='matrix_products' for i in range(previous+1,store-4)), 'Missing QKV projection'
        assert any(cats[i]=='matrix_products' for i in range(store+1,following)), 'Missing attention output projection'
        for i in range(previous+1,store-4):
            if cats[i]=='matrix_products':cats[i]='matrix_qkv'
        for i in range(store+1,following):
            if cats[i]=='matrix_products':cats[i]='matrix_attention_out'
    for act in acts:
        previous=max(i for i in norms if i<act)
        following=min(i for i in norms if i>act)
        assert any(cats[i]=='matrix_products' for i in range(previous+1,act)), 'Missing gate/up projection'
        assert any(cats[i]=='matrix_products' for i in range(act+1,following)), 'Missing down projection'
        for i in range(previous+1,act):
            if cats[i]=='matrix_products':cats[i]='matrix_gate_up'
        for i in range(act+1,following):
            if cats[i]=='matrix_products':cats[i]='matrix_down'
    assert 'matrix_products' not in cats, 'Unattributed matrix kernel'
    assert sum(c=='qk_rmsnorm' for c in cats)==56
    assert sum(c=='rope' for c in cats)==56
    return cats


def summary(values,total):
    return dict(calls=len(values),total_us=sum(values),share_pct=sum(values)*100/total,
        mean_us=statistics.mean(values),median_us=statistics.median(values),
        p95_us=quantile(values,.95),max_us=max(values),
        top_one_pct_of_group=max(values)*100/sum(values))


def process(out,run_id):
    path=out/f'native_{run_id}.sqlite'
    db=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)
    tables=[r[0] for r in db.execute("select name from sqlite_master where type='table'")]
    assert 'CUPTI_ACTIVITY_KIND_KERNEL' in tables, 'No GPU kernel table'
    strings=dict(db.execute('select id,value from StringIds'))
    ranges=[]
    for start,end,text,text_id in db.execute('select start,end,text,textId from NVTX_EVENTS where end is not null order by start'):
        label=text or strings.get(text_id,'')
        if label.startswith('DISCOVERY|'):
            _,case,index=label.split('|')
            ranges.append(dict(start=start,end=end,case=case,index=int(index),kernels=[]))
    assert len(ranges)==1568, ('step range count',len(ranges))
    starts=[r['start'] for r in ranges]
    extras=[]
    count=0
    sql='select start,end,demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ,registersPerThread,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start'
    for start,end,name,*other in db.execute(sql):
        i=bisect.bisect_right(starts,start)-1
        event=dict(start=start,end=end,name=strings[name],us=(end-start)/1000,
                   grid=other[:3],block=other[3:6],registers=other[6],stream=other[7])
        if i<0 or end>ranges[i]['end']:
            extras.append(event)
        else:ranges[i]['kernels'].append(event)
        count+=1
    baseline.dump(out/f'unmatched_{run_id}.json',extras)
    assert not extras, f'{len(extras)} GPU kernels outside step ranges; attribution invalid'
    fixtures=json.loads((out/'workloads.json').read_text())
    expected={x['name']:x['output'] for x in fixtures}
    for case,output in expected.items():
        assert sorted(r['index'] for r in ranges if r['case']==case)==list(range(output))
    groups=collections.defaultdict(list)
    category_groups=collections.defaultdict(list)
    gpu_totals=collections.defaultdict(float)
    step_stats=[]
    examples={}
    for r in ranges:
        events=r['kernels']
        cats=categories(events)
        if r['index'] in (0,1):
            examples[f"{r['case']}/step{r['index']}"]=[dict(name=e['name'],category=c,grid=e['grid'],block=e['block'],registers=e['registers']) for e,c in zip(events,cats)]
        total=sum(e['us'] for e in events)
        per_cat=collections.defaultdict(float)
        for e,c in zip(events,cats):per_cat[c]+=e['us']
        step_stats.append(dict(case=r['case'],step=r['index'],wall_us=(r['end']-r['start'])/1000,
            kernels=len(events),summed_kernel_us=total,categories_us=dict(per_cat)))
        for phase in phase_names(r['index'],expected[r['case']]):
            key=r['case'],phase
            gpu_totals[key]+=total
            for e,c in zip(events,cats):
                groups[key+(e['name'],)].append(e['us'])
                category_groups[key+(c,)].append(e['us'])
    name_rows=[];category_rows=[]
    for (case,phase,name),values in groups.items():
        name_rows.append(dict(run=run_id,case=case,phase=phase,kernel=name,**summary(values,gpu_totals[(case,phase)])))
    for (case,phase,cat),values in category_groups.items():
        category_rows.append(dict(run=run_id,case=case,phase=phase,category=cat,**summary(values,gpu_totals[(case,phase)])))
    for key in gpu_totals:
        rows=[r for r in name_rows if (r['case'],r['phase'])==key]
        for rank,row in enumerate(sorted(rows,key=lambda x:x['total_us'],reverse=True),1):row['time_rank']=rank
        for rank,row in enumerate(sorted(rows,key=lambda x:(x['calls'],x['total_us']),reverse=True),1):row['count_rank']=rank
    baseline.dump(out/f'analysis_{run_id}.json',dict(run=run_id,kernel_count=count,step_count=len(ranges),
        unmatched=0,names=name_rows,categories=category_rows,steps=step_stats,structure_examples=examples))
    return name_rows,category_rows


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main(out):
    baseline.check_manifest(BASELINE)
    names=[];cats=[]
    for run in (1,2,3):
        n,c=process(out,run);names+=n;cats+=c
        print(f'Validated run {run}: {len(n)} phase/name groups',flush=True)
    write_csv(out/'ranking_by_time.csv',sorted(names,key=lambda r:(r['case'],r['phase'],r['run'],r['time_rank'])))
    write_csv(out/'ranking_by_count.csv',sorted(names,key=lambda r:(r['case'],r['phase'],r['run'],r['count_rank'])))
    write_csv(out/'category_runs.csv',cats)
    grouped=collections.defaultdict(list)
    for r in cats:grouped[(r['case'],r['phase'],r['category'])].append(r)
    aggregate=[]
    for (case,phase,cat),rows in grouped.items():
        assert len(rows)==3
        d=dict(case=case,phase=phase,category=cat,calls=rows[0]['calls'])
        for metric in ('share_pct','total_us','median_us','p95_us','max_us'):
            values=[r[metric] for r in rows]
            d[metric+'_median']=statistics.median(values)
            d[metric+'_min']=min(values);d[metric+'_max']=max(values)
        aggregate.append(d)
    write_csv(out/'category_summary.csv',aggregate)
    name_groups=collections.defaultdict(list)
    for row in names:name_groups[(row['case'],row['phase'],row['kernel'])].append(row)
    name_summary=[]
    for (case,phase,kernel),rows in name_groups.items():
        assert len(rows)==3, (case,phase,kernel,len(rows))
        name_summary.append(dict(case=case,phase=phase,kernel=kernel,
            calls_min=min(r['calls'] for r in rows),calls_max=max(r['calls'] for r in rows),
            share_pct_median=statistics.median(r['share_pct'] for r in rows),
            share_pct_min=min(r['share_pct'] for r in rows),share_pct_max=max(r['share_pct'] for r in rows),
            total_us_median=statistics.median(r['total_us'] for r in rows),
            median_us=statistics.median(r['median_us'] for r in rows),
            p95_us=statistics.median(r['p95_us'] for r in rows),max_us=max(r['max_us'] for r in rows)))
    write_csv(out/'kernel_summary.csv',sorted(name_summary,key=lambda r:(r['case'],r['phase'],-r['total_us_median'])))
    baseline.dump(out/'summary.json',dict(categories=aggregate,
        denominator='sum of GPU kernel durations inside per-stage request steps',
        overlapping_phases='full_request=prefill+decode_all; early/late are subsets of decode_all, never add them again',
        source_unchanged=True,unfiltered=True))
    print('SUMMARY COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run-id',type=int)
    args=p.parse_args()
    if args.run_id:
        process(args.out,args.run_id)
        print(f'Run {args.run_id} validated',flush=True)
    else:main(args.out)
