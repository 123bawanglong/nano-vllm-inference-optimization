"""Postprocessing only: validate and summarize the diagnostic experiment."""
import argparse
import collections
import csv
import json
from pathlib import Path
from scripts.compile_boundary.common import baseline, check, trace_kernels


def analyze(row):
    events = sorted(row['kernels'], key=lambda e:e['ts'])
    stores = [i for i,e in enumerate(events) if e['name']=='store_kvcache_kernel']
    assert len(stores)==row['layers']*row['steps'], (row['case'],row['phase'],len(stores))
    target, store_time = [], sum(events[i]['us'] for i in stores)
    groups = collections.defaultdict(list)
    for e in events:
        groups[e['name']].append(e)
    for idx in stores:
        if row['variant']=='native' or row['phase']=='prefill':
            chain=events[idx-4:idx]
            assert len(chain)==4
            assert all('mean_mul_pow_rsqrt' in e['name'] and 'cat_index' not in e['name'] for e in chain[:2]), (row['case'],row['phase'],chain)
            assert all('cat_index_mul_split_sub' in e['name'] for e in chain[2:]), (row['case'],row['phase'],chain)
        else:
            chain=events[idx-2:idx]
            assert all('cat_index_mean_mul_pow_rsqrt_split_sub' in e['name'] for e in chain), chain
        target.extend(chain)
    total=sum(e['us'] for e in events)
    target_time=sum(e['us'] for e in target)
    matmul=sum(e['us'] for e in events if 'gemm' in e['name'].lower() or 'gemv' in e['name'].lower())
    rank=sorted([dict(name=name,calls=len(items),total_us=sum(e['us'] for e in items),
                     pct=sum(e['us'] for e in items)*100/total) for name,items in groups.items()],
                key=lambda r:r['total_us'],reverse=True)
    return {k:row[k] for k in ('case','phase','variant','steps','graph','trace')} | dict(
        kernel_count=len(events), kernels_per_step=len(events)/row['steps'],
        total_kernel_us_per_step=total/row['steps'], norm_rope_us_per_step=target_time/row['steps'],
        norm_rope_pct=100*target_time/total, kv_store_pct=100*store_time/total,
        norm_rope_store_pct=100*(target_time+store_time)/total,
        gemm_gemv_pct=100*matmul/total, norm_rope_kernels_per_step=len(target)/row['steps'],
        top_kernels=rank)


def main(out):
    manifest=check(out)
    micro=json.loads((out/'micro.json').read_text())
    assert micro['complete'] and len(micro['rows'])==9
    provenance=json.loads((out/'micro_provenance.json').read_text())
    micro_dir=Path(provenance['directory'])
    assert baseline.sha(micro_dir/'micro.json')==provenance['result_sha256']
    old_manifest=json.loads((micro_dir/'manifest.json').read_text())
    # Added model diagnostic driver is recorded in the new manifest; the measured micro files are unchanged.
    for name,digest in old_manifest['experiment_sources'].items():
        assert baseline.sha(baseline.ROOT/name)==digest
    numeric={v:json.loads((out/f'numerical_{v}.json').read_text()) for v in ('native','joint')}
    assert all(d['complete'] and len(d['rows'])==9 for d in numeric.values())
    for data in numeric.values():
        assert data['audit']==dict(cache_hits=0,preemptions=0)
    for row in numeric['native']['rows']:
        assert baseline.sha(out/f"reference_{row['case']}.pt")==row['reference_sha256']
    profiles=[]
    for v in ('native','joint'):
        data=json.loads((out/f'profile_{v}.json').read_text())
        assert data['complete'] and len(data['rows'])==26
        assert data['audit']==dict(cache_hits=0,preemptions=0)
        for row in data['rows']:
            assert trace_kernels(out/row['trace'])==row['kernels'], row['trace']
        profiles.extend(analyze(row) for row in data['rows'])
    baseline.dump(out/'hotspots.json',dict(rows=profiles,
        denominator='sum of GPU kernel durations within each stage window, not end-to-end wall time'))
    fields=[k for k in profiles[0] if k!='top_kernels']
    with (out/'hotspots.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in profiles)
    lines=['# Norm → RoPE 编译边界实验（诊断结果）','',
        '结论：融合边界有可测的局部开销，但 Q/K Norm＋RoPE 并非本轮全链路的主要耗时。整体编译把4个kernel减少到2个；数值一致性检查未通过，因此本轮没有产生可宣称的端到端优化收益。','',
        '固定 Qwen3-0.6B、BF16、RTX5080、FlashAttention、CUDA Graph；KV cache为64页×256token。9种负载：B=1/2/4/8，P=64/256，O=256；另加B1/P2048/O32。7种原有输入保持原样，新增B2使用固定种子。只在decode修改编译边界；prefill和KV写入保持原样。','',
        '## 原版的全链路热点','',
        '下表分母是对应窗口的GPU kernel时间之和，不是请求端到端时间。早段采样decode17–32，晚段225–240；短输出控制采样16–31。完整kernel名称排序见 hotspots.json。','',
        '|负载|阶段|GEMM/GEMV占比|Norm＋RoPE占比|加KV写入占比|Norm＋RoPE μs/step|',
        '|---|---|---:|---:|---:|---:|']
    for r in profiles:
        if r['variant']=='native':
            lines.append(f"|{r['case']}|{r['phase']}|{r['gemm_gemv_pct']:.2f}%|{r['norm_rope_pct']:.2f}%|{r['norm_rope_store_pct']:.2f}%|{r['norm_rope_us_per_step']:.2f}|")
    lines+=['','## 真实QKV输入的局部Graph测量','',
        '第一层第一次decode的真实packed QKV视图；Q有16头、K有8头、head_dim128；保留实际stride。每个计时Graph含1000次链调用，10对交替顺序计时，展示中位数。热缓存微基准，GPU未锁频且兼作显示设备；不是模型加速预测。所有case原版4个kernel、整体编译2个kernel，原输入不被修改。','',
        '|负载|原版μs|整体编译μs|局部时间减少|Q最大绝对差|K最大绝对差|',
        '|---|---:|---:|---:|---:|---:|']
    for r in micro['rows']:
        a,b=r['median_us']['native'],r['median_us']['joint']
        lines.append(f"|{r['case']}|{a:.3f}|{b:.3f}|{(1-b/a)*100:.1f}%|{r['errors']['q']['max_abs']}|{r['errors']['k']['max_abs']}|")
    lines+=['','## 全模型数值检查','',
        '原版产生token历史，整体编译版本逐步teacher forcing相同历史。每一步对完整logits做字节哈希；选定步骤保存完整logits，并比较所有28层刚写入的KV槽位，覆盖首步、早期、末步和页边界。避免把采样分叉误当成同输入数值误差；未进行模型质量评测。','',
        '|负载|logits完全一致步数/总步数|抽查logits最大绝对差|抽查KV最大绝对差|',
        '|---|---:|---:|---:|']
    for r in numeric['joint']['rows']:
        selected=[s for s in r['comparisons'] if 'logits' in s]
        lines.append(f"|{r['case']}|{r['exact_steps']}/{r['step_count']}|{max(x['logits']['max_abs'] for x in selected)}|{max(x['written_cache_all_layers']['max_abs'] for x in selected)}|")
    lines+=['','## 决策及限制','',
        '- 这组workload可以量化候选链的调用次数、累计耗时和随batch/context变化的占比。需要把“可融合候选”与“全局最大热点”分开表述。',
        '- 原版每层4个Norm/RoPE节点；整体编译每层2个，28层每个decode step减少56个GPU Graph节点。CUDA Graph已经启用，不能把节点减少说成减少56次CPU启动调用。',
        '- 编译器已能融合每一路Norm→RoPE，Q/K仍分两个kernel；后续手写kernel的价值须和这个编译版本、原版分别比较，并处理数值契约。KV写入没有融合。',
        '- 生成代码显示整体编译省略了Norm输出落地BF16时的中间舍入。这是已确认的计算差异，但没有证明它解释了全部差异。',
        '- exact equality是本次预先选定的严格门槛，失败不等于模型质量一定下降；需要另行定义有根据的容差/质量验收才能接受非逐位相等的实现。本轮没有事后放宽标准。',
        '- 依据事先门槛，未执行10对无profiler端到端性能测试。带profiler时间和局部加速均不应当作已验证的端到端收益。',
        '- 本次用torch.profiler采集真实GPU时间线。此前2024.6.2版Nsight Systems采集缺少GPU事件；这不表示其他已安装版本不可用。',
        '- native/joint各26份阶段trace均保留gzip原始记录，并重新解析校验汇总。每个阶段窗口采集一次；波动和长尾可能影响均值，不能仅凭单次早/晚窗口差异推出稳定的context或batch趋势。全部配置均报告，没有挑选最有利场景。',
        '', '## 复现与证据','',
        '实验入口：scripts/compile_boundary/run_micro.sh 与 run_model_study.sh；后者引用前者固定的结果。数值诊断和profiling分开进程运行。原baseline源文件哈希验证通过，新增实验代码单独记录manifest。',
        '', '原始数据：micro.json、numerical_native.json、numerical_joint.json、profile_native.json、profile_joint.json、hotspots.csv。模型参考张量及compiler_evidence仅供本地复核，不应默认提交Git。compiler_evidence保存的是Inductor自动生成代码，非手写kernel。',
        '', '精度机制的通用背景：[PyTorch官方说明](https://pytorch.org/blog/training-production-ai-models/)指出融合可能改变浮点运算结果；本实验具体的中间舍入变化依据本地实际生成代码判断。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    baseline.dump(out/'verification.json',dict(baseline_unchanged=True,experiment_sources_match=True,
        micro_sources_unchanged=True,workloads=9,profile_windows=52,
        raw_traces_reparsed=True,reference_hashes_verified=True,
        report_driver_sha256=baseline.sha(Path(__file__)),
        micro_exact_gate=micro['exact_gate'],model_exact_gate=all(r['exact_steps']==r['step_count'] for r in numeric['joint']['rows']),
        formal_performance_pairs_run=0,reason='predeclared numerical gate failed'))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
