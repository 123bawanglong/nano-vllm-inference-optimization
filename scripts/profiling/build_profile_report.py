"""Verify target attribution and summarize this specific native profiling run."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/profiling_20260918'
trace=json.loads((OUT/'torch_capture.trace.json').read_text())
events=sorted([e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X'],key=lambda e:e['ts'])
norm='triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0'
rope='triton_poi_fused__to_copy_add_cat_index_mul_split_sub_'
labels=['Q RMSNorm','K RMSNorm','Q RoPE','K RoPE','KV cache store']
expected_names=[norm,norm,rope+'0',rope+'1','store_kvcache_kernel']
expected_grids=[(16,1,1),(8,1,1),(8,1,1),(4,1,1),(1,1,1)]
expected_blocks=[(64,1,1),(64,1,1),(128,1,1),(128,1,1),(128,1,1)]
groups=collections.defaultdict(list)
for event in events:
    for label,name,grid,block in zip(labels,expected_names,expected_grids,expected_blocks):
        if event['name']==name and tuple(event['args']['grid'])==grid and tuple(event['args']['block'])==block:
            groups[label].append(event)
for label in labels:
    assert len(groups[label])==16*28, (label,len(groups[label]))
# Verify these are five consecutive nodes, repeated once per attention layer.
chains=[]
for i,event in enumerate(events):
    if event['name']==norm and event['args']['grid']==[16,1,1]:
        chain=events[i:i+5]
        assert [e['name'] for e in chain]==expected_names
        assert [tuple(e['args']['grid']) for e in chain]==expected_grids
        assert len({e['args']['correlation'] for e in chain})==1
        chains.append(chain)
assert len(chains)==16*28
graph_correlations=sorted({chain[0]['args']['correlation'] for chain in chains})
assert len(graph_correlations)==16
assert all(sum(chain[0]['args']['correlation']==c for chain in chains)==28 for c in graph_correlations)
total_us=sum(e['dur'] for e in events)
target_us=sum(e['dur'] for label in labels for e in groups[label])
timeline_rows=[dict(label=label,calls=len(groups[label]),calls_per_step=len(groups[label])/16,
    total_us=sum(e['dur'] for e in groups[label]),
    average_us=sum(e['dur'] for e in groups[label])/len(groups[label]),
    per_step_us=sum(e['dur'] for e in groups[label])/16,
    percent_gpu_kernel_time=100*sum(e['dur'] for e in groups[label])/total_us) for label in labels]

with (OUT/'ncu_details.csv').open() as f:
    details=list(csv.DictReader(f))
ids=sorted({r['ID'] for r in details},key=int)
assert ids==['0','1','2','3','4']
ncu_rows=[]
for index,label,name,grid,block in zip(ids,labels,expected_names,expected_grids,expected_blocks):
    data=[r for r in details if r['ID']==index]
    assert {r['Kernel Name'] for r in data}=={name}
    assert {r['Grid Size'] for r in data}=={str(grid)}
    assert {r['Block Size'] for r in data}=={str(block)}
    metrics={r['Metric Name']:dict(value=r['Metric Value'],unit=r['Metric Unit'])
             for r in data if r['Metric Name']}
    assert metrics['Duration']['unit']=='us'
    sections={r['Section Name'] for r in data}
    assert {'GPU Speed Of Light Throughput','Launch Statistics','Occupancy',
            'Memory Workload Analysis','Compute Workload Analysis','Scheduler Statistics','Warp State Statistics'} <= sections
    ncu_rows.append(dict(id=int(index),label=label,name=name,grid=grid,block=block,metrics=metrics,
        rules=[r['Rule Description'] for r in data if r.get('Rule Description')]))

for name in ('nsys_capture','torch_capture','ncu_capture'):
    record=json.loads((OUT/f'{name}.json').read_text())
    assert record['baseline_output_matches'] and record['graph']
    manifest=json.loads((ROOT/'results/baseline_20260918_170614/baseline_manifest.json').read_text())
    assert record['source_sha256']==manifest['source_sha256']
    for relative,expected in record['source_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected
evidence=dict(validated=True,graph_steps=16,layers_per_step=28,
    kernel_count=len(events),total_gpu_kernel_us=total_us,target_gpu_kernel_us=target_us,
    target_percent_gpu_kernel_time=100*target_us/total_us,
    target_kernels_per_layer=5,target_kernels_per_step=140,
    timeline=timeline_rows,ncu=ncu_rows,
    ncu_report_sha256=hashlib.sha256((OUT/'native_qk_rope_cache_graph.ncu-rep').read_bytes()).hexdigest(),
    settings=dict(graph_profiling='node',replay='kernel',cache_control='all',clock_control='none',passes_per_kernel=19))
(OUT/'profile_summary.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(f'VALIDATED {len(events)} GPU events; 16 graphs x 28 layers x 5 target kernels; target {100*target_us/total_us:.2f}%')
for row in timeline_rows:
    print(f"{row['label']}: {row['average_us']:.3f} us/call, {row['per_step_us']:.3f} us/step, {row['percent_gpu_kernel_time']:.2f}%")
for row in ncu_rows:
    print(row['id'],row['label'],row['grid'],row['metrics']['Duration'])

lines=['# Native decode profiling — Q/K Norm + RoPE + KV store','',
    '本次使用原版 BF16 Qwen3-0.6B、RTX 5080、FlashAttention、TP=1、CUDA Graph，KV 容量仍为 64×256。主负载 batch1/P64/O256；先完整预热，再采集第17–32步 decode（输入 position 80–95）。未实现 custom CUDA kernel。','',
    '## 结果与工具来源','',
    '- Nsight Systems 2024.6.2 报告没有 GPU kernel 数据；诊断明确提示当前驱动 CUDA 13.2 不受该版本支持。保留原始报告和 nsys_diagnostics.json，不能把它称为成功的 GPU timeline。',
    '- GPU timeline 来自独立进程的 torch.profiler，保留 CUDA Graph，实际采到 GPU kernel events。',
    '- Nsight Compute 2025.1.1 成功采集实际模型 Graph 第一层的 5 个目标节点。报告可用 Windows Nsight Compute 2026.2.1 打开。',
    '- 三次采集均完成整条256-token生成，输出 token ID 与保存的原版 baseline 完全一致；受冻结的原版源文件 hash 未变。','',
    '## 目标链时间线（torch.profiler）','',
    '| 操作 | 16步调用数 | 平均每次 µs | 每步累计 µs | 占全部 GPU kernel 累计时间 |',
    '|---|---:|---:|---:|---:|']
for r in timeline_rows:
    lines.append(f"| {r['label']} | {r['calls']} | {r['average_us']:.3f} | {r['per_step_us']:.3f} | {r['percent_gpu_kernel_time']:.2f}% |")
lines += ['',f'目标链合计每步 {target_us/16:.3f} µs，占本次窗口全部 GPU kernel 累计时间 {100*target_us/total_us:.2f}%。分母不是端到端 wall-clock 时间，不含 kernel 间空隙，也不是推理加速百分比。',
    '矩阵向量计算（cuBLAS GEMV）占本窗口累计 GPU kernel 时间约76.78%，是主要热点。Q/K 链是较小的融合候选，不能将它描述为主要瓶颈。',
    '归因经过核对：RMSNorm 同名 kernel 也被层级 norm 使用。这里用 grid/block 与连续执行顺序区分 Q/K，不能把所有同名 RMSNorm 的耗时算进融合目标。每个 graph 中有28条完整目标链，每条恰好5个节点。','',
    '## Nsight Compute 报告中看到的 5 项','',
    '| Report ID | 操作 | grid blocks | threads/block | NCU Duration µs |',
    '|---:|---|---:|---:|---:|']
for r in ncu_rows:
    lines.append(f"| {r['id']} | {r['label']} | {r['grid'][0]} | {r['block'][0]} | {r['metrics']['Duration']['value']} |")
lines += ['',
    'NCU 配置：graph-profiling=node、replay-mode=kernel、cache-control=all、clock-control=none；每个 kernel 19 passes。这里是清缓存并重放节点后的计数器/耗时，不能与 torch.profiler 的热态 duration 直接相加或替换正式 benchmark。小 kernel、多轮计数器采集和显示任务也会影响解释精度。',
    '五个节点的 grid 仅为 16/8/8/4/1 个 block，相比设备84个SM很小。NCU 提示 grid 太小，不能填满设备；不能仅因涉及显存读写就认定它们已经打满带宽。','',
    '## 如何在界面查看','',
    '打开 native_qk_rope_cache_graph.ncu-rep。先在 Summary 查看五条结果；双击 ID 0 查看 Q RMSNorm，ID 1 是 K RMSNorm；ID 2/3 是 Q/K RoPE；ID 4 是 cache 写入。',
    'Details 中先看 Launch Statistics（Grid/Block/Registers），再看 GPU Speed Of Light Throughput、Memory Workload Analysis 和 Occupancy。NCU 两条 RMSNorm 同名，使用 ID 和 grid 区分。','',
    '## 可验证的融合假设','',
    '若将每层5个目标节点合为1个，在目标路径可将每步140个节点降到28个，理论少112个节点，并减少 Norm→RoPE→store 的中间张量读写。这里指 Graph 内 GPU kernel 节点，不是每步少112次 Python/CUDA API 调用。',
    '这是待验证的设计假设，不是已经获得的性能收益。融合后仍需检查数值/cache正确性、真实内存流量、算子链热态时间和无profiler端到端配对测量。目标链约几个百分点的占比提示整体收益空间有限，不能承诺明显加速。','',
    '## 复现','',
    'WSL 下依次运行 scripts/profiling/run_timeline.sh、run_torch_timeline.sh、analyze_timeline.py、run_ncu.sh、build_profile_report.py。当前脚本针对本次结果目录，驱动和Profiler拒绝覆盖已有报告；重跑前改为新的输出目录。',
    '原始文件：torch_capture.trace.json、kernel_summary.json、ncu_raw.csv、ncu_details.csv、三个 *_capture.json 和 .ncu-rep。工具脚本与本次结果应一并保留。',
    '计数器重放/缓存解释参考：[NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
