"""Build decision report from all raw runs, with separate idle-gated confirmation."""
import argparse
import collections
import json
import statistics
from pathlib import Path
from scripts.discovery.capture import ROOT, baseline

LABELS=dict(matrix_qkv='QKV投影',matrix_attention_out='O投影',matrix_gate_up='MLP gate/up投影',
    matrix_down='MLP down投影',matrix_lm_head='LM head',qk_rmsnorm='Q/K RMSNorm',rope='RoPE',
    kv_write='KV写入',attention='FlashAttention',residual_add_rmsnorm='层间AddRMS',
    input_rmsnorm='首层RMS',final_add_rmsnorm='最终AddRMS',silu_mul='SiLU×up',sampling='采样链',other='其他')
MATRIX=[k for k in LABELS if k.startswith('matrix_')]


def main(out):
    data=out/'confirmation'
    verification=json.loads((out/'verification.json').read_text())
    analyses=[json.loads((data/f'analysis_{i}.json').read_text()) for i in (1,2,3)]
    cases=list(dict.fromkeys(r['case'] for r in analyses[0]['steps']))
    def costs(case,phase,categories):
        return [sum(r['share_pct'] for r in a['categories'] if r['case']==case and r['phase']==phase and r['category'] in categories) for a in analyses]
    def fmt(values):return f'{statistics.median(values):.2f}% [{min(values):.2f}, {max(values):.2f}]'
    def values(category,phase='decode_all'):
        return [statistics.median(costs(c,phase,category)) for c in cases]
    def interval(category,phase='decode_all'):
        v=values(category,phase);return f'{min(v):.2f}–{max(v):.2f}%'
    timing_rows=[]
    for case in cases:
        record=dict(case=case)
        for directory,label in ((out,'initial'),(data,'confirmation')):
            per_run=[]
            for run in (1,2,3):
                rows=json.loads((directory/f'timing_{run}.json').read_text())['rows']
                matching=[r for r in rows if r['case']==case]
                assert len(matching)==2
                per_run.append(statistics.mean(r['e2e_ms'] for r in matching))
            record[label]=per_run
        timing_rows.append(record)
    kernel_count=sum(a['kernel_count'] for a in analyses)
    decision=dict(matrix_decode_share=interval(MATRIX),norm_rope_decode_share=interval(['qk_rmsnorm','rope']),
        norm_rope_cache_decode_share=interval(['qk_rmsnorm','rope','kv_write']),sampler_decode_share=interval(['sampling']),
        hidden_norm_decode_share=interval(['residual_add_rmsnorm','final_add_rmsnorm','input_rmsnorm']),
        silu_decode_share=interval(['silu_mul']),attention_decode_share=interval(['attention']),
        prefill_final_norm_share=interval(['final_add_rmsnorm'],'prefill'),
        selected='Q/K RMSNorm→RoPE prototype',not_largest_hotspot=True,
        reason='Repeated measurable4-node chain, exclusive intermediate consumers, bounded implementation; validate numerical contract first',
        improvement_proven=False,prior_targeted_results_used_as_new_measurements=False)
    baseline.dump(out/'decision.json',decision)
    lines=['# 全链路热点发现与优化对象选择','',
        '**结论：矩阵计算是主要耗时；Q/K RMSNorm→RoPE 是本项目适合优先验证的融合原型，理由是独立边界、中间张量消费者明确和实现范围可控。它不是最大的热点，也尚未证明优化有效。**','',
        '本轮执行“先找耗时→再找可消除开销→比较多个候选→选择原型”。这是带有历史知识的重新评估，不把此前定向实验包装成首次盲目发现。全链路采集没有按目标kernel过滤，候选表包含其他可选方向。','',
        '## 1. 实际执行与数据质量','',
        '- 原有7种固定随机token负载：B=1/4/8 × P=64/256 × O=256；另有B1/P2048/O32。保留原始输入ID、seed及采样设置。',
        '- Qwen3-0.6B、BF16、FlashAttention、TP1、CUDA Graph、64 KV block×256 token；本轮只运行native，未接入整体编译或自写kernel。',
        '- Nsight Systems2026.5.1，CUDA Graph node级采集，覆盖完整请求的全部GPU kernel；每step用NVTX标记，逐条检查GPU事件完整落在step内。',
        '- 初始3轮观察到模型加载前GPU高占用和计时波动：timing第1/2轮启动前GPU99%，第3轮9%；第2轮的实际计时已接近第3轮。快照不能单独证明测量期间干扰的因果关系，全部原样保留，不删除慢样本。',
        '- 另在运行前声明固定3轮确认：每个timing/profile进程启动前检查5次GPU利用率（间隔2秒），全部≤20%才启动。本次6次检查均通过，实测7–10%。该检查不证明运行中绝无后台干扰。',
        f'- 确认集包含21个完整profile请求、{sum(a["step_count"] for a in analyses):,}个模型step、{kernel_count:,}条GPU kernel事件，0条未归属事件；另有3进程×7case×2次=42个无profiler计时请求。',
        '- 权重文件、22个baseline源文件、固定workload和软件包版本均复核；全部native输出token与原baseline一致，0prefix复用、0抢占。10项基础/分析测试通过。',
        '- 本轮先warmup全部case，再测量；不把它和旧baseline数值的差异称作性能变化。带profiler的kernel统计与无profiler的请求延迟分开。',
        '- GPU供Windows显示共同使用且未锁频。kernel持续时间包含可能的调度停顿；median/p95/max均保留，不能把个别长尾全部归因于kernel计算效率。','',
        '## 2. 先看实际耗时','',
        '下表使用确认集完整decode的GPU kernel时间总和为分母。数字是3轮占比的中位数，方括号为最小/最大；不是端到端时间占比。各组合先在每轮内相加，再计算3轮统计，避免错误地相加各类别中位数。','',
        '|配置|全部矩阵投影|FlashAttention|Norm＋RoPE|加KV写入|采样链|',
        '|---|---:|---:|---:|---:|---:|']
    for case in cases:
        lines.append('|'+case+'|'+ '|'.join(fmt(costs(case,'decode_all',cats)) for cats in (MATRIX,['attention'],['qk_rmsnorm','rope'],['qk_rmsnorm','rope','kv_write'],['sampling']))+'|')
    lines+=['','Prefill、early/late分别分析；early/late为完整decode的子集，完整请求=一次prefill+所有decode，绝不重复累加。O256的early为1–32、late224–255；O32为1–15/16–31。以下Norm＋RoPE占比用于判断阶段差异，不能用单个阶段代表全请求。','',
        '|配置|Prefill|早期decode|晚期decode|完整请求|','|---|---:|---:|---:|---:|']
    for case in cases:
        lines.append('|'+case+'|'+'|'.join(fmt(costs(case,p,['qk_rmsnorm','rope'])) for p in ('prefill','decode_early','decode_late','full_request'))+'|')
    representative='b1_p64_o256'
    category_rows=[]
    for category in LABELS:
        rows=[next(r for r in a['categories'] if r['case']==representative and r['phase']=='decode_all' and r['category']==category) for a in analyses]
        category_rows.append(dict(category=category,share=statistics.median(r['share_pct'] for r in rows),
            calls=rows[0]['calls']/255,median_us=statistics.median(r['median_us'] for r in rows),
            p95_us=statistics.median(r['p95_us'] for r in rows),max_us=max(r['max_us'] for r in rows)))
    lines+=['','以B1/P64完整decode为例，将同名kernel按实际调用位置拆成源码角色后，累计时间排名如下。矩阵投影包括LM head，而非只有decoder层内GEMV。median/p95先在每轮该角色的单个GPU kernel事件duration分布上计算，再取三轮中位数；不是整条链的耗时分布。','',
        '|源码角色|GPU时间占比|kernel数/step|单次median μs|单次p95 μs|三轮最大值 μs|',
        '|---|---:|---:|---:|---:|---:|']
    for r in sorted(category_rows,key=lambda x:x['share'],reverse=True):
        lines.append(f"|{LABELS[r['category']]}|{r['share']:.2f}%|{r['calls']:g}|{r['median_us']:.3f}|{r['p95_us']:.3f}|{r['max_us']:.3f}|")
    lines+=['','## 3. 再看调用次数与可消除的开销','',
        '- 两套完整原始kernel排名分别在 `confirmation/ranking_by_time.csv`、`confirmation/ranking_by_count.csv`；包括每case、每phase、每run的全部名称、次数、累计时间、median/p95/max。`kernel_summary.csv`汇总三轮：median/p95列是各轮对应统计量的中位数，不是合并所有事件后的分位数；max为跨轮最大值。',
        '- Q/K Norm各1个kernel、Q/K RoPE各1个kernel，每层4个、每step112个；KV store另28个。通过每层“2norm→2rope→store”实际顺序验证，未把其他同名Norm算进去。',
        '- 层间AddRMS55个、最终AddRMS1个、首层plain RMS1个；SiLU×up28个，实测均已在单一kernel内完成各自调用。采样B1每step7个kernel、B4/B8每step6个；一次linear也可能展开多个kernel。',
        '- Norm输出只被紧邻RoPE消费，独立kernel边界将normalized Q/K写回再读。融合的具体假设是保留这一中间结果继续旋转；必要的归约、RoPE运算和最终Q输出仍要执行。',
        '- Q16头、K8头、head_dim128、BF16：可研究移除的normalized Q/K逻辑写+读为每层每token12KiB，28层合计336KiB/token。它是按张量大小估算的global-memory访问，不是实测DRAM流量，缓存可能已经吸收一部分。',
        '- decode的旋转K只用于cache写入，因此可在后续阶段研究直接写cache；普通prefill的K还被attention直接读取。V的必要搬运、slot=-1和分页边界也必须处理。',
        '- 采样生成代码仍有全词表FP32临时缓冲和多阶段归约；但指数随机数和最后除法已融合进末kernel，不能假设所有源码中间张量都落地。详细证据见源码审计及 `compiler_evidence/`。','',
        '## 4. 同一张表比较候选','',
        '|候选|确认集完整decode占比范围*|能改善什么/当前证据|成本与风险|本轮选择|',
        '|---|---:|---|---|---|',
        f"|专项GEMV/GEMM|{decision['matrix_decode_share']}|主要热点；已拆为5个投影角色，但尚未证明哪种shape存在可恢复效率差距|高；分块、累加精度、多shape回退|保留高收益调查方向，不因cuBLAS而排除|",
        f"|FlashAttention|{decision['attention_decode_share']}|真实耗时显著，当前已是专用attention实现；还缺实现低效率证据|高；cache、mask和归约语义|保留，不把它当naive attention重融|",
        f"|Q/K Norm→RoPE|{decision['norm_rope_decode_share']}|每层4节点，normalized Q/K为独占中间结果，可验证消除读写/边界|中；BF16舍入、head/stride/旋转配对|优先做数值受控的局部原型|",
        f"|上项＋KV写入|{decision['norm_rope_cache_decode_share']}|decode可直接写旋转K，另有V搬运|中高；持久cache污染风险|第二阶段扩展候选|",
        f"|hidden RMS/AddRMS|{decision['hidden_norm_decode_share']}|add+norm已融合，普通residual输出还有消费者；没有直接再少一节点的同等证据|中；FP32 residual sum和BF16边界|保留单核调优候选|",
        f"|SiLU×up / MLP边界|{decision['silu_decode_share']}（仅activation）|SiLU+Mul已融合；跨投影可少物化，但不能把两个GEMM总耗时都当可删开销|跨GEMM高；输出配对和重复计算|暂不作为首个kernel原型|",
        f"|采样链|{decision['sampler_decode_share']}|6/7个节点，仍有多阶段归约/临时张量|中；随机流、选择结果与分布验收较复杂|有价值的竞争候选，列为备选|",
        f"|prefill最终norm提前选行|{decision['prefill_final_norm_share']}（prefill）|只需最后B行，但当前处理全部N行；只影响一次最终norm|中低；输出shape/选行位置|机会明确但预算小，可单列工程改动|",
        '', '*范围为7个case的“三轮占比中位数”的最小到最大。候选占比是成本预算，绝不是保证可消除的比例或端到端收益。',
        '', '## 5. 本轮决策及验收边界','',
        '考虑真实推理接入、数值语义和实现范围，选择 **Q/K RMSNorm→RoPE** 作为第一版原型是合理的：开销在完整执行中可测且重复出现，中间消费者唯一，能明确陈述要减少什么，实现范围小于替换线性层/attention。采样是值得保留的竞争方向，其正确性契约更复杂。',
        '', '该选择不代表全局最优，也没有证明它能提升端到端性能。矩阵产品和attention占据主要耗时；最终Norm选行则是额外发现的低成本小预算机会。若将目标改成最大化模型吞吐而不约束开发成本，应优先深入矩阵/attention效率调查。',
        '', '原型必须以实际compiled baseline为参考：Python中写了cast，不代表编译生成代码保留每一次舍入。该分析阶段未实现 CUDA kernel，也未改变数学计算或采样方法。正确性、微基准和端到端A/B属于实现后的验收。',
        '', '历史补充（与本轮选择证据分开）：上一轮整体torch.compile已证明可以改变融合边界，但未通过当时的严格数值门槛；本次既没有运行该版本，也没有将其局部速度数字当作新发现或已验证收益。',
        '', '## 6. 环境干扰记录和无profiler计时','',
        '下表每格为该进程同case两次请求的平均毫秒数。初始与确认各3轮逐一列出；没有混为一个分布。确认集采用空闲前置检查，但仍有系统噪声，因此性能范围保留。','',
        '|配置|初始run1/2/3 ms|确认run1/2/3 ms|','|---|---|---|']
    for r in timing_rows:
        lines.append(f"|{r['case']}|{' / '.join(f'{v:.2f}' for v in r['initial'])}|{' / '.join(f'{v:.2f}' for v in r['confirmation'])}|")
    lines+=['','## 7. 文件与复核','',
        '- `native_1/2/3.nsys-rep`及SQLite：初始完整原始采集；`confirmation/`：另3次完整采集、idle检查和相应统计。',
        '- `confirmation/ranking_by_time.csv`：累计时间排名；`ranking_by_count.csv`：次数排名；`kernel_summary.csv`：跨轮名称汇总；`category_summary.csv`：源码角色汇总。',
        '- `analysis_*.json`：全部step统计、分类后的首个prefill/decode完整kernel顺序；所有事件无未归属。',
        '- `docs/discovery_source_audit.md`：多候选源码消费者/依赖、现有融合和生成代码审计；`verification.json`：权重/源码/软件与数据完整性校验。',
        '- 复现入口：`scripts/discovery/run.sh`；环境确认：`run_confirmation.sh`；汇总与校验：`finish_analysis.sh`，然后运行`report.py`。没有提交或推送原始报告/大文件。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    baseline.dump(out/'report_metrics.json',dict(decision=decision,timing_runs=timing_rows,
        confirmation_kernel_count=kernel_count,confirmation_steps=sum(a['step_count'] for a in analyses),
        representative_categories=category_rows,report_script_sha256=baseline.sha(Path(__file__))))
    print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
