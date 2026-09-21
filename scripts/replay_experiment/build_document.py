"""Build the concise Markdown from new data and unmodified native screenshots."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/closed_loop_20260921'
DOCS=ROOT/'docs'
ASSETS=DOCS/'closed_loop_20260921_native'
DOCUMENT=DOCS/'算子融合_精简闭环实验_20260921.md'


def main():
    s=json.loads((OUT/'experiment_summary.json').read_text(encoding='utf-8'))
    a=s['ab']
    assert a['all_tokens_exact'] and a['processes']==20 and a['measured_requests']==280
    provenance=json.loads((ASSETS/'provenance.json').read_text(encoding='utf-8'))
    captured={r['file']:r for r in provenance}
    used=[]
    def shot(name,caption):
        file=name+'.jpg'
        assert file in captured,file
        p=ASSETS/file
        assert hashlib.sha256(p.read_bytes()).hexdigest()==captured[file]['sha256'],file
        assert (ROOT/captured[file]['source']).exists(),captured[file]['source']
        used.append(file)
        return f'{caption}\n\n![{caption}](closed_loop_20260921_native/{file})\n'
    native=s['ncu']['native'];f=s['ncu']['final'][0]
    micro=s['micro'];v=s['variants'];d=s['discovery']
    texts=['# nano-vLLM：从热点发现到融合验证',
        '> 2026-09-21 重新实测；这是基于已有项目的受控复现，初版失败来自真实消融测试。',
        '**环境 / workload：** RTX 5080、Qwen3-0.6B、BF16、CUDA Graph；B=1/4/8 × 输入64/256 × 输出256，另加B1/输入2048/输出32；KV=64页×256 token。',
        '## 1．先看全链路，再选优化对象',
        shot('nsys_time',f"我先完整采集：{d['kernel_events']:,} 个 GPU kernel、{d['model_steps']:,} 个 step，全部归属到 NVTX；累计耗时首先指向矩阵计算和 Attention。"),
        shot('nsys_count','再按调用次数排序：高频小 kernel 值得检查，但这张全请求、按名称汇总的表不能直接当作 Q/K Norm 占比。'),
        shot('nsys_sequence','沿 Decode 时间线和源码确认：每层 Q Norm、K Norm、Q RoPE、K RoPE 分开执行，归一化输出马上被 RoPE 读取。'),
        '|候选|本轮判断|\n|---|---|\n|GEMM/GEMV、Attention|分别占 Decode GPU 时间约66.40%～76.08%、13.69%～23.32%；成熟实现的替换范围较大，本轮先保留。|\n|Q/K Norm→RoPE|占3.12%～3.72%，每step共112次；中间读写可以消除，选作原型。|\n|Add+Norm、SiLU×up、采样|前两者已融合；采样仍是候选，本轮先做依赖清楚的 Norm→RoPE。|',
        '## 2．用 NCU 确定要省什么',
        '> 以下原生 NCU 图均为 B1 的 Q RMSNorm；四个节点一并采集，22 passes/node、cold-cache replay，耗时只作诊断。',
        shot('native_roofline','Roofline：点远低于上界；计算强度低，还不能据此断言 DRAM 带宽打满。'),
        shot('native_sol','Speed of Light：四个节点 Compute仅0.11%～0.25%、DRAM仅1.04%～4.52%，先关注小任务的开销。'),
        shot('native_compute','Compute Workload Analysis：Q Norm 的 Executed IPC Active约0.08，没有计算管线饱和的证据。'),
        shot('native_memory','Memory Workload Analysis：结合源码，Norm中间结果每层每token逻辑上写6 KiB再读6 KiB；融合可省这段传递。'),
        shot('native_scheduler','Scheduler Statistics：Q Norm 的 Eligible约0.04、No Eligible约95.71%，可发射工作很少。'),
        shot('native_warp','Warp State Statistics：Long Scoreboard依赖等待突出，配合4～16 blocks的小grid，我优先减少节点和中间访存。'),
        shot('native_source','Source / SASS：实际指令包含SHFL.BFLY归约、MUFU.RSQ和BF16转换，融合时要保留对应数值语义。'),
        '## 3．写原型，先过数值关',
        '我的方案：Q16/K8/D128特化，每warp一个head、每block四个warp，grid=6B；在寄存器里接着算RoPE，把四个节点合成一个。',
        shot('kernel_source','Kernel Specialization负责固定形状和执行布局；归约顺序、FMA与BF16舍入需要另外对齐。'),
        '|版本|本次改动|严格一致的logits step|\n|---|---|---|\n'
        f"|V1|初版融合：Norm后继续用FP32|{v[0]['exact_steps']}/1568|\n"
        f"|V2|恢复Norm→RoPE间的BF16舍入|{v[1]['exact_steps']}/1568|\n"
        f"|V3|再对齐batch=1原生Graph的归约顺序|{v[2]['exact_steps']}/1568|",
        shot('correctness_results','实测发现：只保留BF16存储和FP32计算仍不够；恢复舍入边界并对齐归约后，logits和选定KV位置才全部通过。'),
        'BF16输入→FP32 Norm→寄存器内BF16舍入→FP32 RoPE→BF16输出；三个版本都已有形状特化，不能把对齐单独归功于特化。',
        '## 4．接入模型，再测收益',
        f"真实Decode接入验证：每step的GPU节点 **{s['integration']['native']['total']}→{s['integration']['fused']['total']}**，28个融合节点生效；Prefill和28次KV cache写入保持原实现。",
        shot('micro_results',f"局部Graph微基准：9组数据全部对齐，延迟下降{min(r['latency_reduction_pct'] for r in micro):.2f}%～{max(r['latency_reduction_pct'] for r in micro):.2f}%；这是局部收益。"),
        shot('final_roofline',f"融合后Roofline仍远低于上界；我用独立计时验证收益，不用NCU的{f['duration_us']:.3f} μs冷重放时间计算加速比。"),
        shot('ab_results','端到端：固定10组配对、280次请求、全部保留，所有输出token与冻结baseline一致。'),
        '|B / 输入 / 输出|原生ms|融合ms|配对延迟下降|95%区间|\n|---|---:|---:|---:|---|']
    for r in a['rows']:
        label=r['case'].replace('b','').replace('_p',' / ').replace('_o',' / ')
        texts[-1]+=f"\n|{label}|{r['native_e2e_ms']:.2f}|{r['fused_e2e_ms']:.2f}|{r['e2e_reduction_pct']:.2f}%|[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]%|"
    positive=[r for r in a['rows'] if r['ci95'][0]>0]
    uncertain=[r['case'] for r in a['rows'] if r['ci95'][0]<=0<=r['ci95'][1]]
    texts+= [f"本轮{len(positive)}/7种配置的描述性区间高于0"+('；'+ '、'.join(uncertain)+'尚不能确认稳定收益。' if uncertain else '。'),
        '结果支持减少小节点和中间传递的融合假设；桌面GPU未锁频，区间按配置分别计算，严格一致仅覆盖本轮模型、软件和输入。',
        'Prefill没有替换，其时间差异不计作融合收益；本轮峰值allocated显存相同，没有测出显存收益。',
        '## 5．融合后 NCU 原图复核',
        shot('final_sol',f"Speed of Light：融合后Compute约{f['compute']:.2f}%、DRAM约{f['dram']:.2f}%，仍未接近整卡上限。"),
        shot('final_compute','Compute Workload Analysis：复核融合后的指令和管线压力，不能把四个原节点与一个融合节点的吞吐比当成端到端加速比。'),
        shot('final_memory','Memory Workload Analysis：中间全局数组被消除；逻辑字节节省不等于同量DRAM流量节省。'),
        shot('final_scheduler',f"Scheduler Statistics：融合后Eligible约{f['eligible']:.2f}，仍是小任务，收益不等于GPU已经满载。"),
        shot('final_warp','Warp State Statistics：融合后仍有依赖等待；我保留这一限制，没有把所有stall下降当作成功条件。'),
        shot('final_source','Source / SASS：源码保留BF16舍入，机器指令确认SHFL归约，与数值对拍相互验证。'),
        '## 复现入口',
        '[本轮数据与日志](../results/closed_loop_20260921/) · [运行入口](../scripts/replay_experiment/run.sh) · [汇总脚本](../scripts/replay_experiment/summarize.py) · [截图来源与SHA256](closed_loop_20260921_native/provenance.json)',
        '> 所有插图均为真实窗口原始截图；运行结果图展示脚本从本轮JSON生成的汇总日志，没有重绘图表或模拟终端。',
        '压缩包保留原图、报告、结果和源码；执行脚本仍依赖本机WSL中的模型与CUDA/Python环境，复跑需使用新输出目录。']
    DOCUMENT.write_text('\n\n'.join(texts)+'\n',encoding='utf-8')
    assert len(used)==len(set(used))
    assert all((ASSETS/x).stat().st_size>10000 for x in used)
    links=re.findall(r'\]\(([^)]+)\)',DOCUMENT.read_text(encoding='utf-8'))
    for link in links:
        assert (DOCUMENT.parent/link).exists(),link
    # Include relative data and scripts so document links also work after unpacking.
    archive=DOCS/'算子融合_精简闭环实验_20260921_原图与数据.zip'
    payload=[DOCUMENT,ASSETS/'provenance.json',*(ASSETS/x for x in used),
             *sorted((ROOT/'scripts/replay_experiment').glob('*.py')),
             ROOT/'scripts/replay_experiment/run.sh',
             OUT/'experiment_summary.json',OUT/'discovery_summary.json',
             OUT/'correctness_results.txt',OUT/'micro_results.txt',OUT/'ab_results.txt',
             OUT/'decision_before_prototype.md',
             OUT/'native_full.ncu-rep',OUT/'final_full.ncu-rep',
             OUT/'native_raw.csv',OUT/'final_raw.csv']
    data=OUT/'replay/results/full_project_20260919'
    payload.extend([data/'discovery_recheck/native_1.nsys-rep',
                    data/'manifest.json',data/'ab_summary.json',
                    data/'numerical_fused.json',data/'micro.json',
                    OUT/'numerical_v1_fp32_intermediate.json',
                    OUT/'numerical_v2_bf16_boundary.json',
                    OUT/'replay/src/qk_norm_rope/kernel.cu',
                    OUT/'variants/v1_fp32_intermediate/kernel.cu'])
    payload.extend(sorted(data.glob('ab_??_*.json')))
    # Keep the exact snapshot source and frozen inputs needed by the entry scripts.
    replay=OUT/'replay'
    for directory in ('nanovllm','benchmarks','scripts','src','tests','docs'):
        payload.extend(p for p in (replay/directory).rglob('*')
                       if p.is_file() and '__pycache__' not in p.parts
                       and p.suffix in ('.py','.sh','.cu','.md','.sec'))
    inputs=json.loads((replay/'REPRODUCTION_INPUTS.json').read_text(encoding='utf-8'))
    payload.extend([replay/'REPRODUCTION_INPUTS.json',
                    *(replay/name for name in inputs['copied_inputs']),
                    ROOT/'scripts/fusion_report/build_report.py',
                    ROOT/'docs/superpowers/plans/2026-09-21-concise-replay.md'])
    payload.extend(ROOT/name for name in s['input_sha256'])
    payload.extend(sorted(OUT.glob('*.log')))
    payload.extend(sorted(data.glob('*.log')))
    payload.extend(sorted(data.glob('idle_*.json')))
    payload.extend(sorted(OUT.glob('ranking_*.csv')))
    payload=list(dict.fromkeys(payload))
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in payload:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    audit=dict(document=str(DOCUMENT.relative_to(ROOT)),screenshots=len(used),links=len(links),
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        document_sha256=hashlib.sha256(DOCUMENT.read_bytes()).hexdigest(),
        screenshot_files=used,all_images_native=True,all_links_resolve=True,zip_integrity=True)
    (OUT/'delivery_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
