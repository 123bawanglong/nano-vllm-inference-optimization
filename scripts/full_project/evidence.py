"""Render auditable local data views. Screenshots are explicitly not fake profiler UI."""
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics
import sqlite3

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/full_project_20260919'
ASSETS=ROOT/'docs/project_walkthrough_assets'
OLD=ROOT/'results/discovery_20260918_234946'
LABELS=dict(matrix_qkv='QKV projection',matrix_attention_out='O projection',matrix_gate_up='MLP gate/up',
    matrix_down='MLP down',matrix_lm_head='LM head',qk_rmsnorm='Q/K RMSNorm',rope='RoPE',
    kv_write='KV write',attention='FlashAttention',residual_add_rmsnorm='Residual + RMSNorm',
    input_rmsnorm='Input RMSNorm',final_add_rmsnorm='Final AddRMS',silu_mul='SiLU × up',sampling='Sampling',other='Other')

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def esc(x):return html.escape(str(x))
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def pre(text):return '<pre>'+esc(text)+'</pre>'
def excerpt(path,start,end):
    lines=path.read_text(encoding='utf-8').splitlines()
    return pre('\n'.join(f'{i+1:3} {line}' for i,line in enumerate(lines) if start<=i+1<=end))

PAGES=[]
def page(name,title,body,sources,note):
    ASSETS.mkdir(parents=True,exist_ok=True)
    source_html=''.join('<div>'+esc(str(p.relative_to(ROOT)))+' · SHA256 '+hashlib.sha256(p.read_bytes()).hexdigest()[:16]+'</div>' for p in sources)
    value=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#eef2f6;color:#14263b;font:18px/1.55 "Microsoft YaHei","Noto Sans CJK SC",sans-serif}}
    main{{margin:28px;padding:30px;background:white;border:1px solid #d9e1ec;border-radius:12px}}h1{{font-size:27px;margin:10px 0 18px}}h2{{font-size:21px;margin:20px 0 12px}}.eyebrow{{color:#526b89;font-size:14px;letter-spacing:.4px}}.note{{background:#edf5ff;border-left:4px solid #2774bd;padding:14px;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:16px}}td,th{{padding:9px 12px;border-bottom:1px solid #dce5ee;text-align:left}}th{{background:#edf2f8}}tr:nth-child(even){{background:#f8fafc}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#14263b;color:#ebf3ff;padding:18px;font:14px/1.55 Consolas,"Microsoft YaHei",monospace;border-radius:7px}}.sources{{font:12px/1.7 Consolas,monospace;color:#63778d;overflow-wrap:anywhere;margin-top:20px}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}</style>
    <main><div class="eyebrow">NANO-VLLM 实验数据查看器 · 真实文件自动读取 · 非 Nsight 软件界面</div><h1>{esc(title)}</h1>{body}<div class="note">{esc(note)}</div><div class="sources">数据来源（相对项目根目录）{source_html}</div></main></html>'''
    (ASSETS/f'{name}.html').write_text(value,encoding='utf-8')
    PAGES.append(name)


def main():
    metrics=read(OLD/'report_metrics.json')
    analyses=[read(OLD/'confirmation'/f'analysis_{i}.json') for i in (1,2,3)]
    fresh=OUT/'discovery_recheck/analysis_1.json'
    runs=[[f'历史确认 {a["run"]}',a['kernel_count'],a['step_count'],a['unmatched']] for a in analyses]
    if fresh.exists():
        a=read(fresh);runs.append(['本轮重采集',a['kernel_count'],a['step_count'],a['unmatched']])
    page('01_capture','先确认：抓到的是完整推理链路',table(['采集','GPU kernel 事件','模型 step','未归属事件'],runs)+
         pre('B = 1 / 4 / 8 × P = 64 / 256 × O = 256\n另加 B1 / P2048 / O32\n--trace=cuda,nvtx --cuda-graph-trace=node\n无 kernel-name 过滤；预热后 capture；每个 step 有 NVTX 范围'),
         [OLD/'confirmation/analysis_1.json',OLD/'verification.json']+([fresh] if fresh.exists() else []),
         '先检查事件数量、阶段归属和输出一致性，再相信热点排名。每个 O256 请求是 1 次 Prefill + 255 次 Decode。')
    cats=metrics['representative_categories'] if 'representative_categories' in metrics else next(v for v in metrics.values() if isinstance(v,list) and v and 'category' in v[0])
    ordered=sorted(cats,key=lambda r:r['share'],reverse=True)
    page('02_rankings','两张排名一起看：谁耗时，谁频繁',
         '<h2>B1/P64/O256 · 完整 Decode · 3 轮统计</h2>'+table(['源码角色','GPU kernel 时间占比','每 step 调用数','单次 median μs','单次 p95 μs'],
         [[LABELS[r['category']],f"{r['share']:.2f}%",f"{r['calls']:g}",f"{r['median_us']:.3f}",f"{r['p95_us']:.3f}"] for r in ordered])+pre('调用频率：Q/K Norm 56 + RoPE 56 = 112 kernel / decode step\n数据分母：所有 GPU kernel duration 之和，不是端到端 wall time'),
         [OLD/'report_metrics.json',OLD/'confirmation/ranking_by_time.csv',OLD/'confirmation/ranking_by_count.csv'],
         '矩阵投影和 Attention 才是主要耗时。Norm→RoPE 是高频小链，不能因为名字熟悉就把它说成最大热点。')
    page('02b_frequency','换一种排序：每个 Decode step 反复执行什么？',
         table(['源码角色','调用数 / step','GPU kernel 时间占比','单次 median μs'],
         [[LABELS[r['category']],f"{r['calls']:g}",f"{r['share']:.2f}%",f"{r['median_us']:.3f}"] for r in sorted(cats,key=lambda r:(r['calls'],r['share']),reverse=True)]),
         [OLD/'report_metrics.json',OLD/'confirmation/ranking_by_count.csv'],
         '调用多也不等于必须优化。接下来要查数据依赖、编译器已有融合、可删除的中间读写。')
    if fresh.exists():
        dbpath=fresh.parent/'native_1.sqlite'
        db=sqlite3.connect(f'file:{dbpath.as_posix()}?mode=ro',uri=True)
        start,end=db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='DISCOVERY|b1_p64_o256|1'").fetchone()
        events=db.execute('SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=? ORDER BY k.start LIMIT 11',(start,end)).fetchall()
        page('02c_sequence','回到实际 trace：第一层的执行顺序',
             table(['step 内相对开始 μs','duration μs','实际 kernel 名称'],[[f'{(a-start)/1000:.3f}',f'{(b-a)/1000:.3f}',name] for a,b,name in events]),
             [dbpath], '来自本轮 Nsight Systems SQLite 的真实时间戳。紧邻的两次 Norm、两次 RoPE 和 KV store 是不同节点；这里只展示完整采集中的一个局部。')
        db.close()
    page('03_candidates','结合源码：哪些开销真的能消掉？',
         table(['候选','审查结果','本轮判断'],[
             ['GEMM/GEMV + Attention','合计占大多数时间，仍需专门检查实现余量','不能因库实现就排除；本轮实现成本较高'],
             ['Q/K RMSNorm → RoPE','每层 2 norm + 2 rope；norm 输出仅被 rope 使用','边界和消费者明确，选择有界原型'],
             ['Residual + RMSNorm / SiLU × up','生成的 Triton 已分别融合','不能把已有融合再当新增收益'],
             ['Sampling','多 kernel、有中间张量；随机数和归约语义复杂','保留为竞争候选'],
             ['Prefill final norm → last rows','可考虑先选最后一行，但实测预算很小','不是本轮首选']])+excerpt(ROOT/'nanovllm/models/qwen3.py',70,96),
         [ROOT/'docs/discovery_source_audit.md',ROOT/'nanovllm/models/qwen3.py'],
         '选择依据是实际成本、可删除边界、实现范围和正确性风险；这一步只支持“值得尝试”，还不支持“已有收益”。')
    if (OUT/'ncu_native_raw.csv').exists():
        entries=[]
        for variant in ('native','fused'):
            path=OUT/f'ncu_{variant}_raw.csv'
            if not path.exists():continue
            rows=list(csv.DictReader(path.open()))
            wanted=['gpu__time_duration.sum','launch__grid_size','launch__block_size','launch__registers_per_thread','sm__warps_active.avg.pct_of_peak_sustained_active']
            assert rows[0]['gpu__time_duration.sum']=='us'
            for row in rows:
                if not row['ID']:continue  # the second CSV line contains units
                entries.append([variant,row['ID'],row['Kernel Name'][:62]]+[f'{float(row[k]):.3f}' if '.' in row[k] else row[k] for k in wanted])
        page('05_ncu_comparison','NCU 结构证据：4 个节点 → 1 个节点',
             table(['实现','ID','Kernel','duration μs*','grid','threads/block','reg/thread','occupancy %'],entries),
             [OUT/'ncu_native_raw.csv',OUT/'ncu_fused_raw.csv'],
             '*原始单位见 CSV。冷缓存 kernel replay，不用此 duration 计算热态加速比；吞吐低和小 grid 不能直接证明带宽受限。')
    kernel=ROOT/'src/qk_norm_rope/kernel.cu'
    if kernel.exists():
        page('06_kernel','融合假设落成代码：保留数值边界','<h2>核心片段：warp 映射、BF16 边界、RoPE 输出</h2>'+excerpt(kernel,12,36)+excerpt(kernel,74,93),[kernel],
             '一个 warp 处理一个 head；Q/K 合在一个 launch。Norm 结果先显式舍入到 BF16，再在寄存器里转回 FP32 做 RoPE。没有融合 KV store。')
    validation=OUT/'kernel_validation_final.json'
    if validation.exists():
        data=read(validation)
        rows=data['rows']
        page('07_kernel_validation','先检查正确性，再讨论速度',table(['检查','实测结果'],[
            ['exact_gate',data['exact_gate']],['数值/Graph/输入不变 检查组',len(rows)],
            ['所有 Q/K 元素 exact',all(all(e['exact'] for e in r['errors'].values()) for r in rows)],
            ['所有 Graph replay exact',all(r['graph_exact'] for r in rows)],
            ['所有输入保持不变',all(r['input_unchanged'] for r in rows)],
            ['参数校验通过',', '.join(data['guards'])]])+'<h2>真实首层 fixture（warps=4）</h2>'+table(['fixture','Q max abs','K max abs','exact'],[
                [r['case'],r['errors']['q']['max_abs'],r['errors']['k']['max_abs'],r['errors']['q']['exact'] and r['errors']['k']['exact']] for r in rows[:18] if r['warps']==4]),[validation],
             '展示的是真实测试产物。逐元素一致性只针对已覆盖的输入和固定编译环境，不等于跨平台证明。')
    failure=OUT/'attempt_1/numerical_fused.json'
    if failure.exists():
        failed=read(failure)
        page('07b_model_failure','真实失败：局部测试过了，B1 模型级检查却没过',table(['配置','logits exact / 总 step','是否过关'],[
            [r['case'],f"{r['exact_steps']} / {r['step_count']}",r['exact_steps']==r['step_count']] for r in failed['rows']]),
            [failure,OUT/'attempt_1/kernel_validation_final.json'],
            '本轮第一版的真实结果，已连同源码快照归档。没有删掉失败记录，也没有靠放宽数值阈值进入正式 A/B。')
    diagnosis=OUT/'kernel_reduction_context_diagnosis.json'
    if diagnosis.exists():
        d=read(diagnosis)
        page('07c_diagnosis','定位并修复：实际选中的归约路径不同',
             table(['native B1 执行上下文','XBLOCK','threads/block','Q grid','K grid','求和顺序'],[
             [name,row['configuration']['XBLOCK'],row['block_threads'],row['expected_q_grid_x'],row['expected_k_grid_x'],row['reduction_tree']] for name,row in d['contexts'].items()])+ '<h2>实际模型 28 层 × 17 个 Decode step 的逐层对照</h2>'+table(['配置','检查行数','修复前不同','修复后不同'],[
                 [r['case'],r['rows'],r['mismatching_rows_before'],r['mismatching_rows_after']] for r in d['comparison']]),
             [diagnosis,OUT/'kernel_model_fixture_validation_final.json'],
             '模型接入显式选择 native_graph；没有按参考答案选择路径。294 组捕获 fixture 对拍补充覆盖 Norm、RoPE 和 Graph replay。')
    if (OUT/'micro.json').exists():
        m=read(OUT/'micro.json')
        page('08_micro','局部微基准：融合链自身到底省了多少？',table(['真实首层 fixture','native μs','fused μs','延迟下降','节点数','Q/K 数值'],[
            [r['case'],f"{r['median_us']['native']:.3f}",f"{r['median_us']['fused']:.3f}",f"{100*(1-r['median_us']['fused']/r['median_us']['native']):.1f}%",
             f"{len(r['kernels']['native'])} → {len(r['kernels']['fused'])}",'exact' if all(e['exact'] for e in r['errors'].values()) else 'FAIL'] for r in m['rows']]),
             [OUT/'micro.json'],'CUDA Graph 内重复 1000 次链调用，10 组交替顺序成对测量。这个局部比例不能直接套到整个模型。')
    page('09_integration','接入真实推理：在 CUDA Graph capture 前选择路径',excerpt(ROOT/'scripts/full_project/adapter.py',1,50),
         [ROOT/'scripts/full_project/adapter.py'],'仅符合条件的 Decode 走融合路径；Prefill 原样保留；KV 写入、Attention、O projection 仍执行原实现。')
    if (OUT/'integration_fused.json').exists():
        ns=read(OUT/'integration_native.json');fs=read(OUT/'integration_fused.json')
        page('09b_graph','不是只改了 Python：实际模型 Graph 已替换',table(['实际 Decode step 17','全部 GPU kernel','自写 QK kernel','独立 RoPE','KV store'],[
             [r['variant'],r['counts']['total'],r['counts']['fused'],r['counts']['rope'],r['counts']['kv_store']] for r in (ns,fs)]),
             [OUT/'integration_native.json',OUT/'integration_fused.json'],
             '28 层 × 每层减少 3 个节点 = 每 step 减少 84 个 GPU kernel 节点。CUDA Graph 已启用，不能称为减少 84 次 CPU launch 调用。')
    if (OUT/'numerical_fused.json').exists():
        n=read(OUT/'numerical_fused.json')
        page('10_model_correctness','模型级检查：输入历史相同，再比较 logits 和 KV',table(['配置','logits exact / 总 step','抽查全层 KV 的 step','KV exact'],[
            [r['case'],f"{r['exact_steps']} / {r['step_count']}",','.join(map(str,r['selected_steps'])),all(c.get('cache',{'exact':True})['exact'] for c in r['comparisons'])] for r in n['rows']]),
             [OUT/'numerical_native.json',OUT/'numerical_fused.json'],
             'teacher forcing 固定 native 的 token 历史，避免不同采样路径混淆误差。正式 A/B 还会验证正常采样输出 token 完全一致。')
    summary=OUT/'ab_summary.json'
    if summary.exists():
        s=read(summary)
        page('11_ab','端到端 A/B：10 个独立进程对，全部保留',table(['配置','native ms','fused ms','配对延迟下降','95% CI','decode 下降','prefill 变化'],[
            [r['case'],f"{r['native_e2e_ms']:.2f}",f"{r['fused_e2e_ms']:.2f}",f"{r['e2e_reduction_pct']:.2f}%",f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]%",f"{r['decode_reduction_pct']:.2f}%",f"{r['prefill_reduction_pct']:.2f}%"] for r in s['rows']]),
             [summary],'CI 为对 10 个配对下降率重采样的 percentile bootstrap；这是固定机器/负载上的结果，未包含真实在线服务排队和网络。')
        page('12_variation','不要只看平均值：每一对数据都留下',table(['配置']+[f'对{i}' for i in range(1,11)],
             [[r['case']]+[f'{v:.2f}%' for v in r['paired_e2e_reduction_pct']] for r in s['rows']])+pre(json.dumps(s['memory'],ensure_ascii=False,indent=2)),
             [summary],'正数表示 fused 延迟更低；负数表示这一对发生退化。显存是 PyTorch 进程内 allocated/reserved，不等于显卡总占用。')
    (ASSETS/'pages.json').write_text(json.dumps(PAGES,indent=2),encoding='utf-8')
    print('Evidence pages:',','.join(PAGES))

if __name__=='__main__':main()
