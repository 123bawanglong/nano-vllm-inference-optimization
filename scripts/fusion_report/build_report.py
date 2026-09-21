"""Build a reference-style report from frozen measurements and new NCU exports."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.full_project import evidence as view

DATA = ROOT / 'results/roofline_supplement_20260920'
OLD = ROOT / 'results/full_project_20260919'
ASSETS = ROOT / 'docs/fusion_experiment_assets'
DOCUMENT = ROOT / 'docs/算子融合实验报告_全链路分析到性能验证.md'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def md_table(headers, rows):
    return '\n'.join(['|' + '|'.join(map(str, row)) + '|' for row in [headers, ['---'] * len(headers), *rows]])

def load_ncu(variant):
    with (DATA / f'{variant}_raw.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    units = rows[0]
    def val(row, metric):
        unit = units[metric]
        scale = {'Ghz': 1e9, 'Mhz': 1e6, 'Khz': 1e3, 'hz': 1,
                 'Gbyte/s': 1e9, 'Mbyte/s': 1e6, 'Kbyte/s': 1e3,
                 'Gbyte': 1e9, 'Mbyte': 1e6, 'Kbyte': 1e3}.get(unit, 1)
        return float(row[metric].replace(',', '')) * scale
    result = []
    labels = ['Q RMSNorm', 'K RMSNorm', 'Q RoPE', 'K RoPE'] if variant == 'native' else ['Fused Q/K Norm + RoPE']
    entries = [r for r in rows if r['ID']]
    assert len(entries) == len(labels)
    for label, r in zip(labels, entries):
        v = lambda m: val(r, m)
        perf = (v('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed')
                + v('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed')
                + v('derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2')) * v('smsp__cycles_elapsed.avg.per_second')
        peak = v('derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2') * v('sm__cycles_elapsed.avg.per_second')
        traffic = {
            'DRAM': v('dram__bytes.sum.per_second'),
            'L2': v('derived__lts__lts2xbar_bytes.sum.per_second'),
            'L1': v('derived__l1tex__lsu_writeback_bytes_mem_lgds.sum.per_second'),
        }
        bandwidth = {
            'DRAM': v('dram__bytes.sum.peak_sustained') * v('dram__cycles_elapsed.avg.per_second'),
            'L2': v('derived__lts__lts2xbar_bytes.sum.peak_sustained') * v('lts__cycles_elapsed.avg.per_second'),
            'L1': v('derived__l1tex__lsu_writeback_bytes_mem_lgds.sum.peak_sustained') * v('l1tex__cycles_elapsed.avg.per_second'),
        }
        result.append(dict(label=label, kernel=r['Kernel Name'], duration_us=v('gpu__time_duration.sum'),
            grid=v('launch__grid_size'), block=v('launch__block_size'), regs=v('launch__registers_per_thread'),
            occupancy=v('sm__warps_active.avg.pct_of_peak_sustained_active'),
            compute=v('sm__throughput.avg.pct_of_peak_sustained_elapsed'),
            memory=v('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'),
            dram=v('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'),
            l2=v('lts__throughput.avg.pct_of_peak_sustained_elapsed'),
            l1=v('l1tex__throughput.avg.pct_of_peak_sustained_active'),
            eligible=v('smsp__warps_eligible.avg.per_cycle_active'),
            no_issue=v('smsp__issue_inst0.avg.pct_of_peak_sustained_active'),
            long_scoreboard=v('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'),
            short_scoreboard=v('smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'),
            lg=v('smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio'),
            mio=v('smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio'),
            perf_flops=perf, peak_flops=peak, traffic_Bps=traffic, roof_bandwidth_Bps=bandwidth,
            intensity={level:perf / value for level, value in traffic.items()},
            achieved_roof_pct={level:100 * perf / min(peak, bandwidth[level] * perf / traffic[level]) for level in traffic}))
    return result

def plot(rows, variant):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = dict(DRAM='#c76819', L2='#168978', L1='#3764bb')
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2)) if len(rows) == 4 else plt.subplots(figsize=(10.8, 6.6))
    axes = np.asarray(axes).reshape(-1)
    x = np.logspace(-2, 3, 300)
    for ax, r in zip(axes, rows):
        for level, color in colors.items():
            y = np.minimum(r['peak_flops'], x*r['roof_bandwidth_Bps'][level]) / 1e9
            ax.loglog(x, y, color=color, lw=1.7, label=f'{level} roof')
            ai = r['intensity'][level]
            ax.scatter(ai, r['perf_flops']/1e9, color=color, s=56, zorder=5, edgecolor='white', linewidth=.7)
            ax.annotate(level, (ai, r['perf_flops']/1e9), xytext=(5, 7 if level != 'DRAM' else -15),
                        textcoords='offset points', fontsize=9, color=color)
        ax.set(xlim=(.01, 1000), ylim=(.5, 100000), xlabel='Arithmetic intensity [FLOP/byte]', ylabel='FP32 performance [GFLOP/s]')
        ax.set_title(f"{r['label']} | {r['perf_flops']/1e9:.3f} GFLOP/s | replay {r['duration_us']:.3f} us", fontsize=11)
        ax.grid(which='major', alpha=.2)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle(f"{'Native: four separate kernels' if variant == 'native' else 'Fused: one kernel'} - measured NCU counters", fontsize=17)
    fig.text(.5, .023, 'B1 Q16/K8 D128 | cold-cache kernel replay | FP32 FADD + FMUL + 2*FFMA\n'
             'Reconstructed with NCU section formulas; not a GUI screenshot or a hot-path speedup measurement.',
             ha='center', fontsize=10, color='#46566c')
    fig.tight_layout(rect=(0, .072, 1, .95))
    fig.savefig(ASSETS/f'roofline_{variant}.png', dpi=170)
    fig.savefig(ASSETS/f'roofline_{variant}.svg')
    plt.close(fig)

def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    names = ['01_capture','02_rankings','02b_frequency','02c_sequence','03_candidates','06_kernel',
             '07b_model_failure','07c_diagnosis','10_model_correctness','08_micro','09_integration','09b_graph','11_ab','12_variation']
    for name in names:
        for ext in ['png', 'html']:
            shutil.copy2(ROOT/f'docs/project_walkthrough_assets/{name}.{ext}', ASSETS/f'{name}.{ext}')
    native, fused = load_ncu('native'), load_ncu('fused')
    records = native + fused
    assert [r['grid'] for r in native] == [16,8,8,4]
    assert fused[0]['grid'] == 6 and fused[0]['block'] == 128
    frozen = read(OLD/'manifest.json')
    replay = DATA/'replay_source'
    for name, digest in frozen['sources'].items():
        assert sha(replay/name) == digest, name
    strip = lambda s: re.sub(r'\s+','',re.sub(r'//[^\n]*|/\*.*?\*/','',s,flags=re.S))
    current = ROOT/'src/qk_norm_rope/kernel.cu'
    kernel_copy = replay/'src/qk_norm_rope/kernel.cu'
    assert strip(current.read_text()) == strip(kernel_copy.read_text())
    provenance = dict(date='2026-09-20', historical_measurements='2026-09-19',
        frozen_manifest_sha256=sha(OLD/'manifest.json'),
        current_kernel_sha256=sha(current), frozen_kernel_sha256=sha(kernel_copy),
        current_difference='Comments and whitespace only; current source was not changed',
        frozen_source_count=len(frozen['sources']),
        replay='kernel',cache_control='all',clock_control='none',ncu='2025.1.1',passes_per_kernel=20,
        capture='study.ncu, B1/P64 real first-layer fixture after real engine initialization',
        plot_formula='Exact metric names and unit conversion in build_report.py; corresponding NCU section copied verbatim',
        source_hashes={p.name:sha(p) for p in [DATA/'native_raw.csv', DATA/'fused_raw.csv', DATA/'SingleRoofline.section']})
    (DATA/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (DATA/'derived_metrics.json').write_text(json.dumps(dict(native=native,fused=fused),indent=2)+'\n',encoding='utf-8')
    for variant, rows in [('native',native),('fused',fused)]: plot(rows, variant)
    view.ASSETS = ASSETS
    sources = [DATA/'native_raw.csv', DATA/'fused_raw.csv']
    view.page('ncu_sol', 'Speed of Light · 2026-09-20 补采',
        view.table(['kernel','Compute %','Memory 汇总 %','DRAM %','L2 %','L1/TEX %'],
            [[r['label']]+[f'{r[k]:.3f}' for k in ['compute','memory','dram','l2','l1']] for r in records]), sources,
        '直接读取 NCU 导出值。不同汇总指标有不同口径；Compute 不是 FP32 FLOP/峰值，Memory 不是 DRAM。短 kernel / 冷缓存 replay，仅用于诊断。')
    view.page('ncu_launch','Launch Statistics / Occupancy · 补采',
        view.table(['kernel','grid blocks','threads/block','reg/thread','achieved occupancy %','replay μs'],
            [[r['label']]+[f'{r[k]:.3f}' if k in ['occupancy','duration_us'] else int(r[k]) for k in ['grid','block','regs','occupancy','duration_us']] for r in records]), sources,
        'GPU 有 84 个 SM。融合后仍是小 grid，不能将收益解释为 GPU 已满载；replay duration 不用于热态加速比。')
    view.page('ncu_scheduler','Scheduler / Warp State · 补采',
        view.table(['kernel','eligible warps / cycle','无发射周期 %','Long SB*','Short SB*','LG*','MIO*'],
            [[r['label']]+[f'{r[k]:.3f}' for k in ['eligible','no_issue','long_scoreboard','short_scoreboard','lg','mio']] for r in records])+
        view.pre('* smsp__average_warps_issue_stalled_{reason}_per_issue_active.ratio\n原始导出单位 inst；这些 ratio 不是总运行时间的百分比。\n无发射周期 = smsp__issue_inst0.avg.pct_of_peak_sustained_active'), sources,
        '访存结果的依赖等待不等于 DRAM 带宽已饱和。短 kernel 的跨 pass 比率容易波动，不据此单独作性能结论。')
    view.page('memory_plan','从源码数据依赖核算可消除的中间访问',
        view.pre('原生：Q/K → RMSNorm → global store → global load → RoPE → 输出\n融合：Q/K → RMSNorm → 寄存器内保留 BF16 舍入 → RoPE → 输出')+
        view.table(['核算项','逻辑字节数'],[['Q16 + K8，head_dim=128','3072 个元素'],['BF16 Norm 中间结果','6144 byte = 6 KiB'],['中间结果写一次 + 读一次','12 KiB / 层 / token'],['28 层','336 KiB / token']]),
        [replay/'src/qk_norm_rope/kernel.cu', ROOT/'nanovllm/models/qwen3.py'],
        '源码层面的逻辑访问估算，不是实测 DRAM 节省量；缓存命中、事务粒度和归约分支的重新读取都需要区分。')
    (ASSETS/'pages.json').write_text(json.dumps(view.PAGES),encoding='utf-8')
    micro = read(OLD/'micro.json')['rows']
    ab = read(OLD/'ab_summary.json')['rows']
    assert len(micro)==9 and len(ab)==7
    for r in micro:
        assert r['errors']['q']['exact'] and r['errors']['k']['exact']
        assert len(r['kernels']['native'])==4 and len(r['kernels']['fused'])==1
    replacements = {
        'ROOFLINE_NATIVE':md_table(['kernel','FP32 GFLOP/s','DRAM FLOP/byte','达到对应 DRAM roof 的百分比'],
            [[r['label'],f"{r['perf_flops']/1e9:.3f}",f"{r['intensity']['DRAM']:.3f}",f"{r['achieved_roof_pct']['DRAM']:.3f}%"] for r in native]),
        'SOL_OBSERVATION':f"本次原生 Compute(SM) 约 **{min(r['compute'] for r in native):.2f}%～{max(r['compute'] for r in native):.2f}%**，DRAM 约 **{min(r['dram'] for r in native):.2f}%～{max(r['dram'] for r in native):.2f}%**。结合 Roofline，我没有看到整卡计算或 DRAM 带宽接近饱和的证据。",
        'SCHED_OBSERVATION':f"原生平均 eligible warps/cycle 约 **{min(r['eligible'] for r in native):.3f}～{max(r['eligible'] for r in native):.3f}**，无发射周期约 **{min(r['no_issue'] for r in native):.2f}%～{max(r['no_issue'] for r in native):.2f}%**。结合小 grid，我更关注可用工作量少、等待依赖时缺少其他 warp 顶上，而不是优先堆更多计算指令。",
        'ROOFLINE_FUSED':f"融合节点的 FP32 吞吐约 **{fused[0]['perf_flops']/1e9:.3f} GFLOP/s**，DRAM 口径计算强度约 **{fused[0]['intensity']['DRAM']:.3f} FLOP/byte**，达到对应 DRAM roof 约 **{fused[0]['achieved_roof_pct']['DRAM']:.3f}%**。它仍然远低于 roof。四个原生节点与一个融合节点工作内容不同，所以这不是同一个 kernel 参数调整后的严格横向位移实验，也不把 FP32 指令吞吐比当成整链加速比。",
        'MICRO_TABLE':md_table(['workload','原生 μs','融合 μs','局部延迟下降'],
            [[r['case'],f"{r['median_us']['native']:.3f}",f"{r['median_us']['fused']:.3f}",f"{100*(1-r['median_us']['fused']/r['median_us']['native']):.1f}%"] for r in micro]),
        'AB_TABLE':md_table(['workload','native 平均 ms','fused 平均 ms','配对延迟下降','95% 区间'],
            [[r['case'],f"{r['native_e2e_ms']:.2f}",f"{r['fused_e2e_ms']:.2f}",f"{r['e2e_reduction_pct']:.2f}%",f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]%"] for r in ab]),
    }
    doc = (Path(__file__).with_name('report_template.md')).read_text(encoding='utf-8')
    for key,value in replacements.items():
        assert f'@@{key}@@' in doc
        doc=doc.replace(f'@@{key}@@',value)
    assert '@@' not in doc
    DOCUMENT.write_text(doc,encoding='utf-8')
    print(DOCUMENT)
    print(json.dumps(dict(roofline_native_GFLOPS=[r['perf_flops']/1e9 for r in native],fused_GFLOPS=fused[0]['perf_flops']/1e9)))

if __name__=='__main__': main()
