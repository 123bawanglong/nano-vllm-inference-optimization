"""Replace report figures with unmodified desktop captures, without rerunning GPU work."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/算子融合实验报告_全链路分析到性能验证.md'

def shot(name, caption):
    path = ROOT / 'docs/native_screenshots' / (name + '.jpg')
    assert path.exists(), path
    return f'![{caption}](native_screenshots/{name}.jpg)\n\n*{caption}。保留原始界面，点击图片可放大。*'

def main():
    text = DOC.read_text(encoding='utf-8')
    replacements = {
        'roofline_native.png': [('ncu_qnorm_roofline', 'Nsight Compute 原生界面：Q RMSNorm 的 Single Precision Roofline'), ('ncu_qrope_roofline', 'Nsight Compute 原生界面：Q RoPE 的 Single Precision Roofline')],
        'ncu_sol.png': [('ncu_native_details', 'Nsight Compute 原生 Details：当前选中 Q RMSNorm')],
        'ncu_launch.png': [('ncu_native_summary', 'Nsight Compute 原生 Summary：四个原生节点及其 grid、block、寄存器'), ('ncu_qrope_launch', 'Nsight Compute 原生 Launch Statistics 与 Occupancy：Q RoPE')],
        'ncu_scheduler.png': [('ncu_qrope_scheduler', 'Nsight Compute 原生 Scheduler 与 Warp State：Q RoPE')],
        'memory_plan.png': [],
        '06_kernel.png': [('code_kernel_layout', 'VS Code 原生界面：冻结 CUDA 源码的线程布局与归约模式'), ('code_kernel_math', 'VS Code 原生界面：冻结源码中的 Norm 与 RoPE 计算')],
        '10_model_correctness.png': [('result_correctness', 'VS Code 原生界面：numerical_fused.json 的 B1 P64 正确性记录')],
        'roofline_fused.png': [('ncu_fused_roofline', 'Nsight Compute 原生界面：融合 kernel 的 Single Precision Roofline'), ('ncu_fused_summary', 'Nsight Compute 原生 Summary：只有一个融合节点，grid=6、block=128、寄存器=38')],
        '08_micro.png': [('result_micro', 'VS Code 原生界面：micro.json 的 B1 P64 样本与 median_us')],
        '09_integration.png': [('code_adapter', 'VS Code 原生界面：实际模型 adapter 的 Decode 门控与融合调用')],
        '11_ab.png': [('result_ab', 'VS Code 原生界面：ab_summary.json 的 B1 P64 均值、区间和配对结果')],
        '12_variation.png': [],
    }
    optional = {
        '01_capture.png': [('nsys_overview', 'Nsight Systems 原生界面：完整七请求采集概览')],
        '02_rankings.png': [('nsys_gpu_time', 'Nsight Systems 原生 CUDA GPU Kernel Summary：按累计时间排序')],
        '02b_frequency.png': [('nsys_gpu_count', 'Nsight Systems 原生 CUDA GPU Kernel Summary：按调用次数排序')],
        '02c_sequence.png': [('nsys_sequence', 'Nsight Systems 原生界面：Decode 局部节点与 NVTX')],
        '03_candidates.png': [('code_qwen3', 'VS Code 原生界面：原生 Qwen3 Attention 的 Norm 与 RoPE 调用关系')],
        '07b_model_failure.png': [('result_diagnosis', 'VS Code 原生界面：编译上下文差异及修复前后的误差记录')],
        '07c_diagnosis.png': [],
        '09b_graph.png': [('result_integration', 'VS Code 原生界面：实际融合 Decode trace 的节点统计')],
    }
    for old, entries in optional.items():
        if not entries or all((ROOT/'docs/native_screenshots'/f'{n}.jpg').exists() for n,_ in entries):
            replacements[old] = entries
    for old, entries in replacements.items():
        pattern = r'!\[[^\]]*\]\(fusion_experiment_assets/' + re.escape(old) + r'\)'
        text = re.sub(pattern, lambda m: '\n\n'.join(shot(n,c) for n,c in entries), text)
    text = text.replace('这张图按 NCU section 的口径，', 'NCU 的原生 Roofline 按 section 的口径，')
    text = text.replace('每个小图的 roof 使用该次采样对应频率，图中的标记是从原始计数器计算出来的。', '这里分别展示 Q RMSNorm 和 Q RoPE 的原生图；下面的表汇总四个原生节点的导出计数器。界面纵轴采用 TFLOP/s，表格采用 GFLOP/s，两者相差 1000 倍。')
    text = text.replace('再对照前面的 NCU 资源表，', '再对照原生和融合的 NCU Summary，')
    text = text.replace('图中的等待数值保留导出指标的归一化口径，', '原始导出文件中的等待数值保留指标的归一化口径，')
    text = text.replace('这张表的分母是 **GPU kernel duration 总和**', '正文按算子角色归类的百分比，分母是 **GPU kernel duration 总和**')
    text = text.replace('分享时请把本 MD 与同目录 `fusion_experiment_assets` 一起带上。', '分享时请把本 MD 与同目录 `native_screenshots` 一起带上。')
    if 'fusion_experiment_assets/' not in text:
        text = re.sub(r'> 图片说明：[^\n]*', '> 图片说明：本文插图均为真实 Windows 应用窗口的原生截图，未重绘或拼接。性能分析图来自 Nsight Systems / Nsight Compute；代码和正确性、微基准、A/B 结果图来自 VS Code 打开的原始文件。截图于 2026-09-20 补拍已有实验报告，不代表重新跑了一轮 A/B。每张截图的来源、窗口和时间记录见 [截图来源清单](native_screenshots/provenance.json)。', text)
        text = text.replace('[本报告生成与核验脚本](../scripts/fusion_report/build_report.py)', '[原生截图替换脚本](../scripts/fusion_report/use_native_screenshots.py)、[核验脚本](../scripts/fusion_report/verify_report.py)')
    else:
        text = re.sub(r'> 图片说明：[^\n]*', '> 图片说明：原生截图替换进行中。NCU、源码与实验结果已替换为未重绘、未拼接的真实应用窗口截图；前四张 Nsight Systems 相关图片仍为旧版数据查看器图，尚不是 Systems 原生截图，等待新版 GUI 首次启动完成后替换。截图来源见 [来源清单](native_screenshots/provenance.json)。', text)
    DOC.write_text(text, encoding='utf-8')
    print('Remaining data-view figures:', len(re.findall(r'!\[[^\]]*\]\(fusion_experiment_assets/',text)))

if __name__ == '__main__':
    main()
