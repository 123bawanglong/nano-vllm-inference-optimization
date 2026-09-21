"""Validate and summarize completed stage-0 runs (standard library only)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(out):
    root = Path(__file__).resolve().parents[1]
    manifest_path = out / 'baseline_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest['source_sha256'].items():
        assert digest(root / relative) == expected, f'Source mismatch: {relative}'
    assert digest(out / 'workloads.json') == manifest['workloads_sha256']
    inputs = json.loads((out / 'workloads.json').read_text())
    runs = [json.loads((out / f'process_{i}.json').read_text()) for i in (1,2,3)]
    for index, run in enumerate(runs,1):
        assert run['run_id'] == index
        assert run['source_sha256'] == manifest['source_sha256']
        assert run['manifest_sha256'] == digest(manifest_path)
        assert run['config'] == manifest['config']
        assert run['audit'] == {'cache_hits':0,'preemptions':0}
        assert len(run['rows']) == len(inputs) == 21
        for row,item in zip(run['rows'],inputs):
            assert (row['case'],row['repeat'],row['seed']) == (item['name'],item['repeat'],item['seed'])
            assert row['measured'] == (item['repeat'] > 0)
            assert len(row['output_token_ids']) == item['batch']
            assert all(len(tokens) == item['output'] for tokens in row['output_token_ids'])
            assert row['counts'] == {'prefill':item['batch']*item['prompt'], 'decode':item['batch']*(item['output']-1)}
            assert len(row['step_ms']['prefill']) == 1
            assert len(row['step_ms']['decode']) == item['output']-1
            assert math.isclose(row['decode_step_ms'],statistics.mean(row['step_ms']['decode']),rel_tol=1e-12)
            assert math.isclose(row['output_tokens_per_s'],item['batch']*item['output']*1000/row['e2e_ms'],rel_tol=1e-12)
            assert all(math.isfinite(t) and t > 0 for phase in row['step_ms'].values() for t in phase)
    smoke = json.loads((out / 'layout_smoke.json').read_text())
    assert smoke['logits_all_finite'] and smoke['eager_audit_only']
    assert digest(out / 'native_eager_logits.pt') == smoke['logits_fixture_sha256']
    equal_outputs = all([r['output_token_ids'] for r in run['rows']] ==
                        [r['output_token_ids'] for r in runs[0]['rows']] for run in runs[1:])
    summary = {}
    lines = ['# 原版基线测量结果', '',
             'RTX 5080 / Qwen3-0.6B / BF16 / FlashAttention / TP=1 / decode CUDA Graph / 固定 64×256-token KV pages。', '',
             '3 个独立进程，每场景每进程 1 次预热、2 次正式测量：共 42 个正式批次。先取进程内中位数，再报告进程间中位数及范围。', '',
             '| batch | 输入/输出 tokens（每条） | 完整批次 ms，中位数 [进程范围] | 输出 tok/s | decode step ms |',
             '|---:|---:|---:|---:|---:|']
    for case in dict.fromkeys(x['name'] for x in inputs):
        medians = []
        for run in runs:
            rows = [r for r in run['rows'] if r['case'] == case and r['measured']]
            assert len(rows) == 2
            medians.append({key:statistics.median(r[key] for r in rows) for key in
                            ('e2e_ms','output_tokens_per_s','decode_step_ms','prefill_step_ms','engine_first_token_ms')})
        stats = {key:dict(median=statistics.median(m[key] for m in medians),
                          minimum=min(m[key] for m in medians), maximum=max(m[key] for m in medians))
                 for key in medians[0]}
        summary[case] = dict(process_medians=medians,stats=stats)
        fixture = next(i for i in inputs if i['name']==case)
        e = stats['e2e_ms']
        lines.append(f"| {fixture['batch']} | {fixture['prompt']}/{fixture['output']} | {e['median']:.2f} [{e['minimum']:.2f}, {e['maximum']:.2f}] | {stats['output_tokens_per_s']['median']:.1f} | {stats['decode_step_ms']['median']:.3f} |")
    lines += ['', '完整批次计时包含已 tokenized 请求入队、prefill、decode、sampling；不包含模型初始化与文本编解码。decode step 是完整引擎步骤时间，不是单个 kernel 时间。', '',
              f"检查通过：源文件 hash 一致；42 个正式批次 token 数量正确；prefix cache 命中/抢占均为 0；三个进程输出 token ID 全部一致：{equal_outputs}。",
              '独立 eager smoke 保存第一层布局和 9 次前向的全量有限 logits；该 smoke 的时延不进入上表。', '',
              '初始化耗时（秒，排除在热推理计时外）：' + ', '.join(f"process {r['run_id']}={r['init_seconds']:.2f}" for r in runs) + '。', '',
              '这些是本机原版基线数据，未实现自定义 CUDA kernel，因此没有本项目优化收益结论。三个进程的范围不是置信区间；显示任务、频率与温度可能带来波动。后续微小提升必须通过交错配对 A/B 验证。', '',
              '计时范围见 ../../README.md；原始逐步时间和输出 token ID 见 process_1/2/3.json。']
    (out / 'summary.json').write_text(json.dumps(dict(validated=True,independent_processes=3,
         measured_batches=42,outputs_equal_across_processes=equal_outputs,cases=summary),indent=2)+'\n')
    (out / 'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out',type=Path)
    summarize(parser.parse_args().out)
