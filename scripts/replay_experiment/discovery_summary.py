"""Summarize new complete capture; preserve original role and phase semantics."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/closed_loop_20260921'
raw=OUT/'replay/results/full_project_20260919/discovery_recheck/analysis_1.json'
d=json.loads(raw.read_text())
rows=[]
for case in dict.fromkeys(r['case'] for r in d['categories']):
    cats={r['category']:r for r in d['categories'] if r['case']==case and r['phase']=='decode_all'}
    steps=sum(s['case']==case and s['step']>0 for s in d['steps'])
    rows.append(dict(case=case,matrix_pct=sum(r['share_pct'] for n,r in cats.items() if n.startswith('matrix')),
        attention_pct=cats['attention']['share_pct'],norm_rope_pct=cats['qk_rmsnorm']['share_pct']+cats['rope']['share_pct'],
        norm_calls_per_step=cats['qk_rmsnorm']['calls']/steps,rope_calls_per_step=cats['rope']['calls']/steps))
for order in ('time_rank','count_rank'):
    names=d['names']
    with (OUT/f'ranking_by_{order}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(names[0]));writer.writeheader()
        writer.writerows(sorted(names,key=lambda r:(r['case'],r['phase'],r[order])))
summary=dict(source=str(raw.relative_to(ROOT)),kernel_events=d['kernel_count'],model_steps=d['step_count'],unmatched=d['unmatched'],
    scope='NEW single full seven-request capture; phase-role percentages are sums of GPU kernel durations, not wall time.',rows=rows,
    decision='Prototype Q/K Norm→RoPE: 112 nodes/step and immediate consumers allow fusion. Matrix/Attention dominate; implementation scope and dependency audit defer them, not evidence of optimal library efficiency.')
(OUT/'discovery_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
