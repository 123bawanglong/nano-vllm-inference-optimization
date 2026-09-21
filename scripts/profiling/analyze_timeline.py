"""Inspect Nsys diagnostics and aggregate real GPU events from fallback trace."""
import collections
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/profiling_20260918'
db=sqlite3.connect(OUT/'native_decode_nsys.sqlite')
tables=[x[0] for x in db.execute("select name from sqlite_master where type='table'")]
diagnostics={}
for table in tables:
    if 'DIAGNOSTIC' in table.upper() or 'ANALYSIS' in table.upper():
        diagnostics[table]=dict(columns=[r[1] for r in db.execute(f'pragma table_info("{table}")')],
                               rows=db.execute(f'select * from "{table}"').fetchall())
nsys=dict(tables=tables,diagnostics=diagnostics,has_gpu_kernel_table=any('KERNEL' in t for t in tables))
(OUT/'nsys_diagnostics.json').write_text(json.dumps(nsys,indent=2)+'\n')
print('Nsys diagnostics:',json.dumps(diagnostics)[:4000])
trace=json.loads((OUT/'torch_capture.trace.json').read_text())
events=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X']
assert events, 'No actual GPU kernel events in fallback trace'
groups=collections.defaultdict(list)
for event in events:
    groups[event['name']].append(event)
total=sum(e['dur'] for e in events)
rows=[]
for name,items in groups.items():
    duration=sum(e['dur'] for e in items)
    rows.append(dict(name=name,calls=len(items),total_us=duration,average_us=duration/len(items),
                     kernel_time_pct=100*duration/total,example_args=items[0].get('args')))
rows.sort(key=lambda x:x['total_us'],reverse=True)
result=dict(source='torch.profiler Chrome trace kernel events',steps=16,kernel_count=len(events),
            summed_kernel_us=total,first_kernel_ts=min(e['ts'] for e in events),
            gpu_event_span_us=max(e['ts']+e['dur'] for e in events)-min(e['ts'] for e in events),rows=rows)
(OUT/'kernel_summary.json').write_text(json.dumps(result,indent=2)+'\n')
for row in rows:
    print(f"{row['kernel_time_pct']:6.2f}% {row['calls']:5d} {row['average_us']:8.3f}us {row['name']}")
