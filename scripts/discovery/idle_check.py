"""Before-model environment gate, separate from measured performance results."""
import argparse
import json
import subprocess
import time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
samples=[]
for i in range(5):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
    utilization=int(raw.split(',')[0])
    samples.append(dict(unix_time=time.time(),gpu=raw,utilization_pct=utilization))
    if i<4:time.sleep(2)
args.output.write_text(json.dumps(dict(threshold=20,samples=samples,passed=all(s['utilization_pct']<=20 for s in samples)),indent=2))
assert all(s['utilization_pct']<=20 for s in samples), f'GPU not idle before model load: {samples}'
print('IDLE CHECK:',[s['utilization_pct'] for s in samples],flush=True)
