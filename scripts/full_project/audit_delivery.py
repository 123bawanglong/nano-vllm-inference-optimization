"""Read-only final evidence audit; no GPU operations and no benchmark reruns."""
import csv
import hashlib
import json
import re
import struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/full_project_20260919'
BASE=ROOT/'results/baseline_20260918_170614'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    manifest=read(OUT/'manifest.json');current=sha(OUT/'manifest.json')
    for path,digest in manifest['sources'].items():assert sha(ROOT/path)==digest,path
    baseline=read(BASE/'baseline_manifest.json')
    for path,digest in baseline['source_sha256'].items():assert sha(ROOT/path)==digest,path
    assert sha(BASE/'baseline_manifest.json')==manifest['baseline_sha256']
    assert sha(OUT/'workloads.json')==manifest['workloads_sha256']
    assert sha(ROOT/'docs/superpowers/plans/2026-09-19-full-project.md')==manifest['protocol_sha256']
    model=Path(baseline['model'])
    for name,digest in baseline['model_sha256'].items():assert sha(model/name)==digest,name
    for name,count in [('kernel_validation_final.json',138),('kernel_model_fixture_validation_final.json',294)]:
        result=read(OUT/name)
        assert result['exact_gate'] and len(result['rows'])==count
        assert result['validation_sha256']==sha(ROOT/'scripts/full_project/kernel_validation.py')
        if 'fixture_path' in result:
            assert sha(ROOT/result['fixture_path'])==result['fixture_sha256']
        for path,digest in result['source_sha256'].items():assert sha(ROOT/path)==digest,path
        for row in result['rows']:
            assert row['graph_exact'] and all(e['exact'] and e['finite'] for e in row['errors'].values())
    diagnosis=read(OUT/'kernel_reduction_context_diagnosis.json')
    for name,digest in diagnosis['evidence'].items():assert sha(OUT/name)==digest,name
    for context in diagnosis['contexts'].values():assert sha(OUT/context['ptx_path'])==context['ptx_sha256']
    numerical=read(OUT/'numerical_fused.json')
    assert numerical['manifest_sha256']==current and numerical['complete'] and numerical['exact_gate']
    assert sum(r['step_count'] for r in numerical['rows'])==1568
    for row in numerical['rows']:
        assert row['step_count']==row['exact_steps']
        assert all(c['exact'] and c.get('cache',{'exact':True})['exact'] for c in row['comparisons'])
        assert row['reference_sha256']==sha(OUT/f"reference_{row['case']}.pt")
    micro=read(OUT/'micro.json')
    assert micro['manifest_sha256']==current and micro['complete'] and len(micro['rows'])==9
    for row in micro['rows']:
        assert len(row['kernels']['native'])==4 and len(row['kernels']['fused'])==1
        assert all(e['exact'] for e in row['errors'].values())
        assert all(len(t)==10 and all(v>0 for v in t) for t in row['times_us'].values())
    for variant,total in [('native',405),('fused',321)]:
        result=read(OUT/f'integration_{variant}.json')
        assert result['manifest_sha256']==current and result['replacement_verified']
        assert result['counts']['total']==total and result['counts']['kv_store']==28
    reference={(r['case'],r['repeat']):r['output_token_ids'] for r in read(BASE/'process_1.json')['rows']}
    summary=read(OUT/'ab_summary.json')
    assert summary['manifest_sha256']==current and len(summary['rows'])==7
    for name,digest in summary['input_hashes'].items():
        path=OUT/name
        assert sha(path)==digest
        data=read(path)
        assert data['manifest_sha256']==current and data['all_tokens_exact']
        for row in data['rows']:
            assert row['output_token_ids']==reference[row['case'],row['repeat']]
    ncu=[]
    for variant,count in [('native',4),('fused',1)]:
        rows=list(csv.DictReader((OUT/f'ncu_{variant}_raw.csv').open()))
        assert rows[0]['gpu__time_duration.sum']=='us'
        kernels=[r for r in rows if r['ID']]
        assert len(kernels)==count
        ncu.append(dict(variant=variant,kernels=count,report_sha256=sha(OUT/f'{variant}_qk_rope.ncu-rep')))
    discovery=read(OUT/'discovery_recheck/analysis_1.json')
    assert discovery['kernel_count']==661895 and discovery['step_count']==1568 and discovery['unmatched']==0
    doc=ROOT/'docs/PROJECT_WALKTHROUGH.md'
    content=doc.read_text(encoding='utf-8')
    links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',content)
    for link in links:
        if '://' not in link:assert (doc.parent/link).exists(),link
    screenshots=[]
    for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',content):
        p=doc.parent/link;data=p.read_bytes()
        assert data[:8]==b'\x89PNG\r\n\x1a\n'
        w,h=struct.unpack('>II',data[16:24]);assert w>=800 and h>=200
        screenshots.append(dict(path=link,width=w,height=h,sha256=sha(p)))
    result=dict(passed=True,baseline_source_files=len(baseline['source_sha256']),model_hashes_verified=True,
        manifest_sha256=current,numerical_steps=1568,micro_cases=9,ab_processes=20,ab_requests=280,
        screenshots=screenshots,ncu=ncu,markdown_links_checked=len(links),document_sha256=sha(doc),
        distinction='Artifact consistency audit; performance conclusions come from the paired experiment, not this audit.')
    (OUT/'delivery_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS',len(screenshots),'screenshots;',len(links),'links; all frozen sources and measurement hashes verified')

if __name__=='__main__':main()
