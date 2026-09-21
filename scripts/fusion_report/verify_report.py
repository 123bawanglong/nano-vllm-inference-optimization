"""Verify report links, source identities and the data used in its tables."""
from pathlib import Path
import hashlib
import json
import re
import statistics
import argparse
from PIL import Image
import zipfile

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/roofline_supplement_20260920'
OLD=ROOT/'results/full_project_20260919'
DOC=ROOT/'docs/算子融合实验报告_全链路分析到性能验证.md'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-pending', action='store_true', help='Verify a clearly labeled draft without creating the final archive')
    args=parser.parse_args()
    doc=DOC.read_text(encoding='utf-8')
    assert '@@' not in doc
    links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc)
    local=[(DOC.parent/p).resolve() for p in links if not p.startswith('https://')]
    for p in local:assert p.exists(),str(p)
    images=[(DOC.parent/p).resolve() for p in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',doc)]
    assert len(images)>=15
    pending=[p for p in images if p.parent.name!='native_screenshots']
    if not args.allow_pending:
        assert not pending, 'Old generated figures remain'
    captures=read(ROOT/'docs/native_screenshots/provenance.json')
    by_name={c['name']:c for c in captures}
    for p in images:
        if p in pending:
            continue
        assert p.parent.name=='native_screenshots',p
        capture=by_name[p.stem]
        assert capture['file']==p.name and capture['sha256']==sha(p)
        assert (ROOT/capture['source']).exists(),capture
        assert capture['window']['app'] and capture['capturedAt']
        with Image.open(p) as im:
            assert im.format=='JPEG' and p.suffix=='.jpg'
            w,h=im.size
            im.verify()
        assert w>=1000 and h>=200,(p,w,h)
        assert (w,h)==(capture['width'],capture['height'])
    manifest=read(OLD/'manifest.json')
    snapshot=DATA/'replay_source'
    for name,digest in manifest['sources'].items():assert sha(snapshot/name)==digest,name
    provenance=read(DATA/'provenance.json')
    assert sha(ROOT/'src/qk_norm_rope/kernel.cu')==provenance['current_kernel_sha256']
    for name,digest in provenance['source_hashes'].items():assert sha(DATA/name)==digest
    for v,n in [('native',4),('fused',1)]:
        log=(DATA/f'{v}.log').read_text()
        assert '==ERROR==' not in log and f'NCU captured {v}' in log
        assert log.count('20 passes')==n
        assert (DATA/f'{v}_roofline.ncu-rep').stat().st_size>10000
    for name,count in [('kernel_validation_final.json',138),('kernel_model_fixture_validation_final.json',294)]:
        d=read(OLD/name);assert d['exact_gate'] and len(d['rows'])==count
        assert d['source_sha256']['src/qk_norm_rope/kernel.cu']==provenance['frozen_kernel_sha256']
    numerical=read(OLD/'numerical_fused.json')
    assert numerical['exact_gate'] and numerical['complete']
    assert sum(r['exact_steps'] for r in numerical['rows'])==1568
    assert numerical['manifest_sha256']==sha(OLD/'manifest.json')
    ab=read(OLD/'ab_summary.json')
    assert ab['all_tokens_exact'] and ab['processes']==20 and ab['measured_requests']==280
    assert ab['manifest_sha256']==sha(OLD/'manifest.json')
    for name,digest in ab['input_hashes'].items():assert sha(OLD/name)==digest,name
    for r in ab['rows']:
        reductions=[100*(p['native']['e2e_ms']-p['fused']['e2e_ms'])/p['native']['e2e_ms'] for p in r['pairs']]
        assert abs(statistics.mean(reductions)-r['e2e_reduction_pct'])<1e-9
        expected=f"|{r['case']}|{r['native_e2e_ms']:.2f}|{r['fused_e2e_ms']:.2f}|{r['e2e_reduction_pct']:.2f}%|"
        assert expected in doc
    assert sum(r['ci95'][0]>0 for r in ab['rows'])==6
    reductions=[]
    for r in read(OLD/'micro.json')['rows']:
        assert r['errors']['q']['exact'] and r['errors']['k']['exact']
        assert len(r['kernels']['native'])==4 and len(r['kernels']['fused'])==1
        reduction=100*(1-r['median_us']['fused']/r['median_us']['native']);reductions.append(reduction)
        assert f"|{r['case']}|{r['median_us']['native']:.3f}|{r['median_us']['fused']:.3f}|{reduction:.1f}%|" in doc
    ns=read(OLD/'discovery_recheck/analysis_1.json')
    assert (ns['kernel_count'],ns['step_count'],ns['unmatched'])==(661895,1568,0)
    report=dict(passed=True,document_sha256=sha(DOC),local_links=len(local),images=len(images),
        image_hashes={p.name:sha(p) for p in images},frozen_sources=len(manifest['sources']),
        historical_numerical_steps=1568,historical_ab_processes=20,historical_measured_requests=280,
        local_reduction_range=[min(reductions),max(reductions)],
        supplement_ncu_kernels={'native':4,'fused':1},current_source_untouched_sha256=provenance['current_kernel_sha256'],
        distinction='Historical A/B and numerical evidence verified; only NCU/Roofline was newly collected on 2026-09-20.')
    report['screenshot_provenance']='docs/native_screenshots/provenance.json'
    report['screenshot_type']='Unmodified native application window captures; Nsight GUI and VS Code raw source/results.'
    report['native_screenshot_revision_complete']=not pending
    report['pending_figures']=[p.name for p in pending]
    if pending:
        (ROOT/'docs/native_screenshots/verification_draft.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'DRAFT VERIFIED: {len(images)-len(pending)} native images; {len(pending)} generated figures still pending replacement. Final archive not updated.')
        return
    (ROOT/'docs/native_screenshots/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=DOC.with_name('算子融合实验报告_含图.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(DOC,DOC.name)
        for p in dict.fromkeys(images):z.write(p,Path('native_screenshots')/p.name)
        for name in ['provenance.json','verification.json']:
            z.write(ROOT/'docs/native_screenshots'/name,Path('native_screenshots')/name)
        z.writestr('阅读说明.txt','解压后打开 Markdown 文件；保持 native_screenshots 与 MD 在同一目录。\n插图均为原生应用窗口截图，未重绘或拼接。性能界面来自 Nsight，源码及实验 JSON 来自 VS Code。\n原始报告和源码链接需在原项目目录中查看；来源清单和核验记录已打包。\n')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len(z.namelist())==len(set(images))+4
    print(f'PASS: {len(images)} images, {len(local)} local links, 12 frozen sources, 20 historical AB files verified.')
    print(archive)

if __name__=='__main__':main()
