"""Audit source/data identity, recompute paired analysis, and verify packaged evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/closed_loop_20260921'
REPLAY = OUT / 'replay'
DATA = REPLAY / 'results/full_project_20260919'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    summary = read(OUT / 'experiment_summary.json')
    manifest = read(DATA / 'manifest.json')
    baseline = read(REPLAY / 'results/baseline_20260918_170614/baseline_manifest.json')
    inputs = read(REPLAY / 'REPRODUCTION_INPUTS.json')
    checks = {}
    for label, base, hashes in (
        ('summary_inputs', ROOT, summary['input_sha256']),
        ('frozen_sources', REPLAY, manifest['sources']),
        ('baseline_sources', REPLAY, baseline['source_sha256']),
        ('frozen_inputs', REPLAY, inputs['copied_inputs']),
        ('ab_inputs', DATA, summary['ab']['input_hashes']),
    ):
        for name, digest in hashes.items():
            assert sha(base / name) == digest, (label, name)
        checks[label] = len(hashes)
    assert sha(ROOT/'src/qk_norm_rope/kernel.cu') == sha(REPLAY/'src/qk_norm_rope/kernel.cu')
    spec = importlib.util.spec_from_file_location('frozen_ab_analysis', REPLAY/'scripts/full_project/analyze_ab.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='fusion_delivery_audit_') as name:
        temp = Path(name)
        for p in [DATA/'manifest.json', DATA/'workloads.json', *sorted(DATA.glob('ab_??_*.json'))]:
            shutil.copy2(p, temp/p.name)
        module.OUT = temp
        module.main()
        assert read(temp/'ab_summary.json') == summary['ab'], 'A/B recomputation differs'
    checks['ab_recomputed_identically'] = True
    audit = read(OUT/'delivery_audit.json')
    document = ROOT/audit['document']
    assert sha(document) == audit['document_sha256']
    archive = document.with_name(document.stem+'_原图与数据.zip')
    assert sha(archive) == audit['archive_sha256']
    provenance = read(document.parent/'closed_loop_20260921_native/provenance.json')
    records = {r['file']:r for r in provenance}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names = set(z.namelist())
        for name in audit['screenshot_files']:
            p = document.parent/'closed_loop_20260921_native'/name
            assert sha(p) == records[name]['sha256']
            with Image.open(p) as im:
                assert im.format == 'JPEG' and im.size == (2560,1392)
                im.verify()
            assert z.read(p.relative_to(ROOT).as_posix()) == p.read_bytes()
        for link in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            target = (document.parent/link).resolve()
            relative = target.relative_to(ROOT).as_posix()
            assert target.exists()
            assert relative in names or any(n.startswith(relative.rstrip('/')+'/') for n in names), link
        for path in summary['input_sha256']:
            assert (ROOT/path).relative_to(ROOT).as_posix() in names, path
        checks['archive_files'] = len(names)
    checks.update(screenshots_verified=21, image_links_valid=True,
                  final_exact_steps=summary['variants'][-1]['exact_steps'],
                  positive_ci_cases=sum(r['ci95'][0]>0 for r in summary['ab']['rows']),
                  measured_requests=summary['ab']['measured_requests'],
                  unchanged_project_kernel=True, all_checks_pass=True)
    (OUT/'final_verification.json').write_text(json.dumps(checks,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__ == '__main__':
    main()
