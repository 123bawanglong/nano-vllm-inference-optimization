"""Check paired statistics and reject accidentally mixed experiment inputs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from scripts.full_project import analyze_ab
import hashlib


class PairedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.previous=analyze_ab.OUT
        analyze_ab.OUT=Path(self.temp.name)
        self.out=analyze_ab.OUT
        (self.out/'manifest.json').write_text('{}')
        digest=hashlib.sha256(b'{}').hexdigest()
        workloads=[dict(name=f'case{i}',repeat=r,seed=i*10+r,batch=1,prompt=64,output=256)
                   for i in range(7) for r in (1,2)]
        (self.out/'workloads.json').write_text(json.dumps(workloads))
        for pair in range(1,11):
            for variant,elapsed in (('native',100.),('fused',90.)):
                rows=[dict(case=w['name'],repeat=w['repeat'],seed=w['seed'],batch=1,input_length=64,
                    output_length=256,e2e_ms=elapsed,decode_step_ms=elapsed/256,prefill_step_ms=elapsed/256,
                    engine_first_token_ms=elapsed/256,output_tokens_per_s=256000/elapsed,
                    step_ms=dict(decode=[elapsed/256]*255)) for w in workloads]
                order=[f'case{i}' for i in range(7)]
                shift=(pair-1)%7;order=order[shift:]+order[:shift]
                rows.sort(key=lambda r:(order.index(r['case']),r['repeat']))
                value=dict(pair=pair,variant=variant,manifest_sha256=digest,all_tokens_exact=True,
                    audit=dict(cache_hits=0,preemptions=0),rows=rows,case_order=order,
                    peak_allocated_bytes=1024,peak_reserved_bytes=2048,steady_allocated_bytes=1024)
                (self.out/f'ab_{pair:02d}_{variant}.json').write_text(json.dumps(value))

    def tearDown(self):
        analyze_ab.OUT=self.previous
        self.temp.cleanup()

    def test_known_paired_reduction_and_throughput(self):
        with contextlib.redirect_stdout(io.StringIO()):analyze_ab.main()
        value=json.loads((self.out/'ab_summary.json').read_text())
        self.assertEqual(value['measured_requests'],280)
        for row in value['rows']:
            self.assertAlmostEqual(row['e2e_reduction_pct'],10.)
            self.assertAlmostEqual(row['throughput_gain_pct'],100/9)
            self.assertTrue(all(abs(x-10.)<1e-10 for x in row['ci95']))

    def test_rejects_old_manifest_and_duplicate_repeat(self):
        path=self.out/'ab_01_fused.json'
        original=json.loads(path.read_text())
        value=dict(original,manifest_sha256='stale')
        path.write_text(json.dumps(value))
        with self.assertRaises(AssertionError):analyze_ab.main()
        original['rows'][1]=original['rows'][0]
        path.write_text(json.dumps(original))
        with self.assertRaises(AssertionError):analyze_ab.main()

    def test_rejects_truncated_case_order_with_complete_rows(self):
        for variant in ('native','fused'):
            path=self.out/f'ab_01_{variant}.json'
            value=json.loads(path.read_text())
            value['case_order']=value['case_order'][:6]
            path.write_text(json.dumps(value))
        with self.assertRaisesRegex(AssertionError,'workload'):analyze_ab.main()
