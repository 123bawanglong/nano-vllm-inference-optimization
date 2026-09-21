import unittest
from scripts.compile_boundary.common import make_workloads


class WorkloadsTests(unittest.TestCase):
    def test_grid_counts_lengths_and_capacity(self):
        rows = make_workloads()
        self.assertEqual(len(rows), 27)
        self.assertEqual(len({r['name'] for r in rows}), 9)
        for r in rows:
            self.assertEqual(len(r['prompts']), r['batch'])
            self.assertTrue(all(len(x) == r['prompt'] for x in r['prompts']))
            self.assertLessEqual(r['batch'] * r['prompt'], 4096)
            self.assertLessEqual(r['batch'] * ((r['prompt'] + r['output'] + 255)//256), 64)
    def test_original_inputs_preserved_and_new_batch_reproducible(self):
        from scripts.compile_boundary.common import baseline, BASELINE
        import json
        old = json.loads((BASELINE/'workloads.json').read_text())
        new = make_workloads()
        for row in old:
            self.assertIn(row,new)
        self.assertEqual(new, make_workloads())
    def test_warmup_and_measured_prefixes_differ(self):
        rows = make_workloads()
        starts = [tuple(p[:64]) for r in rows for p in r['prompts']]
        self.assertEqual(len(starts),len(set(starts)))


if __name__ == '__main__':
    unittest.main()
