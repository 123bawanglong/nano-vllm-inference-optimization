"""The public runner must isolate runs and preserve correctness gates."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from scripts import experiment


class ExperimentRunnerTests(unittest.TestCase):
    def test_prepare_requires_new_output_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            with self.assertRaisesRegex(ValueError, 'model'):
                experiment.configuration('prepare', out, None)
            cfg = experiment.configuration('prepare', out, Path(tmp))
            self.assertEqual(cfg['model'], str(Path(tmp).resolve()))
            out.mkdir()
            with self.assertRaises(FileExistsError):
                experiment.configuration('prepare', out, Path(tmp))

    def test_later_stages_use_saved_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'run.json').write_text(json.dumps({'model': '/models/Qwen3-0.6B'}))
            self.assertEqual(experiment.configuration('ab', out, None)['model'], '/models/Qwen3-0.6B')
            with self.assertRaisesRegex(ValueError, 'model'):
                experiment.configuration('ab', out, out)

    def test_ab_is_alternating_independent_processes(self):
        tasks = experiment.commands('ab', Path('/tmp/run'))
        timing = [cmd for _, cmd in tasks if 'timing' in cmd]
        self.assertEqual(len(timing), 20)
        self.assertEqual([cmd[cmd.index('--variant') + 1] for cmd in timing[:4]],
                         ['native', 'fused', 'fused', 'native'])
        self.assertTrue(all(cmd[:3] == [experiment.sys.executable, '-m', 'scripts.study'] for cmd in timing))
        self.assertEqual(len([cmd for _, cmd in tasks if 'scripts.discovery.idle_check' in cmd]), 20)

    def test_profile_systems_is_unfiltered_graph_nodes(self):
        tasks = experiment.commands('profile', Path('/tmp/run'), tool='systems', nsys='/opt/nsys')
        profiles = [cmd for _, cmd in tasks if '--cuda-graph-trace=node' in cmd]
        self.assertEqual(len(profiles), 3)
        self.assertTrue(all(cmd[0] == '/opt/nsys' and '--trace=cuda,nvtx' in cmd for cmd in profiles))
        self.assertTrue(all(not any('kernel-name' in arg for arg in cmd) for cmd in profiles))
        self.assertEqual(tasks[-1][0], 'rankings_summary')

    def test_compute_collects_requested_sections_and_source(self):
        tasks = experiment.commands('profile', Path('/tmp/run'), tool='compute')
        captures = [cmd for _, cmd in tasks if '--graph-profiling' in cmd]
        self.assertEqual(len(captures), 2)
        for cmd in captures:
            for section in ('SpeedOfLight_HierarchicalSingleRooflineChart', 'MemoryWorkloadAnalysis',
                            'ComputeWorkloadAnalysis', 'SchedulerStats', 'WarpStateStats', 'SourceCounters'):
                self.assertIn(section, cmd)
            self.assertIn('--import-source', cmd)

    def test_failed_task_stops_sequence_and_log_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)
            sentinel = out / 'should-not-exist'
            jobs = [('fail', [experiment.sys.executable, '-c', 'raise SystemExit(7)']),
                    ('later', [experiment.sys.executable, '-c', f'open({str(sentinel)!r}, "w").close()'])]
            with self.assertRaises(RuntimeError):
                experiment.run_tasks(jobs, out, experiment.os.environ.copy())
            self.assertFalse(sentinel.exists())
            original = (out / 'logs/fail.log').read_bytes()
            with self.assertRaises(FileExistsError):
                experiment.run_tasks(jobs, out, experiment.os.environ.copy())
            self.assertEqual((out / 'logs/fail.log').read_bytes(), original)

    def test_validation_runs_reference_before_fused(self):
        tasks = experiment.commands('validate', Path('/tmp/run'))
        labels = [name for name, _ in tasks]
        self.assertLess(labels.index('numerical_native'), labels.index('numerical_fused'))
        self.assertLess(labels.index('integration_native'), labels.index('integration_fused'))
        self.assertEqual(labels[0], 'kernel_validation')

    def test_environment_keeps_fresh_baseline_separate_from_fixtures(self):
        out = Path('/tmp/run').resolve()
        env = experiment.environment(out, {'model': '/models/qwen'})
        self.assertEqual(env['QK_RUN_DIR'], str(out))
        self.assertEqual(env['QK_BASELINE_DIR'], str(out / 'baseline'))
        self.assertEqual(env['QK_MODEL'], '/models/qwen')


if __name__ == '__main__':
    unittest.main()
