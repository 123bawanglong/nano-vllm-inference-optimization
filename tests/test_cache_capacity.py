import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'nanovllm/utils/cache_capacity.py'
spec = importlib.util.spec_from_file_location('capacity', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class CapacityTests(unittest.TestCase):
    def test_explicit_capacity_does_not_drift_with_free_memory(self):
        self.assertEqual(module.choose_cache_capacity(100,64),64)
        self.assertEqual(module.choose_cache_capacity(95,64),64)
    def test_auto_retains_upstream_capacity(self):
        self.assertEqual(module.choose_cache_capacity(100,-1),100)
    def test_insufficient_memory_fails_instead_of_changing_experiment(self):
        with self.assertRaisesRegex(RuntimeError,'exceeds'):
            module.choose_cache_capacity(63,64)
    def test_no_available_memory_fails(self):
        with self.assertRaises(RuntimeError):
            module.choose_cache_capacity(0,-1)

if __name__ == '__main__': unittest.main()
