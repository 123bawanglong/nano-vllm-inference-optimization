import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = (Path(__file__).resolve().parents[1]/'src/nano-vllm-cuda-mixed')
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.find_spec('nanovllm.layers.cuda_fused')


def reference_rms(x, residual, weight, eps):
    summed = x.float() + residual.float()
    # Installed torch.compile eliminates intermediate low-precision casts.
    out = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
    return out, summed.to(x.dtype)


class FusedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)
        if SPEC:
            from nanovllm.layers.cuda_fused import add_rms_norm, silu_and_mul
            cls.rms = staticmethod(add_rms_norm)
            cls.silu = staticmethod(silu_and_mul)

    def setUp(self):
        self.assertIsNotNone(SPEC, 'CUDA fusion implementation is missing')

    def close(self, actual, expected):
        tol = {torch.float32: (2e-5, 2e-6), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}[expected.dtype]
        torch.testing.assert_close(actual, expected, rtol=tol[0], atol=tol[1])

    @torch.inference_mode()
    def test_numerical_shapes_dtypes_and_no_mutation(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for shape in ((1, 1), (3, 127), (2, 3, 1024), (17, 3072), (2, 8192), (0, 1024)):
                with self.subTest(dtype=dtype, shape=shape):
                    x = torch.randn(shape, dtype=dtype, device='cuda')
                    r = torch.randn_like(x)
                    w = torch.randn(shape[-1], dtype=dtype, device='cuda')
                    before = (x.clone(), r.clone(), w.clone())
                    expected = reference_rms(x, r, w, 1e-6)
                    actual = self.rms(x, r, w, 1e-6)
                    self.close(actual[0], expected[0])
                    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                    for a, b in zip((x, r, w), before):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    gates = torch.randn((*shape[:-1], shape[-1] * 2), dtype=dtype, device='cuda')
                    g, u = gates.chunk(2, -1)
                    self.close(self.silu(gates), F.silu(g) * u)

    @torch.inference_mode()
    def test_zeros_large_values_and_cancellation(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.randn(4, 1024, device='cuda', dtype=dtype) * 100
            r = -x
            w = torch.ones(1024, device='cuda', dtype=dtype)
            for a, b in zip(self.rms(x, r, w, 1e-6), reference_rms(x, r, w, 1e-6)):
                self.close(a, b)
            gates = torch.zeros(4, 6144, device='cuda', dtype=dtype)
            self.close(self.silu(gates), gates[:, :3072])

    @torch.inference_mode()
    def test_nondefault_stream_and_graph_replay(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x = torch.randn(4, 1024, device='cuda', dtype=torch.bfloat16)
            r = torch.randn_like(x)
            w = torch.ones(1024, device='cuda', dtype=x.dtype)
            gates = torch.randn(4, 6144, device='cuda', dtype=x.dtype)
            for _ in range(3):
                self.rms(x, r, w, 1e-6)
                self.silu(gates)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                y, residual = self.rms(x, r, w, 1e-6)
                activation = self.silu(gates)
            x.fill_(2)
            gates.fill_(1)
            graph.replay()
            expected = reference_rms(x, r, w, 1e-6)
        stream.synchronize()
        self.close(y, expected[0])
        self.close(residual, expected[1])
        g, u = gates.chunk(2, -1)
        self.close(activation, F.silu(g) * u)

    @torch.inference_mode()
    def test_matches_actual_compiled_baseline(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('original_norm', ROOT.parent/'nano-vllm-upstream/nanovllm/layers/layernorm.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        spec_a = importlib.util.spec_from_file_location('original_activation', ROOT.parent/'nano-vllm-upstream/nanovllm/layers/activation.py')
        module_a = importlib.util.module_from_spec(spec_a)
        spec_a.loader.exec_module(module_a)
        for dtype in (torch.float16, torch.bfloat16):
            norm = module.RMSNorm(1024).cuda().to(dtype)
            norm.weight.copy_(torch.randn_like(norm.weight))
            x = torch.randn(128, 1024, dtype=dtype, device='cuda')
            r = torch.randn_like(x)
            for a, b in zip(self.rms(x, r, norm.weight, norm.eps), norm.add_rms_forward(x, r)):
                self.assertLess(((a.float()-b.float()).norm()/b.float().norm()).item(), 5e-4)
            gates = torch.randn(128, 6144, dtype=dtype, device='cuda')
            a, b = self.silu(gates), module_a.SiluAndMul().cuda()(gates)
            self.assertLess(((a.float()-b.float()).norm()/b.float().norm()).item(), 5e-4)

    @torch.inference_mode()
    def test_contiguous_unaligned_silu_storage_offset(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            buf = torch.randn(3 * 6144 + 1, device='cuda', dtype=dtype)
            x = buf[1:].view(3, 6144)
            self.assertTrue(x.is_contiguous())
            self.assertNotEqual(x.data_ptr() % (4 * x.element_size()), 0)
            g, u = x.chunk(2, -1)
            self.close(self.silu(x), F.silu(g) * u)
            self.close(self.silu(x), self.silu(x.clone()))

    @torch.inference_mode()
    def test_integrated_fallbacks_and_switches(self):
        import os
        from nanovllm.layers.layernorm import RMSNorm
        from nanovllm.layers.activation import SiluAndMul
        previous = os.environ.get('NANOVLLM_FUSED_OPS')
        try:
            for mode in ('none', 'add_rms', 'silu', 'all'):
                os.environ['NANOVLLM_FUSED_OPS'] = mode
                norm = RMSNorm(1024).cuda().bfloat16()
                act = SiluAndMul().cuda()
                self.assertEqual(norm.use_cuda_add_rms, mode in ('add_rms', 'all'))
                self.assertEqual(act.use_cuda_silu, mode in ('silu', 'all'))
                # A strided input must use the original compiled fallback.
                x = torch.randn(2, 2048, device='cuda', dtype=torch.bfloat16)[:, ::2]
                r = torch.randn_like(x)
                for a, b in zip(norm(x, r), reference_rms(x, r, norm.weight, norm.eps)):
                    self.close(a, b)
                gates = torch.randn(2, 2048, device='cuda', dtype=x.dtype)[:, ::2]
                g, u = gates.chunk(2, -1)
                self.close(act(gates), F.silu(g) * u)
            # FP32 original implementation mutates x. Preserve that behavior via fallback.
            norm = RMSNorm(128).cuda().float()
            x = torch.randn(2, 128, device='cuda')
            r = torch.randn_like(x)
            before = x.clone()
            reference_input = x.clone()
            expected = norm.add_rms_forward(reference_input, r)
            actual = norm(x, r)
            for a, b in zip(actual, expected):
                self.close(a, b)
            self.assertEqual(actual[0].data_ptr() == actual[1].data_ptr(),
                             expected[0].data_ptr() == expected[1].data_ptr())
            self.close(x, reference_input)
            self.assertFalse(torch.equal(x, before))
        finally:
            if previous is None:
                os.environ.pop('NANOVLLM_FUSED_OPS', None)
            else:
                os.environ['NANOVLLM_FUSED_OPS'] = previous

    @torch.inference_mode()
    def test_reject_invalid_direct_inputs(self):
        x = torch.ones(2, 8, device='cuda')
        w = torch.ones(8, device='cuda')
        with self.assertRaises(RuntimeError):
            self.rms(x, x.double(), w, 1e-6)
        with self.assertRaises(RuntimeError):
            self.silu(x[:, ::2])
        with self.assertRaises(RuntimeError):
            self.silu(torch.ones(2, 7, device='cuda'))
        with self.assertRaises(RuntimeError):
            self.silu(x.cpu())


if __name__ == '__main__':
    unittest.main(verbosity=2)
