import collections
import unittest
from scripts.discovery.analyze import categories, phase_names, quantile


class DiscoveryAnalysisTests(unittest.TestCase):
    def test_windows_cover_full_decode_without_discarding_middle(self):
        for output,early,late in ((256,32,32),(32,15,16)):
            counts=collections.Counter(p for i in range(output) for p in phase_names(i,output))
            self.assertEqual(counts['full_request'],output)
            self.assertEqual(counts['prefill'],1)
            self.assertEqual(counts['decode_all'],output-1)
            self.assertEqual(counts['decode_early'],early)
            self.assertEqual(counts['decode_late'],late)

    def test_sampler_prefix_random_kernel_and_projection_roles(self):
        norm='triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0'
        rope='triton_poi_fused__to_copy_add_cat_index_mul_split_sub_0'
        names=['embedding_kernel',norm]
        for i in range(28):
            if i:names.append(norm)
            names += ['gemv_qkv',norm,norm,rope,rope,'store_kvcache_kernel','flash::attention',
                      'gemv_output',norm,'gemv_gateup','triton_poi_fused_mul_silu_split_0','gemv_down']
        names += [norm,'index_kernel','gemv_head',
            'triton_red_fused_amax_0','triton_per_fused_amax_1','triton_poi_fused_softmax_2',
            'triton_red_fused_softmax_3','triton_per_fused_softmax_4',
            'void at::native::distribution_elementwise_grid_stride_kernel',
            'triton_red_fused_softmax_argmax_exponential_5']
        result=categories([dict(name=n) for n in names])
        counts=collections.Counter(result)
        self.assertEqual(counts['sampling'],7)
        self.assertEqual(counts['qk_rmsnorm'],56)
        self.assertEqual(counts['residual_add_rmsnorm'],55)
        self.assertEqual(counts['final_add_rmsnorm'],1)
        self.assertEqual(result[names.index('index_kernel')],'other')
        for role in ('qkv','attention_out','gate_up','down'):
            self.assertEqual(counts['matrix_'+role],28)
        self.assertEqual(counts['matrix_lm_head'],1)
        fewer=names.copy();fewer.remove('triton_per_fused_amax_1')
        self.assertEqual(collections.Counter(categories([dict(name=n) for n in fewer]))['sampling'],6)
        missing=names.copy();missing.remove('gemv_qkv')
        with self.assertRaises(AssertionError):categories([dict(name=n) for n in missing])
        expanded=names.copy();expanded.insert(expanded.index('gemv_down'),'gemm_down_partial')
        self.assertEqual(collections.Counter(categories([dict(name=n) for n in expanded]))['matrix_down'],29)
        names.remove('store_kvcache_kernel')
        with self.assertRaises(AssertionError):categories([dict(name=n) for n in names])

    def test_quantile_retains_long_tail(self):
        self.assertAlmostEqual(quantile([1,2,3,10],.95),8.95)


if __name__=='__main__':unittest.main()
