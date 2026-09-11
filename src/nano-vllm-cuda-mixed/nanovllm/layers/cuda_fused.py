"""Inference-only CUDA fusion; build once before CUDA Graph capture."""
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def enabled(op):
    mode = os.environ.get('NANOVLLM_FUSED_OPS', 'all')
    if mode not in ('none', 'add_rms', 'silu', 'all'):
        raise ValueError('NANOVLLM_FUSED_OPS must be none, add_rms, silu or all')
    return mode in ('all', op)


def supported(x):
    return (x.is_cuda and x.is_contiguous() and x.ndim >= 1
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and not (torch.is_grad_enabled() and x.requires_grad))


@lru_cache(maxsize=1)
def extension():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Warm up CUDA fusion before CUDA Graph capture')
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('MAX_JOBS', '2')
    return load(
        name='nanovllm_cuda_fused_v1',
        sources=[str(root / 'csrc' / 'fused_ops.cpp'), str(root / 'csrc' / 'fused_ops.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--fmad=false', '-lineinfo'],
        verbose=os.environ.get('NANOVLLM_CUDA_BUILD_VERBOSE') == '1',
    )


def add_rms_norm(x, residual, weight, eps):
    return extension().add_rms_norm(x, residual, weight, eps)


def silu_and_mul(x):
    return extension().silu_and_mul(x)
