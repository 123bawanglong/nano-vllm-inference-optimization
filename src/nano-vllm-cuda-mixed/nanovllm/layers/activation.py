import torch
from torch import nn
import torch.nn.functional as F
from nanovllm.layers.cuda_fused import enabled, supported, silu_and_mul


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()
        self.use_cuda_silu = enabled('silu')

    @torch.compile
    def compiled_forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cuda_silu and supported(x) and x.size(-1) > 0 and x.size(-1) % 2 == 0:
            return silu_and_mul(x)
        return self.compiled_forward(x)
