import torch
from torch import nn
from nanovllm.layers.cuda_fused import add_rms_norm, enabled, supported


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.use_cuda_add_rms = enabled('add_rms')

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            if (self.use_cuda_add_rms and x.dtype in (torch.float16, torch.bfloat16)
                    and supported(x) and supported(residual)
                    and supported(self.weight) and 0 < x.size(-1) <= 8192
                    and x.shape == residual.shape and x.dtype == residual.dtype == self.weight.dtype
                    and x.device == residual.device == self.weight.device):
                return tuple(add_rms_norm(x, residual, self.weight, self.eps))
            return self.add_rms_forward(x, residual)
