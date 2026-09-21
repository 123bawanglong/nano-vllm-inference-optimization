"""Opt-in decode fusion; install before model creation / CUDA Graph capture."""
import torch


def install():
    from src.qk_norm_rope import fused, build
    from nanovllm.models.qwen3 import Qwen3Attention
    from nanovllm.utils.context import get_context
    build()
    original = Qwen3Attention.forward

    def forward(self, positions, hidden_states):
        eligible = (not get_context().is_prefill and not self.qkv_bias
                    and hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16
                    and self.head_dim == 128 and self.num_heads == 16
                    and self.num_kv_heads == 8
                    and self.q_norm.eps == self.k_norm.eps)
        if not eligible:
            return original(self, positions, hidden_states)
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = fused(q, k, self.q_norm.weight, self.k_norm.weight,
                     positions, self.rotary_emb.cos_sin_cache, self.q_norm.eps,
                     reduction_mode='native_graph')
        return self.o_proj(self.attn(q, k, v).flatten(1, -1))

    Qwen3Attention.forward = forward
    return original
