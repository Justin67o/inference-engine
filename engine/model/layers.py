import torch
import torch.nn.functional as F
from torch import nn

from engine.attention.paged import decode_attention, prefill_attention
from engine.attention.reference import attend

try:
    from engine.attention.triton_paged import paged_decode_attention
except ImportError:     # CPU-only PyTorch (CI) has no Triton
    paged_decode_attention = None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)

def rope_cos_sin(positions, head_dim, base, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]    # [T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)                    # [T, D]
    return emb.cos().to(dtype), emb.sin().to(dtype)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):                                   # x: [T, heads, D]
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.kv_h = cfg.num_attention_heads, cfg.num_key_value_heads
        self.d = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.h
        bias = cfg.model_type == "qwen2"
        self.q_proj = nn.Linear(cfg.hidden_size, self.h * self.d, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kv_h * self.d, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kv_h * self.d, bias=bias)
        self.o_proj = nn.Linear(self.h * self.d, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, segments, layer, cache, write, batch, chunks=()):                            # x: [N, hidden]
        N = x.shape[0]
        q = apply_rope(self.q_proj(x).view(N, self.h, self.d), cos, sin)
        k = apply_rope(self.k_proj(x).view(N, self.kv_h, self.d), cos, sin)
        v = self.v_proj(x).view(N, self.kv_h, self.d)
        cache.write(layer, write, k, v)     # before the gather, so each decode sees its own token
        out = torch.empty(N, self.h * self.d, device=x.device, dtype=x.dtype)
        i = 0
        for s in segments:      
            if s.start == 0:    # prefills, one SDPA call each, on this step's KV
                out[i:i + s.n] = prefill_attention(q[i:i + s.n], k[i:i + s.n], v[i:i + s.n])
            i += s.n

        if batch is not None:       # decodes
            rows, tables, lens, M = batch
            decode = paged_decode_attention if x.is_cuda and paged_decode_attention else decode_attention
            out[rows] = decode(q[rows], cache.k[layer], cache.v[layer], tables, lens, M, cache.bs)

        for row, n, slots, pos in chunks:   # verify: causal over the cache, which already holds these n
            out[row:row + n] = attend(q[row:row + n], cache.k[layer][slots], cache.v[layer][slots], pos)
        return self.o_proj(out)
    

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)
    def forward(self, x, cos, sin, segments, layer, cache, write, batch, chunks=()):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, segments, layer, cache, write, batch, chunks)
        return x + self.mlp(self.post_attention_layernorm(x))

