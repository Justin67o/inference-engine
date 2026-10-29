import math, torch, torch.nn as nn, torch.nn.functional as F

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

    def forward(self, x, cos, sin):                            # x: [T, hidden]
        T = x.shape[0]
        q = apply_rope(self.q_proj(x).view(T, self.h, self.d), cos, sin)
        k = apply_rope(self.k_proj(x).view(T, self.kv_h, self.d), cos, sin)
        v = self.v_proj(x).view(T, self.kv_h, self.d)
        rep = self.h // self.kv_h                              # GQA: share KV heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))       # [heads, T, D]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)   # [heads, T, T]
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = (probs @ v).transpose(0, 1).reshape(T, self.h * self.d)
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
    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))

