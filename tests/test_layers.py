import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from engine.model.layers import RMSNorm, apply_rope, rope_cos_sin



@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_matches_hf(dtype):
    torch.manual_seed(0)
    dim, eps = 896, 1e-6
    ref = Qwen2RMSNorm(dim, eps)
    mine = RMSNorm(dim, eps)

    # Random weight: with the default all-ones weight a missing multiply would go unnoticed
    weight = torch.randn(dim)
    with torch.no_grad():
        ref.weight.copy_(weight)
        mine.weight.copy_(weight)
    ref, mine = ref.to(dtype), mine.to(dtype)

    x = torch.randn(20, dim, dtype=dtype)
    expected, actual = ref(x), mine(x)

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), (actual - expected).abs().max().item()

@pytest.fixture(scope="module")
def cfg():
    return AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_cos_sin_matches_hf(cfg, dtype):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    base = cfg.rope_parameters["rope_theta"]
    # Positions far from 0 too: errors in the angle grow with position
    positions = torch.cat([torch.arange(20), torch.arange(5000, 5020)])

    ref = Qwen2RotaryEmbedding(cfg)
    x = torch.empty(0, dtype=dtype)                  # HF only reads x for its dtype and device
    ref_cos, ref_sin = ref(x, positions[None])       # HF wants positions as [batch, T]
    cos, sin = rope_cos_sin(positions, head_dim, base, dtype)

    assert cos.shape == (len(positions), head_dim)
    assert torch.equal(cos, ref_cos[0])
    assert torch.equal(sin, ref_sin[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apply_rope_matches_hf(cfg, dtype):
    torch.manual_seed(0)
    T = 20
    heads, kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = cfg.hidden_size // heads
    cos, sin = rope_cos_sin(torch.arange(T), head_dim, cfg.rope_parameters["rope_theta"], dtype)

    q = torch.randn(T, heads, head_dim, dtype=dtype)      # my layout: [T, heads, D]
    k = torch.randn(T, kv_heads, head_dim, dtype=dtype)

    # HF's layout is [batch, heads, T, D], with cos/sin as [batch, T, D]
    ref_q, ref_k = apply_rotary_pos_emb(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos[None], sin[None]
    )

    assert torch.equal(apply_rope(q, cos, sin), ref_q[0].transpose(0, 1))
    assert torch.equal(apply_rope(k, cos, sin), ref_k[0].transpose(0, 1))
