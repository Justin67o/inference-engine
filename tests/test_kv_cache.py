import pytest
import torch
from transformers import AutoConfig

from engine.attention.reference import attend
from engine.model.layers import Attention, apply_rope, rope_cos_sin


@pytest.fixture(scope="module")
def cfg():
    return AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def qkv(attn, x, cos, sin):
    # The same projections and RoPE as Attention.forward, stopping before attention
    T = x.shape[0]
    q = apply_rope(attn.q_proj(x).view(T, attn.h, attn.d), cos, sin)
    k = apply_rope(attn.k_proj(x).view(T, attn.kv_h, attn.d), cos, sin)
    v = attn.v_proj(x).view(T, attn.kv_h, attn.d)
    return q, k, v

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attend_matches_attention(cfg, dtype):
    torch.manual_seed(0)
    T = 20
    attn = Attention(cfg).to(dtype)
    x = torch.randn(T, cfg.hidden_size, dtype=dtype)
    pos = torch.arange(T)
    cos, sin = rope_cos_sin(pos, attn.d, cfg.rope_parameters["rope_theta"], dtype)

    with torch.inference_mode():
        q, k, v = qkv(attn, x, cos, sin)
        actual = attn.o_proj(attend(q, k, v, pos))
        expected = attn(x, cos, sin)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), (actual - expected).abs().max().item()

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attend_last_token_only(cfg, dtype):
    torch.manual_seed(0)
    T = 20
    attn = Attention(cfg).to(dtype)
    x = torch.randn(T, cfg.hidden_size, dtype=dtype)
    pos = torch.arange(T)
    cos, sin = rope_cos_sin(pos, attn.d, cfg.rope_parameters["rope_theta"], dtype)

    with torch.inference_mode():
        q, k, v = qkv(attn, x, cos, sin)
        actual = attend(q[-1:], k, v, pos[-1:]) # decode: one query, whole cache
        expected = attend(q, k, v, pos)[-1:] # last row of the full pass

    torch.testing.assert_close(actual, expected)