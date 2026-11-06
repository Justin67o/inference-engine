import pytest
import torch
import math
from engine.cache.kv_cache import KVCache, Segment
from transformers import AutoConfig
import copy
from engine.model.model import Model
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

def attention(attn, x, cos, sin):
    # Attention.forward

    q, k, v = qkv(attn, x, cos, sin)
    T, rep = x.shape[0], attn.h // attn.kv_h
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    q, k, v = (t.transpose(0, 1) for t in (q, k, v))
    scores = q @ k.transpose(-2, -1) / math.sqrt(attn.d)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return attn.o_proj((probs @ v).transpose(0, 1).reshape(T, attn.h * attn.d))


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
        expected = attention(attn, x, cos, sin)

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_one_segment_matches_part4(cfg, dtype):
    torch.manual_seed(0)
    T, layer = 20, 3
    attn = Attention(cfg).to(dtype)
    x = torch.randn(T, cfg.hidden_size, dtype=dtype)
    cos, sin = rope_cos_sin(torch.arange(T), attn.d, cfg.rope_parameters["rope_theta"], dtype)
    cache = KVCache(cfg, attn.d, max_len=64, dtype=dtype, device="cpu")

    with torch.inference_mode():
        actual = attn(x, cos, sin, [Segment(cache, start=0, n=T)], layer)
        expected = attention(attn, x, cos, sin)
        _, k, v = qkv(attn, x, cos, sin)

    assert torch.equal(actual, expected), (actual - expected).abs().max().item()
    assert torch.equal(cache.k[layer, :T], k)            # this layer's rows were written
    assert torch.equal(cache.v[layer, :T], v)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_two_segments_match_each_alone(cfg, dtype):
    torch.manual_seed(0)
    attn = Attention(cfg).to(dtype)
    theta = cfg.rope_parameters["rope_theta"]
    xa, xb = torch.randn(7, cfg.hidden_size, dtype=dtype), torch.randn(3, cfg.hidden_size, dtype=dtype)
    pos = torch.cat([torch.arange(7), torch.arange(3)])   # each sequence starts at position 0
    cos, sin = rope_cos_sin(pos, attn.d, theta, dtype)

    def new_cache():
        return KVCache(cfg, attn.d, max_len=64, dtype=dtype, device="cpu")

    with torch.inference_mode():
        both = attn(torch.cat([xa, xb]), cos, sin,
                    [Segment(new_cache(), 0, 7), Segment(new_cache(), 0, 3)], layer=0)
        a = attn(xa, cos[:7], sin[:7], [Segment(new_cache(), 0, 7)], layer=0)
        b = attn(xb, cos[7:], sin[7:], [Segment(new_cache(), 0, 3)], layer=0)

    torch.testing.assert_close(both, torch.cat([a, b]))

@pytest.fixture(scope="module")
def tiny_model(cfg):
    # Real Qwen shapes per layer, but 2 layers and a small vocab so it fits on the CPU
    small = copy.deepcopy(cfg)
    small.num_hidden_layers, small.vocab_size = 2, 1000
    torch.manual_seed(0)
    return Model(small).eval()


def test_model_returns_last_token_logits_per_segment(tiny_model):
    m = tiny_model
    a, b = torch.randint(1000, (7,)), torch.randint(1000, (3,))

    with torch.inference_mode():
        both = m(torch.cat([a, b]), [Segment(m.new_cache(16), 0, 7), Segment(m.new_cache(16), 0, 3)])
        alone_a = m(a, [Segment(m.new_cache(16), 0, 7)])
        alone_b = m(b, [Segment(m.new_cache(16), 0, 3)])

    assert both.shape == (2, 1000)                     # one row per segment, not per token
    torch.testing.assert_close(both, torch.cat([alone_a, alone_b]))


def test_model_decode_from_cache_matches_prefill(tiny_model):
    m = tiny_model
    ids = torch.randint(1000, (8,))

    with torch.inference_mode():
        full = m(ids, [Segment(m.new_cache(16), 0, 8)])   # all 8 tokens at once
        cache = m.new_cache(16)
        m(ids[:7], [Segment(cache, 0, 7)])                # prefill 7 tokens
        step = m(ids[7:], [Segment(cache, 7, 1)])         # decode the 8th from the cache

    torch.testing.assert_close(step, full)
