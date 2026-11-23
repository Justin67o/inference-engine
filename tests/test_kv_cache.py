import copy
import json
import math
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

from engine.attention.reference import attend
from engine.cache.kv_cache import PagedKVCache, Segment, fresh
from engine.generate import generate
from engine.model.layers import Attention, apply_rope, rope_cos_sin
from engine.model.model import Model

PROMPTS_FILE = Path(__file__).parent / "prompts_200.jsonl"
PROMPTS = [json.loads(line)["prompt"] for line in PROMPTS_FILE.read_text().splitlines()][:24]
MAX_NEW_TOKENS = 32
STOP_IDS = (151645, 151643)     # <|im_end|>, <|endoftext|>: some prompts stop early
NEAR_TIE = 0.5
m1_cache = {}

@torch.inference_mode()
def m1_generate(model, ids):
    """ Rerun the whole sequence each step. Returns (new tokens, top-2 gap at each step)."""
    seq, gaps = ids, []

    for _ in range(MAX_NEW_TOKENS):
        cache, seg = fresh(model, len(seq))
        logits = model(seq, [seg], cache)[0]
        top = logits.float().topk(2).values
        gaps.append((top[0] - top[1]).item())
        seq = torch.cat([seq, logits.argmax()[None]])
        if seq[-1].item() in STOP_IDS:
            break
    return seq[len(ids):], gaps

def compare_to_m1(models, tok, dtype, batch_size):
    """Runs generate() in batches of batch_size; returns (prompt index, step, M1's gap) per mismatch."""
    _, mine = models(dtype)
    prompts = [tok(p, return_tensors="pt").input_ids[0].cuda() for p in PROMPTS]
    if dtype not in m1_cache: # slow; compute once per dtype
        m1_cache.clear()
        m1_cache[dtype] = [m1_generate(mine, ids) for ids in prompts]
    ref = m1_cache[dtype]

    out = []
    for k in range(0, len(prompts), batch_size):    # consecutive prompts: mixed lengths and categories
        out += generate(mine, prompts[k:k + batch_size], MAX_NEW_TOKENS, STOP_IDS)

    mismatches = []
    for i, (tokens, (ref_tokens, gaps)) in enumerate(zip(out, ref)):
        if not torch.equal(tokens, ref_tokens):
            n = min(len(tokens), len(ref_tokens))
            diff = (tokens[:n] != ref_tokens[:n]).nonzero()
            step = diff[0].item() if len(diff) else n
            mismatches.append((i, step, gaps[step]))
    return mismatches

@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 5, 8])
def test_fp32_generate_identical_to_m1(models, tok, batch_size):
    mismatches = compare_to_m1(models, tok, torch.float32, batch_size)
    assert not mismatches, f"(prompt, step, M1 top-2 gap): {mismatches}"

@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 5, 8])
def test_bf16_generate_differs_from_m1_only_at_near_ties(models, tok, batch_size):
    # bf16 can't be token-identical: the cache changes matmul shapes, so near-tied tokens flip
    mismatches = compare_to_m1(models, tok, torch.bfloat16, batch_size)
    real = [m for m in mismatches if m[2] >= NEAR_TIE]
    assert not real, f"not near-ties (prompt, step, M1 top-2 gap): {real}"

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
    cache = PagedKVCache(cfg, attn.d, num_blocks=4, block_size=16, dtype=dtype, device="cpu")
    slots = torch.arange(T)                              # table [0, 1]: slot = position

    with torch.inference_mode():
        actual = attn(x, cos, sin, [Segment([0, 1], start=0, n=T)], layer, cache, slots, [slots])
        expected = attention(attn, x, cos, sin)
        _, k, v = qkv(attn, x, cos, sin)

    tol = {} if dtype == torch.float32 else {"atol": 1e-2, "rtol": 1.6e-2}   # SDPA rounds differently from Part 4
    torch.testing.assert_close(actual, expected, **tol)
    assert torch.equal(cache.k[layer, slots], k)         # this layer's rows were written
    assert torch.equal(cache.v[layer, slots], v)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_two_segments_match_each_alone(cfg, dtype):
    torch.manual_seed(0)
    attn = Attention(cfg).to(dtype)
    theta = cfg.rope_parameters["rope_theta"]
    xa, xb = torch.randn(7, cfg.hidden_size, dtype=dtype), torch.randn(3, cfg.hidden_size, dtype=dtype)
    pos = torch.cat([torch.arange(7), torch.arange(3)])   # each sequence starts at position 0
    cos, sin = rope_cos_sin(pos, attn.d, theta, dtype)

    def new_cache():
        return PagedKVCache(cfg, attn.d, num_blocks=2, block_size=16, dtype=dtype, device="cpu")

    sa, sb = torch.arange(7), torch.arange(16, 19)       # A in block 0, B in block 1
    with torch.inference_mode():
        both = attn(torch.cat([xa, xb]), cos, sin, [Segment([0], 0, 7), Segment([1], 0, 3)],
                    0, new_cache(), torch.cat([sa, sb]), [sa, sb])
        a = attn(xa, cos[:7], sin[:7], [Segment([0], 0, 7)], 0, new_cache(), sa, [sa])
        b = attn(xb, cos[7:], sin[7:], [Segment([1], 0, 3)], 0, new_cache(), sb, [sb])

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
        both = m(torch.cat([a, b]), [Segment([0], 0, 7), Segment([1], 0, 3)], m.new_cache(2))
        alone_a = m(a, [Segment([0], 0, 7)], m.new_cache(1))
        alone_b = m(b, [Segment([0], 0, 3)], m.new_cache(1))

    assert both.shape == (2, 1000)                     # one row per segment, not per token
    torch.testing.assert_close(both, torch.cat([alone_a, alone_b]))


def test_model_decode_from_cache_matches_prefill(tiny_model):
    m = tiny_model
    ids = torch.randint(1000, (20,))

    with torch.inference_mode():
        full = m(ids, [Segment([0, 1], 0, 20)], m.new_cache(2))   # all 20 tokens at once
        cache = m.new_cache(4)
        m(ids[:19], [Segment([3, 1], 0, 19)], cache)               # prefill 19 tokens
        step = m(ids[19:], [Segment([3, 1], 19, 1)], cache)        # decode the 20th: block 1, offset 3

    torch.testing.assert_close(step, full)

def test_scattered_blocks_match_contiguous(tiny_model):
    m, ids = tiny_model, torch.randint(1000, (40,))       # 40 tokens = 3 blocks of 16
    with torch.inference_mode():
        a = m(ids, [Segment([0, 1, 2], 0, 40)], m.new_cache(8))
        b = m(ids, [Segment([7, 2, 5], 0, 40)], m.new_cache(8))
    assert torch.equal(a, b)
