import random

import pytest
import torch

from engine.attention.paged import decode_attention, prefill_attention
from engine.attention.reference import attend


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 7, 128])
def test_prefill_matches_attend(n):
    torch.manual_seed(0)
    q = torch.randn(n, 14, 64, device="cuda")
    k = torch.randn(n, 2, 64, device="cuda")
    v = torch.randn(n, 2, 64, device="cuda")
    ref = attend(q, k, v, torch.arange(n, device="cuda"))
    torch.testing.assert_close(prefill_attention(q, k, v), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("seed", range(5))
def test_decode_matches_attend(seed):
    torch.manual_seed(seed); rnd = random.Random(seed)
    bs, num_blocks, h, kv_h, d = 16, 64, 14, 2, 64
    k_cache = torch.randn(num_blocks * bs, kv_h, d, device="cuda")
    v_cache = torch.randn(num_blocks * bs, kv_h, d, device="cuda")

    lens = [1, 16, 17, 33] + [rnd.randint(1, 120) for _ in range(4)]   # block edges + random
    free = rnd.sample(range(num_blocks), num_blocks)                   # shuffled: out-of-order blocks
    tables = [[free.pop() for _ in range(-(-n // bs))] for n in lens]
    width = max(map(len, tables))
    padded = torch.tensor([t + [0] * (width - len(t)) for t in tables], device="cuda")
    q = torch.randn(len(lens), h, d, device="cuda")

    out = decode_attention(q, k_cache, v_cache, padded, torch.tensor(lens, device="cuda"), max(lens), bs)

    for b, n in enumerate(lens):            # oracle: each sequence alone, contiguous context
        slots = torch.tensor([tables[b][p // bs] * bs + p % bs for p in range(n)], device="cuda")
        ref = attend(q[b:b + 1], k_cache[slots], v_cache[slots], torch.tensor([n - 1], device="cuda"))
        torch.testing.assert_close(out[b:b + 1], ref, atol=1e-4, rtol=1e-4)