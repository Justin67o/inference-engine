import math

import torch
import torch.nn.functional as F


def prefill_attention(q, k, v):     # q: [n, h, d]  k, v: [n, kv_h, d] 
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(0, 1).reshape(q.shape[1], -1)


def decode_attention(q, k_cache, v_cache, tables, lens, M, bs):
    """One new token per sequence, attending over its cached context
    q: [B, h, d]    k_cache, v_cache: [slots, kv_h, d] (one layer)
    tables: [B, max_blocks] long, padded with 0     lens: [B] including the
    new token   M: max(lens) as an int"""

    B, h, d = q.shape
    kv_h = k_cache.shape[1]

    pos = torch.arange(M, device=q.device)
    slot_idx = tables[:, pos // bs] * bs + pos % bs         # [B, M]
    K = k_cache[slot_idx].transpose(1, 2)                   # [B, kv_h, M, d]
    V = v_cache[slot_idx].transpose(1, 2)
    q = q.view(B, kv_h, h // kv_h, d)                       # group query heads by their KV head (head j uses j // rep)
    scores = (q @ K.transpose(-2, -1)) / math.sqrt(d)       # [B, kv_h, rep, M]
    pad = pos[None, :] >= lens[:, None]                     # [B, M]
    scores = scores.masked_fill(pad[:, None, None, :], float("-inf"))
    out = torch.softmax(scores.float(), -1).to(q.dtype) @ V # [B, kv_h, rep, d]
    return out.reshape(B, h * d)