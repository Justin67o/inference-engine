import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decoded_kernel(q_ptr, k_ptr, v_ptr, tables_ptr, lens_ptr, out_ptr, table_stride, scale,
                          KV_H: tl.constexpr, REP: tl.constexpr, REP_P: tl.constexpr,
                          D: tl.constexpr, BS: tl.constexpr):
    """One program per (sequence, KV head). Its REP query heads walk the sequence's block table one
    block at a time with an online softmax, reading each K/V block once from the cache"""

    b = tl.program_id(0)                # sequence
    g = tl.program_id(1)                # KV head
    n = tl.load(lens_ptr + b)           # context length including new token
    r = tl.arange(0, REP_P)             # query heads sharing KV head, g, padded to REP_P
    dd = tl.arange(0, D)
    t = tl.arange(0, BS)                # offsets within a block
    head_ok = r < REP
    q_off = (b * KV_H * REP + g * REP + r)[:, None] * D + dd[None, :]
    q = tl.load(q_ptr + q_off, mask=head_ok[:, None], other=0.0)     # [REP_P, D]

    m = tl.full([REP_P], float("-inf"), tl.float32)     # running max of the scores
    s_sum = tl.zeros([REP_P], tl.float32)               # running softmax denominator
    acc = tl.zeros([REP_P, D], tl.float32)             # running softmax-weighted sum of V

    for i in range (tl.cdiv(n, BS)):
        block = tl.load(tables_ptr + b * table_stride + i)
        ok = i * BS + t < n                           # is it past the end of the sequence
        kv_off = ((block * BS + t) * KV_H + g)[:, None] * D + dd[None, :]       # [BS, D]
        k = tl.load(k_ptr + kv_off, mask=ok[:, None], other=0.0)
        v = tl.load(v_ptr + kv_off, mask=ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale      # [REP_P, BS]
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)       # rescales what we summed so far
        p = tl.exp(s - m_new[:, None])
        s_sum = s_sum * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m = m_new

    out = acc / s_sum[:, None]
    tl.store(out_ptr + q_off, out.to(out_ptr.dtype.element_ty), mask=head_ok[:, None])


def paged_decode_attention(q, k_cache, v_cache, tables, lens, M, bs):
    """Same arguments and result as paged.decode_attention, without building the gathered copy.
    M is unused: each sequence stops at its own length"""

    B, h, d = q.shape
    kv_h = k_cache.shape[1]
    assert bs >= 16 and d >= 16, "tl.dot needs every dimension to be at least 16"
    assert k_cache.is_contiguous() and v_cache.is_contiguous()

    q = q.contiguous()
    out = torch.empty_like(q)
    rep = h // kv_h
    _paged_decoded_kernel[(B, kv_h)](
        q, k_cache, v_cache, tables, lens, out, tables.stride(0), 1 / math.sqrt(d),
        KV_H=kv_h, REP=rep, REP_P=max(16, triton.next_power_of_2(rep)), D=d, BS=bs, num_warps=2)

    return out.reshape(B, h * d)
