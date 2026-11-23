import torch.nn.functional as F


def prefill_attention(q, k, v):     # q: [n, h, d]  k, v: [n, kv_h, d] 
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(0, 1).reshape(q.shape[1], -1)