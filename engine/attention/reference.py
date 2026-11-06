import math

import torch


def attend(q, k, v, q_pos): # q: [n, h, d]   k, v: [S, kv_h, d]
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]) # [h, n, S]
    key_pos = torch.arange(k.shape[1], device=q.device)
    future = key_pos[None, None, :] > q_pos[None, :, None] # causal mask
    probs = torch.softmax(scores.masked_fill(future, float("-inf")).float(), -1)

    return (probs.to(q.dtype) @ v).transpose(0, 1).reshape(q.shape[1], -1)