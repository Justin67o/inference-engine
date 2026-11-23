from dataclasses import dataclass

import torch


class PagedKVCache:
    """K and V for every layer, as num_blocks * block_size flat slots shared by all sequences.
    Which slots belong to which sequence is the block manager's work, not the cache"""

    def __init__(self, cfg, head_dim, num_blocks, block_size, dtype, device="cuda"):
        self.bs, self.num_blocks = block_size, num_blocks
        shape = (cfg.num_hidden_layers, num_blocks * block_size, cfg.num_key_value_heads, head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty(shape, dtype=dtype, device=device)


    def write(self, layer, slots, k, v):
        """slots: [N] long tensor on the cache's device; k, v: [N, kv_h, d]"""
        self.k[layer, slots] = k
        self.v[layer, slots] = v


@dataclass
class Segment:
    table:list[int]     # the sequence's physical block ids, in order
    start: int          # absolute position of this segment's first new token
    n: int              # number of new tokens (prompt length in prefill, 1 in decode)


def fresh(model, n, bs=16):
    """A new cache and Segment for one n-token sequence on contiguus blocks"""
    nb = -(-n // bs)
    return model.new_cache(nb, bs), Segment(list(range(nb)), 0, n)
    