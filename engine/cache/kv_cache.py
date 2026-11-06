from dataclasses import dataclass

import torch


class KVCache:
    def __init__(self, cfg, head_dim, max_len, dtype, device="cuda"):
        shape = (cfg.num_hidden_layers, max_len, cfg.num_key_value_heads, head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty(shape, dtype=dtype, device=device)


@dataclass
class Segment:
    cache: KVCache
    start: int # absolute position of this segments first new token
    n: int # number of new tokens (prompt length in prefill, 1 in decode)
