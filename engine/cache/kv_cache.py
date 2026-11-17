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


class CachePool:
    """A fixed set of preallocated caches, lent to running requests and returned when they finish."""


    def __init__(self, make_cache, size):
        self.size = size
        self.free = [make_cache() for _ in range(size)]

    def has_free(self):
        return len(self.free) > 0

    def take(self):
        return self.free.pop()

    def give_back(self, cache):
        if any(c is cache for c in self.free):
            raise ValueError("cache returned twice")
        self.free.append(cache)

    @property
    def num_free(self):
        return len(self.free)