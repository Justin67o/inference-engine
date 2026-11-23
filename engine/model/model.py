import torch
from torch import nn

from engine.cache.block_manager import slot_ids
from engine.cache.kv_cache import PagedKVCache
from engine.model.layers import DecoderLayer, RMSNorm, rope_cos_sin


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, ids, segments, cache):                         # ids: [N], every segment's new tokens
        x = self.embed_tokens(ids)
        dev, bs = ids.device, cache.bs
        positions = torch.cat([torch.arange(s.start, s.start + s.n, device=ids.device) for s in segments])
        cos, sin = rope_cos_sin(positions, self.layers[0].self_attn.d,
                                self.cfg.rope_parameters["rope_theta"], x.dtype)
        # where each new token's K/V goes, for the whole step at once
        write = torch.tensor([sl for s in segments for sl in slot_ids(s.table, s.start, s.n, bs)], device=dev)
        # decode segments, batched once per step for every layer's gather
        dec, rows, i = [], [], 0

        for s in segments:
            if s.start > 0:
                assert s.n == 1, "no chunked prefill: a segment with start > 0 is a decode"
                dec.append(s); rows.append(i)
            i += s.n

        batch = None
        if dec:
            width = max(len(s.table) for s in dec)
            tables = torch.tensor([s.table + [0] * (width - len(s.table)) for s in dec], device=dev)
            lens = [s.start + 1 for s in dec]
            batch = (torch.tensor(rows, device=dev), tables, torch.tensor(lens, device=dev), max(lens))
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, segments, i, cache, write, batch) 
        last = torch.tensor([s.n for s in segments]).cumsum(0) - 1 # each segment's last row in x
        return self.lm_head(self.norm(x[last.to(x.device)]))               # [num_segments, vocab]

    def new_cache(self, num_blocks, block_size=16):
        w = self.embed_tokens.weight
        return PagedKVCache(self.cfg, self.layers[0].self_attn.d, num_blocks, block_size, w.dtype, w.device)