from dataclasses import dataclass

import torch
from torch import nn

from engine.cache.block_manager import slot_ids
from engine.cache.kv_cache import PagedKVCache
from engine.model.layers import DecoderLayer, RMSNorm, rope_cos_sin


@dataclass
class StepInputs:
    """What run() needs besides the token ids. prepare() builds it from Python segments;
    the CUDA graph runner fills the same fields from static buffers instead"""
    positions: torch.Tensor     # [N] absolute position of each new token
    write: torch.Tensor         # [N] cache slot for each new token's K/V
    segments: list              # Attention reads it for prefills; [] when there are none
    batch: tuple | None         # decodes: (rows, tables, lens, M); rows None = every row
    chunks: list                # verify segments (rows, n, slots, positions)
    keep: torch.Tensor | None   # rows whose logits to return; None = every row


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    
    def forward(self, ids, segments, cache):        # ids: [N] every segment's new tokens
        return self.run(ids, self.prepare(segments, ids.device, cache.bs), cache)


    def prepare(self, segments, dev, bs):
        """Python lists to device tensors, the part of a step that a CUDA graph can't replay"""
        positions = torch.cat([torch.arange(s.start, s.start + s.n, device=dev) for s in segments])

        # where each new token's KV goes, for the whole step at once
        write = torch.tensor([sl for s in segments for sl in slot_ids(s.table, s.start, s.n, bs)], device=dev)

        # decode segment, batched once per step for every layer's gather
        dec, rows, chunks, keep, i = [], [], [], [], 0

        for s in segments:
            if s.start > 0 and s.n == 1:    # decode
                dec.append(s)
                rows.append(i)
            elif s.start > 0:               # verify: n new tokens on a cached prefix
                slots = torch.tensor(slot_ids(s.table, 0, s.start + s.n, bs), device=dev)
                chunks.append((i, s.n, slots, torch.arange(s.start, s.start + s.n, device=dev)))
            keep += range(i, i + s.n) if s.start > 0 else [i + s.n - 1]
            i += s.n

        batch = None
        if dec:
            width = max(len(s.table) for s in dec)
            tables = torch.tensor([s.table + [0] * (width - len(s.table)) for s in dec], device=dev)
            lens = [s.start + 1 for s in dec]
            batch = (torch.tensor(rows, device=dev), tables, torch.tensor(lens, device=dev), max(lens))
        return StepInputs(positions, write, segments, batch, chunks, torch.tensor(keep, device=dev))


    def run(self, ids, inp, cache):
        """Tensor ops only. With no prefills or chunks, rows and keep None, and fixed shapes,
        a CUDA graph can capture it"""
        x = self.embed_tokens(ids)
        cos, sin = rope_cos_sin(inp.positions, self.layers[0].self_attn.d,
                                self.cfg.rope_parameters["rope_theta"], x.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, inp.segments, i, cache, inp.write, inp.batch, inp.chunks)
        if inp.keep is not None:
            x = x[inp.keep]
        return self.lm_head(self.norm(x))
    
    
    def new_cache(self, num_blocks, block_size=16):
        w = self.embed_tokens.weight
        return PagedKVCache(self.cfg, self.layers[0].self_attn.d, num_blocks, block_size, w.dtype, w.device)




