import torch
from torch import nn

from engine.cache.kv_cache import KVCache
from engine.model.layers import DecoderLayer, RMSNorm, rope_cos_sin


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, ids, segments):                         # ids: [N], every segment's new tokens
        x = self.embed_tokens(ids)
        positions = torch.cat([torch.arange(s.start, s.start + s.n, device=ids.device) for s in segments])
        cos, sin = rope_cos_sin(positions, self.layers[0].self_attn.d,
                                self.cfg.rope_parameters["rope_theta"], x.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, segments, i) 
        last = torch.tensor([s.n for s in segments]).cumsum(0) - 1 # each segment's last row in x
        return self.lm_head(self.norm(x[last.to(x.device)]))               # [num_segments, vocab]

    def new_cache(self, max_len):
        w = self.embed_tokens.weight
        return KVCache(self.cfg, self.layers[0].self_attn.d, max_len, w.dtype, w.device)