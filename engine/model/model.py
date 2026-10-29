import torch.nn as nn
from engine.model.layers import RMSNorm, DecoderLayer, rope_cos_sin

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    def forward(self, ids, positions):                         # ids, positions: [T]
        x = self.embed_tokens(ids)
        cos, sin = rope_cos_sin(positions, self.layers[0].self_attn.d,
                                self.cfg.rope_parameters["rope_theta"], x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))                      # [T, vocab]
    