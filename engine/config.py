import os
from dataclasses import dataclass, fields


@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_num_seqs: int = 32                  # cap on RUNNING sequences = caches in the pool
    max_num_batched_tokens: int = 2048      # cap on tokens per step (prefills + decodes)
    max_model_len: int = 2048               # per-sequence cache length: prompt + max_tokens
    watermark: float = 0.1                  # share of KV blocks admission leaves free for decodes to grow
    num_blocks: int = 0                     # KV blocks of 16 tokens

    def __post_init__(self):
        if self.max_num_seqs > self.max_num_batched_tokens:
            raise ValueError("max_num_seqs must be <= max_num_batched_tokens")
        if not 0 <= self.watermark < 1:
            raise ValueError("watermark must be between 0 and 1")


    @classmethod
    def from_env(cls):
        """The defaults, overriden by ENGINE_<FIELD> environment variables,
        e.g. ENGINE_MAX_NUM_BATCHED_TOKENS=256"""
        overrides = {}
        for f in fields(cls):
            value = os.environ.get(f"ENGINE_{f.name.upper()}")
            if value is not None:
                overrides[f.name] = type(f.default)(value)
        return cls(**overrides)