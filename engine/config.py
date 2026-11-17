from dataclasses import dataclass


@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_num_seqs: int = 32                  # cap on RUNNING sequences = caches in the pool
    max_num_batched_tokens: int = 2048      # cap on tokens per step (prefills + decodes)
    max_model_len: int = 2048               # per-sequence cache length: prompt + max_tokens

    def __post_init__(self):
        if self.max_num_seqs > self.max_num_batched_tokens:
            raise ValueError("max_num_seqs must be <= max_num_batched_tokens")

        