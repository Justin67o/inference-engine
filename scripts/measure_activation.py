"""Measure peak activation memory during the largest prefill the engine allows"""
import torch

from engine.cache.kv_cache import fresh
from engine.config import EngineConfig
from engine.model.loader import load_model

MiB = 2**20


@torch.inference_mode()
def measure_prefill_peak(model, n_tokens):
    """Bytes the forward pass needs on top of what is already allocated (weights, caches)"""
    cache, seg = fresh(model, n_tokens)       # allocate before the baseline: not activations
    ids = torch.randint(0, model.cfg.vocab_size, (n_tokens,), device="cuda")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    base_resv = torch.cuda.memory_reserved()
    model(ids, [seg], cache)
    torch.cuda.synchronize()
    act_alloc = torch.cuda.max_memory_allocated() - base_alloc
    act_resv = torch.cuda.max_memory_reserved() - base_resv
    del cache
    torch.cuda.empty_cache()
    return act_alloc, act_resv


if __name__ == "__main__":
    cfg = EngineConfig.from_env()
    model, mcfg = load_model(cfg.model)
    # Largest prefill: one prompt filling the whole token budget, capped by max_model_len
    # A single long segment is the worst case because reference attention is O(T^2)
    n = min(cfg.max_num_batched_tokens, cfg.max_model_len)
    print(f"weights: {torch.cuda.memory_allocated() / MiB:.0f} MiB")
    for t in (256, 512, 1024, n):
        a, r = measure_prefill_peak(model, t)
        print(f"prefill {t:5d} tokens: activations {a / MiB:7.1f} MiB allocated, "
            f"{r / MiB:7.1f} MiB reserved")
    free, total = torch.cuda.mem_get_info()
    print(f"free after load: {free / MiB:.0f} of {total / MiB:.0f} MiB")
    