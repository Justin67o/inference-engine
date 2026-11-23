import torch

from engine.cache.kv_cache import Segment


@torch.inference_mode()
def generate(model, prompts, max_new_tokens, stop_ids=()):
    """Greedy decoding with a KV cache and static batching: one prefill for all prompts,
    then one decode step per token for every sequence that hasn't stopped.

    prompts: list of [T] prompt token ids on the model's device. Returns a list of new token ids, [N] each."""

    device = prompts[0].device
    lengths = [0] * len(prompts)        # tokens already in each cache
    new = [[] for _ in prompts]         
    active = list(range(len(prompts)))  # indices of prompts that haven't stopped yet

    bs = 16
    need = [-(-(len(p) + max_new_tokens) // bs) for p in prompts]
    cache = model.new_cache(sum(need), bs)
    starts = [sum(need[:i]) for i in range(len(prompts))]
    tables = [list(range(s, s + n))[::-1] for s, n in zip(starts, need)]

    # prefill: each whole prompt is one segment of a single forward pass
    ids = torch.cat(prompts)
    segments = [Segment(tables[i], 0, len(prompts[i])) for i in active]

    for _ in range(max_new_tokens):
        next_ids = model(ids, segments, cache).argmax(-1).tolist()     # one per active sequence
        for i, s, t in zip(active, segments, next_ids):
            lengths[i] += s.n
            new[i].append(t)

        active = [i for i in active if new[i][-1] not in stop_ids]
        if not active:
            break

        # Decode: each active sequence feeds back its newest token, one position past its cache
        ids = torch.tensor([new[i][-1] for i in active], device=device)
        segments = [Segment(tables[i], lengths[i], 1) for i in active]

    return [torch.tensor(n, dtype=torch.long, device=device) for n in new]


