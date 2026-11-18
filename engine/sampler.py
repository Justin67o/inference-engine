import torch


def sample(logits, temperature, top_k, top_p, generators=None):
    """Picks one token per row, each with its own settings.
    
    logits [B, V]. Settings are Python lists of length B: temperature (0 = greedy),
    top_k (0 = off), top_p (1.0 = off), and generators (a seeded torch.Generator, or None
    to use the global one). Returns [B] token ids."""

    greedy = logits.argmax(-1)
    if all(t == 0 for t in temperature):        # fast path: all greedy
        return greedy

    dev, vocab = logits.device, logits.shape[-1]
    temp = torch.tensor(temperature, device=dev)
    logits = logits.float() / temp.clamp(min=1e-5)[:, None]

    if any(k > 0 for k in top_k) or any(p < 1.0 for p in top_p):
        # Sort once, highest first: top-k and top-p both cut off the tail of each sorted row
        logits, ids = logits.sort(-1, descending=True)
        k = torch.tensor([k if k > 0 else vocab for k in top_k], device=dev)
        logits = logits.masked_fill(torch.arange(vocab, device=dev) >= k[:, None], float("-inf"))
        probs = logits.softmax(-1)
        p = torch.tensor(top_p, device=dev)
        probs = probs.masked_fill(probs.cumsum(-1) - probs > p[:, None], 0.0)   # keeps the top token
    else:                                                                       # fast path: no sort needed
        probs, ids = logits.softmax(-1), None


    picked = torch.multinomial(probs, 1).squeeze(1)     # multinomial renormalizes
    for i, g in enumerate(generators or []):            # seeded rows: redo with their own
        if g is not None:
            picked[i] = torch.multinomial(probs[i], 1, generator=g)[0]

    if ids is not None:
        picked = ids.gather(-1, picked[:, None]).squeeze(1)     # rank in sorted row -> token id

    return torch.where(temp == 0, greedy, picked)