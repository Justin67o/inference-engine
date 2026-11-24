import torch

from engine.cache.kv_cache import Segment


def verify(p_logits, draft_tokens, q_probs, temperature):
    """Decides which draft tokens the target accepts, for one sequence.
    
    p_logits [k+1, V]: target logits; row i predicts the position of draft tokens i, since
    the target's input starts with the last accepted token. 
    draft_tokens: list of k ids.
    q_probs: k draft distributions [V], made at the same temperature (unused when greedy)
    Returns the accepted drafts + one more token, a correction after a rejection or a bonus
    if every draft was accepted. Always 1 to k+1 tokens"""

    k = len(draft_tokens)

    if temperature == 0:        # greedy, accept while drafts match argmax
        return accept_greedy(p_logits.argmax(-1).tolist(), draft_tokens)


    p = torch.softmax(p_logits.float() / temperature, -1)
    for i, t in enumerate(draft_tokens):
        q = q_probs[i].to(p)
        r = torch.rand(()).item()

        if r * q[t] < p[i, t]:      # accept with prob min(1, p/q), no division
            continue
        residual = (p[i] - q).clamp(min=0)
        if residual.sum() <= 0:     # p == q up to rounding, sample from p
            residual = p[i]
        return draft_tokens[:i] + [torch.multinomial(residual, 1).item()]
    return draft_tokens + [torch.multinomial(p[k], 1).item()]


def propose_ngram(tokens, k, n_max=3, n_min=1):
    """Prompt lookup: find the latest earlier occurrence of the last n tokens (longest n first)
    and proposes the up to k tokens that followed it. Returns [] when nothing matches"""
    for n in range(min(n_max, len(tokens) - 1), n_min - 1, -1):
        tail = tokens[-n:]
        for i in range(len(tokens) - n - 1, -1, -1):    # latest first, the match must have a follower
            if tokens[i:i + n] == tail:
                return tokens[i + n: i + n + k]
    return []


def rollback(blocks, r):
    """After a verify step the target cache is valid for r.length - 1 positions: everything but the
    newest token, which is the next step's input. Rejected drafts' K/V sits past that and gets
    overwritten next step; free the blocks lying past it. The draft cache is valid for the 
    same prefix at most, drafts past the first rejection are wrong, and aftr a full accept it still
    lacks the last draft"""
    blocks.trim(r, r.length - 1)
    r.draft_len = min(r.draft_len, r.length - 1)


def sample_draft(logits, temps):
    """One draft token per row: argmax when greedy, else a sample from softmax(logits / T).
    Returns the tokens [n] and the distributions they came from [n, V], unused when greedy"""
    t = torch.tensor(temps, device=logits.device).unsqueeze(1)
    probs = torch.softmax(logits.float() / t.clamp(min=1e-6), -1)
    toks = torch.where(t.squeeze(1) == 0, logits.argmax(-1), torch.multinomial(probs, 1).squeeze(1))
    return toks, probs


def draft_loop(model, cache, work, device):
    """work: (request, k) pairs, k >= 1. A catch-up pass feeds each request every accepted token
    its draft cache lacks, then k - 1 one-token passes feed the drafts back. The last draft is never
    fed, so each request ends with r.length + k - 1 cached positions.
    Returns {r: (k draft tokens, [k, V] draft probabibilities)}"""

    toks = {r: [] for r, _ in work}
    probs = {r: [] for r, _ in work}

    ids, segs = [], []
    for r, _ in work:       # catch-up: draft_len ... length - 1, newest token included
        ids += (r.prompt_ids + r.output_ids)[r.draft_len:]
        segs.append(Segment(r.block_table, r.draft_len, r.length - r.draft_len))
        r.draft_len = r.length

    logits = model(torch.tensor(ids, device=device), segs, cache)
    rows = [s.n if s.start > 0 else 1 for s in segs]    # a prefill returns only its last row
    last = logits[torch.tensor(rows, device=device).cumsum(0) - 1]

    active = work

    while True:
        t, p = sample_draft(last, [r.temperature for r, _ in active])
        for (r, _), ti, pi in zip(active,t.tolist(), p):
            toks[r].append(ti)
            probs[r].append(pi)
        active = [(r, k) for r, k in active if len(toks[r]) < k]
        if not active:
            break
        segs = [Segment(r.block_table, r.draft_len, 1) for r, _ in active]
        last = model(torch.tensor([toks[r][-1] for r, _ in active], device=device), segs, cache)

        for r, _ in active:
            r.draft_len += 1

    return {r: (toks[r], torch.stack(probs[r])) for r, _ in work}


def accept_greedy(target, draft_tokens):
    """target: the target's argmax ids, k+1 of them"""
    for i, t in enumerate(draft_tokens):
        if t != target[i]:
            return draft_tokens[:i] + [target[i]]
        
    return draft_tokens + [target[len(draft_tokens)]]