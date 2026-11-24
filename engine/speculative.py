import torch


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
        target = p_logits.argmax(-1).tolist()
        for i, t in enumerate(draft_tokens):
            if t != target[i]:
                return draft_tokens[:i] + [target[i]]
        return draft_tokens + [target[k]]


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
