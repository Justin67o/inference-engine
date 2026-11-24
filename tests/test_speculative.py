import torch

from engine.speculative import verify


def logits(*rows):                      # logits whose softmax is exactly these probability rows
    return torch.tensor(rows).log()     # log(0) = -inf, so a zero probability stays exactly zero


# Greedy: the argmax of each row is 1, 0, 3
P_GREEDY = logits([.1, .7, .1, .1], [.7, .1, .1, .1], [.1, .1, .1, .7])


def test_greedy_accepts_matching_drafts_plus_bonus():
    assert verify(P_GREEDY, [1, 0], None, 0) == [1, 0, 3]


def test_greedy_stops_at_first_mismatch_with_correction():
    assert verify(P_GREEDY, [1, 2], None, 0) == [1, 0]


def test_greedy_first_draft_wrong_gives_only_the_correction():
    assert verify(P_GREEDY, [2, 0], None, 0) == [1]


def test_greedy_no_drafts_is_plain_decode():
    assert verify(P_GREEDY[:1], [], None, 0) == [1]


# Sampled: hand-made target p and draft q over 4 tokens
P = [.6, .3, .1, .0]
Q = [.2, .3, .1, .4]


def test_q_equal_p_always_accepts():
    torch.manual_seed(0)
    p, q = logits(P, P, P), torch.tensor(P)
    for _ in range(200):
        out = verify(p, [0, 1], [q, q], 1.0)
        assert out[:2] == [0, 1] and len(out) == 3


def test_draft_the_target_rules_out_is_always_rejected():
    torch.manual_seed(0)
    p, q = logits(P, P), torch.tensor(Q)
    for _ in range(200):
        assert verify(p, [3], [q], 1.0) == [0]      # residual max(p - q, 0) is [.4, 0, 0, 0]


def test_acceptance_rate_is_sum_of_min_p_q():
    torch.manual_seed(0)
    p, q, n = logits(P, P), torch.tensor(Q), 20_000
    accepted = 0
    for _ in range(n):
        d = torch.multinomial(q, 1).item()          # the draft proposes from q
        accepted += len(verify(p, [d], [q], 1.0)) == 2
    assert abs(accepted / n - 0.6) < 0.02           # min(P, Q) = .2 + .3 + .1 + 0


def test_first_token_follows_p_not_q():
    torch.manual_seed(0)
    p, q, n = logits(P, P), torch.tensor(Q), 20_000
    counts = torch.zeros(4)
    for _ in range(n):
        d = torch.multinomial(q, 1).item()
        counts[verify(p, [d], [q], 1.0)[0]] += 1
    torch.testing.assert_close(counts / n, torch.tensor(P), atol=0.02, rtol=0)


def test_temperature_is_applied_to_p():
    torch.manual_seed(0)
    p_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]] * 2)
    q = torch.softmax(p_logits[0] / 0.5, -1)        # draft distribution at the same temperature
    for _ in range(200):
        assert verify(p_logits, [0], [q], 0.5)[0] == 0