import pytest
import torch

from engine.cache.kv_cache import Segment, fresh
from engine.config import EngineConfig
from engine.engine import Engine
from engine.request import Request
from engine.speculative import propose_ngram, verify


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

def test_lookup_proposes_what_followed_the_match():
    assert propose_ngram([1, 2, 3, 4, 9, 1, 2], k=3) == [3, 4, 9]

def test_lookup_prefers_the_longest_ngram():      # 1-gram [2] would give 7; 2-gram [1, 2] gives 3
    assert propose_ngram([1, 2, 3, 5, 2, 7, 1, 2], k=1) == [3]

def test_lookup_prefers_the_latest_match():
    assert propose_ngram([5, 1, 6, 5, 1, 8, 5, 1], k=1) == [8]

def test_lookup_short_near_the_end_and_empty_without_a_match():
    assert propose_ngram([1, 2, 1], k=5) == [2, 1]
    assert propose_ngram([1, 2, 3], k=5) == []



def test_verify_segment_matches_one_token_at_a_time(tiny_model):
    torch.manual_seed(0)
    ids = torch.randint(1000, (12,))
    cache, seg = fresh(tiny_model, 12)
    tiny_model(ids[:8], [Segment(seg.table, 0, 8)], cache)
    together = tiny_model(ids[8:], [Segment(seg.table, 8, 4)], cache)          # [4, V]
    one_by_one = torch.cat([tiny_model(ids[8 + j:9 + j], [Segment(seg.table, 8 + j, 1)], cache)
                            for j in range(4)])
    torch.testing.assert_close(together, one_by_one)



COPY_TEXT = ("The lighthouse keeper wrote the same three lines in his log every night: the wind, "
             "the sea, and the number of ships that passed. In forty years the third number was "
             "never zero, and he never missed a night.")
CHAT = [  # copy tasks: the answer repeats the prompt, so lookup drafts and gets accepted
    "Repeat the following text exactly, with nothing before or after it:\n\n" + COPY_TEXT,
    "Fix the spelling and repeat the rest unchanged:\n\n" + COPY_TEXT.replace("night", "nite"),
]
RAW = ["def fibonacci(n):\n    ", "List the planets of the solar system in order from the Sun:\n1."]


@pytest.mark.gpu
def test_prompt_lookup_matches_plain_greedy_on_qwen(models, tok):
    _, mine = models(torch.float32)     # fp32: bf16 rounding would flip near-ties between the two paths
    ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True)).input_ids for p in CHAT]
    ids += [tok(p).input_ids for p in RAW]

    def run(k):
        cfg = EngineConfig(max_num_seqs=4, max_num_batched_tokens=1024, max_model_len=512,
                           num_speculative_tokens=k)
        e = Engine(mine, cfg, eos_ids=[tok.eos_token_id])
        reqs = [Request(i, p, max_tokens=96) for i, p in enumerate(ids)]
        for r in reqs:
            e.add_request(r)
        e.run_until_idle()
        return [r.output_ids for r in reqs], e

    plain, _ = run(0)
    spec, e = run(4)
    for i, (p, s) in enumerate(zip(plain, spec)):
        j = next((j for j, (a, b) in enumerate(zip(p, s)) if a != b), min(len(p), len(s)))
        assert s == p, f"request {i} diverges at token {j}: plain {tok.decode(p[j:j + 5])!r}, " \
                       f"spec {tok.decode(s[j:j + 5])!r}"
    assert e.spec_accepted > 0
    print(f"\naccepted {e.spec_accepted}/{e.spec_drafted} = {e.spec_accepted / e.spec_drafted:.0%}")
    