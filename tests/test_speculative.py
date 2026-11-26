import copy
import gc
from collections import Counter

import pytest
import torch
from scipy.stats import chi2_contingency

from engine.cache.kv_cache import Segment, fresh
from engine.config import EngineConfig
from engine.engine import Engine
from engine.model.loader import load_model
from engine.model.model import Model
from engine.request import Request
from engine.speculative import propose_ngram, verify


def logits(*rows):                      # logits whose softmax is exactly these probability rows
    return torch.tensor(rows).log()     # log(0) = -inf, so a zero probability stays exactly zero


def qwen_prompts(tok):
    ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True)).input_ids for p in CHAT]
    return ids + [tok(p).input_ids for p in RAW]


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


@pytest.mark.gpu
def test_sampled_lookup_keeps_the_target_distribution(models, tok):
    _, mine = models(torch.float32)
    # the prompt repeats itself, so lookup always has a draft after the first token
    prompt = tok("My favourite colours are red, blue and green. My favourite colours are").input_ids
    n = 10_000

    def second_tokens(k):
        torch.manual_seed(k)
        cfg = EngineConfig(max_num_seqs=256, max_num_batched_tokens=4096, max_model_len=64,
                           num_speculative_tokens=k)
        e = Engine(mine, cfg)                      # no eos_ids: every request runs to max_tokens
        reqs = [Request(i, prompt, max_tokens=6, temperature=1.0) for i in range(n)]
        for r in reqs:
            e.add_request(r)
        e.run_until_idle()
        return Counter(r.output_ids[1] for r in reqs), e

    plain, _ = second_tokens(0)
    spec, e = second_tokens(4)
    common = [t for t, c in plain.items() if c >= 20]
    table = [[c[t] for t in common] + [n - sum(c[t] for t in common)] for c in (plain, spec)]
    if table[0][-1] + table[1][-1] == 0:          # no rare tokens: drop the empty bucket
        table = [row[:-1] for row in table]
    pvalue = chi2_contingency(table).pvalue
    print(f"\n{len(table[0])} buckets, accepted {e.spec_accepted}/{e.spec_drafted}, p = {pvalue:.3f}")
    assert e.spec_drafted > e.spec_accepted > 0    # drafts made and some rejected: residual exercised
    assert pvalue > 0.01


def test_rollback_keeps_tables_exactly_the_cached_length(tiny_model):
    torch.manual_seed(0)
    cfg = EngineConfig(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=64,
                       num_speculative_tokens=4)
    e = Engine(tiny_model, cfg)
    # prompt ends with a repeat, so lookup always drafts; a random model rejects most of it
    reqs = [Request(i, torch.randint(1000, (5,)).tolist() * 3, max_tokens=30) for i in range(3)]
    for r in reqs:
        e.add_request(r)
    bm = e.scheduler.blocks
    while e.step():
        running = e.scheduler.running
        for r in running:
            assert len(r.block_table) == bm.blocks_for(r.length - 1)
        assert sum(len(r.block_table) for r in running) + len(bm.free) == bm.num_blocks
    assert e.spec_drafted > e.spec_accepted > -1    # rejections happened, so rollback had work


def test_draft_cache_shares_the_target_block_ids(tiny_model):
    cfg = EngineConfig(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=64)
    e = Engine(tiny_model, cfg, num_blocks=8, draft=tiny_model)
    assert e.draft_cache.num_blocks == e.cache.num_blocks == e.scheduler.blocks.num_blocks
    assert e.draft_cache.k is not e.cache.k                 # separate storage


@pytest.mark.gpu
def test_1_5b_target_loads_beside_0_5b_draft(models, tok):
    models.unload()
    cfg = EngineConfig(model="Qwen/Qwen2.5-1.5B-Instruct", draft_model="Qwen/Qwen2.5-0.5B-Instruct")
    e = Engine.from_config(cfg)
    for m in (e.model, e.draft):
        assert m.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr()  # tied, not copied
    assert e.cache.num_blocks < 4096                         # sized to the GPU, not the default
    r = Request(0, tok("The capital of France is").input_ids, max_tokens=8)
    e.add_request(r)
    e.run_until_idle()
    assert r.finish_reason in ("stop", "length") and r.output_ids
    del e
    gc.collect()
    torch.cuda.empty_cache()                                 # ~4.7 GB: free it for later GPU tests


def run_tiny(model, k, draft=None, reqs_ids=None):
    cfg = EngineConfig(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=64,
                       num_speculative_tokens=k)
    e = Engine(model, cfg, draft=draft)
    reqs = [Request(i, p, max_tokens=30) for i, p in enumerate(reqs_ids)]
    for r in reqs:
        e.add_request(r)
    e.run_until_idle()
    return [r.output_ids for r in reqs], e


def test_target_as_its_own_draft_accepts_everything(tiny_model):
    torch.manual_seed(0)
    ids = [torch.randint(1000, (n,)).tolist() for n in (5, 9, 17)]
    plain, _ = run_tiny(tiny_model, 0, reqs_ids=ids)
    spec, e = run_tiny(tiny_model, 4, draft=tiny_model, reqs_ids=ids)
    assert spec == plain
    assert e.spec_accepted == e.spec_drafted > 0     # any stale draft K/V would cause a rejection


def test_different_draft_still_matches_greedy(tiny_model):
    cfg = copy.deepcopy(tiny_model.cfg)
    torch.manual_seed(1)
    other = Model(cfg).eval()                         # different weights: most drafts are rejected
    torch.manual_seed(0)
    ids = [torch.randint(1000, (n,)).tolist() for n in (5, 9, 17)]
    plain, _ = run_tiny(tiny_model, 0, reqs_ids=ids)
    spec, e = run_tiny(tiny_model, 4, draft=other, reqs_ids=ids)
    assert spec == plain
    assert e.spec_drafted > e.spec_accepted           # truncation had work to do


def test_batch_with_diverging_acceptance_matches_plain(tiny_model):
    other = copy.deepcopy(tiny_model)             # close to the target: some drafts accepted, some not
    torch.manual_seed(1)
    with torch.no_grad():
        for w in other.parameters():
            w.add_(0.2 * w.std() * torch.randn_like(w))
    first = [torch.randint(1000, (n,)).tolist() for n in (5, 9, 17)]
    later = [torch.randint(1000, (n,)).tolist() for n in (3, 12)]

    def run(k):
        cfg = EngineConfig(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=64,
                           num_speculative_tokens=k)
        e = Engine(tiny_model, cfg, num_blocks=6, draft=other if k else None)
        reqs = [Request(i, p, max_tokens=m) for i, (p, m) in enumerate(zip(first, (30, 7, 20)))]
        reqs.append(Request(3, first[0], max_tokens=20, temperature=1.0, seed=5))  # never drafts
        for r in reqs:
            e.add_request(r)
        for _ in range(3):
            e.step()
        for i, p in enumerate(later, start=len(reqs)):
            reqs.append(Request(i, p, max_tokens=25))
            e.add_request(reqs[-1])
        e.run_until_idle()
        assert len(e.scheduler.blocks.free) == e.scheduler.blocks.num_blocks
        return [r.output_ids for r in reqs], e

    plain, _ = run(0)
    spec, e = run(4)
    assert spec == plain
    assert e.spec_drafted > e.spec_accepted > 0
    assert e.scheduler.preemptions > 0


def top2(model, ids):               # the target's two best next tokens and the logit gap between them
    cache, seg = fresh(model, len(ids))
    top = model(torch.tensor(ids, device="cuda"), [seg], cache)[-1].float().topk(2)
    return top.indices.tolist(), (top.values[0] - top.values[1]).item()


@pytest.mark.gpu
def test_0_5b_draft_matches_plain_greedy_on_1_5b(models, tok):
    models.unload()
    target, _ = load_model("Qwen/Qwen2.5-1.5B-Instruct")     # bf16: fp32 1.5B won't fit in 6 GB
    draft, _ = load_model("Qwen/Qwen2.5-0.5B-Instruct")
    ids = qwen_prompts(tok)

    def run(k):
        cfg = EngineConfig(max_num_seqs=4, max_num_batched_tokens=1024, max_model_len=512,
                           num_speculative_tokens=k)
        e = Engine(target, cfg, eos_ids=[tok.eos_token_id], draft=draft if k else None)
        reqs = [Request(i, p, max_tokens=96) for i, p in enumerate(ids)]
        for r in reqs:
            e.add_request(r)
        e.run_until_idle()
        return [r.output_ids for r in reqs], e

    plain, _ = run(0)
    spec, e = run(4)
    for i, (p, s) in enumerate(zip(plain, spec)):
        if s == p:
            continue
        j = next(j for j, (a, b) in enumerate(zip(p + [-1], s + [-1])) if a != b)
        best, gap = top2(target, ids[i] + p[:j])     # bf16 rounding may flip only a near-tie
        pj, sj = (p + [-1])[j], (s + [-1])[j]      # -1: that run had already stopped
        print(f"\nrequest {i} diverges at token {j}: plain {pj}, spec {sj}, top-2 {best}, gap {gap:.3f}")
        assert gap < 0.2 and sj in best, f"request {i} diverges at {j} with gap {gap:.3f}: a real bug"
    assert e.spec_accepted > 0
    print(f"\naccepted {e.spec_accepted}/{e.spec_drafted} = {e.spec_accepted / e.spec_drafted:.0%}")
