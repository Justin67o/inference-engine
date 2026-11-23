import threading

import pytest
import torch

from engine.config import EngineConfig
from engine.engine import Engine
from engine.generate import generate
from engine.request import Request, State

VOCAB = 1000



def make_engine(model, **limits):
    limits = {"max_num_seqs": 4, "max_num_batched_tokens": 64, "max_model_len": 64} | limits
    return Engine(model, EngineConfig(**limits))


def prompts(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(VOCAB, (int(torch.randint(2, 20, (1,), generator=g)),), generator=g)
            for _ in range(n)]


def alone(model, prompt, max_tokens):              # reference: static-batch generate, one prompt
    return generate(model, [prompt], max_tokens)[0].tolist()


def test_matches_generate_when_all_arrive_together(tiny_model):
    engine, ps = make_engine(tiny_model), prompts(4)
    reqs = [Request(i, p.tolist(), max_tokens=10) for i, p in enumerate(ps)]
    for r in reqs:
        engine.add_request(r)
    engine.run_until_idle()
    for r, p in zip(reqs, ps):
        assert r.output_ids == alone(tiny_model, p, 10)


def test_matches_generate_when_requests_join_mid_run(tiny_model):
    engine, ps = make_engine(tiny_model), prompts(6, seed=1)
    reqs = [Request(i, p.tolist(), max_tokens=8 + i) for i, p in enumerate(ps)]
    for r in reqs[:2]:
        engine.add_request(r)
    for _ in range(3):                             # these two are decoding when the rest arrive
        engine.step()
    for r in reqs[2:]:
        engine.add_request(r)
    engine.run_until_idle()
    for r, p in zip(reqs, ps):
        assert r.output_ids == alone(tiny_model, p, r.max_tokens)


def test_more_requests_than_seqs_all_finish_and_return_their_blocks(tiny_model):
    engine = make_engine(tiny_model, max_num_seqs=2)
    reqs = [Request(i, p.tolist(), max_tokens=5) for i, p in enumerate(prompts(5, seed=2))]
    for r in reqs:
        engine.add_request(r)
    engine.run_until_idle()
    assert all(r.state is State.FINISHED and len(r.output_ids) == 5 for r in reqs)
    bm = engine.scheduler.blocks
    assert len(bm.free) == bm.num_blocks and not engine.scheduler.running


def test_on_token_streams_every_token_and_done_only_at_the_end(tiny_model):
    engine, seen = make_engine(tiny_model), []
    r = Request(0, prompts(1)[0].tolist(), max_tokens=6, on_token=lambda t, d: seen.append((t, d)))
    engine.add_request(r)
    engine.run_until_idle()
    assert [t for t, _ in seen] == r.output_ids
    assert [d for _, d in seen] == [False] * 5 + [True]


def test_same_seed_same_output_alone_or_in_a_batch(tiny_model):
    p = prompts(1, seed=3)[0].tolist()

    def run(extra):
        engine = make_engine(tiny_model)
        r = Request(0, p, max_tokens=10, temperature=1.0, seed=1234)
        engine.add_request(r)
        for i, q in enumerate(prompts(extra, seed=4)):
            engine.add_request(Request(i + 1, q.tolist(), max_tokens=10, temperature=1.0))
        engine.run_until_idle()
        return r.output_ids

    assert run(0) == run(3)


def test_add_request_rejects_a_request_that_does_not_fit(tiny_model):
    engine = make_engine(tiny_model)
    with pytest.raises(ValueError):
        engine.add_request(Request(0, [1] * 60, max_tokens=10))   # 70 > max_model_len 64
    assert engine.inbox.empty()


def test_run_forever_in_a_thread(tiny_model):
    engine, ps = make_engine(tiny_model), prompts(3, seed=5)
    finished = threading.Semaphore(0)
    reqs = [Request(i, p.tolist(), max_tokens=5, on_token=lambda t, d: d and finished.release())
            for i, p in enumerate(ps)]
    thread = threading.Thread(target=engine.run_forever, daemon=True)
    thread.start()
    for r in reqs:
        engine.add_request(r)                      # from this thread, while the engine runs
    for _ in reqs:
        assert finished.acquire(timeout=30)
    engine.stop()
    thread.join(timeout=10)
    assert not thread.is_alive()
    for r, p in zip(reqs, ps):
        assert r.output_ids == alone(tiny_model, p, 5)


class LetterTok:                    # fake tokenizer: token id -> one letter
    @staticmethod
    def decode(ids):
        return "".join(chr(ord("a") + i % 26) for i in ids)


def run_one(model, p, **kw):
    eos_ids, tok = kw.pop("eos_ids", ()), kw.pop("tokenizer", None)
    engine = Engine(model, EngineConfig(max_num_seqs=4, max_num_batched_tokens=64,
                                        max_model_len=64), eos_ids, tok)
    r = Request(0, p, **kw)
    engine.add_request(r)
    engine.run_until_idle()
    bm = engine.scheduler.blocks
    assert r.state is State.FINISHED and len(bm.free) == bm.num_blocks
    return r


def test_max_tokens_finishes_with_length(tiny_model):
    r = run_one(tiny_model, prompts(1)[0].tolist(), max_tokens=7)
    assert len(r.output_ids) == 7 and r.finish_reason == "length"


def test_eos_stops_early_and_ignore_eos_runs_on(tiny_model):
    p = prompts(1, seed=6)[0].tolist()
    full = alone(tiny_model, torch.tensor(p), 12)
    eos = full[4]
    cut = full[: full.index(eos) + 1]               # up to and including the first EOS

    r = run_one(tiny_model, p, max_tokens=12, eos_ids={eos})
    assert r.output_ids == cut and r.finish_reason == "stop"

    r = run_one(tiny_model, p, max_tokens=12, eos_ids={eos}, ignore_eos=True)
    assert r.output_ids == full and r.finish_reason == "length"


def test_stop_string_stops_at_first_match(tiny_model):
    p = prompts(1, seed=7)[0].tolist()
    full = alone(tiny_model, torch.tensor(p), 12)
    text = LetterTok.decode(full)
    stop = text[5:7]                                 # two letters the model really produces
    k = next(i for i in range(1, len(full) + 1) if stop in LetterTok.decode(full[:i]))

    r = run_one(tiny_model, p, max_tokens=12, stop=[stop], tokenizer=LetterTok)
    assert r.output_ids == full[:k] and r.finish_reason == "stop"


def test_stop_strings_without_tokenizer_are_rejected(tiny_model):
    engine = make_engine(tiny_model)
    with pytest.raises(ValueError):
        engine.add_request(Request(0, [1, 2], max_tokens=5, stop=["x"]))

def test_abort_frees_the_blocks_and_stops_tokens(tiny_model):
    engine, seen = make_engine(tiny_model), []
    p = prompts(1)[0].tolist()
    r = Request(0, p, max_tokens=20, on_token=lambda t, d: seen.append(t))
    engine.add_request(r)
    engine.step(); engine.step()                    # prefill + one decode
    engine.abort_request(r)
    engine.run_until_idle()
    assert len(seen) == 2 and r.finish_reason == "abort" and r.block_table == []
    bm = engine.scheduler.blocks
    assert len(bm.free) == bm.num_blocks and not engine.scheduler.running


def test_abort_while_waiting_never_runs(tiny_model):
    engine = make_engine(tiny_model, max_num_seqs=1)
    a, b = (Request(i, p.tolist(), max_tokens=5) for i, p in enumerate(prompts(2)))
    engine.add_request(a); engine.add_request(b)
    engine.step()                                   # a runs, b waits (only 1 seq)
    engine.abort_request(b)
    engine.run_until_idle()
    assert b.output_ids == [] and b.state is State.FINISHED
    assert len(a.output_ids) == 5                   # the other request is unaffected


def test_abort_after_finish_is_a_no_op(tiny_model):
    engine = make_engine(tiny_model)
    r = Request(0, prompts(1)[0].tolist(), max_tokens=3)
    engine.add_request(r)
    engine.run_until_idle()
    engine.abort_request(r)
    engine.step()
    assert r.finish_reason == "length"


def test_outputs_do_not_depend_on_arrival_order(tiny_model):
    ps = prompts(6, seed=8)

    def run(order):
        engine = make_engine(tiny_model, max_num_seqs=2)  # few slots: order decides who batches with whom
        reqs = {i: Request(i, ps[i].tolist(), max_tokens=6,
                           temperature=1.0 if i % 2 else 0.0, seed=100 + i) for i in order}
        for i in order:
            engine.add_request(reqs[i])
        engine.run_until_idle()
        return {i: r.output_ids for i, r in reqs.items()}

    first = run([0, 1, 2, 3, 4, 5])
    for order in ([5, 4, 3, 2, 1, 0], [3, 0, 5, 1, 4, 2]):
        assert run(order) == first


def assert_no_leaks(engine, reqs):
    s = engine.scheduler
    assert len(s.blocks.free) == s.blocks.num_blocks
    assert len(set(s.blocks.free)) == s.blocks.num_blocks       # no block freed twice
    assert not s.running and not s.waiting
    assert engine.inbox.empty() and engine.aborts.empty()
    assert all(r.state is State.FINISHED and r.block_table == [] for r in reqs)


def test_no_leaked_blocks_after_every_kind_of_finish(tiny_model):
    ps = prompts(6, seed=9)
    eos = alone(tiny_model, ps[1], 10)[3]
    engine = Engine(tiny_model, EngineConfig(max_num_seqs=2, max_num_batched_tokens=64,
                                             max_model_len=64), eos_ids={eos})
    reqs = [Request(i, p.tolist(), max_tokens=10) for i, p in enumerate(ps)]
    reqs[2].ignore_eos = True                       # guarantees one "length"
    for r in reqs:
        engine.add_request(r)
    with pytest.raises(ValueError):                 # rejected: never enters the engine
        engine.add_request(Request(99, [1] * 60, max_tokens=10))
    engine.step()                                   # reqs 0 and 1 run, the rest wait
    engine.abort_request(reqs[0])                   # abort while running
    engine.abort_request(reqs[5])                   # abort while waiting
    engine.run_until_idle()

    assert {"stop", "length", "abort"} <= {r.finish_reason for r in reqs}
    assert_no_leaks(engine, reqs)

    # The freed blocks are reused cleanly: a second wave still matches generate
    wave = [Request(i, p.tolist(), max_tokens=8, ignore_eos=True)
            for i, p in enumerate(prompts(4, seed=10))]
    for r in wave:
        engine.add_request(r)
    engine.run_until_idle()
    for r in wave:
        assert r.output_ids == alone(tiny_model, torch.tensor(r.prompt_ids), 8)
    assert_no_leaks(engine, wave)


def test_preemption_does_not_change_outputs(tiny_model):
    ps = prompts(4, seed=9)
    cfg = EngineConfig(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=64)
    engine = Engine(tiny_model, cfg, num_blocks=6)     # 4 seqs * up to 4 blocks: must preempt
    reqs = [Request(i, p.tolist(), max_tokens=30) for i, p in enumerate(ps)]
    for r in reqs:
        engine.add_request(r)
    engine.run_until_idle()
    for r, p in zip(reqs, ps):
        assert r.output_ids == alone(tiny_model, p, 30)
    assert_no_leaks(engine, reqs)