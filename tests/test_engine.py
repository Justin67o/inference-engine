import copy
import threading

import pytest
import torch
from transformers import AutoConfig

from engine.config import EngineConfig
from engine.engine import Engine
from engine.generate import generate
from engine.model.model import Model
from engine.request import Request, State

VOCAB = 1000


@pytest.fixture(scope="module")
def tiny_model():
    # Real Qwen shapes per layer, but 2 layers and a small vocab so it runs on the CPU
    cfg = copy.deepcopy(AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"))
    cfg.num_hidden_layers, cfg.vocab_size = 2, VOCAB
    torch.manual_seed(0)
    return Model(cfg).eval()


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


def test_more_requests_than_seqs_all_finish_and_return_their_caches(tiny_model):
    engine = make_engine(tiny_model, max_num_seqs=2)
    reqs = [Request(i, p.tolist(), max_tokens=5) for i, p in enumerate(prompts(5, seed=2))]
    for r in reqs:
        engine.add_request(r)
    engine.run_until_idle()
    assert all(r.state is State.FINISHED and len(r.output_ids) == 5 for r in reqs)
    pool = engine.scheduler.pool
    assert pool.num_free == pool.size and not engine.scheduler.running


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
