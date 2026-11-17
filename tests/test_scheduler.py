import pytest

from engine.cache.kv_cache import CachePool
from engine.config import EngineConfig
from engine.request import Request, State
from engine.scheduler import Scheduler


def make(max_num_seqs=4, max_num_batched_tokens=16, max_model_len=64, pool_size=None):
    cfg = EngineConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                       max_model_len=max_model_len)
    return Scheduler(cfg, CachePool(object, pool_size or max_num_seqs))


def req(i, prompt_len, max_tokens=8):
    return Request(id=i, prompt_ids=[1] * prompt_len, max_tokens=max_tokens)


def ids(segs):
    return [r.id for r, _, _ in segs]


def test_admits_in_arrival_order_with_whole_prompt_prefill():
    s = make()
    for i in range(3):
        s.add(req(i, prompt_len=i + 2))
    segs = s.schedule()
    assert [(r.id, start, n) for r, start, n in segs] == [(0, 0, 2), (1, 0, 3), (2, 0, 4)]
    assert all(r.state is State.RUNNING and r.cache is not None for r, _, _ in segs)
    assert not s.waiting


def test_running_sequences_decode_one_token_at_the_right_position():
    s = make()
    s.add(req(0, prompt_len=5))
    (r, _, _), = s.schedule()
    r.output_ids.append(7)                        # what the engine does after the prefill
    assert s.schedule() == [(r, 5, 1)]            # the new token goes in at position 5
    r.output_ids.append(8)
    assert s.schedule() == [(r, 6, 1)]


def test_decodes_come_before_new_prefills():
    s = make()
    s.add(req(0, prompt_len=3))
    s.schedule()[0][0].output_ids.append(7)
    s.add(req(1, prompt_len=4))
    assert [(r.id, start, n) for r, start, n in s.schedule()] == [(0, 3, 1), (1, 0, 4)]


def test_max_num_seqs_caps_running():
    s = make(max_num_seqs=2)
    for i in range(3):
        s.add(req(i, prompt_len=1))
    assert ids(s.schedule()) == [0, 1]
    assert [r.id for r in s.waiting] == [2]


def test_token_budget_holds_back_a_prompt_that_does_not_fit():
    s = make(max_num_batched_tokens=10)
    s.add(req(0, prompt_len=6))
    s.add(req(1, prompt_len=6))                   # 6 + 6 > 10
    assert ids(s.schedule()) == [0]
    s.running[0].output_ids.append(7)
    assert ids(s.schedule()) == [0, 1]            # 1 decode + 6 prefill = 7 <= 10


def test_decodes_use_up_budget():
    s = make(max_num_seqs=4, max_num_batched_tokens=6)
    for i in range(3):
        s.add(req(i, prompt_len=1))
    for r, _, _ in s.schedule():
        r.output_ids.append(7)
    s.add(req(3, prompt_len=4))                   # 3 decodes + 4 = 7 > 6
    assert ids(s.schedule()) == [0, 1, 2]


def test_first_come_first_served_no_skipping_ahead():
    s = make(max_num_batched_tokens=10)
    s.add(req(0, prompt_len=8))
    s.add(req(1, prompt_len=8))                   # doesn't fit this step
    s.add(req(2, prompt_len=1))                   # would fit, but must not jump the queue
    assert ids(s.schedule()) == [0]
    assert [r.id for r in s.waiting] == [1, 2]


def test_empty_pool_blocks_admission():
    s = make(max_num_seqs=4, pool_size=1)
    s.add(req(0, prompt_len=1))
    s.add(req(1, prompt_len=1))
    assert ids(s.schedule()) == [0]


def test_finish_frees_the_cache_for_the_next_request():
    s = make(max_num_seqs=4, pool_size=1)
    s.add(req(0, prompt_len=1))
    s.add(req(1, prompt_len=1))
    (r0, _, _), = s.schedule()
    cache = r0.cache
    s.finish(r0)
    assert r0.state is State.FINISHED and r0.cache is None and not s.running
    (r1, _, _), = s.schedule()
    assert r1.id == 1 and r1.cache is cache
    s.finish(r1)
    assert s.pool.num_free == s.pool.size


def test_nothing_to_do_returns_empty():
    assert make().schedule() == []


@pytest.mark.parametrize("prompt_len, max_tokens", [(17, 1), (60, 8)])
def test_add_rejects_requests_that_can_never_run(prompt_len, max_tokens):
    s = make(max_num_batched_tokens=16, max_model_len=64)
    with pytest.raises(ValueError):
        s.add(req(0, prompt_len, max_tokens))
