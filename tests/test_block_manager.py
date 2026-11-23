from types import SimpleNamespace

import pytest

from hypothesis import given, settings, strategies as st
from engine.cache.block_manager import BlockManager, bytes_per_block, num_blocks_for, slot_ids

QWEN_05B = SimpleNamespace(num_hidden_layers=24, num_key_value_heads=2)


def seq(n_tokens):
    """Stand-in for a Request: just a token count and a block table"""
    return SimpleNamespace(n=n_tokens, block_table=[])


def test_bytes_per_block_qwen():
    # 2 (K, V) * 24 layers * 2 kv heads * 64 dims * 2 bytes (bf16) * 16 tokens
    assert bytes_per_block(QWEN_05B, head_dim=64, dtype_bytes=2) == 196_608


def test_num_blocks_subtracts_activations_and_headroom():
    assert num_blocks_for(free_bytes=1000, activation_bytes=100, block_bytes=10) == 80  # 900 - 100


def test_num_blocks_rounds_down():
    assert num_blocks_for(free_bytes=1000, activation_bytes=100, block_bytes=30) == 26  # 800 / 30


def test_num_blocks_raises_when_nothing_fits():
    with pytest.raises(ValueError):
        num_blocks_for(free_bytes=1000, activation_bytes=900, block_bytes=10)


# --- plain unit tests: pin down the basic arithmetic ---

def test_blocks_for_is_ceiling_division():
    bm = BlockManager(num_blocks=8)
    assert [bm.blocks_for(n) for n in (0, 1, 16, 17, 32, 33)] == [0, 1, 1, 2, 2, 3]


def test_ensure_grows_only_when_crossing_a_block_boundary():
    bm = BlockManager(num_blocks=8)
    s = seq(10)
    assert bm.ensure(s, 10) and len(s.block_table) == 1
    assert bm.ensure(s, 16) and len(s.block_table) == 1   # still fits
    assert bm.ensure(s, 17) and len(s.block_table) == 2   # crossed into block 2


def test_ensure_fails_without_partial_allocation():
    bm = BlockManager(num_blocks=2)
    s = seq(40)                                            # needs 3 blocks
    assert not bm.ensure(s, 40)
    assert s.block_table == [] and len(bm.free) == 2


def test_release_returns_every_block():
    bm = BlockManager(num_blocks=4)
    s = seq(50)
    bm.ensure(s, 50)
    bm.release(s)
    assert s.block_table == [] and sorted(bm.free) == [0, 1, 2, 3]


def test_slot_ids_crosses_block_boundary():
    # tokens 14..18 of a sequence owning blocks [7, 3]: last 2 slots of block 7, first 3 of block 3
    assert slot_ids([7, 3], start=14, n=5) == [126, 127, 48, 49, 50]


# --- property tests ---

ops = st.lists(
    st.tuples(st.sampled_from(["admit", "grow", "finish", "preempt"]),
              st.integers(0, 1000),        # which live sequence (taken mod len(live))
              st.integers(1, 200)),        # tokens to admit or add
    max_size=300)


def check_invariants(bm, live):
    owned = [b for s in live for b in s.block_table]
    assert len(owned) == len(set(owned))                        # no block in two tables
    assert len(owned) + len(bm.free) == bm.num_blocks           # nothing leaked or duplicated
    assert set(owned) | set(bm.free) == set(range(bm.num_blocks))
    for s in live:
        assert len(s.block_table) == bm.blocks_for(s.n)         # tables are exactly the right size


@settings(max_examples=200)
@given(ops)
def test_allocator_invariants(op_list):
    bm = BlockManager(num_blocks=64)
    live, waiting = [], []
    for op, i, n in op_list:
        if op == "admit":
            s = waiting.pop(0) if waiting else seq(n)           # preempted sequences come back first
            if bm.ensure(s, s.n):
                live.append(s)
            else:
                assert s.block_table == []
                waiting.insert(0, s)
        elif live:
            k = i % len(live)
            s = live[k]
            if op == "grow":
                before, free_before = list(s.block_table), len(bm.free)
                need = bm.blocks_for(s.n + n) - len(before)
                ok = bm.ensure(s, s.n + n)
                assert ok == (need <= free_before)              # fails exactly when blocks run out
                if ok:
                    s.n += n
                    assert s.block_table[:len(before)] == before  # existing blocks never move
                else:
                    assert s.block_table == before and len(bm.free) == free_before
            elif op == "finish":
                bm.release(live.pop(k))
            elif op == "preempt":
                bm.release(s)
                waiting.insert(0, live.pop(k))                  # recompute later, front of the queue
        check_invariants(bm, live)


@given(st.data())
def test_slot_ids_land_in_owned_blocks(data):
    bs = data.draw(st.sampled_from([8, 16, 32]))
    table = data.draw(st.lists(st.integers(0, 255), min_size=1, max_size=8, unique=True))
    start = data.draw(st.integers(0, len(table) * bs - 1))
    n = data.draw(st.integers(1, len(table) * bs - start))
    slots = slot_ids(table, start, n, bs)
    assert len(slots) == n == len(set(slots))                   # one distinct slot per token
    for p, s in zip(range(start, start + n), slots):
        assert s // bs == table[p // bs]                        # right physical block
        assert s % bs == p % bs                                 # right offset inside it