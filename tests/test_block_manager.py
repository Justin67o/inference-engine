from types import SimpleNamespace

import pytest

from engine.cache.block_manager import bytes_per_block, num_blocks_for

QWEN_05B = SimpleNamespace(num_hidden_layers=24, num_key_value_heads=2)


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