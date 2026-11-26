import pytest

from engine.config import EngineConfig


def test_from_env_overrides_only_what_is_set(monkeypatch):
    monkeypatch.setenv("ENGINE_MAX_NUM_BATCHED_TOKENS", "256")
    cfg = EngineConfig.from_env()
    assert cfg.max_num_batched_tokens == 256
    assert cfg.max_num_seqs == EngineConfig().max_num_seqs


def test_from_env_still_validates(monkeypatch):
    monkeypatch.setenv("ENGINE_MAX_NUM_BATCHED_TOKENS", "16")    # below max_num_seqs (32)
    with pytest.raises(ValueError):
        EngineConfig.from_env()


def test_from_env_reads_bools(monkeypatch):
    monkeypatch.setenv("ENGINE_CUDA_GRAPHS", "0")
    assert EngineConfig.from_env().cuda_graphs is False
    monkeypatch.setenv("ENGINE_CUDA_GRAPHS", "true")
    assert EngineConfig.from_env().cuda_graphs is True
    