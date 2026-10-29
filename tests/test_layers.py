import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from engine.model.layers import RMSNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_matches_hf(dtype):
    torch.manual_seed(0)
    dim, eps = 896, 1e-6
    ref = Qwen2RMSNorm(dim, eps)
    mine = RMSNorm(dim, eps)

    # Random weight: with the default all-ones weight a missing multiply would go unnoticed
    weight = torch.randn(dim)
    with torch.no_grad():
        ref.weight.copy_(weight)
        mine.weight.copy_(weight)
    ref, mine = ref.to(dtype), mine.to(dtype)

    x = torch.randn(20, dim, dtype=dtype)
    expected, actual = ref(x), mine(x)

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), (actual - expected).abs().max().item()
