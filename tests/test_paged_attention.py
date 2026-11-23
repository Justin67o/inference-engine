import pytest
import torch

from engine.attention.paged import prefill_attention
from engine.attention.reference import attend


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 7, 128])
def test_prefill_matches_attend(n):
    torch.manual_seed(0)
    q = torch.randn(n, 14, 64, device="cuda")
    k = torch.randn(n, 2, 64, device="cuda")
    v = torch.randn(n, 2, 64, device="cuda")
    ref = attend(q, k, v, torch.arange(n, device="cuda"))
    torch.testing.assert_close(prefill_attention(q, k, v), ref, atol=1e-4, rtol=1e-4)