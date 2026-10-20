import torch


def test_cuda_available():
    assert torch.cuda.is_available()


def test_bf16_matmul():
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    assert (x @ x).shape == (64, 64)
