import torch


def gpu_time_ms(fn, iters=20):
    for _ in range(3):  # warm up
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

# Memory bandwidth: copying reads N bytes and writes N bytes
src = torch.empty(2**30, dtype=torch.uint8, device="cuda")  # 1 GiB
dst = torch.empty_like(src)
ms = gpu_time_ms(lambda: dst.copy_(src))
print(f"bandwidth ≈ {2 * src.numel() / (ms / 1e3) / 1e9:.0f} GB/s")

# Compute: an n×n matmul does 2·n³ floating-point operations
n = 4096
a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
ms = gpu_time_ms(lambda: a @ a)
print(f"compute ≈ {2 * n**3 / (ms / 1e3) / 1e12:.1f} TFLOP/s")