import torch

print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"free {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print("matmul ok", tuple(y.shape))
