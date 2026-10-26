import time

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name, dtype=torch.bfloat16).cuda().eval()

messages = [{"role": "user", "content": "What is the capital of France?"}]
text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt").input_ids.cuda()   # [1, T]

N_STEPS = 500
N_RUNS = 5


def forward(x):
    return model(x, use_cache=False, logits_to_keep=1).logits   # [1, 1, vocab]


with torch.inference_mode():
    # untimed warm-up with the same call, over a spread of lengths we will hit
    for L in range(ids.shape[1], ids.shape[1] + N_STEPS + 1, 16):
        forward(torch.zeros(1, L, dtype=torch.long, device="cuda"))
    torch.cuda.synchronize()

    runs = []
    for r in range(N_RUNS):
        x = ids.clone()
        times = []
        for step in range(N_STEPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            next_id = forward(x)[:, -1, :].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
            x = torch.cat([x, next_id], dim=1)       # no EOS break: we want the full length range
        runs.append(times)

runs = np.array(runs)                                # [N_RUNS, N_STEPS]
seq_lens = np.arange(ids.shape[1], ids.shape[1] + N_STEPS)   # length fed in each step
med = np.median(runs, axis=0)
lo, hi = runs.min(axis=0), runs.max(axis=0)
np.save("results/step_time_runs.npy", runs)

plt.figure(figsize=(7, 4))
plt.fill_between(seq_lens, lo, hi, alpha=0.25, label="min-max over runs")
plt.plot(seq_lens, med, linewidth=1.2, label=f"median of {N_RUNS} runs")
plt.xlabel("Sequence length (tokens fed to model)")
plt.ylabel("Time per step (ms)")
plt.title(f"{name}: no KV cache, full re-forward each step")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/step_time_vs_seqlen.png", dpi=150)
print("saved results/step_time_vs_seqlen.png")
