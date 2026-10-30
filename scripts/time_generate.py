"""Single-request tokens/s for the naive (no KV cache) generate.

    python -m scripts.time_generate"""

import argparse
import statistics
import time

import torch
from transformers import AutoTokenizer

from engine.generate import generate
from engine.model.loader import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Explain why the sky is blue.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    model, _ = load_model(args.model) # bf16 on the GPU
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids[0].cuda()

    generate(model, ids, 8) # warm-up: CUDA kernel loading and cuBLAS setup

    rates = []

    for _ in range(args.runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = generate(model, ids, args.max_new_tokens) # no stop tokens: always the full length
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rates.append(len(out) / elapsed)

    print(f"prompt {len(ids)} tokens, {args.max_new_tokens} new tokens, {args.runs} runs")
    print("tokens/s per run: " + ", ".join(f"{r:.1f}" for r in rates))
    print(f"median: {statistics.median(rates):.1f} tokens/s "
          f"({1000 / statistics.median(rates):.1f} ms/token)")
    print("output:", repr(tok.decode(out)[:200]))


if __name__ == "__main__":
    main()