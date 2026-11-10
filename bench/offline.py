"""Offline throughput: run a fixed prompt set through generate() in static batches,
calling the engine directly (no server), and report output tokens/s.

    python -m bench.offline --engine mine --batch-size 8 --output results/offline/mine_b8.json
    python -m bench.offline -- engine hf --batch-size 8 --output results/offline/hf_b8.json
    """


import argparse
import json
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.env import get_env
from engine.generate import generate
from engine.model.loader import load_model

def make_runner(engine, name, tok):
    """Returns run(texts, n) -> number of new tokens generated. Every sequence makes exactly n."""
    if engine == "mine":
        model, _ = load_model(name)     # bf16 on the GPU

        def run(texts, n):
            ids = [tok(t, return_tensors="pt").input_ids[0].cuda() for t in texts]
            out = generate(model, ids, n)       # no stop ids, so never stops early
            return sum(len(o) for o in out)
        
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()

        def run(texts, n):
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                # Plain greedy, the same work as mine: Qwen's config adds repetition_penalty=1.1
                out = model.generate(
                    **enc, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                    repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                    pad_token_id=tok.pad_token_id)
            return (out.shape[1] - enc.input_ids.shape[1]) * len(texts)
    return run


def time_all(run, prompts, batch_size, n):
    """One pass over every prompt in static batches. Returns (new tokens, seconds)."""

    tokens = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    for k in range(0, len(prompts), batch_size):
        tokens += run(prompts[k:k + batch_size], n)
    torch.cuda.synchronize()
    return tokens, time.perf_counter() - start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["mine", "hf"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompts", default="bench/prompts/synthetic_200.jsonl")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128, help="new tokens per prompt, exactly")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default=None, help="defaults to the engine name")
    args = parser.parse_args()
    args.label = args.label or args.engine

    with open(args.prompts, encoding="utf-8") as f:
        prompts = [json.loads(line)["prompt"] for line in f][:args.num_prompts]


    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"       # HF batches: pad on the left, as in hf_server.py
    run = make_runner(args.engine, args.model, tok)

    run(prompts[:args.batch_size], 8) # warm-up: CUDA kernel loading and cuBLAS setup
    runs = []

    for i in range(args.runs):
        tokens, seconds = time_all(run, prompts, args.batch_size, args.max_tokens)
        runs.append({"tokens": tokens, "seconds": seconds, "tok_s": tokens / seconds})
        print(f"run {i + 1}: {tokens} tokens in {seconds:.1f} s = {tokens / seconds:.1f} tok/s")
    env = get_env()     # right after the runs, GPU still under load

    metrics = {"throughput_tok_s": statistics.median(r["tok_s"] for r in runs), "runs": runs}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "metrics": metrics, "env": env}, f, indent=2)

    print(f"median: {metrics['throughput_tok_s']:.1f} tok/s -> {args.output}")


if __name__ == "__main__":
    main()