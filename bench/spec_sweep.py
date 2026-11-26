"""Speculative decoding sweeps, engine called directly.
    python -m bench.spec_sweep --sweep k     --output results/m5/k_sweep.json
    python -m bench.spec_sweep --sweep batch --output results/m5/batch_sweep.json
"""
import argparse, gc, json, os, time
import torch
from transformers import AutoTokenizer, GenerationConfig
from bench.env import get_env
from engine.config import EngineConfig
from engine.engine import Engine
from engine.model.loader import load_model
from engine.request import Request

TARGET, DRAFT = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
NUM_BLOCKS = 1000           # 1.5B + 0.5B fit 1026 (test log, 2026-11-25)


def load_prompts(kind, tok, n):
    with open(f"bench/prompts/spec_{kind}.jsonl") as f:
        texts = [json.loads(l)["prompt"] for l in f]
    texts = [texts[i % len(texts)] for i in range(n)]
    return [tok(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                        add_generation_prompt=True)).input_ids for t in texts]


def run_one(model, draft, prompts, eos, k, batch, temperature, max_tokens):
    cfg = EngineConfig(max_num_seqs=batch, num_speculative_tokens=k)
    e = Engine(model, cfg, eos, num_blocks=NUM_BLOCKS, draft=draft)
    reqs = [Request(id=i, prompt_ids=p, max_tokens=max_tokens, temperature=temperature)
            for i, p in enumerate(prompts)]
    for r in reqs:
        e.add_request(r)
    steps = 0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    while e.step():
        steps += 1
    torch.cuda.synchronize(); secs = time.perf_counter() - t0
    del e; gc.collect(); torch.cuda.empty_cache()

    tokens = sum(len(r.output_ids) for r in reqs)
    s = {f: sum(getattr(r, f"spec_{f}") for r in reqs)
         for f in ("passes", "drafted", "accepted", "rejected")}
    out = {"k": k, "batch": batch, "temperature": temperature, "tokens": tokens,
           "seconds": secs, "tok_s": tokens / secs, "steps": steps, **s}
    if s["passes"]:
        a = s["accepted"] / max(1, s["accepted"] + s["rejected"])
        out["alpha"] = a
        out["tokens_per_pass"] = (s["accepted"] + s["passes"]) / s["passes"]
        out["predicted_tokens_per_pass"] = (1 - a ** (k + 1)) / (1 - a) if a < 1 else k + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["k", "batch", "cost"], required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TARGET)
    eos = GenerationConfig.from_pretrained(TARGET).eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    target, _ = load_model(TARGET)
    draft, _ = load_model(DRAFT)

    rows = []
    def run(label, model, d, kind, n, k, batch, temperature=0.0):
        row = run_one(model, d, load_prompts(kind, tok, n), eos, k, batch,
                      temperature, args.max_tokens)
        row.update(drafter=label, prompts=kind)
        rows.append(row)
        print({key: round(v, 3) if isinstance(v, float) else v for key, v in row.items()},
              flush=True)

    run("warmup", target, draft, "chat", 2, 4, 2)    # kernel loading, discarded below
    rows.clear()

    if args.sweep == "k":
        for kind in ("chat", "code", "summ"):
            run("none", target, None, kind, 8, 0, 1)
            for label, d in (("draft", draft), ("lookup", None)):
                for k in (2, 4, 6):
                    run(label, target, d, kind, 8, k, 1)
    elif args.sweep == "batch":
        for b in (1, 4, 16):
            run("none", target, None, "chat", 4 * b, 0, b)
            run("draft", target, draft, "chat", 4 * b, 4, b)
            run("draft", target, draft, "chat", 4 * b, 4, b, temperature=1.0)
    else:                                   # cost: seconds per step, draft vs target
        for b in (1, 4, 16):
            for label, m in (("target", target), ("draft_as_target", draft)):
                run(label, m, None, "chat", 4 * b, 0, b)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "rows": rows, "env": get_env()}, f, indent=2)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
