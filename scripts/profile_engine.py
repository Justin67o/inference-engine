"""Profile engine.step() at steady-state decode (every request past prefill, batch stays full).
    python -m scripts.profile_engine --batch 32
    python -m scripts.profile_engine --batch 192
    python -m scripts.profile_engine --batch 32 --k 4      # prompt lookup
Prints top ops by GPU time, CPU time per phase, kernels per step and GPU-busy share,
saves a summary to results/m6/ and a trace to trace_b{B}_k{k}.json (open at ui.perfetto.dev).
"""
import argparse
import json
import os
import time

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function

import engine.engine as engine_module
from bench.env import get_env
from engine.config import EngineConfig
from engine.engine import Engine
from engine.request import Request

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PHASES = ("step", "schedule", "draft", "forward", "sample")


def labelled(name, fn):
    def wrapper(*args, **kwargs):
        with record_function(name):
            return fn(*args, **kwargs)
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k", type=int, default=0, help="prompt-lookup drafts per decode")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    e = Engine.from_config(EngineConfig(model=MODEL, max_num_seqs=args.batch,
                                        num_speculative_tokens=args.k))
    with open("bench/prompts/synthetic_200.jsonl") as f:
        texts = [json.loads(line)["prompt"] for line in f]
    reqs = [Request(id=i, prompt_ids=e.tokenizer(texts[i % len(texts)]).input_ids,
                    max_tokens=512, ignore_eos=True) for i in range(args.batch)]
    for r in reqs:
        e.add_request(r)

    for _ in range(300):                    # prefill everyone
        e.step()
        if all(r.output_ids for r in reqs):
            break
    else:
        raise SystemExit("not every request was admitted; lower --batch")
    for _ in range(10):                     # decode warm-up (Triton compile for this shape)
        e.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        e.step()
    torch.cuda.synchronize()
    ms_step = (time.perf_counter() - t0) * 1e3 / args.steps

    e.scheduler.schedule = labelled("schedule", e.scheduler.schedule)
    e._draft = labelled("draft", e._draft)
    e.model.forward = labelled("forward", e.model.forward)
    engine_module.sample = labelled("sample", engine_module.sample)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            with record_function("step"):
                e.step()
        torch.cuda.synchronize()
        prof_ms_step = (time.perf_counter() - t0) * 1e3 / args.steps

    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=15))

    # record_function labels also show up as GPU-side annotations: skip them in both
    kernels = [ev for ev in prof.events()
               if ev.device_type == DeviceType.CUDA and ev.name not in PHASES]
    gpu_ms_step = sum(ev.time_range.elapsed_us() for ev in kernels) / 1e3 / args.steps
    avg = {ev.key: ev for ev in prof.key_averages() if ev.device_type == DeviceType.CPU}
    summary = {
        "batch": args.batch, "k": args.k, "steps": args.steps,
        "ms_per_step": ms_step,                     # unprofiled: the real number
        "ms_per_step_profiled": prof_ms_step,
        "gpu_kernel_ms_per_step": gpu_ms_step,
        "gpu_busy": gpu_ms_step / prof_ms_step,
        "kernels_per_step": len(kernels) / args.steps,
        "cpu_ms_per_step": {p: avg[p].cpu_time_total / 1e3 / args.steps
                            for p in PHASES if p in avg},
        "env": get_env(),
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "env"}, indent=2))

    os.makedirs("results/m6", exist_ok=True)
    with open(f"results/m6/profile_b{args.batch}_k{args.k}.json", "w") as f:
        json.dump(summary, f, indent=2)
    prof.export_chrome_trace(f"trace_b{args.batch}_k{args.k}.json")


if __name__ == "__main__":
    main()