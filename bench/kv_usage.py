import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BS = 16     # M4 block size


def live_spans(times, gap_factor):
    """(start, end) of each stretch where a request was running. A gap in its tokens longer than
    gap_factor x its median token spacing means it was preempted; it held no blocks until readmitted"""
    gaps = np.diff(times)
    if len(gaps) == 0:
        return [(times[0], times[-1])]
    cut = np.flatnonzero(gaps > gap_factor * np.median(gaps))
    starts = np.r_[times[0], times[cut + 1]]
    ends = np.r_[times[cut], times[-1]]
    return list(zip(starts, ends))


def timeline(run, prompt_tokens, gap_factor=6, dt=0.05):
    """Running sequences and KV slots over time, rebuilt from each request's token times.
    A request holds cache from its first token to its last (admission is one prefill step
    earlier, so this slightly undercounts). Gaps from preemption are not counted"""
    reqs = [r for r in run["results"] if r["ok"] and r["token_times"]]
    t0 = min(r["sent"] for r in run["results"])
    end = max(r["token_times"][-1] for r in reqs)
    t = np.arange(0, end - t0, dt)

    cfg = run["server"]["config"]
    paged = "num_blocks" in cfg                     # only M4 has a block pool
    if paged:
        capacity = cfg["num_blocks"] * BS
    else:
        capacity = cfg["max_num_seqs"] * cfg["max_model_len"]

    running = np.zeros_like(t)
    tokens = np.zeros_like(t)                       # slots holding a real token
    held = np.zeros_like(t)                         # slots reserved by running sequences
    gaps = 0                                        # preemptions seen as gaps, to check against the engine's count
    for r in reqs:
        times = np.array(r["token_times"]) - t0
        live = np.zeros(len(t), dtype=bool)
        spans = live_spans(times, gap_factor)
        gaps += len(spans) - 1
        for start, end in spans:
            live |= (t >= start) & (t <= end)
        n = prompt_tokens + np.searchsorted(times, t, side="right")    # prompt + tokens so far
        running += live
        tokens += live * n
        held += live * (-(-n // BS) * BS if paged else cfg["max_model_len"])
    return {"t": t, "running": running, "tokens": tokens, "held": held, "capacity": capacity, "gaps": gaps}


def summary(tl):
    busy = tl["running"] > 0
    return {
        "peak_running": int(tl["running"].max()),
        "mean_running": float(tl["running"][busy].mean()),
        "peak_pool_used": float(tl["held"].max() / tl["capacity"]),      # share of KV memory reserved
        "mean_pool_used": float(tl["held"][busy].mean() / tl["capacity"]),
        "useful": float(tl["tokens"][busy].sum() / tl["held"][busy].sum()),  # reserved slots holding tokens
        "gaps": tl["gaps"],
    }


def main():
    parser = argparse.ArgumentParser(description="Concurrency and KV memory use from results files.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--prompt-tokens", type=int, default=200, help="synthetic_200 prompts are 200 tokens")
    parser.add_argument("--plot", help="running sequences and KV memory over time, one line per file")
    args = parser.parse_args()

    print("| label | rate | peak running | mean running | peak KV reserved | mean KV reserved | useful | preemptions | gaps seen | file |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    tls = []
    for path in args.paths:
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        tl = timeline(run, args.prompt_tokens)
        s = summary(tl)
        pre = (run.get("engine_stats") or {}).get("preemptions", "n/a")
        print(f"| {run['config']['label']} | {run['config']['rate']:g} | {s['peak_running']} | "
              f"{s['mean_running']:.1f} | {s['peak_pool_used']:.0%} | {s['mean_pool_used']:.0%} | "
              f"{s['useful']:.0%} | {pre} | {s['gaps']} | {os.path.basename(path)} |")
        tls.append((f"{run['config']['label']} @ {run['config']['rate']:g}/s", tl))

    if args.plot:
        fig, (ax_run, ax_mem) = plt.subplots(1, 2, figsize=(11, 4))
        for label, tl in tls:
            (line,) = ax_run.plot(tl["t"], tl["running"], label=label)
            ax_mem.plot(tl["t"], tl["held"] / tl["capacity"] * 100, color=line.get_color())
            ax_mem.plot(tl["t"], tl["tokens"] / tl["capacity"] * 100, color=line.get_color(), linestyle="--")
        ax_run.set_ylabel("running sequences")
        ax_run.legend(fontsize=8, loc="upper right")         # colour = run
        ax_mem.legend([Line2D([], [], color="gray"), Line2D([], [], color="gray", linestyle="--")],
                      ["reserved", "holding tokens"], fontsize=8, loc="center right")   # style = what
        ax_mem.set_ylabel("% of KV pool")
        ax_mem.set_ylim(0, 105)
        for ax in (ax_run, ax_mem):
            ax.set_xlabel("time since first request (s)")
            ax.grid(alpha=0.3)
        fig.suptitle("Concurrency and KV memory, same KV budget")
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"\nsaved {args.plot}")


if __name__ == "__main__":
    main()