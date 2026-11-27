import argparse
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

# (key in the row dict, title shown in the table header), in display order
COLUMNS = [
    ("label", "label"),
    ("rate", "rate"),
    ("ok", "ok"),
    ("errors", "err"),
    ("budget", "budget"),
    ("ttft_p50", "TTFT p50"),
    ("ttft_p95", "TTFT p95"),
    ("tpot_p50", "TPOT p50"),
    ("tpot_p95", "TPOT p95"),
    ("e2e_p50", "E2E p50"),
    ("e2e_p95", "E2E p95"),
    ("e2e_p99", "E2E p99"),
    ("throughput", "tok/s"),
    ("commit", "commit"),
]


def budget(run):
    """max_num_batched_tokens from the server's /health, or None (e.g. the HF baseline)."""
    server = run.get("server") or {}
    return (server.get("config") or {}).get("max_num_batched_tokens")


def load_runs(paths):
    """Load results files. Skips files without a metrics section."""
    runs = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            run = json.load(f)

        if "metrics" not in run:
            print(f"skipping {path}: no metrics")
            continue

        run["path"] = path
        runs.append(run)

    return runs


def pct(section, key):
    """Returns section[key], or None if the whole section is None"""
    return None if section is None else section[key]

def to_row(run):
    """Flatten one run into the columns used by the table and plot."""
    config = run["config"]
    metrics = run["metrics"]
    env = run.get("env") or {}
    git = env.get("git") or {}
    commit = git.get("commit")
    return {
        "label": config.get("label", ""),
        "rate": config["rate"],
        "budget": budget(run),
        "ok": metrics["num_ok"],
        "errors": metrics["num_errors"],
        "ttft_p50": pct(metrics["ttft"], "p50"),
        "ttft_p95": pct(metrics["ttft"], "p95"),
        "tpot_p50": pct(metrics["tpot"], "p50"),
        "tpot_p95": pct(metrics["tpot"], "p95"),
        "e2e_p50": pct(metrics["e2e"], "p50"),
        "e2e_p95": pct(metrics["e2e"], "p95"),
        "e2e_p99": pct(metrics["e2e"], "p99"),
        "throughput": metrics["throughput_tok_s"],
        "tokens_per_req": metrics.get("output_tokens", 0) / metrics["num_ok"] if metrics["num_ok"] else None,
        "commit": commit[:7] if commit else None,    
        "dirty": git.get("dirty"),
        "path": run["path"],
    }


def plot_budget_sweep(rows, out_path):
    """p95 TTFT and p95 TPOT against max_num_batched_tokens, one line per rate.
    Repeated runs at the same budget and rate are averaged."""
    rows = [r for r in rows if r["budget"] is not None and r["tpot_p95"] is not None]

    by_rate = defaultdict(list)
    for r in rows:
        by_rate[r["rate"]].append(r)

    ticks = sorted({r["budget"] for r in rows})
    fig, (ax_ttft, ax_tpot) = plt.subplots(1, 2, figsize=(10, 4))
    for rate, runs in sorted(by_rate.items()):
        budgets = sorted({r["budget"] for r in runs})
        ttft = [mean(r["ttft_p95"] for r in runs if r["budget"] == b) for b in budgets]
        tpot = [mean(r["tpot_p95"] for r in runs if r["budget"] == b) * 1000 for b in budgets]
        ax_ttft.plot(budgets, ttft, marker="o", label=f"rate {rate:g}")
        ax_tpot.plot(budgets, tpot, marker="o", label=f"rate {rate:g}")

    for ax, ylabel in ((ax_ttft, "p95 TTFT (s)"), (ax_tpot, "p95 TPOT (ms)")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(b) for b in ticks])
        ax.set_xlabel("max_num_batched_tokens")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Token budget tradeoff")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fmt(x):
    """One table cell: n/a for None, 2 decimals for floats, everything else as-is."""
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)

def format_table(rows):
    """Markdown table of runs, sorted by label then rate."""

    rows = sorted(rows, key=lambda r: (r["label"], r["rate"]))

    header = "| " + " | ".join(title for _, title in COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [header, separator]

    for row in rows:
        if row["dirty"]:
            row = {**row, "commit": f"{row['commit']}*"}
        cells = []
        for key, _ in COLUMNS:
            cells.append(fmt(row[key]))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)

def plot_latency_vs_rate(rows, out_path):
    """E2E latency (p50 dashed, p95 solid) vs request rate, one colour per label.
    Lines go through the mean of repeated runs; dots show each run's p95."""
    rows = [r for r in rows if r["e2e_p95"] is not None]   # skip all-failed runs

    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, runs in by_label.items():
        rates = sorted({r["rate"] for r in runs})
        p50 = [mean(r["e2e_p50"] for r in runs if r["rate"] == x) for x in rates]
        p95 = [mean(r["e2e_p95"] for r in runs if r["rate"] == x) for x in rates]

        (line,) = ax.plot(rates, p95, marker="o", label=f"{label} p95")
        color = line.get_color()
        ax.plot(rates, p50, marker="o", linestyle="--", color=color, label=f"{label} p50")
        ax.scatter([r["rate"] for r in runs], [r["e2e_p95"] for r in runs], color=color, alpha=0.3)

    ax.set_xlabel("request rate (req/s)")
    ax.set_yscale("log")          # M3 is 10x slower; linear would flatten everything else
    ax.set_ylabel("end-to-end latency (s, log scale)")
    ax.set_title("Latency vs request rate")
    ax.grid(alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Part 9 final figures -------------------------------------------------------------

# Fixed order and colours (categorical slots 1-3), so each system keeps its colour in every figure
SYSTEMS = [("engine", "This engine", "#257de0"),
           ("hf", "Hugging Face (tuned)", "#f47230"),
           ("vllm", "vLLM", "#00b894")]
COLOR = {key: color for key, _, color in SYSTEMS}
TEXT, MUTED, GRID = "#191919", "#666666", "#cccccc"
WORKLOADS = {"fixed": "Fixed lengths (128 tokens each)", "mixed": "Mixed lengths (16-505 tokens, mean 158)"}
FINAL_NAME = re.compile(r"^(engine|hf|vllm)_(fixed|mixed)_(?:rate([\d.]+)|burst)_run\d+\.json$")

# 300-request burst, fixed lengths, 1402 MHz locked. M5 (speculative decoding) is off in a
# burst of independent requests, so it has no bar; acceptance_vs_k.png covers it.
ABLATION = [
    ("M0\nHF, batch 16", "hf", ["results/hf_tune/hf_b16_w500_burst_fixed.json"]),
    ("M3\ncontinuous\nbatching", "engine", ["results/m4_kv/m3_burst.json"]),
    ("M4\npaged KV,\n192 seqs", "engine", ["results/m4_kv/m4_192_burst.json"]),
    ("M4\n+ Triton\ndecode kernel", "engine", ["results/m4_kv/m4_triton_192_burst_locked2.json"]),
]


def load_final(final_dir):
    """Rows from bench/run_all.sh results. The file name gives system, workload and rate (None = burst)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(final_dir, "*.json"))):
        m = FINAL_NAME.match(os.path.basename(path))
        if not m:
            continue
        row = to_row(load_runs([path])[0])
        row.update(system=m[1], workload=m[2], rate=float(m[3]) if m[3] else None)
        rows.append(row)
    return rows


def style(ax):
    """Recessive axes: light grid, no top or right spine, muted ticks."""
    ax.grid(alpha=0.6, linewidth=0.6, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)


def plain_log(ax):
    """Log y axis labelled 0.05, 1, 20 rather than 5x10^-2."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def avg_by_rate(runs, key):
    rates = sorted({r["rate"] for r in runs})
    return rates, [mean(r[key] for r in runs if r["rate"] == x) for x in rates]


def plot_final_latency(rows, out_path):
    """Throughput, p95 TTFT and E2E (p50 solid, p95 dashed) against rate, one row per workload.
    Lines go through the mean of the repeated runs. The grey dotted line is the offered load."""
    rows = [r for r in rows if r["rate"] is not None and r["e2e_p50"] is not None]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for i, wl in enumerate(WORKLOADS):
        wl_rows = [r for r in rows if r["workload"] == wl]
        if not wl_rows:
            continue
        rates = sorted({r["rate"] for r in wl_rows})
        per_req = mean(r["tokens_per_req"] for r in wl_rows)
        axes[i, 0].plot(rates, [x * per_req for x in rates], ls=":", color=MUTED, lw=1.5, label="offered load")
        for system, name, color in SYSTEMS:
            runs = [r for r in wl_rows if r["system"] == system]
            if not runs:
                continue
            axes[i, 0].plot(*avg_by_rate(runs, "throughput"), marker="o", ms=6, lw=2, color=color, label=name)
            axes[i, 1].plot(*avg_by_rate(runs, "ttft_p95"), marker="o", ms=6, lw=2, color=color, label=name)
            axes[i, 2].plot(*avg_by_rate(runs, "e2e_p50"), marker="o", ms=6, lw=2, color=color)
            axes[i, 2].plot(*avg_by_rate(runs, "e2e_p95"), marker="o", ms=5, lw=1.5, ls="--", color=color, alpha=0.75)
        axes[i, 0].set_ylabel(f"{WORKLOADS[wl]}\n\noutput tokens/s", color=TEXT)
        axes[i, 1].set_ylabel("p95 time to first token (s, log)", color=TEXT)
        axes[i, 2].set_ylabel("end-to-end latency (s, log)", color=TEXT)
        for ax in axes[i, 1:]:
            plain_log(ax)
        for ax in axes[i]:
            style(ax)
    for ax in axes[1]:
        ax.set_xlabel("request rate (req/s)", color=TEXT)
    for ax, title in zip(axes[0], ("Throughput", "Time to first token", "End-to-end latency")):
        ax.set_title(title, color=TEXT, loc="left", fontweight="bold")
    handles = [Line2D([], [], color=c, lw=2, marker="o", label=n) for _, n, c in SYSTEMS] + [
        Line2D([], [], color=MUTED, lw=2, label="E2E p50"),
        Line2D([], [], color=MUTED, lw=1.5, ls="--", label="E2E p95"),
        Line2D([], [], color=MUTED, lw=1.5, ls=":", label="offered load")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Latency vs request rate: RTX 3050 6 GB Laptop, Qwen2.5-0.5B, 1402 MHz locked. "
                 "HF doesn't stream, so its TTFT equals its E2E", color=MUTED, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ablation(final_rows, out_path):
    """Burst throughput at each milestone, then the tuned HF and vLLM references from the final run.
    Bars are the mean of the runs; dots are single runs."""
    def bursts(system):
        return [r["throughput"] for r in final_rows
                if r["system"] == system and r["workload"] == "fixed" and r["rate"] is None]

    bars = [(label, system, [to_row(r)["throughput"] for r in load_runs(paths)])
            for label, system, paths in ABLATION]
    bars.append(("M6\n+ CUDA graphs\n(final)", "engine", bursts("engine")))
    n_milestones = len(bars)
    bars.append(("Hugging Face\ntuned,\nbatch 300", "hf", bursts("hf")))
    bars.append(("vLLM", "vllm", bursts("vllm")))
    bars = [b for b in bars if b[2]]

    fig, ax = plt.subplots(figsize=(11, 5))
    xs = list(range(len(bars)))
    xs = [x + (0.6 if x >= n_milestones else 0) for x in xs]   # gap before the references
    for x, (label, system, values) in zip(xs, bars):
        height = mean(values)
        reference = x > n_milestones - 1 + 0.3
        ax.bar(x, height, width=0.7, color=COLOR[system], alpha=0.55 if reference else 1.0,
               edgecolor="white", linewidth=2)
        if len(values) > 1:
            ax.scatter([x] * len(values), values, color=TEXT, s=12, zorder=3)
        ax.text(x, max(values) + 50, f"{height:,.0f}", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_xticks(xs, [b[0] for b in bars], fontsize=8.5)
    top = max(mean(b[2]) for b in bars)
    ax.set_ylim(0, top * 1.15)
    ax.axvline(n_milestones - 0.2, color=GRID, ls="--", lw=1)
    ax.text(n_milestones - 0.1, top * 1.13, "references, final run", color=MUTED, fontsize=8.5, va="top")
    ax.text(n_milestones - 0.3, top * 1.13, "milestones", color=MUTED, fontsize=8.5, va="top", ha="right")
    ax.set_ylabel("output tokens/s", color=TEXT)
    ax.set_title("Throughput by milestone: 300-request burst, fixed lengths, 1402 MHz locked",
                 color=TEXT, loc="left", fontweight="bold")
    style(ax)
    ax.grid(axis="x", visible=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draft_cost_ratio(cost_path):
    """c = draft step time / target step time at batch 1, from spec_sweep's cost rows."""
    with open(cost_path, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    tok_s = {r["drafter"]: r["tok_s"] for r in rows if r["batch"] == 1}
    return tok_s["target"] / tok_s["draft_as_target"]


def plot_acceptance(k_sweep_path, cost_path, out_path):
    """Acceptance rate and speedup over k = 0 against k, for each prompt set and drafter."""
    with open(k_sweep_path, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    rows = [r for r in rows if r["drafter"] != "warmup"]
    prompts = ["chat", "code", "summ"]
    colors = dict(zip(prompts, [c for _, _, c in SYSTEMS]))
    drafters = {"draft": ("0.5B draft model", "-", "o"), "lookup": ("prompt lookup", "--", "s")}

    fig, (ax_a, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5))
    for p in prompts:
        base = next(r["tok_s"] for r in rows if r["prompts"] == p and r["k"] == 0)
        for d, (name, ls, marker) in drafters.items():
            runs = sorted((r for r in rows if r["prompts"] == p and r["drafter"] == d), key=lambda r: r["k"])
            ks = [r["k"] for r in runs]
            ax_a.plot(ks, [r["alpha"] for r in runs], ls=ls, marker=marker, ms=7, lw=2, color=colors[p],
                      label=f"{p}, {name}")
            ax_s.plot([0] + ks, [1.0] + [r["tok_s"] / base for r in runs], ls=ls, marker=marker, ms=7, lw=2,
                      color=colors[p], label=f"{p}, {name}")
    ax_s.axhline(1.0, color=MUTED, lw=1)
    ax_a.set_ylim(0, 1)
    ax_a.set_ylabel("acceptance rate α", color=TEXT)
    ax_s.set_ylabel("speedup over no speculation (×)", color=TEXT)
    ax_a.set_xticks([2, 4, 6])
    ax_s.set_xticks([0, 2, 4, 6])
    for ax, title in ((ax_a, "Acceptance rate"), (ax_s, "Speedup")):
        ax.set_xlabel("draft length k", color=TEXT)
        ax.set_title(title, color=TEXT, loc="left", fontweight="bold")
        style(ax)
    handles = [Line2D([], [], color=colors[p], lw=2, label=p) for p in prompts] + [
        Line2D([], [], color=MUTED, lw=2, ls=ls, marker=m, label=n) for n, ls, m in drafters.values()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    c = draft_cost_ratio(cost_path)
    fig.suptitle(f"Speculative decoding, Qwen2.5-1.5B target, batch 1, greedy. "
                 f"Draft cost ratio c = {c:.2f} (a 0.5B draft step costs {c:.0%} of a 1.5B target step)",
                 color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_final_table(rows):
    """Markdown table of the final run: mean over repeated runs per system, workload and rate."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["workload"], r["rate"] if r["rate"] is not None else float("inf"), r["system"])].append(r)
    keys = ["throughput", "ttft_p50", "ttft_p95", "tpot_p50", "e2e_p50", "e2e_p95"]
    lines = ["| workload | rate | system | runs | tok/s | TTFT p50 (s) | TTFT p95 (s) | TPOT p50 (ms) | E2E p50 (s) | E2E p95 (s) | errors |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {key: i for i, (key, _, _) in enumerate(SYSTEMS)}
    for (wl, rate, system), runs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], order[kv[0][2]])):
        vals = {}
        for k in keys:
            xs = [r[k] for r in runs if r[k] is not None]
            vals[k] = mean(xs) if xs else None
        tpot = "n/a" if vals["tpot_p50"] is None else f"{vals['tpot_p50'] * 1000:.1f}"
        lines.append(f"| {wl} | {'burst' if rate == float('inf') else f'{rate:g}'} | {system} | {len(runs)} | "
                     f"{vals['throughput']:,.0f} | {fmt(vals['ttft_p50'])} | {fmt(vals['ttft_p95'])} | {tpot} | "
                     f"{fmt(vals['e2e_p50'])} | {fmt(vals['e2e_p95'])} | {sum(r['errors'] for r in runs)} |")
    return "\n".join(lines)


def final_report(final_dir, fig_dir, k_sweep, cost):
    rows = load_final(final_dir)
    if not rows:
        print(f"no run_all.sh results in {final_dir}")
        return
    print(format_final_table(rows))
    plot_final_latency(rows, os.path.join(fig_dir, "latency_vs_rate.png"))
    plot_ablation(rows, os.path.join(fig_dir, "ablation.png"))
    plot_acceptance(k_sweep, cost, os.path.join(fig_dir, "acceptance_vs_k.png"))
    print(f"\nsaved latency_vs_rate.png, ablation.png, acceptance_vs_k.png in {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description="Results table and latency-vs-rate plot.")
    parser.add_argument("paths", nargs="*", help="results JSON files, e.g. results/*.json")
    parser.add_argument("--plot", default="docs/figures/latency_vs_rate.png")
    parser.add_argument("--sweep-plot", help="also plot p95 TTFT/TPOT against the token budget")
    parser.add_argument("--final", help="bench/run_all.sh results dir: table plus the three final figures")
    parser.add_argument("--figures", default="docs/figures", help="where --final saves its figures")
    parser.add_argument("--k-sweep", default="results/m5/k_sweep.json")
    parser.add_argument("--cost", default="results/m5/cost.json")
    args = parser.parse_args()

    if args.final:
        final_report(args.final, args.figures, args.k_sweep, args.cost)
        return

    runs = load_runs(args.paths)
    if not runs:
        print("no runs to report")
        return

    rows = [to_row(r) for r in runs]
    print(format_table(rows))
    plot_latency_vs_rate(rows, args.plot)
    print(f"\nsaved {args.plot}")
    if args.sweep_plot:
        plot_budget_sweep(rows, args.sweep_plot)
        print(f"\nsaved {args.sweep_plot}")


if __name__ == "__main__":
    main()
    