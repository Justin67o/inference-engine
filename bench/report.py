import json
import argparse
import os
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt

# (key in the row dict, title shown in the table header), in display order
COLUMNS = [
    ("label", "label"),
    ("rate", "rate"),
    ("ok", "ok"),
    ("errors", "err"),
    ("ttft_p50", "TTFT p50"),
    ("tpot_p50", "TPOT p50"),
    ("e2e_p50", "E2E p50"),
    ("e2e_p95", "E2E p95"),
    ("e2e_p99", "E2E p99"),
    ("throughput", "tok/s"),
    ("commit", "commit"),
]


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
        "ok": metrics["num_ok"],
        "errors": metrics["num_errors"],
        "ttft_p50": pct(metrics["ttft"], "p50"),
        "ttft_p95": pct(metrics["ttft"], "p95"),
        "tpot_p50": pct(metrics["tpot"], "p50"),
        "e2e_p50": pct(metrics["e2e"], "p50"),
        "e2e_p95": pct(metrics["e2e"], "p95"),
        "e2e_p99": pct(metrics["e2e"], "p99"),
        "throughput": metrics["throughput_tok_s"],
        "commit": commit[:7] if commit else None,    
        "dirty": git.get("dirty"),
        "path": run["path"],
    }

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
    ax.set_ylabel("end-to-end latency (s)")
    ax.set_title("Latency vs request rate")
    ax.grid(alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Results table and latency-vs-rate plot.")
    parser.add_argument("paths", nargs="+", help="results JSON files, e.g. results/*.json")
    parser.add_argument("--plot", default="docs/figures/latency_vs_rate.png")
    args = parser.parse_args()

    runs = load_runs(args.paths)
    if not runs:
        print("no runs to report")
        return

    rows = [to_row(r) for r in runs]
    print(format_table(rows))
    plot_latency_vs_rate(rows, args.plot)
    print(f"\nsaved {args.plot}")


if __name__ == "__main__":
    main()
