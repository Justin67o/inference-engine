import json

import pytest

from bench.report import COLUMNS, fmt, format_table, load_runs, plot_latency_vs_rate, to_row


def make_run(rate=0.2, label="hf-baseline", tpot=None, env=True, dirty=False):
    """A results-file dict in the format bench/loadgen.py writes."""
    run = {
        "config": {"label": label, "rate": rate},
        "metrics": {
            "num_ok": 10,
            "num_errors": 0,
            "ttft": {"p50": 4.0, "p95": 9.0, "p99": 9.5},
            "tpot": tpot,
            "e2e": {"p50": 4.0, "p95": 9.0, "p99": 9.5},
            "throughput_tok_s": 22.2,
        },
        "results": [],
    }
    if env:
        run["env"] = {"git": {"commit": "b218ba3dcb31cfed65b97485b927bc6ee32a77f5", "dirty": dirty}}
    return run


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# --- load_runs ---

def test_load_runs_skips_files_without_metrics(tmp_path, capsys):
    new = write_json(tmp_path / "new.json", make_run())
    old = write_json(tmp_path / "old.json", {"config": {"rate": 1.0}, "results": []})

    runs = load_runs([new, old])

    assert len(runs) == 1
    assert runs[0]["path"] == new          # each run remembers its file
    assert "skipping" in capsys.readouterr().out


# --- to_row ---

def test_to_row_flattens_metrics_and_shortens_commit():
    row = to_row({**make_run(rate=0.4), "path": "x.json"})
    assert row["rate"] == 0.4
    assert row["e2e_p95"] == 9.0
    assert row["commit"] == "b218ba3"      # first 7 characters
    assert row["dirty"] is False


def test_to_row_non_streaming_has_no_tpot():
    row = to_row({**make_run(tpot=None), "path": "x.json"})
    assert row["tpot_p50"] is None


def test_to_row_streaming_tpot():
    row = to_row({**make_run(tpot={"p50": 0.03, "p95": 0.05, "p99": 0.06}), "path": "x.json"})
    assert row["tpot_p50"] == pytest.approx(0.03)


def test_to_row_without_env_section():
    # Files written before bench/env.py existed still load, with blank git columns.
    row = to_row({**make_run(env=False), "path": "x.json"})
    assert row["commit"] is None
    assert row["dirty"] is None


# --- fmt ---

def test_fmt():
    assert fmt(None) == "n/a"
    assert fmt(4.693390451997402) == "4.69"
    assert fmt(10) == "10"                 # ints are not given decimals
    assert fmt("hf-baseline") == "hf-baseline"


# --- format_table ---

def rows_for(*runs):
    return [to_row({**run, "path": f"{i}.json"}) for i, run in enumerate(runs)]


def test_format_table_shape():
    lines = format_table(rows_for(make_run(), make_run(rate=0.4))).splitlines()
    assert len(lines) == 2 + 2             # header + separator + one line per run
    assert lines[0] == "| " + " | ".join(title for _, title in COLUMNS) + " |"
    assert lines[1] == "| " + " | ".join("---" for _ in COLUMNS) + " |"
    for line in lines:                     # every line has one cell per column
        assert line.count("|") == len(COLUMNS) + 1


def test_format_table_sorts_by_label_then_rate():
    rows = rows_for(make_run(rate=0.4), make_run(rate=0.1), make_run(rate=0.2, label="engine"))
    body = format_table(rows).splitlines()[2:]
    assert [line.split(" | ")[:2] for line in body] == [
        ["| engine", "0.20"],
        ["| hf-baseline", "0.10"],
        ["| hf-baseline", "0.40"],
    ]


def test_format_table_does_not_modify_rows():
    rows = rows_for(make_run(rate=0.4, dirty=True), make_run(rate=0.1))
    format_table(rows)
    assert rows[0]["rate"] == 0.4          # caller's order unchanged
    assert rows[0]["commit"] == "b218ba3"  # no "*" written into the caller's row


def test_format_table_marks_dirty_runs():
    table = format_table(rows_for(make_run(rate=0.1, dirty=True), make_run(rate=0.2)))
    line_01, line_02 = table.splitlines()[2:]
    assert "b218ba3*" in line_01
    assert "b218ba3*" not in line_02


def test_format_table_shows_na_for_missing_tpot():
    line = format_table(rows_for(make_run(tpot=None))).splitlines()[2]
    assert "n/a" in line


# --- plot_latency_vs_rate ---

def test_plot_writes_png(tmp_path):
    rows = rows_for(make_run(rate=0.1), make_run(rate=0.2), make_run(rate=0.2), make_run(rate=0.4))
    out = tmp_path / "figures" / "plot.png"   # folder doesn't exist yet: the function must create it

    plot_latency_vs_rate(rows, str(out))

    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"   # a real PNG, not an empty file


def test_plot_skips_all_failed_runs(tmp_path):
    failed = make_run(rate=0.8)
    failed["metrics"].update({"num_ok": 0, "num_errors": 10, "ttft": None, "e2e": None})
    rows = rows_for(make_run(rate=0.1), failed)
    out = tmp_path / "plot.png"

    plot_latency_vs_rate(rows, str(out))   # must not crash on the run with no latencies

    assert out.exists()
