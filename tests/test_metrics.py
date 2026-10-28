import pytest

from bench.metrics import per_request, percentiles, summarize

FAKE = [
    {"sent": 0.0, "token_times": [1.0, 1.1, 1.2, 1.3], "n": 4, "ok": True},   # streaming
    {"sent": 0.5, "token_times": [3.0],                "n": 4, "ok": True},   # non-streaming
    {"sent": 0.6, "token_times": [0.7],                "n": 4, "ok": False},  # error
]


def test_per_request_streaming():
    m = per_request({"sent": 0.0, "token_times": [1.0, 1.1, 1.2, 1.3], "n": 4, "ok": True})
    assert m["ttft"] == pytest.approx(1.0)
    assert m["tpot"] == pytest.approx(0.1)
    assert m["e2e"] == pytest.approx(1.3)


def test_per_request_non_streaming_has_no_tpot():
    m = per_request({"sent": 0.5, "token_times": [3.0], "n": 128, "ok": True})
    assert m["tpot"] is None
    assert m["ttft"] == m["e2e"] == pytest.approx(2.5)


def test_per_request_e2e_identity():
    # Uneven gaps: TPOT is an average, and the identity must still hold exactly.
    times = [2.0, 2.05, 2.3, 2.31, 2.9]
    m = per_request({"sent": 1.2, "token_times": times, "n": len(times), "ok": True})
    assert m["ttft"] + (len(times) - 1) * m["tpot"] == pytest.approx(m["e2e"])


def test_percentiles_1_to_100():
    # numpy interpolates linearly: position = q/100 * (count - 1).
    p = percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5)
    assert p["p95"] == pytest.approx(95.05)
    assert p["p99"] == pytest.approx(99.01)


def test_percentiles_empty_is_none():
    assert percentiles([]) is None


def test_summarize_mixed_run():
    s = summarize(FAKE)
    assert s["num_ok"] == 2
    assert s["num_errors"] == 1
    assert s["ttft"]["p50"] == pytest.approx(1.75)   # from [1.0, 2.5]
    assert s["tpot"]["p50"] == pytest.approx(0.1)    # only the streaming request
    assert s["e2e"]["p50"] == pytest.approx(1.9)     # from [1.3, 2.5]
    # Useful tokens (8, the error's are excluded) over the whole run window (0.0 to 3.0).
    assert s["throughput_tok_s"] == pytest.approx(8 / 3.0)


def test_summarize_all_failed():
    s = summarize([FAKE[2]])
    assert s["num_ok"] == 0
    assert s["num_errors"] == 1
    assert s["ttft"] is None and s["tpot"] is None and s["e2e"] is None
    assert s["throughput_tok_s"] == 0.0   # the run happened and produced nothing useful


def test_summarize_empty():
    s = summarize([])
    assert s["num_ok"] == 0
    assert s["num_errors"] == 0
    assert s["throughput_tok_s"] is None   # no run window, so not measurable
