import numpy as np

def per_request(results):
    """Latency metrics for one request, in seconds. TPOT is None when the server didn't stream

    Args:
        results: List of dicts, each with keys "sent", "token_times", "n", "ok".

    Returns:
        A list of dicts, each with keys "sent", "ttft", "tpot", "e2e",
        """
    times = results["token_times"]
    sent = results["sent"]
    ttft = times[0] - sent
    if len(times) < 2:
        tpot = None
    else:
        tpot = (times[-1] - times[0]) / (len(times) - 1)
    e2e = times[-1] - sent
    return {"ttft": ttft, "tpot": tpot, "e2e": e2e}

def percentiles(values):
    """Compute the 50th, 95th, and 99th percentiles of a list of values.

    Args:
        values: List of numeric values.
    """
    if len(values) == 0:
        return None
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99)}

def summarize(results):
    """Summarize the results of a loadgen run.

    Latencies (ttft, tpot, e2e) are in seconds. Throughput is in output tokens/s.
    Only successful requests are included in the summary.
    Args:
        results: List of dicts, each with keys "sent", "token_times", "n", "ok".

    Returns:
        A dict with summary statistics.
    """
    if not results:
        return {"num_ok": 0, "num_errors": 0, "ttft": None, "tpot": None,
            "e2e": None, "throughput_tok_s": None}

    
    ok = [r for r in results if r["ok"]]
    per = [per_request(r) for r in ok]

    ttfts = [p["ttft"] for p in per]
    e2es = [p["e2e"] for p in per]
    tpots = [p["tpot"] for p in per if p["tpot"] is not None]

    start = min(r["sent"] for r in results)
    end = max(r["token_times"][-1] for r in results if r["token_times"])
    tokens = sum(r["n"] for r in ok)
    throughput = tokens / (end - start) if end > start else 0.0

    return {
        "num_ok": len(ok),
        "num_errors": len(results) - len(ok),
        "ttft": percentiles(ttfts),
        "tpot": percentiles(tpots),
        "e2e": percentiles(e2es),
        "throughput_tok_s": throughput,
    }
