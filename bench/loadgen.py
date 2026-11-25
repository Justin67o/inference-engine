import argparse
import asyncio
import json
import os
import random
import time

import httpx

from bench.env import get_env
from bench.metrics import summarize


async def one_request(client, url, prompt, max_tokens, results):
    sent, token_times, ok, spec = time.perf_counter(), [], False, None
    body = {"prompt": prompt, 
            "max_tokens": max_tokens, 
            "ignore_eos": True,     # exactly max_tokens, like the HF baseline's min_new_tokens
            "stream": True}
    
    try:
        async with client.stream("POST", url, json=body) as response:
            ok = response.status_code == 200
            if not ok:
                print(f"Error: HTTP {response.status_code}")

            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    if '"finish_reason": "error"' in line:      # the engine died mid-request
                        ok = False
                        print("Error: the server ended the request with an error")
                    else:
                        token_times.append(time.perf_counter())
                        chunk = json.loads(line[6:])
                        spec = chunk.get("spec", spec)
    except httpx.HTTPError as e:    # dropped connection, or no data for --read-timeout seconds
        ok = False
        print(f"Error: {type(e).__name__}: {e}")

    if not token_times:
        token_times = [time.perf_counter()]
    results.append({"sent": sent, 
                    "token_times": token_times,
                    "n": max_tokens,
                    "ok": ok,
                    "spec": spec,
                    })


def server_info(url):
    """The server's /health JSON (my server reports its EngineConfig there), or None"""
    try:
        r = httpx.get(url.replace("/v1/completions", "/health"), timeout=5)
        return r.json() if r.status_code == 200 else None
    except(httpx.HTTPError, ValueError):
        return None


def preemptions(url):
    """preemptions_total from the server's /metrics, or None (HF baseline, M3)"""
    try:
        r = httpx.get(url.replace("/v1/completions", "/metrics"), timeout=5)
    except httpx.HTTPError:
        return None
    for line in r.text.splitlines():
        if line.startswith("preemptions_total "):
            return float(line.split()[1])
    return None


async def run(url, prompts, rate_per_s, max_tokens, read_timeout=None):
    results = []
    limits = httpx.Limits(max_connections=1000)
    timeout = httpx.Timeout(None, read=read_timeout)   # no limit on connect/write, only on silence

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = []
        for p in prompts:
            tasks.append(asyncio.create_task(
                one_request(client, url, p, max_tokens, results)))
            await asyncio.sleep(random.expovariate(rate_per_s))

        await asyncio.gather(*tasks)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001/v1/completions")
    parser.add_argument("--rate", type=float, required=True, help="requests per second")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompts", default="bench/prompts/synthetic_200.jsonl")
    parser.add_argument("--num-requests", type=int, required=True, help="number of requests to send")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20, help="number of warmup requests to send")
    parser.add_argument("--output", required=True, help="output file to write results to")
    parser.add_argument("--label", default="", help="what is being measured")
    parser.add_argument("--read-timeout", type=float, default=600,
                        help="seconds without any data before a request counts as failed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.prompts, encoding="utf-8") as f:
        all_prompts = [json.loads(line)["prompt"] for line in f]

    prompts = [all_prompts[i % len(all_prompts)] for i in range(args.num_requests)]

    server = server_info(args.url)

    if args.warmup > 0:
        warm_prompts = [all_prompts[i % len(all_prompts)] for i in range(args.warmup)]
        asyncio.run(run(args.url, warm_prompts, 100.0, args.max_tokens, args.read_timeout))
        print(f"warm-up: {args.warmup} done")
    random.seed(args.seed)

    before = preemptions(args.url)
    results = asyncio.run(run(args.url, prompts, args.rate, args.max_tokens, args.read_timeout))
    after = preemptions(args.url)
    env = get_env() # right after the run, so GPU state is still under load
    metrics = summarize(results)

    engine_stats = {"preemptions": None if before is None or after is None else after - before}
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "server": server, "metrics": metrics, "env": env,
                       "results": results, "engine_stats": engine_stats}, f, indent=2)

    
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()