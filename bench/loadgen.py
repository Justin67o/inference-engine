import asyncio, argparse
import random
import time
import json
import httpx

async def one_request(client, url, prompt, max_tokens, results):
    sent, token_times = time.perf_counter(), []
    body = {"prompt": prompt, 
            "max_tokens": max_tokens, 
            "stream": True}
    async with client.stream("POST", url, json=body) as response:
        ok = response.status_code == 200
        if not ok:
            print(f"Error: HTTP {response.status_code}")
        

        async for line in response.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                token_times.append(time.perf_counter())
            

    
    if not token_times:
        token_times = [time.perf_counter()]
    results.append({"sent": sent, 
                    "token_times": token_times,
                    "n": max_tokens,
                    "ok": ok
                    })

async def run(url, prompts, rate_per_s, max_tokens):
    results = []
    limits = httpx.Limits(max_connections=1000)

    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
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
    args = parser.parse_args()
    

    with open(args.prompts, encoding="utf-8") as f:
        all_prompts = [json.loads(line)["prompt"] for line in f]

    prompts = [all_prompts[i % len(all_prompts)] for i in range(args.num_requests)]

    if args.warmup > 0:
        warm_prompts = [all_prompts[i % len(all_prompts)] for i in range(args.warmup)]
        asyncio.run(run(args.url, warm_prompts, 100.0, args.max_tokens))
        print(f"warm-up: {args.warmup} requests sent")
    random.seed(args.seed)

    results = asyncio.run(run(args.url, prompts, args.rate, args.max_tokens))

    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
  
if __name__ == "__main__":
    main()