"""Writes mixed_200.jsonl: synthetic_200's prompts, each with its own max_tokens.
python -m bench.prompts.make_mixed
"""

import json
import math
import random

LO, HI, SEED = 16, 512, 0


def draw(rng):
    """Log-uniform on [LO, HI]: as many requests in 16-32 as in 256-512; mean ~143, near the fixed 128"""
    return round(math.exp(rng.uniform(math.log(LO), math.log(HI))))


if __name__ == "__main__":
    rng = random.Random(SEED)
    with open("bench/prompts/synthetic_200.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        row["max_tokens"] = draw(rng)
    with open("bench/prompts/mixed_200.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    lens = [r["max_tokens"] for r in rows]
    print(
        f"{len(lens)} requests, max_tokens min {min(lens)} max {max(lens)} mean {sum(lens) / len(lens):.0f}"
    )
