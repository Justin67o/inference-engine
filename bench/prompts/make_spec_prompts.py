"""Writes spec_chat / spec_code / spec_summ.jsonl for the speculative sweeps.
    python -m bench.prompts.make_spec_prompts
"""
import json

CHAT = [
    "Explain why the sky is blue.",
    "What are the main causes of inflation?",
    "Give me three tips for learning a new language.",
    "How does a refrigerator keep food cold?",
    "What is the difference between a virus and a bacterium?",
    "Describe how a bill becomes a law in the United States.",
    "Why do cats purr?",
    "What should I pack for a week-long hiking trip?",
    "Explain recursion to a ten-year-old.",
    "What are the pros and cons of electric cars?",
    "How do vaccines train the immune system?",
    "Write a short poem about autumn.",
    "What is the theory of plate tectonics?",
    "How can I improve my sleep?",
    "Explain what a neural network is in simple terms.",
    "What happened during the French Revolution?",
]

CODE_FUNCS = [
    "def mean(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / len(xs)",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    "def flatten(nested):\n    out = []\n    for item in nested:\n        if isinstance(item, list):\n            out.extend(flatten(item))\n        else:\n            out.append(item)\n    return out",
    "def word_counts(text):\n    counts = {}\n    for w in text.lower().split():\n        counts[w] = counts.get(w, 0) + 1\n    return counts",
    "def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n        if xs[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    "def merge(a, b):\n    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]",
    "def chunks(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
]
CODE_TASKS = ["Add type hints and a docstring to this function:",
              "Rewrite this function with clearer variable names:"]

def summ_passages(path, n, words=300):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text = text[text.find("It is a truth universally acknowledged"):].split()
    return [" ".join(text[i * words:(i + 1) * words]) for i in range(n)]

def write(name, prompts):
    with open(f"bench/prompts/spec_{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"id": i, "prompt": p}) + "\n" for i, p in enumerate(prompts))

if __name__ == "__main__":
    write("chat", CHAT)
    write("code", [f"{t}\n\n```python\n{fn}\n```" for t in CODE_TASKS for fn in CODE_FUNCS])
    write("summ", [f"Summarize this passage in 3 sentences:\n\n{p}" for p in
                   summ_passages("bench/prompts/source/prideandprejudice.txt", 16)])