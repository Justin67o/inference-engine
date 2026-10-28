import json
import re

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", model_max_length=10**9)

N_PROMPT_TOKENS = 200
N_PROMPTS = 200
with open ("bench/prompts/source/prideandprejudice.txt", encoding="utf-8") as f:
    text = f.read()

marker = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
newline = text.find("\n", marker)
start = newline + 1

end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
body = text[start:end].strip()
body = re.sub(r"\[_?Copyright[^\]]*\]", "", body)
body = re.sub(r"\[Illustration[^\]]*\]", "", body)
body = re.sub(r"\n{3,}", "\n\n", body) 


ids = tokenizer.encode(body, add_special_tokens=False)

assert len(ids) >= N_PROMPT_TOKENS * N_PROMPTS, f"Not enough tokens in the source text. Found {len(ids)}, need {N_PROMPT_TOKENS * N_PROMPTS}."

prompt_ids = []
for i in range(0, N_PROMPTS * N_PROMPT_TOKENS, N_PROMPT_TOKENS):
    prompt_ids.append(ids[i:i+N_PROMPT_TOKENS])

prompts = []
for window in prompt_ids:
    prompts.append(tokenizer.decode(window))


with open("bench/prompts/synthetic_200.jsonl", "w", encoding="utf-8") as f:
    for i, prompt in enumerate(prompts):
        n = len(tokenizer.encode(prompt, add_special_tokens=False))
        f.write(json.dumps({"id": i, "prompt": prompt, "n_tokens": n}) + "\n")
