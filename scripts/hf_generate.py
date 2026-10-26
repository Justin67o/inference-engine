import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name, dtype=torch.bfloat16).cuda().eval()

messages = [{"role": "user", "content": "What is the capital of France?"}]
text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt").input_ids.cuda()   # [1, T]

output = model.generate(ids, max_new_tokens=20, temperature=1, do_sample=False)


with torch.inference_mode():
    original_ids = ids.clone()
    original_logits = model(original_ids).logits                    # [1, T, vocab]
    for step in range(30):
        t0 = time.perf_counter()
        logits = model(ids).logits                    # [1, T, vocab]
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)  # [1, 1]
        ids = torch.cat([ids, next_id], dim=1)
        torch.cuda.synchronize()
        print(step, ids.shape[1], f"{(time.perf_counter()-t0)*1e3:.1f} ms")
        if next_id.item() == tok.eos_token_id:
            break

print(torch.equal(output, ids))  

def top_5_tokens(logits, temperature=1.0):
    logits = logits[0, -1, :] / temperature
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=5, dim=-1)
    for prob, idx in zip(top_probs, top_ids):
        token = tok.convert_ids_to_tokens(idx.item())
        print(f"{token}: {prob.item():.4f}")


top_5_tokens(original_logits, temperature=1.0)