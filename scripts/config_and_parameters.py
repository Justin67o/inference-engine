import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name, dtype=torch.bfloat16).cuda().eval()

messages = [{"role": "user", "content": "What is the capital of France?"}]
text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt").input_ids.cuda()   # [1, T]

cfg = model.config
print(cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
      cfg.num_key_value_heads, cfg.intermediate_size, cfg.rope_parameters["rope_theta"],
      cfg.tie_word_embeddings)
n = sum(p.numel() for p in model.parameters())     # total parameter count
print(f"{n / 1e6:.0f}M parameters")
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_allocated() / 2**30, "GiB on the GPU")