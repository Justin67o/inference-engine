import json
from pathlib import Path

import pytest
import torch

PROMPTS_FILE = Path(__file__).parent / "prompts_200.jsonl"
PROMPTS = [json.loads(line)["prompt"] for line in PROMPTS_FILE.read_text().splitlines()[:50]]

@pytest.mark.gpu
@pytest.mark.parametrize("prompt", PROMPTS)
def test_fp32_logits_match(fp32_models, tok, prompt):
    ref, mine = fp32_models
    ids = tok(prompt, return_tensors="pt").input_ids[0].cuda()
    pos = torch.arange(len(ids), device="cuda")
    with torch.inference_mode():
        diff = (ref(ids[None]).logits[0] - mine(ids, pos)).abs().max().item()
    assert diff < 1e-3, f"max abs logit diff {diff:.2e}"