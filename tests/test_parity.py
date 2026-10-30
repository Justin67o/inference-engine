import json
import warnings
from pathlib import Path

import pytest
import torch

PROMPTS_FILE = Path(__file__).parent / "prompts_200.jsonl"
PROMPTS = [json.loads(line)["prompt"] for line in PROMPTS_FILE.read_text().splitlines()]

MAX_NEW_TOKENS = 32
NEAR_TIE = 0.5 # top-2 logit gap below which a greedy mismatch counts as bf16 noise
MAX_NEAR_TIE_RATE = 0.05 # many near-ties at once point to a real bug, not noise
near_ties = []

@pytest.mark.gpu
@pytest.mark.parametrize("prompt", PROMPTS[:50])
def test_fp32_logits_match(models, tok, prompt):
    ref, mine = models(torch.float32)
    ids = tok(prompt, return_tensors="pt").input_ids[0].cuda()
    pos = torch.arange(len(ids), device="cuda")
    with torch.inference_mode():
        diff = (ref(ids[None]).logits[0] - mine(ids, pos)).abs().max().item()
    assert diff < 1e-3, f"max abs logit diff {diff:.2e}"

def top2_gap(logits):
    top = logits.float().topk(2).values
    return (top[0] - top[1]).item()


@pytest.mark.gpu
@pytest.mark.parametrize("prompt", PROMPTS)
def test_bf16_greedy_matches(models, tok, prompt):
    ref, mine = models(torch.bfloat16)
    ids = tok(prompt, return_tensors="pt").input_ids[0].cuda()

    with torch.inference_mode():
        # Plain greedy: Qwen's generation config adds repetition_penalty=1.1, which
        # changes the chosen token even with do_sample=False. use_cache=False makes
        # HF recompute everything each step, the same shapes my loop uses.
        out = ref.generate(
            ids[None], max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=False,
            repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
            output_logits=True, return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        ref_tokens = out.sequences[0, len(ids):]

        seq, my_logits = ids, []
        for _ in range(len(ref_tokens)): # HF may stop early at EOS
            logits = mine(seq, torch.arange(len(seq), device="cuda"))[-1]
            my_logits.append(logits)
            seq = torch.cat([seq, logits.argmax()[None]])
        my_tokens = seq[len(ids):]

    if torch.equal(my_tokens, ref_tokens):
        return

    i = (my_tokens != ref_tokens).nonzero()[0].item()
    ref_gap, my_gap = top2_gap(out.logits[i][0]), top2_gap(my_logits[i])
    msg = (f"first mismatch at step {i}: HF {tok.decode(ref_tokens[i])!r}, "
            f"mine {tok.decode(my_tokens[i])!r}; top-2 gap HF {ref_gap:.3f}, mine {my_gap:.3f}")
    assert ref_gap < NEAR_TIE, f"not a near-tie, {msg}"
    near_ties.append(prompt)
    warnings.warn(f"near-tie, {msg}")

@pytest.mark.gpu
def test_bf16_near_tie_rate():
    # Runs after the greedy tests above (pytest keeps file order)
    assert len(near_ties) <= MAX_NEAR_TIE_RATE * len(PROMPTS), (
        f"{len(near_ties)} of {len(PROMPTS)} prompts diverged at near-ties; "
        "that many suggests a real bug"
    )