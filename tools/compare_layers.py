"""Run the same prompt through Hugging Face's Qwen and my model, layer by layer.
Prints the max difference at every checkpoint and the first one that diverges.

    python -m tools.compare_layers --prompt "The capital of France is"
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.model.loader import load_model

def capture(model, names):
    """Register a forward hook on each named submodule; returns (outputs, handles)"""
    outputs, handles = {}, []

    for name in names:
        def hook(module, inputs, output, name=name):
            out = output[0] if isinstance(output, tuple) else output # HF attention returns a tuple
            outputs[name] = out.detach().float().reshape(-1, out.shape[-1]) # [T, hidden]
        handles.append(model.get_submodule(name).register_forward_hook(hook))
    return outputs, handles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="flag a checkpoint when max diff /max |ref| exceeds this")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    mine, cfg = load_model(args.model, dtype=dtype)
    ref = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager").cuda().eval()
    tok = AutoTokenizer.from_pretrained(args.model)

    # Module names match HF's once "model." is stripped, so one list serves both
    names = ["embed_tokens"]
    for i in range(cfg.num_hidden_layers):
        names += [f"layers.{i}.self_attn", f"layers.{i}.mlp", f"layers.{i}"]
    names += ["norm", "lm_head"]
    ref_names = [n if n == "lm_head" else f"model.{n}" for n in names]

    mine_out, mine_handles = capture(mine, names)
    ref_out, ref_handles = capture(ref, ref_names)

    ids = tok(args.prompt, return_tensors="pt").input_ids[0].cuda()
    with torch.inference_mode():
        mine(ids, torch.arange(len(ids), device="cuda"))
        ref(ids[None])
    for h in mine_handles + ref_handles:
        h.remove()

    first_bad = None
    print(f"{'checkpoint':<22}{'max diff':>12}{'max |ref|':>12}{'relative':>12}")
    for name, ref_name in zip(names, ref_names):
        a, b = mine_out[name], ref_out[ref_name]
        diff, scale = (a - b).abs().max().item(), b.abs().max().item()
        rel = diff / scale
        bad = rel > args.rtol

        if bad and first_bad is None:
            first_bad = name
        print(f"{name:<22}{diff:>12.2e}{scale:>12.2e}{rel:>12.2e}{' <-- differs' if bad else ''}")

    print()
    if first_bad is None:
        print(f"All {len(names)} checkpoints match (relative diff <= {args.rtol}).")
    else:
        print(f"First divergence: {first_bad}")

if __name__ == "__main__":
    main()