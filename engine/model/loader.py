import glob
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig

from engine.model.model import Model


def load_model(name, dtype=torch.bfloat16, device="cuda"):
    path = snapshot_download(name, allow_patterns=["*.json", "*.safetensors"])
    cfg = AutoConfig.from_pretrained(path)
    model = Model(cfg).to(dtype)
    state = {}
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        state.update(load_file(f))
    state = {k.removeprefix("model."): v for k, v in state.items()}
    if cfg.tie_word_embeddings:
        state["lm_head.weight"] = state["embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return model.to(device).eval(), cfg