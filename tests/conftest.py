import gc

import pytest
import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="session")
def models():
    """Returns load(dtype) -> (ref, mine), both on the GPU.
    
    Only one dtype is kept loaded: loading a pair peaks at ~3.5 GB of RAM,
    so the previous pair is freed before the next one loads."""

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")

    from transformers import AutoModelForCausalLM

    from engine.model.loader import load_model

    loaded = {}

    def load(dtype):
        if dtype not in loaded:
            loaded.clear() # free the other dtype's pair before loading this one
            gc.collect()
            torch.cuda.empty_cache()

            # Eager attention: HF's default (SDPA) orders its operations differently
            ref = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=dtype, attn_implementation="eager").cuda().eval()
            mine, _ = load_model(MODEL, dtype=dtype)
            loaded[dtype] = (ref, mine)

        return loaded[dtype]
    return load
