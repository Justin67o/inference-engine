import pytest
import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="session")
def fp32_models():
    """(ref, mine) in fp32 on the GPU, loaded once for the whole test run."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from transformers import AutoModelForCausalLM

    from engine.model.loader import load_model

    # Eager attenion: HF's default (SDPA) orders its operations differently
    ref = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager").cuda().eval()

    mine, _ = load_model(MODEL, dtype=torch.float32)
    return ref, mine