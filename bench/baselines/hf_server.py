import asyncio
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_BATCH_SIZE = int(os.environ.get("HF_MAX_BATCH", "16"))    # 16 = the M0 baseline
WINDOW_SIZE = float(os.environ.get("HF_WINDOW", "0.02"))      # seconds to collect a batch

queue: asyncio.Queue = asyncio.Queue()

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    NAME, dtype=torch.bfloat16).cuda().eval()
tok = AutoTokenizer.from_pretrained(NAME)
tok.padding_side = "left"


class Request(BaseModel):
    prompt: str
    max_tokens: int = 128


def run_batch(reqs):
    enc = tok([r.prompt for r in reqs], return_tensors="pt", padding=True).to("cuda")
    n = max(r.max_tokens for r in reqs)
    with torch.inference_mode():
        # Plain greedy, as in offline.py: Qwen's config adds repetition_penalty=1.1 even when greedy
        out = model.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                             repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
    outs = out[:, enc.input_ids.shape[1]:]   # every row has n tokens: trim each to its own max_tokens
    return [tok.decode(o[:r.max_tokens], skip_special_tokens=True) for o, r in zip(outs, reqs)]


async def batcher():
    loop = asyncio.get_running_loop()
    while True:
        batch = [await queue.get()]
        deadline = loop.time() + WINDOW_SIZE
        while len(batch) < MAX_BATCH_SIZE and (left := deadline - loop.time()) > 0:
            try:
                batch.append(await asyncio.wait_for(queue.get(), left))
            except TimeoutError:
                break
        print(f"Running batch of {len(batch)} requests", flush=True)
        texts = await asyncio.to_thread(run_batch, [r for r, _ in batch])
        for (_, fut), text in zip(batch, texts):
            fut.set_result(text)


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(batcher())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/completions")
async def complete(req: Request):
    fut = asyncio.get_running_loop().create_future()
    await queue.put((req, fut))
    return {"text": await fut}


@app.get("/health")
async def health():
    return {"status": "ok", "config": {"engine": "hf", "max_batch_size": MAX_BATCH_SIZE,
                                       "window_s": WINDOW_SIZE}}