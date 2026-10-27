import asyncio
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_BATCH_SIZE = 16
WINDOW_SIZE = 0.02
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
        out = model.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

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