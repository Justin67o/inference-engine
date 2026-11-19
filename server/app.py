import asyncio
import itertools
import json
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine.config import EngineConfig
from engine.engine import Engine
from engine.request import Request
from server.detokenizer import Detokenizer

_ids = itertools.count()


@asynccontextmanager
async def lifespan(app):
    engine = Engine.from_config(EngineConfig()) # load here, not at import time
    thread = threading.Thread(target=engine.run_forever, name="engine", daemon=True)
    thread.start()
    app.state.engine, app.state.thread = engine, thread
    yield
    engine.stop()           # run_forever returns after its current step
    thread.join(timeout=10)


app = FastAPI(lifespan=lifespan)


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(128, ge=1)
    temperature: float = Field(0.0, ge=0.0)
    top_k: int = Field(0, ge=0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    stream: bool = True

def sse(rid, text, finish_reason):
    chunk = {"id": f"cmpl-{rid}", "object": "text_completion",
             "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}]}
    return f"data: {json.dumps(chunk)}\n\n"


async def stream(r, events, tokenizer):
    detok = Detokenizer(tokenizer)
    while True:
        token_id, done = await events.get()
        text = detok.push(token_id)
        if done:
            text += detok.flush()
        if text or done:        # a half-finished emoji yields "": don't send anything
            yield sse(r.id, text, r.finish_reason if done else None)
        if done:
            break
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(body: CompletionRequest):
    engine = app.state.engine
    loop = asyncio.get_running_loop()
    events = asyncio.Queue()

    def on_token(token_id, done):   # runs on the engine thread
        loop.call_soon_threadsafe(events.put_nowait, (token_id, done))

    stop = [body.stop] if isinstance(body.stop, str) else (body.stop or [])
    r = Request(id=next(_ids), prompt_ids=engine.tokenizer.encode(body.prompt),
                max_tokens=body.max_tokens, temperature=body.temperature,
                top_k=body.top_k, top_p=body.top_p, seed=body.seed, stop=stop,
                ignore_eos=body.ignore_eos, on_token=on_token)
    try:
        engine.add_request(r)
    except ValueError as e:         # prompt too long or other errors
        raise HTTPException(status_code=400, detail=str(e))
    return StreamingResponse(stream(r, events, engine.tokenizer),
                             media_type="text/event-stream")


@app.get("/health")
async def health():
    thread = getattr(app.state, "thread", None)
    if thread is None or not thread.is_alive():
        raise HTTPException(status_code=503, detail="engine thread is not running")
    return {"status": "ok"}