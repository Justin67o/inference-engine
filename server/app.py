import asyncio
import itertools
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from engine.config import EngineConfig
from engine.engine import Engine
from engine.request import Request
from server.detokenizer import Detokenizer, StopStringFilter
from server.metrics import RequestTimer, watch

_ids = itertools.count()


@asynccontextmanager
async def lifespan(app):
    cfg = EngineConfig.from_env()
    print(f"engine config: {cfg}", flush=True)
    engine = Engine.from_config(cfg) # load here, not at import time
    watch(engine)
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

def sse(rid, text, finish_reason, extra=None):
    chunk = {"id": f"cmpl-{rid}", "object": "text_completion",
             "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}]}

    if extra:
        chunk.update(extra)
    return f"data: {json.dumps(chunk)}\n\n"


async def stream(r, events, engine):
    detok = Detokenizer(engine.tokenizer)
    stops = StopStringFilter(r.stop)
    done = False
    try:
        while not done:
            token_id, done = await events.get()
            if token_id is None:    # the engine died: end the stream instead of hanging
                yield sse(r.id, "", "error")
                break
            text = detok.push(token_id)
            if done:
                text += detok.flush()
            text = stops.push(text)
            if done:
                text += stops.flush()
            if text or done:        # a half-finished emoji yields "": don't send anything
                extra = None
                if done:
                    extra = {"spec": {"passes": r.spec_passes, "drafted": r.spec_drafted,
                                      "accepted": r.spec_accepted, "rejected": r.spec_rejected}}
                yield sse(r.id, text, r.finish_reason if done else None, extra)
        yield "data: [DONE]\n\n"
    finally:
        if not done:                # client left early, free its slot and cache
            engine.abort_request(r)


@app.post("/v1/completions")
async def completions(body: CompletionRequest):
    timer = RequestTimer()
    engine = app.state.engine
    loop = asyncio.get_running_loop()
    events = asyncio.Queue()
    
    def on_token(token_id, done):   # runs on the engine thread; token_id None = engine error
        if token_id is not None:
            timer.token(done)
            if done:
                timer.finish(r)
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
    except RuntimeError as e:       # the engine thread died
        raise HTTPException(status_code=503, detail=str(e))
    return StreamingResponse(stream(r, events, engine),
                             media_type="text/event-stream")


@app.get("/health")
async def health():
    thread = getattr(app.state, "thread", None)
    if thread is None or not thread.is_alive():
        raise HTTPException(status_code=503, detail="engine thread is not running")
    return {"status": "ok", "config": asdict(app.state.engine.cfg)}



@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)