import json
import re
import socket
import threading
import time

import httpx
import pytest
import torch
import uvicorn
from fastapi.testclient import TestClient

from engine.config import EngineConfig
from engine.engine import Engine
from engine.generate import generate
from engine.request import State
from server.app import app


class LetterTok:                    # fake tokenizer: "a".."z" <-> 0..25
    def encode(self, text):
        return [ord(c) - ord("a") for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(ord("a") + i % 26) for i in ids)


TOK = LetterTok()


@pytest.fixture(scope="module")
def engine(tiny_model):
    engine = Engine(tiny_model, EngineConfig(max_num_seqs=4, max_num_batched_tokens=64,
                                             max_model_len=64), tokenizer=TOK)
    thread = threading.Thread(target=engine.run_forever, daemon=True)
    thread.start()
    app.state.engine, app.state.thread = engine, thread
    yield engine
    engine.stop()
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def client(engine):
    return TestClient(app)      # no `with`: that would run the lifespan and load the real model


def wait_until(cond, timeout=10):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture(scope="module")
def server_url(engine):
    """A real uvicorn server on a free port, so a test can drop the connection mid-stream."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))                 # port 0: the OS picks a free one
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    wait_until(lambda: server.started)
    yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    server.should_exit = True
    thread.join(timeout=10)



def expected_text(model, prompt, max_tokens):
    ids = torch.tensor(TOK.encode(prompt))
    return TOK.decode(generate(model, [ids], max_tokens)[0].tolist())


def sse_chunks(resp):
    """Checks the SSE framing and returns the JSON chunks before [DONE]."""
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    blocks = resp.text.split("\n\n")
    assert blocks[-1] == ""                         # every event ends with a blank line
    assert all(b.startswith("data: ") for b in blocks[:-1])
    data = [b.removeprefix("data: ") for b in blocks[:-1]]
    assert data[-1] == "[DONE]"
    return [json.loads(d) for d in data[:-1]]


def test_stream_format_and_text(client, tiny_model):
    resp = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 6})
    chunks = sse_chunks(resp)

    assert len({c["id"] for c in chunks}) == 1      # one id for the whole stream
    assert re.fullmatch(r"cmpl-\d+", chunks[0]["id"])
    for c in chunks:
        assert c["object"] == "text_completion"
        assert len(c["choices"]) == 1 and c["choices"][0]["index"] == 0
    reasons = [c["choices"][0]["finish_reason"] for c in chunks]
    assert reasons[:-1] == [None] * (len(chunks) - 1) and reasons[-1] == "length"
    text = "".join(c["choices"][0]["text"] for c in chunks)
    assert text == expected_text(tiny_model, "hello", 6)


def test_stop_string_is_cut_from_the_text(client, tiny_model):
    full = expected_text(tiny_model, "world", 12)
    stop = full[5:7]
    resp = client.post("/v1/completions",
                       json={"prompt": "world", "max_tokens": 12, "stop": stop})
    chunks = sse_chunks(resp)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert "".join(c["choices"][0]["text"] for c in chunks) == full[: full.find(stop)]


@pytest.mark.parametrize("bad", [{"max_tokens": 0}, {"temperature": -1}, {"top_p": 0},
                                 {"top_k": -1}])
def test_invalid_params_are_422(client, bad):
    assert client.post("/v1/completions", json={"prompt": "abc"} | bad).status_code == 422


def test_request_that_does_not_fit_is_400(client):
    resp = client.post("/v1/completions", json={"prompt": "a" * 60, "max_tokens": 10})
    assert resp.status_code == 400
    assert "max_model_len" in resp.json()["detail"]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200 and resp.json()["status"] == "ok"
    assert resp.json()["config"]["max_num_batched_tokens"] == 64



def test_client_disconnect_aborts_and_frees_the_cache(engine, server_url, monkeypatch):
    step = engine.step

    def slow_step():                            # 50 tokens take 2.5 s: we hang up long before
        time.sleep(0.05)
        return step()

    aborted, abort = [], engine.abort_request
    monkeypatch.setattr(engine, "step", slow_step)
    monkeypatch.setattr(engine, "abort_request", lambda r: (aborted.append(r), abort(r)))

    with httpx.stream("POST", f"{server_url}/v1/completions",
                      json={"prompt": "ab", "max_tokens": 50}) as resp:
        assert resp.status_code == 200
        next(resp.iter_lines())                 # the first token arrived: the request is running
    # leaving the `with` closes the connection mid-stream

    wait_until(lambda: aborted)
    r = aborted[0]
    wait_until(lambda: r.state is State.FINISHED)
    assert r.finish_reason == "abort" and r.cache is None and len(r.output_ids) < 50
    pool = engine.scheduler.pool
    wait_until(lambda: pool.num_free == pool.size and not engine.scheduler.running)
