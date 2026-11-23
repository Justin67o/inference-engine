import asyncio
import json

import httpx

from bench.loadgen import one_request


def chunk(text, finish_reason=None):
    return "data: " + json.dumps({"choices": [{"text": text, "finish_reason": finish_reason}]}) + "\n\n"


def send(handler):
    """One request through bench.loadgen against a fake server; returns its results entry"""
    results = []

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await one_request(client, "http://test/v1/completions", "hi", 2, results)

    asyncio.run(go())
    return results[0]


def test_ok_stream_counts_tokens():
    r = send(lambda req: httpx.Response(200, text=chunk("a") + chunk("b", "length") + "data: [DONE]\n\n"))
    assert r["ok"] and len(r["token_times"]) == 2


def test_error_finish_marks_the_request_failed():
    r = send(lambda req: httpx.Response(200, text=chunk("a") + chunk("", "error") + "data: [DONE]\n\n"))
    assert not r["ok"] and len(r["token_times"]) == 1     # the error chunk is not a token


def test_dropped_connection_fails_one_request_not_the_run():
    def handler(req):
        raise httpx.ReadError("connection reset")
    r = send(handler)
    assert not r["ok"] and len(r["token_times"]) == 1     # the placeholder time


def test_http_error_status_is_not_ok():
    assert not send(lambda req: httpx.Response(503, json={"detail": "engine stopped"}))["ok"]
