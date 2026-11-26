from collections import deque
from queue import Queue
from types import SimpleNamespace

from prometheus_client import generate_latest

from server.metrics import watch


def test_watch_exports_blocks_and_preemptions():
    blocks = SimpleNamespace(free=deque(range(7)), num_blocks=10)
    sched = SimpleNamespace(blocks=blocks, preemptions=3, waiting=deque(), running=[])
    watch(SimpleNamespace(scheduler=sched, inbox=Queue(), spec_drafted=8, spec_accepted=5))

    text = generate_latest().decode()
    assert "kv_free_blocks 7.0" in text
    assert "kv_total_blocks 10.0" in text
    assert "preemptions_total 3.0" in text
    assert "spec_draft_tokens_total 8.0" in text
    assert "spec_accepted_tokens_total 5.0" in text

    sched.preemptions = 5               # read at scrape time, not copied at watch()
    blocks.free.popleft()
    text = generate_latest().decode()
    assert "preemptions_total 5.0" in text
    assert "kv_free_blocks 6.0" in text