import time

from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_client.core import CounterMetricFamily

QUEUE_DEPTH = Gauge("queue_depth", "Requests waiting to be admitted")
RUNNING = Gauge("running_sequences", "Sequences in the batch")
TOKENS = Counter("tokens_generated", "Output tokens; tokens/s = rate(tokens_generated_total[1m])")
TTFT = Histogram("ttft_seconds", "Request arrival to first token",
                 buckets=(.025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60))
TPOT = Histogram("tpot_seconds", "Mean time per output token after the first",
                 buckets=(.005, .01, .02, .03, .05, .075, .1, .25, .5, 1))
KV_FREE = Gauge("kv_free_blocks", "KV cache blocks not owned by any sequence")
KV_TOTAL = Gauge("kv_total_blocks", "KV cache blocks in the pool")
SPEC_RATE = Histogram("spec_acceptance_rate", "Per request: accepted / drafted draft tokens",
                      buckets=(.1, .2, .3, .4, .5, .6, .7, .8, .9, 1))


class _EngineCollector:
    engine = None
    def collect(self):
        e = self.engine
        if e is not None:
            yield CounterMetricFamily("preemptions", "Sequences evicted to free KV blocks",
                                      value=e.scheduler.preemptions)
            yield CounterMetricFamily("spec_draft_tokens", "Draft tokens sent for verification",
                                      value=e.spec_drafted)
            yield CounterMetricFamily("spec_accepted_tokens", "Draft tokens the target accepted",
                                      value=e.spec_accepted)
            



_COLLECTOR = _EngineCollector()
REGISTRY.register(_COLLECTOR)


def watch(engine):
    """Gauges read engine state at scrape time, so the engine itself needs no metric code"""
    QUEUE_DEPTH.set_function(lambda: engine.inbox.qsize() + len(engine.scheduler.waiting))
    RUNNING.set_function(lambda: len(engine.scheduler.running))
    KV_FREE.set_function(lambda: len(engine.scheduler.blocks.free))
    KV_TOTAL.set_function(lambda: engine.scheduler.blocks.num_blocks)
    _COLLECTOR.engine = engine


class RequestTimer:
    """One per request. token() runs on the engine thread, prometheus_client is thread-safe"""

    def __init__(self):
        self.start = time.perf_counter()
        self.first = None
        self.n = 0

    def token(self, done):
        now = time.perf_counter()
        self.n += 1
        TOKENS.inc()
        if self.first is None:
            self.first = now
            TTFT.observe(now - self.start)
        if done and self.n > 1:         # TPOT is undefined for a 1-token output
            TPOT.observe((now- self.first) / (self.n - 1))


    def finish(self, r):
        if r.spec_drafted:
            SPEC_RATE.observe(r.spec_accepted / r.spec_drafted)
    