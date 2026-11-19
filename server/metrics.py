import time

from prometheus_client import Counter, Gauge, Histogram

QUEUE_DEPTH = Gauge("queue_depth", "Requests waiting to be admitted")
RUNNING = Gauge("running_sequences", "Sequences in the batch")
TOKENS = Counter("tokens_generated", "Output tokens; tokens/s = rate(tokens_generated_total[1m])")
TTFT = Histogram("ttft_seconds", "Request arrival to first token",
                 buckets=(.025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60))
TPOT = Histogram("tpot_seconds", "Mean time per output token after the first",
                 buckets=(.005, .01, .02, .03, .05, .075, .1, .25, .5, 1))


def watch(engine):
    """Gauges read engine state at scrape time, so the engine itself needs no metric code"""
    QUEUE_DEPTH.set_function(lambda: engine.inbox.qsize() + len(engine.scheduler.waiting))
    RUNNING.set_function(lambda: len(engine.scheduler.running))


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
        if done and self.n > 1:         # TOPT is undefined for a 1-token output
            TPOT.observe((now- self.first) / (self.n - 1))