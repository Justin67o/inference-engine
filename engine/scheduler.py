from collections import deque

from engine.request import State


class Scheduler:
    """Decides, before every step, which sequences run and how many tokens each feeds in."""


    def __init__(self, cfg, pool):
        self.cfg, self.pool = cfg, pool
        self.waiting = deque()          # arrival order: first in, first admitted
        self.running = []

    def check(self, r):
        """Raises ValueError for a request that could never run."""
        if len(r.prompt_ids) > self.cfg.max_num_batched_tokens:
            raise ValueError("prompt is longer than max_num_batched_tokens")
        if len(r.prompt_ids) + r.max_tokens > self.cfg.max_model_len:
            raise ValueError("prompt + max_tokens is longer than max_model_len")

    def add(self, r):
        self.check(r)
        self.waiting.append(r)

    def schedule(self):
        """Returns (request, start, n) for each sequence in this step: decodes first, then prefills."""

        segs, budget = [], self.cfg.max_num_batched_tokens
        for r in self.running:          # every running sequence decodes 1 token
            segs.append((r, r.length -1, 1))
            budget -= 1

        while(self.waiting
              and len(self.running) < self.cfg.max_num_seqs
              and len(self.waiting[0].prompt_ids) <= budget
              and self.pool.has_free()):
            r = self.waiting.popleft()
            r.state, r.cache = State.RUNNING, self.pool.take()

            self.running.append(r)
            segs.append((r, 0, len(r.prompt_ids)))      # prefill the whole prompt
            budget -= len(r.prompt_ids)
        return segs

    def finish(self, r):
        r.state = State.FINISHED
        self.running.remove(r)
        self.pool.give_back(r.cache)
        r.cache = None

