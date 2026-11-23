from collections import deque

from engine.request import State


class Scheduler:
    """Decides, before every step, which sequences run and how many tokens each feeds in."""


    def __init__(self, cfg, blocks):
        self.cfg, self.blocks = cfg, blocks
        self.waiting = deque()          # arrival order: first in, first admitted
        self.running = []

    def check(self, r):
        """Raises ValueError for a request that could never run."""
        if len(r.prompt_ids) > self.cfg.max_num_batched_tokens:
            raise ValueError("prompt is longer than max_num_batched_tokens")
        if len(r.prompt_ids) + r.max_tokens > self.cfg.max_model_len:
            raise ValueError("prompt + max_tokens is longer than max_model_len")
        if self.blocks.blocks_for(len(r.prompt_ids) + r.max_tokens) > self.blocks.num_blocks:
            raise ValueError("request needs more KV blocks than the cache has")

    def add(self, r):
        self.check(r)
        self.waiting.append(r)

    def schedule(self):
        """Returns (request, start, n) for each sequence in this step: decodes first, then prefills."""

        segs, budget = [], self.cfg.max_num_batched_tokens
        for r in self.running:          # every running sequence decodes 1 token
            if not self.blocks.ensure(r, r.length):
                raise RuntimeError("out of KV blocks for a running sequence (needs preemption)")
            segs.append((r, r.length -1, 1))
            budget -= 1

        while(self.waiting
              and len(self.running) < self.cfg.max_num_seqs
              and len(self.waiting[0].prompt_ids) <= budget
              and self.blocks.ensure(self.waiting[0], len(self.waiting[0].prompt_ids))):
            r = self.waiting.popleft()
            r.state = State.RUNNING

            self.running.append(r)
            segs.append((r, 0, len(r.prompt_ids)))      # prefill the whole prompt
            budget -= len(r.prompt_ids)
        return segs

    def finish(self, r):
        r.state = State.FINISHED
        self.running.remove(r)
        self.blocks.release(r)

    def abort(self, r):
        """Drops r wherever it is. A no-op if it has already finished."""
        if r.state is State.WAITING:
            if r in self.waiting:
                self.waiting.remove(r)
            r.state = State.FINISHED
        elif r.state is State.RUNNING:
            self.finish(r)      # frees the cache, same as a normal finish
        r.finish_reason = r.finish_reason or "abort"
        
