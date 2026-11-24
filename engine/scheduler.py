from collections import deque

from engine.request import State


class Scheduler:
    """Decides, before every step, which sequences run and how many tokens each feeds in."""


    def __init__(self, cfg, blocks):
        self.cfg, self.blocks = cfg, blocks
        self.waiting = deque()          # arrival order: preempted requests go back to the front
        self.running = []               # admission order: oldest first
        self.watermark = int(cfg.watermark * blocks.num_blocks)
        self.preemptions = 0            # total evictions (for tests and metrics)

    def check(self, r):
        """Raises ValueError for a request that could never run."""
        total = len(r.prompt_ids) + r.max_tokens
        if total > self.cfg.max_model_len:
            raise ValueError("prompt + max_tokens is longer than max_model_len")
        if total - 1 > self.cfg.max_num_batched_tokens:     # a preempted request re-prefills all but its last token
            raise ValueError("prompt + max_tokens is longer than max_num_batched_tokens")
        if self.blocks.blocks_for(total) > self.blocks.num_blocks - self.watermark:
            raise ValueError("request needs more KV blocks than the cache has")

    def add(self, r):
        self.check(r)
        self.waiting.append(r)


    def lookahead(self, r):
        """Draft room for r: verify returns up to k+1 tokens, so never draft past max_tokens"""
        return max(0, min(self.cfg.num_speculative_tokens, r.max_tokens - len(r.output_ids) - 1))

    
    def schedule(self):
        """Returns (request, start, n) for each sequence in this step: decodes first, then prefills.
        A decode that can't get a block preempts the newest running sequence, possibly itself"""

        segs, budget = [], self.cfg.max_num_batched_tokens
        queue, preempted = deque(self.running), False

        while queue:
            r = queue.popleft()     # oldest first
            k = self.lookahead(r)

            if not self.blocks.ensure(r, r.length + k):
                k = 0       # drafts are optional, drop before evicting anyone
                while not self.blocks.ensure(r, r.length + k):
                    victim = queue.pop() if queue else r    # newest first, r itself last
                    self.preempt(victim)
                    preempted = True
                    if victim is r:
                        break
            if r.state is State.RUNNING:
                segs.append((r, r.length - 1, 1 + k))       # n is the most tokens r may feed in
                budget -= 1 + k

        if preempted:           # blocks are short, anything admitted now would be evicted
            return segs

        while self.waiting and len(self.running) < self.cfg.max_num_seqs:
            r = self.waiting[0]
            
            if r.length > budget or not self.can_admit(r):
                break  

            self.waiting.popleft()
            self.blocks.ensure(r, r.length)
            r.state = State.RUNNING
            self.running.append(r)
            segs.append((r, 0, r.length))   # prompt, plus any output from before a preemption
            budget -= r.length
        return segs


    def can_admit(self, r):
        """True if r's prefill fits while leaving watermark free"""
        return self.blocks.blocks_for(r.length) <= len(self.blocks.free) - self.watermark


    def preempt(self, r):
        """Frees r's blocks and puts it at the front of the queue. It keeps output_ids, so it
        comes back with a prefill of prompt + output (recompute)"""
        self.preemptions += 1
        self.running.remove(r)
        self.blocks.release(r)
        r.draft_len = 0     # the draft's K/V lived in those blocks too
        r.state = State.WAITING
        self.waiting.appendleft(r)


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
        
