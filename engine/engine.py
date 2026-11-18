import queue

import torch

from engine.cache.kv_cache import CachePool, Segment
from engine.model.loader import load_model
from engine.sampler import sample
from engine.scheduler import Scheduler


class Engine:
    """Continuous batching: before every step the scheduler picks which sequences run, then one
    forward pass advances each of them by one token. Requests arrive through `inbox`."""

    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.device = model.embed_tokens.weight.device
        pool = CachePool(lambda: model.new_cache(cfg.max_model_len), cfg.max_num_seqs)
        self.scheduler = Scheduler(cfg, pool)
        self.inbox = queue.Queue()          # thread-safe: the server puts, the engine gets
        self._stop = False

    @classmethod
    def from_config(cls, cfg):
        model, _ = load_model(cfg.model)
        return cls(model, cfg)

    def add_request(self, r):
        """Safe to call from any thread. Raises ValueError for a request that could never run."""
        self.scheduler.check(r)         # check here, so the error reaches the caller
        if r.seed is not None:
            r.generator = torch.Generator(device=self.device).manual_seed(r.seed)
        self.inbox.put(r)

    @torch.inference_mode()
    def step(self):
        """Admits new arrivals, runs one forward pass and samples one token per sequence.
        Returns False if there was nothing to run."""
        self._take_new()
        segs = self.scheduler.schedule()
        if not segs:
            return False

        ids = []
        for r, start, n in segs:        # prefill: the whole prompt; decode: newest token
            ids += r.prompt_ids if start == 0 else [r.output_ids[-1]]
        logits = self.model(torch.tensor(ids, device=self.device),
                            [Segment(r.cache, start, n) for r, start, n in segs])

        reqs = [r for r, _, _ in segs]
        tokens = sample(logits, [r.temperature for r in reqs], [r.top_k for r in reqs],
                        [r.top_p for r in reqs], [r.generator for r in reqs]).tolist()

        for r, t in zip(reqs, tokens):
            r.output_ids.append(t)
            done = len(r.output_ids) >= r.max_tokens
            if r.on_token is not None:
                r.on_token(t, done)
            if done:
                self.scheduler.finish(r)
        return True

    def run_until_idle(self):
        """Runs steps on the calling thread until every request has finished (tests, offline)."""
        while self.step():
            pass

    def run_forever(self):
        """The engine thread's loop: step while there is work, sleep while idle."""
        while not self._stop:
            if not self.step():
                r = self.inbox.get()        # idle: block until something arrives
                if r is not None:           # None is stop()'s wake-up call
                    self.scheduler.add(r)

    def stop(self):
        """Safe to call from any thread: run_forever returns after its current step."""
        self._stop = True
        self.inbox.put(None)

    def _take_new(self):
        while True:
            try:
                r = self.inbox.get_nowait()
            except queue.Empty:
                return
            if r is not None:
                self.scheduler.add(r)