"""CUDA graphs for decode-only steps. GPU idles between launches
in our findings, so a graph records one step's launches per batch 
bucket and replays them all with one launch, reducing overhead"""
import torch

from engine.model.model import StepInputs

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


class GraphRunner:
    """Copies each step's inputs into static buffers, then replays the smallest bucket that
    holds the batch. Rows past the real batch are padding, they write to and read from a
    dummy block that no sequence owns, and their logits are dropped"""


    def __init__(self, model, cache, max_batch, max_blocks, dummy_block):
        self.model, self.cache, self.dummy = model, cache, dummy_block
        self.buckets = [b for b in BUCKETS if b < max_batch] + [max_batch]
        self.max_blocks, self.replays = max_blocks, 0
        dev, bs = model.embed_tokens.weight.device, cache.bs
        self.ids = torch.zeros(max_batch, dtype=torch.long, device=dev)
        self.positions = torch.zeros(max_batch, dtype=torch.long, device=dev)
        self.write = torch.full((max_batch,), dummy_block * bs, dtype=torch.long, device=dev)
        self.lens = torch.ones(max_batch, dtype=torch.long, device=dev)
        # fixed width, so tables.stride(0) (a Python int at kernel launch) is the same on replay
        self.tables = torch.full((max_batch, max_blocks), dummy_block, dtype=torch.long, device=dev)
        for kv in (cache.k, cache.v):           # padding rows read it, so keep it finite
            kv[:, dummy_block * bs:(dummy_block + 1) * bs] = 0
        self.graphs, self.out = {}, {}
        self._capture()


    def _forward(self, b):
        """The first b rows of the buffers through run(), decode-only form"""
        batch = (None, self.tables[:b], self.lens[:b], self.max_blocks * self.cache.bs)
        inp = StepInputs(self.positions[:b], self.write[:b], [], batch, [], None)
        return self.model.run(self.ids[:b], inp, self.cache)

    @torch.inference_mode()
    def _capture(self):
        # warm up on a side stream first: Triton compiles, cuBLAS picks its kernels
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for b in self.buckets:
                self._forward(b)
        torch.cuda.current_stream().wait_stream(s)

        pool = None
        for b in reversed(self.buckets):        # largest first; the rest share its memory pool
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self.out[b] = self._forward(b)
            pool = g.pool()
            self.graphs[b] = g

    def fits(self, segments):
        return len(segments) <= self.buckets[-1] and all(len(s.table) <= self.max_blocks for s in segments)

    def run(self, ids, segments):
        """ids: one token per segment, every segment a decode (start > 0, n == 1).
        Returns [len(segments), vocab] logits: a view the next replay overwrites"""
        n, bs = len(segments), self.cache.bs
        b = next(b for b in self.buckets if b >= n)
        w = max(len(s.table) for s in segments)
        self.ids[:n].copy_(torch.tensor(ids))
        self.positions[:n].copy_(torch.tensor([s.start for s in segments]))
        self.write[:n].copy_(torch.tensor([s.table[s.start // bs] * bs + s.start % bs for s in segments]))
        self.lens[:n].copy_(torch.tensor([s.start + 1 for s in segments]))
        # columns past w hold stale ids from earlier steps; lens stops the kernel before them
        self.tables[:n, :w].copy_(torch.tensor([s.table + [self.dummy] * (w - len(s.table)) for s in segments]))
        # padding rows: undo whatever a bigger batch left there
        self.write[n:b] = self.dummy * bs
        self.lens[n:b] = 1
        self.tables[n:b, 0] = self.dummy
        self.graphs[b].replay()
        self.replays += 1
        return self.out[b][:n]