import copy
from dataclasses import replace

import pytest
import torch

from engine.cache.kv_cache import Segment
from engine.config import EngineConfig
from engine.cuda_graph import GraphRunner
from engine.engine import Engine
from engine.request import Request

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")


def test_decode_only_inputs_match_forward(tiny_model):
    """run() with rows and keep None (the form the CUDA graph uses) gives the same logits as
    forward() on an all-decode step. 17 tokens puts one sequence across a block boundary"""
    torch.manual_seed(0)
    cache = tiny_model.new_cache(8, 16)
    tables, lens = [[0, 1], [2, 3], [4, 5]], [5, 17, 9]
    with torch.inference_mode():
        prompt = torch.randint(1000, (sum(lens),))
        tiny_model(prompt, [Segment(t, 0, n) for t, n in zip(tables, lens)], cache)

        ids = torch.randint(1000, (3,))
        segs = [Segment(t, n, 1) for t, n in zip(tables, lens)]
        eager = tiny_model(ids, segs, cache)
        inp = tiny_model.prepare(segs, ids.device, cache.bs)
        graph_form = replace(inp, segments=[], batch=(None, *inp.batch[1:]), keep=None)
        got = tiny_model.run(ids, graph_form, cache)

    assert torch.equal(got, eager)


@pytest.fixture(scope="module")
def gpu_tiny(tiny_model):
    return copy.deepcopy(tiny_model).cuda()     # fp32: graph vs eager should match exactly


@cuda
def test_replay_matches_the_same_step_launched_op_by_op(gpu_tiny):
    """5 sequences padded to bucket 8: replay gives the logits run() gives on the same buffers"""
    torch.manual_seed(0)
    cache = gpu_tiny.new_cache(11, 16)                   # block 10 is the dummy
    runner = GraphRunner(gpu_tiny, cache, max_batch=8, max_blocks=4, dummy_block=10)
    tables, lens = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], [5, 17, 9, 30, 2]
    with torch.inference_mode():
        prompt = torch.randint(1000, (sum(lens),), device="cuda")
        gpu_tiny(prompt, [Segment(t, 0, n) for t, n in zip(tables, lens)], cache)
        segs = [Segment(t, n, 1) for t, n in zip(tables, lens)]
        got = runner.run(torch.randint(1000, (5,)).tolist(), segs).clone()
        want = runner._forward(8)[:5]                    # same buffers, eager

    assert runner.replays == 1
    assert torch.equal(got, want)


@cuda
@pytest.mark.parametrize("n", [4, 5])                    # on a bucket; padded 5 -> 8
def test_engine_with_graphs_matches_eager(gpu_tiny, n):
    """Requests finish at different steps, so the batch moves across buckets; 40 tokens on
    prompts up to 19 cross block boundaries mid-run"""
    g = torch.Generator().manual_seed(n)
    prompts = [torch.randint(1000, (int(torch.randint(2, 20, (1,), generator=g)),), generator=g).tolist()
               for _ in range(n)]

    def run(graphs):
        e = Engine(gpu_tiny, EngineConfig(max_num_seqs=8, max_num_batched_tokens=256,
                                          max_model_len=64, cuda_graphs=graphs))
        reqs = [Request(i, p, max_tokens=20 + 5 * i, ignore_eos=True) for i, p in enumerate(prompts)]
        for r in reqs:
            e.add_request(r)
        e.run_until_idle()
        return [r.output_ids for r in reqs], e.graphs

    eager, _ = run(False)
    graphed, runner = run(True)
    assert runner.replays > 0
    assert graphed == eager
    