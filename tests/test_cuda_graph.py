from dataclasses import replace

import torch

from engine.cache.kv_cache import Segment


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