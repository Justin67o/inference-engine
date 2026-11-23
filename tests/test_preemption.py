import torch

from engine.config import EngineConfig
from engine.engine import Engine
from engine.request import Request, State

VOCAB = 1000


def prompts(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(VOCAB, (int(torch.randint(2, 20, (1,), generator=g)),), generator=g).tolist()
            for _ in range(n)]


def run(model, ps, num_blocks=None, **sampling):
    """Runs every prompt to 40 tokens; num_blocks=None gives the default, roomy pool"""
    cfg = EngineConfig(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=64)
    engine = Engine(model, cfg, num_blocks=num_blocks)
    reqs = [Request(i, p, max_tokens=40, **sampling) for i, p in enumerate(ps)]
    for r in reqs:
        engine.add_request(r)
    engine.run_until_idle()
    return engine.scheduler, reqs


def test_greedy_outputs_survive_preemption(tiny_model):
    ps = prompts(4)
    _, roomy = run(tiny_model, ps)
    s, tight = run(tiny_model, ps, num_blocks=6)     # ~1.5 requests' worth of blocks
    assert s.preemptions > 0                         # otherwise this tests nothing
    assert [r.output_ids for r in tight] == [r.output_ids for r in roomy]
    assert all(r.state is State.FINISHED for r in tight)
    assert len(s.blocks.free) == s.blocks.num_blocks and not s.running


def test_seeded_sampling_survives_preemption(tiny_model):
    ps = prompts(4, seed=1)
    _, roomy = run(tiny_model, ps, temperature=1.0, seed=7)
    s, tight = run(tiny_model, ps, num_blocks=6, temperature=1.0, seed=7)
    assert s.preemptions > 0
    assert [r.output_ids for r in tight] == [r.output_ids for r in roomy]