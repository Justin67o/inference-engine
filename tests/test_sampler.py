import math

import torch

from engine.sampler import sample

V = 10


def settings(b, temperature=1.0, top_k=0, top_p=1.0):
    return [temperature] * b, [top_k] * b, [top_p] * b


def probs_logits(*probs):                          # logits whose softmax is exactly these probs
    return torch.tensor([math.log(p) for p in probs])


def test_temperature_zero_is_greedy():
    logits = torch.randn(8, V)
    assert torch.equal(sample(logits, *settings(8, temperature=0.0)), logits.argmax(-1))


def test_top_k_1_is_greedy():
    logits = torch.randn(8, V)
    for _ in range(20):
        assert torch.equal(sample(logits, *settings(8, top_k=1)), logits.argmax(-1))


def test_top_k_only_samples_the_k_most_likely():
    logits = torch.randn(1, V).repeat(1000, 1)
    top3 = set(logits[0].topk(3).indices.tolist())
    assert set(sample(logits, *settings(1000, top_k=3)).tolist()) == top3


def test_top_p_keeps_the_smallest_set_reaching_p():
    logits = probs_logits(0.5, 0.3, 0.2).repeat(1000, 1)
    assert set(sample(logits, *settings(1000, top_p=0.6)).tolist()) == {0, 1}   # 0.5 < 0.6 ≤ 0.8
    assert set(sample(logits, *settings(1000, top_p=0.4)).tolist()) == {0}      # 0.5 ≥ 0.4


def test_top_p_tiny_still_keeps_the_top_token():
    logits = torch.randn(8, V)
    assert torch.equal(sample(logits, *settings(8, top_p=1e-6)), logits.argmax(-1))


def test_temperature_1_matches_the_softmax():
    torch.manual_seed(0)
    logits = probs_logits(0.6, 0.3, 0.1).repeat(20_000, 1)
    counts = torch.bincount(sample(logits, *settings(20_000)), minlength=3) / 20_000
    torch.testing.assert_close(counts, torch.tensor([0.6, 0.3, 0.1]), atol=0.02, rtol=0)


def test_high_temperature_flattens_low_temperature_sharpens():
    torch.manual_seed(0)
    logits = probs_logits(0.6, 0.3, 0.1).repeat(20_000, 1)
    share_top = {t: (sample(logits, *settings(20_000, temperature=t)) == 0).float().mean()
                 for t in (0.5, 1.0, 2.0)}
    assert share_top[0.5] > share_top[1.0] > share_top[2.0]


def test_each_row_uses_its_own_settings():
    logits = probs_logits(0.5, 0.3, 0.2).repeat(3, 1)
    for _ in range(50):                            # greedy, top-k 1 and top-p 0.4 all give token 0
        assert sample(logits, [0.0, 1.0, 1.0], [0, 1, 0], [1.0, 1.0, 0.4]).tolist() == [0, 0, 0]


def test_same_seed_same_tokens():
    logits = torch.randn(1, V)
    runs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(1234)
        runs.append([sample(logits, *settings(1), generators=[g]).item() for _ in range(30)])
    assert runs[0] == runs[1]


def test_seeded_row_ignores_the_rest_of_the_batch():
    logits = torch.randn(4, V)
    g = torch.Generator().manual_seed(1234)
    alone = [sample(logits[2:3], *settings(1), generators=[g]).item() for _ in range(30)]
    g = torch.Generator().manual_seed(1234)
    in_batch = [sample(logits, *settings(4), generators=[None, None, g, None])[2].item()
                for _ in range(30)]
    assert alone == in_batch


def test_seeded_row_ignores_the_rest_of_the_batch_with_top_k():
    logits = torch.randn(4, V)
    g = torch.Generator().manual_seed(1234)
    alone = [sample(logits[2:3], *settings(1, top_k=5), generators=[g]).item() for _ in range(30)]
    g = torch.Generator().manual_seed(1234)
    in_batch = [sample(logits, *settings(4, top_k=5), generators=[None, None, g, None])[2].item()
                for _ in range(30)]
    assert alone == in_batch
