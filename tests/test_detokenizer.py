import pytest

from server.detokenizer import Detokenizer

TEXTS = [
    "Hello, world! How are you today?",
    "I love pizza 🍕🍕, planets 🪐 and flamingos 🦩!",
    "Family: 👨‍👩‍👧‍👦, flags: 🇨🇦🇯🇵",
    "我爱自然语言处理。今天天气很好！",
    "龘鬱 are rare characters split across tokens",
    "Mixed: café, naïve, 日本語, emoji 😀 end",
]


def stream(tok, ids):
    d = Detokenizer(tok)
    pieces = [d.push(t) for t in ids]
    return pieces, d.flush()


@pytest.mark.parametrize("text", TEXTS)
def test_streamed_pieces_join_to_full_decode(tok, text):
    ids = tok(text).input_ids
    pieces, rest = stream(tok, ids)
    assert "".join(pieces) + rest == tok.decode(ids)
    assert rest == ""


@pytest.mark.parametrize("text", TEXTS)
def test_never_emits_replacement_character(tok, text):
    pieces, _ = stream(tok, tok(text).input_ids)
    assert not any("\ufffd" in p for p in pieces)


@pytest.mark.parametrize("char", ["🪐", "🦩", "龘"])
def test_multi_token_character_is_held_back(tok, char):
    ids = tok(char).input_ids
    assert len(ids) > 1                     # otherwise this test proves nothing
    pieces, _ = stream(tok, ids)
    assert pieces == [""] * (len(ids) - 1) + [char]


def test_special_tokens_are_skipped(tok):
    pieces, rest = stream(tok, tok("Hi").input_ids + [tok.eos_token_id])
    assert "".join(pieces) + rest == "Hi"


def test_flush_returns_unfinished_bytes(tok):
    ids = tok("🪐").input_ids[:-1]          # cut the emoji short
    pieces, rest = stream(tok, ids)
    assert "".join(pieces) == ""
    assert rest.endswith("\ufffd")
