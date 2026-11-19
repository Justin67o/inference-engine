import pytest

from server.detokenizer import Detokenizer, StopStringFilter

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


def run_filter(stop, pieces):
    f = StopStringFilter(stop)
    out = [f.push(p) for p in pieces]
    return out, f.flush()


def test_stop_string_split_across_pieces_is_removed():
    out, rest = run_filter(["\n\n"], ["Hello", "\n", "\n"])
    assert "".join(out) + rest == "Hello"


def test_partial_match_is_held_then_released():
    out, rest = run_filter(["STOP"], ["a S", "T", "x"])
    assert out == ["a ", "", "STx"]
    assert rest == ""


def test_unfinished_prefix_released_on_flush():
    out, rest = run_filter(["STOP"], ["end ST"])   # e.g. finished on max_tokens
    assert out == ["end "]
    assert rest == "ST"


def test_text_after_stop_is_dropped():
    out, rest = run_filter(["."], ["Hi. More", " text"])
    assert out == ["Hi", ""]
    assert rest == ""


def test_earliest_of_several_stop_strings_wins():
    out, _ = run_filter(["world", "lo"], ["hello world"])
    assert out == ["hel"]


def test_no_stop_strings_passes_through():
    out, rest = run_filter([], ["a", "b"])
    assert out == ["a", "b"] and rest == ""
