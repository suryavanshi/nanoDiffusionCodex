"""Basic smoke tests for tokenizer and config-free components."""

from nano_diffusion.data.tokenizer import ByteTokenizer


def test_smoke() -> None:
    assert True


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    ids, attention_mask = tokenizer.encode("def f(x):\n    return x + 1\n", max_length=64)

    assert len(ids) == 64
    assert len(attention_mask) == 64
    assert ids[0] == tokenizer.bos_token_id
    assert tokenizer.decode(ids).startswith("def f")
