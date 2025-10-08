import pytest

from mini_rsa.codec import (
    PUNCTUATION_MAP,
    decrypt_numbers,
    encrypt_text,
    map_character,
    map_number,
    tokenize_cipher_text,
)
from mini_rsa.core import _validate_block  # ensure module import


def test_map_character_letter_space_and_punctuation() -> None:
    assert map_character("A", include_punctuation=False) == (True, 1)
    assert map_character(" ", include_punctuation=False) == (True, 32)
    for symbol, value in PUNCTUATION_MAP.items():
        assert map_character(symbol, include_punctuation=True) == (True, value)


def test_map_character_rejects_unknown_symbol() -> None:
    assert map_character("@", include_punctuation=False) == (False, None)
    assert map_character("@", include_punctuation=True) == (False, None)


def test_map_number_round_trip_with_punctuation() -> None:
    assert map_number(1, include_punctuation=True) == "A"
    assert map_number(32, include_punctuation=False) == " "
    for symbol, value in PUNCTUATION_MAP.items():
        assert map_number(value, include_punctuation=True) == symbol
        assert map_number(value, include_punctuation=False) == ""
    assert map_number(999, include_punctuation=True) == ""


def test_tokenize_cipher_text_filters_invalid_tokens() -> None:
    assert tokenize_cipher_text("10 abc 42\t-7 99") == [10, 42, -7, 99]
    assert tokenize_cipher_text("") == []


def test_encrypt_text_uses_callback_and_produces_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_encrypt_block(value: int, e: int, n: int, use_large_numbers: bool) -> int:
        calls.append((value, e, n, use_large_numbers))
        return value + 1

    result = encrypt_text(
        text="HI!",
        encrypt_block_fn=fake_encrypt_block,
        include_punctuation=True,
        use_large_numbers=True,
        n=33,
        e=3,
    )
    assert result.cipher_blocks == [8 + 1, 9 + 1, PUNCTUATION_MAP["!"] + 1]
    assert result.plain_blocks == [8, 9, PUNCTUATION_MAP["!"]]
    assert result.skipped_characters == []
    assert len(result.trace) == len(result.cipher_blocks)
    assert all(isinstance(entry, str) and "Encrypting" in entry for entry in result.trace)
    assert calls == [
        (8, 3, 33, True),
        (9, 3, 33, True),
        (PUNCTUATION_MAP["!"], 3, 33, True),
    ]


def test_encrypt_text_records_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    result = encrypt_text(
        text="A@B",
        encrypt_block_fn=lambda v, *_: v,
        include_punctuation=False,
        use_large_numbers=False,
        n=33,
        e=3,
    )
    assert result.plain_blocks == [1, 2]
    assert result.skipped_characters == ["@"]


def test_encrypt_text_validation_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="message_block must be less than modulus n"):
        encrypt_text(
            text="g",
            encrypt_block_fn=lambda v, *_: v,
            include_punctuation=False,
            use_large_numbers=False,
            n=5,
            e=3,
        )


def test_decrypt_numbers_validation_message() -> None:
    with pytest.raises(ValueError, match="cipher_block must be less than modulus n"):
        decrypt_numbers([10], lambda *args, **kwargs: 0, use_large_numbers=False, d=7, n=5)


def test_decrypt_numbers_passes_pow(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_decrypt_block(cipher: int, d: int, n: int, use_large_numbers: bool) -> int:
        calls.append((cipher, d, n, use_large_numbers))
        return cipher - 1

    numbers = decrypt_numbers([10, 11], fake_decrypt_block, use_large_numbers=True, d=7, n=33)
    assert numbers == [9, 10]
    assert calls == [(10, 7, 33, True), (11, 7, 33, True)]
