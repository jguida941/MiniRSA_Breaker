from __future__ import annotations

import pytest
from hypothesis import assume, given, strategies as st

from mini_rsa import core
from mini_rsa.core import EncryptionResult, decrypt_text_blocks, encrypt_text, mod_inverse
from mini_rsa.codec import map_character, map_number
from rsa import RSACalculator

PRIME_CHOICES = [
    (3, 11),
    (5, 13),
    (7, 19),
    (11, 23),
    (13, 29),
]


def random_prime_pair():
    return st.sampled_from(PRIME_CHOICES)


@given(
    prime_pair=random_prime_pair(),
    message=st.text(
        alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;"),
        min_size=0,
        max_size=64,
    ),
    include_punctuation=st.booleans(),
)
def test_round_trip_with_allowed_symbols(
    prime_pair, message: str, include_punctuation: bool
) -> None:
    p, q = prime_pair
    n = p * q
    e = 3
    phi = (p - 1) * (q - 1)
    if core.gcd(e, phi) != 1:
        e = 5
    d = core.mod_inverse(e, phi)

    result: EncryptionResult = encrypt_text(message, e, n, include_punctuation=include_punctuation)
    decrypted = decrypt_text_blocks(
        result.cipher_blocks,
        d,
        n,
        include_punctuation=include_punctuation,
    )

    expected_plaintext = "".join(
        map_number(block, include_punctuation) for block in result.plain_blocks
    )
    assert decrypted == expected_plaintext
    assert all(block < n for block in result.plain_blocks)


@given(
    message=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            min_codepoint=0,
            max_codepoint=255,
        ),
        min_size=1,
        max_size=128,
    )
)
def test_encryption_reports_skipped_characters(message: str) -> None:
    p, q = 11, 23
    n = p * q
    e = 3

    result = encrypt_text(message, e, n, include_punctuation=False)
    skipped_set = {ch for ch in message if not (ch.isalpha() or ch.isspace())}
    assert set(result.skipped_characters) >= skipped_set
    assert len(result.cipher_blocks) == len(result.plain_blocks)


@given(
    phi=st.integers(min_value=2, max_value=5000),
    e=st.integers(min_value=2, max_value=5000),
)
def test_mod_inverse_property(phi: int, e: int) -> None:
    assume(core.gcd(e, phi) == 1)
    assume(phi > 1)
    inv = mod_inverse(e, phi)
    assert (e * inv) % phi == 1


@given(
    prime_pair=random_prime_pair(),
    message=st.text(
        alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;"),
        min_size=0,
        max_size=48,
    ),
)
def test_core_matches_qt_calculator(prime_pair, message: str) -> None:
    p, q = prime_pair
    n = p * q
    e = 3
    phi = (p - 1) * (q - 1)
    if core.gcd(e, phi) != 1:
        e = 5
    d = core.mod_inverse(e, phi)

    core_result = encrypt_text(message, e, n, include_punctuation=True)
    qt_result = RSACalculator.encrypt_block

    qt_cipher = []
    for char in message:
        if char.isalpha():
            number = ord(char.lower()) - 96
        elif char == " ":
            number = 32
        elif char in core.PUNCTUATION_MAP:
            number = core.PUNCTUATION_MAP[char]
        else:
            continue
        qt_cipher.append(qt_result(number, e, n))

    assert core_result.cipher_blocks == qt_cipher
    decrypted = decrypt_text_blocks(core_result.cipher_blocks, d, n, include_punctuation=True)
    assert decrypted == "".join(map_number(block, True) for block in core_result.plain_blocks)


@given(
    phi=st.integers(min_value=2, max_value=5000),
    e=st.integers(min_value=2, max_value=5000),
)
def test_mod_inverse_rejects_non_coprime(phi: int, e: int) -> None:
    assume(phi > 1)
    assume(core.gcd(e, phi) != 1)
    with pytest.raises(ValueError):
        mod_inverse(e, phi)


@given(
    block=st.integers(min_value=0, max_value=2048),
    modulus=st.integers(min_value=1, max_value=2048),
)
def test_encrypt_block_rejects_block_at_or_above_modulus(block: int, modulus: int) -> None:
    assume(block >= modulus)
    with pytest.raises(ValueError):
        core.encrypt_block(block, e=3, n=modulus)


@given(
    block=st.integers(min_value=0, max_value=2048),
    modulus=st.integers(min_value=1, max_value=2048),
)
def test_decrypt_block_rejects_block_at_or_above_modulus(block: int, modulus: int) -> None:
    assume(block >= modulus)
    with pytest.raises(ValueError):
        core.decrypt_block(block, d=7, n=modulus)


@given(
    cipher_text=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            min_codepoint=0,
            max_codepoint=255,
        ),
        min_size=0,
        max_size=256,
    )
)
def test_tokenize_cipher_text_ignores_non_numeric_tokens(cipher_text: str) -> None:
    expected: list[int] = []
    for token in cipher_text.split():
        try:
            expected.append(int(token))
        except ValueError:
            continue
    assert core.tokenize_cipher_text(cipher_text) == expected


@given(number=st.integers(min_value=1, max_value=26))
def test_map_number_round_trip(number: int) -> None:
    char = map_number(number, include_punctuation=False)
    assert len(char) == 1
    assert map_character(char, include_punctuation=False) == (True, number)


@given(
    a=st.integers(min_value=2, max_value=50),
    b=st.integers(min_value=2, max_value=50),
)
def test_is_prime_rejects_composites(a: int, b: int) -> None:
    composite = a * b
    assume(composite > 1)
    assert not core.is_prime(composite)


@given(
    e=st.integers(min_value=2, max_value=2048),
    phi=st.integers(min_value=2, max_value=2048),
)
def test_ensure_coprime_property(e: int, phi: int) -> None:
    if core.gcd(e, phi) == 1:
        core.ensure_coprime(e, phi)
    else:
        with pytest.raises(ValueError):
            core.ensure_coprime(e, phi)
