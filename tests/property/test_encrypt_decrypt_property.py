from __future__ import annotations

from hypothesis import given, strategies as st

from mini_rsa import core
from mini_rsa.core import EncryptionResult, decrypt_text_blocks, encrypt_text, map_number
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
def test_round_trip_with_allowed_symbols(prime_pair, message: str, include_punctuation: bool) -> None:
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
