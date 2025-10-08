import math

import pytest

from mini_rsa import core
from mini_rsa.core import (
    MIN_MODULUS,
    MIN_PRIME_SIZE,
    MIN_SECURE_PRIME,
    PUNCTUATION_MAP,
    calculate_entropy,
    decrypt_numbers,
    ensure_coprime,
    encrypt_text,
    encrypt_block,
    encrypt_with_reference,
    gcd,
    is_prime,
    map_character,
    map_number,
    mod_inverse,
    tokenize_cipher_text,
    validate_entropy_bounds,
    validate_prime_pair,
)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (0, 0, 0),
        (54, 24, 6),
        (24, 54, 6),
        (-24, 54, 6),
        (17, 31, 1),
    ],
)
def test_gcd(a: int, b: int, expected: int) -> None:
    assert gcd(a, b) == expected


def test_mod_inverse_round_trip() -> None:
    phi = (3 - 1) * (11 - 1)
    inv = mod_inverse(3, phi)
    assert (3 * inv) % phi == 1


def test_mod_inverse_failure() -> None:
    with pytest.raises(ValueError):
        mod_inverse(2, 20)


def test_mod_inverse_invalid_phi() -> None:
    with pytest.raises(ValueError):
        mod_inverse(3, 0)


def test_mod_inverse_without_sympy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", False)
    monkeypatch.setattr(core, "sympy_mod_inverse", None)
    assert mod_inverse(3, 20) == 7
    with pytest.raises(ValueError):
        mod_inverse(4, 20)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2, True), (17, True), (1, False), (9, False), (97, True), (100, False)],
)
def test_is_prime(value: int, expected: bool) -> None:
    assert is_prime(value) is expected


def test_is_prime_without_sympy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", False)
    monkeypatch.setattr(core, "sympy_isprime", None)
    assert is_prime(17) is True
    assert is_prime(25) is False
    assert is_prime(2) is True
    assert is_prime(4) is False


def test_validate_prime_pair_happy_path() -> None:
    errors, warnings = validate_prime_pair(3, 11)
    assert errors == []
    assert warnings == [f"For better security, use primes ≥ {MIN_SECURE_PRIME}"]


def test_validate_prime_pair_errors() -> None:
    errors, warnings = validate_prime_pair(4, 4)
    assert "4 is not prime" in errors
    assert "p and q must be different primes" in errors
    assert any("at least" in error for error in errors)
    assert warnings == [f"For better security, use primes ≥ {MIN_SECURE_PRIME}"]


def test_validate_prime_pair_too_small() -> None:
    errors, _ = validate_prime_pair(1, 3)
    assert f"at least {MIN_PRIME_SIZE}" in " ".join(errors)


def test_calculate_entropy_matches_log2() -> None:
    modulus = MIN_MODULUS
    expected = math.log2(modulus)
    assert calculate_entropy(modulus) == pytest.approx(expected)


def test_calculate_entropy_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        calculate_entropy(0)


def test_encrypt_block_matches_pow() -> None:
    modulus = 3 * 11
    message = 4
    e = 3
    assert encrypt_block(message, e, modulus) == pow(message, e, modulus)


def test_encrypt_block_large_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    if not core.SYMPY_AVAILABLE:
        pytest.skip("Sympy not available for large number path")
    result = encrypt_block(4, 3, 33, use_large_numbers=True)
    assert result == pow(4, 3, 33)


def test_map_character_and_number_round_trip_space_and_punctuation() -> None:
    for punctuation, value in PUNCTUATION_MAP.items():
        accepted, mapped = map_character(punctuation, include_punctuation=True)
        assert accepted and mapped == value
        assert map_number(value, include_punctuation=True) == punctuation

    accepted, mapped = map_character(" ", include_punctuation=False)
    assert accepted and mapped == 32
    assert map_number(32, include_punctuation=False) == " "

    accepted, mapped = map_character("$", include_punctuation=False)
    assert not accepted and mapped is None

    assert map_number(999, include_punctuation=True) == ""


def test_decrypt_numbers_matches_encrypt_text() -> None:
    p, q, e = 5, 13, 5
    n = p * q
    phi = (p - 1) * (q - 1)
    d = mod_inverse(e, phi)
    message = "HELLO"
    result = encrypt_text(message, e, n, include_punctuation=False)
    decrypted_numbers = decrypt_numbers(result.cipher_blocks, d, n)
    assert decrypted_numbers == result.plain_blocks


def test_decrypt_text_blocks_full_round_trip() -> None:
    p, q, e = 5, 13, 5
    n = p * q
    phi = (p - 1) * (q - 1)
    d = mod_inverse(e, phi)
    message = "HELLO WORLD!"
    result = encrypt_text(message, e, n, include_punctuation=True)
    decrypted_text = core.decrypt_text_blocks(
        result.cipher_blocks,
        d,
        n,
        include_punctuation=True,
    )
    expected_plaintext = "".join(map_number(num, True) for num in result.plain_blocks)
    assert decrypted_text == expected_plaintext


def test_tokenize_cipher_text_filters_invalid_tokens() -> None:
    cipher_numbers = tokenize_cipher_text("10 abc 42\t-7 99")
    assert cipher_numbers == [10, 42, -7, 99]


def test_validate_entropy_bounds_success_and_failure() -> None:
    modulus = 97 * 101
    validate_entropy_bounds(modulus, expected_min_bits=5)
    with pytest.raises(ValueError):
        validate_entropy_bounds(modulus, expected_min_bits=20)


def test_ensure_coprime_checks() -> None:
    ensure_coprime(3, 20)
    with pytest.raises(ValueError):
        ensure_coprime(4, 20)


def test_encrypt_with_reference_matches_encrypt_text() -> None:
    p, q, e = 3, 11, 3
    n = p * q
    message = "TEST"
    reference = encrypt_with_reference(message, e, n)
    result = encrypt_text(message, e, n, include_punctuation=True)
    assert reference == result.cipher_blocks


def test_generate_secure_primes_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "CRYPTO_AVAILABLE", False)
    monkeypatch.setattr(core, "number", None)

    choices = []

    def fake_choice(seq):
        value = seq[len(choices)]
        choices.append(value)
        return value

    monkeypatch.setattr(core.random, "choice", fake_choice)
    p, q = core.generate_secure_primes(16)
    assert p != q
    assert p in choices and q in choices


class DummyNumber:
    def __init__(self):
        self.calls = 0

    def getPrime(self, bits: int) -> int:  # noqa: N802 - mimics library API
        self.calls += 1
        if self.calls == 1:
            return 101
        return 103


def test_generate_secure_primes_crypto_path(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyNumber()
    monkeypatch.setattr(core, "CRYPTO_AVAILABLE", True)
    monkeypatch.setattr(core, "number", dummy)
    p, q = core.generate_secure_primes(16)
    assert p == 101 and q == 103


def test_generate_secure_primes_requires_positive_bits() -> None:
    with pytest.raises(ValueError):
        core.generate_secure_primes(0)
