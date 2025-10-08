import math

import pytest

from mini_rsa import core
from mini_rsa.codec import (
    PUNCTUATION_MAP,
    map_character,
    map_number,
    tokenize_cipher_text,
)
from mini_rsa.core import (
    EncryptionResult,
    MIN_MODULUS,
    MIN_PRIME_SIZE,
    MIN_SECURE_PRIME,
    _ensure_int,
    _validate_block,
    _validate_modulus,
    calculate_entropy,
    decrypt_block,
    decrypt_numbers,
    ensure_coprime,
    encrypt_block,
    encrypt_text,
    encrypt_with_reference,
    gcd,
    is_prime,
    mod_inverse,
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
    with pytest.raises(ValueError) as exc:
        mod_inverse(2, 20)
    assert str(exc.value) == "inverse of 2 (mod 20) does not exist"


def test_mod_inverse_invalid_phi() -> None:
    with pytest.raises(ValueError, match="phi must be positive"):
        mod_inverse(3, 0)
    with pytest.raises(ValueError) as exc:
        mod_inverse(3, -5)
    assert str(exc.value) == "phi must be positive"


def test_mod_inverse_phi_equals_one_sympy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", True)
    monkeypatch.setattr(core, "sympy_mod_inverse", core.sympy_mod_inverse)
    with pytest.raises(ValueError, match=r"inverse of 3 \(mod 1\) does not exist"):
        mod_inverse(3, 1)


def test_mod_inverse_phi_equals_one_without_sympy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", False)
    monkeypatch.setattr(core, "sympy_mod_inverse", None)
    assert mod_inverse(3, 1) == 0


def test_mod_inverse_returns_positive_result() -> None:
    result = mod_inverse(7, 40)
    assert result == 23
    assert 0 < result < 40


def test_mod_inverse_without_sympy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", False)
    monkeypatch.setattr(core, "sympy_mod_inverse", None)
    assert mod_inverse(3, 20) == 7
    with pytest.raises(ValueError):
        mod_inverse(4, 20)


def test_mod_inverse_requires_positive_phi_message() -> None:
    with pytest.raises(ValueError, match="phi must be positive"):
        mod_inverse(3, 0)


def test_mod_inverse_sympy_path_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", True)

    def fake_mod_inverse(e: int, phi: int) -> int:
        assert e == 3 and phi == 20
        return 7

    monkeypatch.setattr(core, "sympy_mod_inverse", fake_mod_inverse)
    assert mod_inverse(3, 20) == 7


def test_mod_inverse_sympy_path_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", True)

    def fake_mod_inverse(e: int, phi: int) -> int:
        raise ValueError("no inverse")

    monkeypatch.setattr(core, "sympy_mod_inverse", fake_mod_inverse)
    with pytest.raises(ValueError, match="no inverse"):
        mod_inverse(3, 20)


def test_mod_inverse_sympy_path_other_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", True)

    class Boom(Exception):
        pass

    def fake_mod_inverse(e: int, phi: int) -> int:
        raise Boom("boom")

    monkeypatch.setattr(core, "sympy_mod_inverse", fake_mod_inverse)
    with pytest.raises(ValueError) as exc:
        mod_inverse(3, 20)
    assert str(exc.value) == "Failed to compute modular inverse via sympy"


def test_mod_inverse_sympy_missing_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", True)
    monkeypatch.setattr(core, "sympy_mod_inverse", None)
    assert mod_inverse(3, 20) == 7


def test_mod_inverse_extended_gcd_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", False)
    monkeypatch.setattr(core, "sympy_mod_inverse", None)
    assert mod_inverse(17, 3120) == 2753


def test_mod_inverse_fallback_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", False)
    monkeypatch.setattr(core, "sympy_mod_inverse", None)
    with pytest.raises(ValueError) as exc:
        mod_inverse(4, 20)
    assert str(exc.value) == "Modular inverse does not exist"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2, True),
        (17, True),
        (1, False),
        (9, False),
        (25, False),
        (27, False),
        (97, True),
        (100, False),
    ],
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
    assert is_prime(9) is False


def test_is_prime_sympy_unavailable_but_flag_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", True)
    monkeypatch.setattr(core, "sympy_isprime", None)
    assert is_prime(17) is True
    assert is_prime(18) is False


def test_is_prime_sympy_called(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_isprime(n: int) -> bool:
        calls["count"] += 1
        return n == 17

    monkeypatch.setattr(core, "SYMPY_AVAILABLE", True)
    monkeypatch.setattr(core, "sympy_isprime", fake_isprime)
    assert is_prime(17) is True
    assert is_prime(18) is False
    assert calls["count"] == 2


def test_validate_prime_pair_happy_path() -> None:
    errors, warnings = validate_prime_pair(3, 11)
    assert errors == []
    assert warnings == [f"For better security, use primes ≥ {MIN_SECURE_PRIME}"]


def test_validate_prime_pair_accepts_minimum_prime_size() -> None:
    errors, warnings = validate_prime_pair(2, 3)
    assert "Primes must be at least" not in " ".join(errors)


def test_validate_prime_pair_minimum_thresholds() -> None:
    errors, warnings = validate_prime_pair(3, 2)
    assert "Primes must be at least" not in " ".join(errors)

    errors, warnings = validate_prime_pair(MIN_SECURE_PRIME, 13)
    assert warnings == []

    errors, warnings = validate_prime_pair(13, MIN_SECURE_PRIME)
    assert warnings == []


def test_validate_prime_pair_errors() -> None:
    errors, warnings = validate_prime_pair(4, 4)
    assert "4 is not prime" in errors
    assert "p and q must be different primes" in errors
    assert any("at least" in error for error in errors)
    assert warnings == [f"For better security, use primes ≥ {MIN_SECURE_PRIME}"]


def test_validate_prime_pair_too_small() -> None:
    errors, _ = validate_prime_pair(1, 3)
    assert f"at least {MIN_PRIME_SIZE}" in " ".join(errors)


def test_validate_prime_pair_modulus_message() -> None:
    errors, _ = validate_prime_pair(3, 7)
    assert "n = p×q must be at least 33 to handle all characters" in errors


def test_calculate_entropy_matches_log2() -> None:
    modulus = MIN_MODULUS
    expected = math.log2(modulus)
    assert calculate_entropy(modulus) == pytest.approx(expected)


def test_calculate_entropy_rejects_non_positive() -> None:
    with pytest.raises(ValueError) as exc:
        calculate_entropy(0)
    assert str(exc.value) == "Modulus must be positive"


def test_calculate_entropy_handles_one() -> None:
    assert calculate_entropy(1) == pytest.approx(0.0)


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


def test_encrypt_block_default_use_large_numbers_false(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_pow_mod(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
        calls["count"] += 1
        assert use_large_numbers is False
        return pow(base, exponent, modulus)

    monkeypatch.setattr(core, "_pow_mod", fake_pow_mod)
    assert encrypt_block(3, 5, 11) == pow(3, 5, 11)
    assert calls["count"] == 1


def test_encrypt_block_rejects_non_int_inputs() -> None:
    with pytest.raises(TypeError, match="message_block must be an int"):
        encrypt_block("2", 3, 33)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="public exponent e must be an int"):
        encrypt_block(2, "3", 33)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="modulus n must be an int"):
        encrypt_block(2, 3, "33")  # type: ignore[arg-type]


def test_encrypt_block_validates_block_range_and_modulus() -> None:
    with pytest.raises(ValueError, match="Modulus n must be positive"):
        encrypt_block(1, 3, 0)
    with pytest.raises(ValueError, match="message_block must be non-negative"):
        encrypt_block(-1, 3, 33)
    with pytest.raises(ValueError, match="message_block must be less than modulus n"):
        encrypt_block(40, 3, 33)


def test_encrypt_block_accepts_sympy_integer() -> None:
    if not core.SYMPY_AVAILABLE:
        pytest.skip("Sympy not available for Integer coercion test")
    message = core.Integer(4)
    exponent = core.Integer(3)
    modulus = core.Integer(33)
    result = encrypt_block(message, exponent, modulus)
    assert result == pow(4, 3, 33)


def test_pow_mod_without_large_numbers_skips_sympy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "SYMPY_AVAILABLE", True)

    class ExplodingInteger:
        def __call__(self, *args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("Integer should not be used when use_large_numbers is False")

    monkeypatch.setattr(core, "Integer", ExplodingInteger())
    result = core._pow_mod(3, 5, 11, use_large_numbers=False)
    assert result == pow(3, 5, 11)


def test_pow_mod_with_large_numbers_uses_sympy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    class TrackingInteger(int):
        def __new__(cls, value):
            calls["count"] += 1
            return int.__new__(cls, value)

    monkeypatch.setattr(core, "SYMPY_AVAILABLE", True)
    monkeypatch.setattr(core, "Integer", TrackingInteger)
    result = core._pow_mod(3, 5, 11, use_large_numbers=True)
    assert result == pow(3, 5, 11)
    assert calls["count"] == 3


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
    assert map_number(27, include_punctuation=False) == ""
    assert map_number(1, include_punctuation=True) == "A"
    assert map_number(3, include_punctuation=True) == "C"
    assert map_number(26, include_punctuation=True) == "Z"


def test_decrypt_numbers_matches_encrypt_text() -> None:
    p, q, e = 5, 13, 5
    n = p * q
    phi = (p - 1) * (q - 1)
    d = mod_inverse(e, phi)
    message = "HELLO"
    result = encrypt_text(message, e, n, include_punctuation=False)
    decrypted_numbers = decrypt_numbers(result.cipher_blocks, d, n)
    assert decrypted_numbers == result.plain_blocks


def test_decrypt_numbers_respects_use_large_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_pow_mod(cipher: int, d: int, n: int, use_large_numbers: bool) -> int:
        calls["count"] += 1
        assert use_large_numbers is True
        return pow(cipher, d, n)

    monkeypatch.setattr(core, "decrypt_block", fake_pow_mod)
    decrypt_numbers([1, 2, 3], 7, 33, use_large_numbers=True)
    assert calls["count"] == 3


def test_decrypt_numbers_default_use_large_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    flags = []

    def fake_decrypt_block(cipher: int, d: int, n: int, use_large_numbers: bool) -> int:
        flags.append(use_large_numbers)
        return cipher

    monkeypatch.setattr(core, "decrypt_block", fake_decrypt_block)
    decrypt_numbers([1, 2], 7, 33)
    assert flags == [False, False]


def test_decrypt_numbers_validates_inputs() -> None:
    with pytest.raises(TypeError, match="private exponent d must be an int"):
        core.decrypt_numbers([1], d="7", n=33)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="modulus n must be an int"):
        core.decrypt_numbers([1], d=7, n="33")  # type: ignore[arg-type]


def test_decrypt_block_default_use_large_numbers_false(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_pow_mod(cipher: int, d: int, n: int, use_large_numbers: bool) -> int:
        calls["use_large_numbers"] = use_large_numbers
        return pow(cipher, d, n)

    monkeypatch.setattr(core, "_pow_mod", fake_pow_mod)
    assert decrypt_block(4, 7, 33) == pow(4, 7, 33)
    assert calls["use_large_numbers"] is False


def test_decrypt_block_validates_inputs() -> None:
    with pytest.raises(ValueError) as exc:
        decrypt_block(40, 7, 33)
    assert "less than modulus" in str(exc.value)


def test_decrypt_block_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="cipher_block must be an int"):
        decrypt_block("10", 7, 33)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="private exponent d must be an int"):
        decrypt_block(10, "7", 33)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="modulus n must be an int"):
        decrypt_block(10, 7, "33")  # type: ignore[arg-type]


def test_decrypt_block_validates_range_and_modulus() -> None:
    with pytest.raises(ValueError, match="Modulus n must be positive"):
        decrypt_block(1, 7, 0)
    with pytest.raises(ValueError, match="cipher_block must be non-negative"):
        decrypt_block(-1, 7, 33)
    with pytest.raises(ValueError, match="cipher_block must be less than modulus n"):
        decrypt_block(100, 7, 33)


def test_decrypt_block_accepts_sympy_integer() -> None:
    if not core.SYMPY_AVAILABLE:
        pytest.skip("Sympy not available for Integer coercion test")
    cipher = core.Integer(4)
    private = core.Integer(7)
    modulus = core.Integer(33)
    result = decrypt_block(cipher, private, modulus)
    assert result == pow(4, 7, 33)


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


def test_decrypt_text_blocks_respects_punctuation_flag() -> None:
    cipher_numbers = [15, 7, 2, 6, 25]
    assert core.decrypt_text_blocks(cipher_numbers, d=7, n=33, include_punctuation=False) == ""
    assert core.decrypt_text_blocks(cipher_numbers, d=7, n=33, include_punctuation=True) == ".,!?;"


def test_decrypt_text_blocks_passes_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_decrypt_numbers(numbers, d, n, use_large_numbers):
        captured["use_large_numbers"] = use_large_numbers
        return numbers

    monkeypatch.setattr(core, "decrypt_numbers", fake_decrypt_numbers)
    result = core.decrypt_text_blocks([1, 32], d=7, n=33, include_punctuation=False, use_large_numbers=True)
    assert result == "A "
    assert captured["use_large_numbers"] is True


def test_decrypt_text_blocks_default_excludes_punctuation() -> None:
    cipher_numbers = [15, 7, 2, 6, 25]
    assert core.decrypt_text_blocks(cipher_numbers, d=7, n=33) == ""


def test_decrypt_text_blocks_defaults_passed_to_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_decrypt_numbers(numbers, d, n, use_large_numbers):
        captured["use_large_numbers"] = use_large_numbers
        return numbers

    monkeypatch.setattr(core, "decrypt_numbers", fake_decrypt_numbers)
    result = core.decrypt_text_blocks([1, 32], d=7, n=33)
    assert result == "A "
    assert captured == {"use_large_numbers": False}


def test_tokenize_cipher_text_filters_invalid_tokens() -> None:
    cipher_numbers = tokenize_cipher_text("10 abc 42\t-7 99")
    assert cipher_numbers == [10, 42, -7, 99]


def test_validate_entropy_bounds_success_and_failure() -> None:
    modulus = 97 * 101
    validate_entropy_bounds(modulus, expected_min_bits=5)
    validate_entropy_bounds(modulus, expected_min_bits=math.log2(modulus))
    with pytest.raises(ValueError, match="Entropy .* below"):
        validate_entropy_bounds(modulus, expected_min_bits=20)


def test_ensure_coprime_checks() -> None:
    ensure_coprime(3, 20)
    with pytest.raises(ValueError) as exc:
        ensure_coprime(4, 20)
    assert str(exc.value) == "e=4 is not coprime with φ(n)=20"


def test_internal_validators() -> None:
    assert _ensure_int(5, "value") == 5
    assert _ensure_int(True, "value") == 1
    if core.SYMPY_AVAILABLE:
        assert _ensure_int(core.Integer(7), "value") == 7
    with pytest.raises(TypeError) as exc:
        _ensure_int("x", "value")
    assert str(exc.value) == "value must be an int, got str"

    with pytest.raises(ValueError) as exc:
        _validate_modulus(0)
    assert str(exc.value) == "Modulus n must be positive"

    # n=1 is accepted; ensure no exception is raised
    _validate_modulus(1)

    with pytest.raises(ValueError) as exc:
        _validate_block(-1, 10, "cipher")
    assert str(exc.value) == "cipher must be non-negative"

    with pytest.raises(ValueError) as exc:
        _validate_block(10, 10, "cipher")
    assert str(exc.value) == "cipher must be less than modulus n"

    # zero is valid input and should not raise
    _validate_block(0, 10, "cipher")


def test_encrypt_with_reference_matches_encrypt_text() -> None:
    p, q, e = 3, 11, 3
    n = p * q
    message = "TEST"
    reference = encrypt_with_reference(message, e, n)
    result = encrypt_text(message, e, n, include_punctuation=True)
    assert reference == result.cipher_blocks


def test_encrypt_with_reference_forces_punctuation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_encrypt_text(text: str, e: int, n: int, include_punctuation: bool, use_large_numbers: bool) -> EncryptionResult:
        calls["include_punctuation"] = include_punctuation
        calls["use_large_numbers"] = use_large_numbers
        return EncryptionResult(cipher_blocks=[1], plain_blocks=[1], skipped_characters=[], trace=[])

    monkeypatch.setattr(core, "encrypt_text", fake_encrypt_text)
    encrypt_with_reference("A", 3, 33)
    assert calls["include_punctuation"] is True
    assert calls["use_large_numbers"] is False


def test_encrypt_text_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default excludes punctuation and does not use large numbers
    calls = {"pow_calls": 0}

    def fake_pow_mod(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
        calls["pow_calls"] += 1
        assert use_large_numbers is False
        return pow(base, exponent, modulus)

    monkeypatch.setattr(core, "_pow_mod", fake_pow_mod)
    result = encrypt_text("HI!", e=3, n=33)
    assert "!" in result.skipped_characters
    assert calls["pow_calls"] == 2  # only H and I processed
    assert all("Encrypting:" in entry for entry in result.trace)


def test_encrypt_text_includes_punctuation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_pow_mod(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
        calls.append((base, use_large_numbers))
        return base  # deterministic

    monkeypatch.setattr(core, "_pow_mod", fake_pow_mod)
    result = encrypt_text("!;", e=3, n=33, include_punctuation=True, use_large_numbers=True)
    assert result.cipher_blocks == [29, 31]
    assert calls == [(29, True), (31, True)]
    assert result.skipped_characters == []
    assert result.trace is not None


def test_encrypt_text_validates_inputs() -> None:
    with pytest.raises(TypeError, match="modulus n must be an int"):
        core.encrypt_text("A", e=3, n="33")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="public exponent e must be an int"):
        core.encrypt_text("A", e="3", n=33)  # type: ignore[arg-type]


def test_encrypt_text_processes_after_skipped_character(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_pow_mod(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
        calls.append(base)
        return base

    monkeypatch.setattr(core, "_pow_mod", fake_pow_mod)
    result = encrypt_text("A@B", e=3, n=33, include_punctuation=False, use_large_numbers=False)
    assert result.plain_blocks == [1, 2]
    assert result.skipped_characters == ["@"]
    assert calls == [1, 2]


def test_generate_secure_primes_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "CRYPTO_AVAILABLE", False)
    monkeypatch.setattr(core, "number", None)

    choices = []

    def fake_choice(seq):
        assert all(core.is_prime(x) for x in seq)
        value = seq[len(choices) % len(seq)]
        choices.append(value)
        return value

    monkeypatch.setattr(core.random, "choice", fake_choice)
    p, q = core.generate_secure_primes(16)
    assert p in choices and q in choices
    assert p != q
    assert core.is_prime(p) and core.is_prime(q)


class DummyNumber:
    def __init__(self):
        self.calls = 0

    def getPrime(self, bits: int) -> int:  # noqa: N802 - mimics library API
        assert bits == 16
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
    assert dummy.calls == 2


def test_generate_secure_primes_crypto_bits_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    bits_recorded = []

    class Recorder:
        def getPrime(self, bits: int) -> int:  # noqa: N802
            bits_recorded.append(bits)
            return 101 if not bits_recorded or bits_recorded.count(bits) == 1 else 103

    monkeypatch.setattr(core, "CRYPTO_AVAILABLE", True)
    monkeypatch.setattr(core, "number", Recorder())
    result = core.generate_secure_primes(32)
    assert result[0] != result[1]
    assert bits_recorded == [32, 32]


def test_generate_secure_primes_crypto_path_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyNumberRepeat:
        def __init__(self):
            self.calls = 0

        def getPrime(self, bits: int) -> int:  # noqa: N802
            self.calls += 1
            if self.calls < 3:
                return 101
            return 103

    dummy = DummyNumberRepeat()
    monkeypatch.setattr(core, "CRYPTO_AVAILABLE", True)
    monkeypatch.setattr(core, "number", dummy)
    p, q = core.generate_secure_primes(16)
    assert p == 101 and q == 103
    assert dummy.calls == 3


def test_generate_secure_primes_handles_missing_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "CRYPTO_AVAILABLE", True)
    monkeypatch.setattr(core, "number", None)

    def fake_choice(seq):
        assert all(core.is_prime(x) for x in seq)
        return seq[0]

    monkeypatch.setattr(core.random, "choice", fake_choice)
    p, q = core.generate_secure_primes(16)
    assert core.is_prime(p) and core.is_prime(q)


def test_generate_secure_primes_fallback_list_contains_only_primes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "CRYPTO_AVAILABLE", False)
    monkeypatch.setattr(core, "number", None)

    def fake_choice(seq):
        assert all(core.is_prime(x) for x in seq)
        return seq[0]

    monkeypatch.setattr(core.random, "choice", fake_choice)
    p, q = core.generate_secure_primes(16)
    assert core.is_prime(p) and core.is_prime(q)


def test_generate_secure_primes_requires_positive_bits_message() -> None:
    with pytest.raises(ValueError) as exc:
        core.generate_secure_primes(0)
    assert str(exc.value) == "bits must be positive"

    with pytest.raises(ValueError) as exc:
        core.generate_secure_primes(1)
    assert str(exc.value) == "bits must be >= 2"
