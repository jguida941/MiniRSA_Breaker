from __future__ import annotations

import math
import random
from numbers import Integral
from typing import Iterable, List, Sequence, Tuple

from .codec import (
    PUNCTUATION_MAP,
    REVERSE_PUNCTUATION_MAP,
    decrypt_numbers as codec_decrypt_numbers,
    encrypt_text as codec_encrypt_text,
    map_character,
    map_number,
    tokenize_cipher_text,
)
from .models import EncryptionResult

try:
    from sympy import Integer, isprime as sympy_isprime, mod_inverse as sympy_mod_inverse

    SYMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    Integer = int  # type: ignore[assignment]
    sympy_isprime = None  # type: ignore[assignment]
    sympy_mod_inverse = None  # type: ignore[assignment]
    SYMPY_AVAILABLE = False

try:
    from Crypto.Util import number

    CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    number = None  # type: ignore[assignment]
    CRYPTO_AVAILABLE = False

MIN_PRIME_SIZE = 2
MIN_SECURE_PRIME = 11
MIN_MODULUS = 33


def gcd(a: int, b: int) -> int:
    """Euclidean algorithm."""
    first, second = abs(a), abs(b)
    while second:
        first, second = second, first % second
    return first


def mod_inverse(e: int, phi: int) -> int:
    """Return modular inverse of e mod phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is not None:
        try:
            return int(sympy_mod_inverse(e, phi))
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ValueError("Failed to compute modular inverse via sympy") from exc

    from math import gcd as _gcd  # local import avoids top-level dependency

    if _gcd(e, phi) != 1:
        raise ValueError("Modular inverse does not exist")

    try:
        return pow(e % phi, -1, phi)
    except ValueError as exc:  # pragma: no cover - unexpected for coprime inputs
        raise ValueError("Modular inverse does not exist") from exc


def is_prime(n: int) -> bool:
    """Trial division (uses sympy when available)."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(math.isqrt(n))
    candidate = 3
    while candidate <= limit:
        if n % candidate == 0:
            return False
        candidate += 2
    return True


def validate_prime_pair(p: int, q: int) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if not is_prime(p):
        errors.append(f"{p} is not prime")
    if not is_prime(q):
        errors.append(f"{q} is not prime")
    if p == q:
        errors.append("p and q must be different primes")
    if p < MIN_PRIME_SIZE or q < MIN_PRIME_SIZE:
        errors.append(f"Primes must be at least {MIN_PRIME_SIZE}")
    if p < MIN_SECURE_PRIME or q < MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")
    if p * q < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")
    return errors, warnings


def calculate_entropy(n: int) -> float:
    if n <= 0:
        raise ValueError("Modulus must be positive")
    return math.log2(n)


def validate_entropy_bounds(n: int, expected_min_bits: float) -> None:
    entropy = calculate_entropy(n)
    if entropy < expected_min_bits:
        raise ValueError(f"Entropy {entropy:.2f} bits is below expected minimum {expected_min_bits}")


def _pow_mod(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), Integer(modulus)))
    return pow(base, exponent, modulus)


def _ensure_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, Integral):
        converted = int(value)
        return converted
    raise TypeError(f"{name} must be an int, got {type(value).__name__}")


def _validate_modulus(n: int) -> None:
    if n <= 0:
        raise ValueError("Modulus n must be positive")


def _validate_block(block: int, n: int, label: str) -> None:
    if block < 0:
        raise ValueError(f"{label} must be non-negative")
    if block >= n:
        raise ValueError(f"{label} must be less than modulus n")


def encrypt_block(message_block: int, e: int, n: int, use_large_numbers: bool = False) -> int:
    message_block_int = _ensure_int(message_block, "message_block")
    exponent = _ensure_int(e, "public exponent e")
    modulus = _ensure_int(n, "modulus n")
    _validate_modulus(modulus)
    _validate_block(message_block_int, modulus, "message_block")
    return _pow_mod(message_block_int, exponent, modulus, use_large_numbers)


def decrypt_block(cipher_block: int, d: int, n: int, use_large_numbers: bool = False) -> int:
    cipher_block_int = _ensure_int(cipher_block, "cipher_block")
    exponent = _ensure_int(d, "private exponent d")
    modulus = _ensure_int(n, "modulus n")
    _validate_modulus(modulus)
    _validate_block(cipher_block_int, modulus, "cipher_block")
    return _pow_mod(cipher_block_int, exponent, modulus, use_large_numbers)


def encrypt_text(
    text: str,
    e: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> EncryptionResult:
    return codec_encrypt_text(
        text=text,
        encrypt_block_fn=encrypt_block,
        include_punctuation=include_punctuation,
        use_large_numbers=use_large_numbers,
        n=_ensure_int(n, "modulus n"),
        e=_ensure_int(e, "public exponent e"),
    )


def encrypt_with_reference(text: str, e: int, n: int) -> List[int]:
    """Convenience wrapper used in legacy tests/examples."""
    return encrypt_text(
        text=text,
        e=e,
        n=n,
        include_punctuation=True,
        use_large_numbers=False,
    ).cipher_blocks


def decrypt_numbers(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    use_large_numbers: bool = False,
) -> List[int]:
    return codec_decrypt_numbers(
        cipher_numbers,
        decrypt_block_fn=decrypt_block,
        use_large_numbers=use_large_numbers,
        d=_ensure_int(d, "private exponent d"),
        n=_ensure_int(n, "modulus n"),
    )


def decrypt_text_blocks(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    numbers = decrypt_numbers(
        cipher_numbers,
        d,
        n,
        use_large_numbers=use_large_numbers,
    )
    return "".join(map_number(number, include_punctuation) for number in numbers)


def ensure_coprime(e: int, phi: int) -> None:
    if gcd(e, phi) != 1:
        raise ValueError(f"e={e} is not coprime with φ(n)={phi}")


def generate_secure_primes(bits: int) -> Tuple[int, int]:
    if bits <= 0:
        raise ValueError("bits must be positive")
    if bits == 1:
        raise ValueError("bits must be >= 2")

    if CRYPTO_AVAILABLE and number is not None:
        p = number.getPrime(bits)
        q = number.getPrime(bits)
        while p == q:
            q = number.getPrime(bits)
        return p, q

    fallback_primes = [
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
    ]
    p = random.choice(fallback_primes)
    remaining = [candidate for candidate in fallback_primes if candidate != p]
    q = random.choice(remaining)
    return p, q


__all__ = [
    "EncryptionResult",
    "PUNCTUATION_MAP",
    "REVERSE_PUNCTUATION_MAP",
    "calculate_entropy",
    "decrypt_block",
    "decrypt_numbers",
    "decrypt_text_blocks",
    "encrypt_block",
    "encrypt_text",
    "encrypt_with_reference",
    "ensure_coprime",
    "gcd",
    "generate_secure_primes",
    "is_prime",
    "map_character",
    "map_number",
    "mod_inverse",
    "tokenize_cipher_text",
    "validate_entropy_bounds",
    "validate_prime_pair",
]
