from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from sympy import Integer, isprime as sympy_isprime, mod_inverse as sympy_mod_inverse

    SYMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Integer = int  # type: ignore[assignment]
    sympy_isprime = None  # type: ignore[assignment]
    sympy_mod_inverse = None  # type: ignore[assignment]
    SYMPY_AVAILABLE = False

try:
    from Crypto.Util import number

    CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    number = None  # type: ignore[assignment]
    CRYPTO_AVAILABLE = False

MIN_PRIME_SIZE = 2
MIN_SECURE_PRIME = 11
MIN_MODULUS = 33

PUNCTUATION_MAP: Dict[str, int] = {".": 27, ",": 28, "!": 29, "?": 30, ";": 31}
REVERSE_PUNCTUATION_MAP: Dict[int, str] = {v: k for k, v in PUNCTUATION_MAP.items()}


@dataclass(slots=True)
class EncryptionResult:
    """Structured container returned by encrypt_text."""

    cipher_blocks: List[int]
    plain_blocks: List[int]
    skipped_characters: List[str]
    trace: List[str]


AllowedSymbol = str


def gcd(a: int, b: int) -> int:
    """Compute the greatest common divisor using Euclid's algorithm."""
    first, second = abs(a), abs(b)
    while second:
        first, second = second, first % second
    return first


def mod_inverse(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is not None:
        try:
            return int(sympy_mod_inverse(e, phi))
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Failed to compute modular inverse via sympy") from exc

    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        inner_gcd, inner_x, inner_y = extended_gcd(b % a, a)
        adj_x = inner_y - (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def is_prime(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(math.isqrt(n))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def validate_prime_pair(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
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

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def calculate_entropy(n: int) -> float:
    """Return the approximate entropy (log2) of the modulus."""
    if n <= 0:
        raise ValueError("Modulus must be positive")
    return math.log2(n)


def _pow_mod(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), Integer(modulus)))
    return pow(base, exponent, modulus)


def encrypt_block(message_block: int, e: int, n: int, use_large_numbers: bool = False) -> int:
    """Encrypt a numeric message block with RSA."""
    return _pow_mod(message_block, e, n, use_large_numbers)


def decrypt_block(cipher_block: int, d: int, n: int, use_large_numbers: bool = False) -> int:
    """Decrypt a numeric cipher block with RSA."""
    return _pow_mod(cipher_block, d, n, use_large_numbers)


def map_character(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(char.lower()) - 96
    if char.isspace():
        return True, 32
    if include_punctuation and char in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return False, None


def map_number(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def encrypt_text(
    text: str,
    e: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> EncryptionResult:
    """Encrypt text and return structured details."""
    cipher_blocks: List[int] = []
    plain_blocks: List[int] = []
    skipped: List[str] = []
    trace: List[str] = []

    for character in text:
        accepted, mapped = map_character(character, include_punctuation)
        if not accepted or mapped is None:
            skipped.append(character)
            continue

        encrypted = encrypt_block(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def decrypt_numbers(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> List[int]:
    """Decrypt RSA cipher numbers back to their numeric representation."""
    return [
        decrypt_block(cipher, d, n, use_large_numbers)
        for cipher in cipher_numbers
    ]


def decrypt_text_blocks(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def generate_secure_primes(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("bits must be positive")

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


def tokenize_cipher_text(cipher_text: str) -> List[int]:
    """Parse space separated cipher numbers, ignoring invalid tokens."""
    numbers: List[int] = []
    for token in cipher_text.split():
        try:
            numbers.append(int(token))
        except ValueError:
            continue
    return numbers


def validate_entropy_bounds(n: int, expected_min_bits: float) -> None:
    """Raise ValueError when entropy drops below the expected lower bound."""
    entropy = calculate_entropy(n)
    if entropy < expected_min_bits:
        raise ValueError(f"Entropy {entropy:.2f} bits is below expected minimum {expected_min_bits}")


def ensure_coprime(e: int, phi: int) -> None:
    """Validate that e and phi are coprime."""
    if gcd(e, phi) != 1:
        raise ValueError(f"e={e} is not coprime with φ(n)={phi}")


def encrypt_with_reference(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, e, n, include_punctuation=True, use_large_numbers=False)
    return result.cipher_blocks
