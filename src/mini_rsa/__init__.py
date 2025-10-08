"""MiniRSA public API."""

from .codec import (
    PUNCTUATION_MAP,
    REVERSE_PUNCTUATION_MAP,
    map_character,
    map_number,
    tokenize_cipher_text,
)
from .core import (
    EncryptionResult,
    calculate_entropy,
    decrypt_numbers,
    decrypt_text_blocks,
    encrypt_block,
    encrypt_text,
    ensure_coprime,
    gcd,
    generate_secure_primes,
    is_prime,
    mod_inverse,
    validate_prime_pair,
)

__all__ = [
    "EncryptionResult",
    "PUNCTUATION_MAP",
    "REVERSE_PUNCTUATION_MAP",
    "calculate_entropy",
    "decrypt_numbers",
    "decrypt_text_blocks",
    "encrypt_block",
    "encrypt_text",
    "ensure_coprime",
    "gcd",
    "generate_secure_primes",
    "is_prime",
    "map_character",
    "map_number",
    "mod_inverse",
    "tokenize_cipher_text",
    "validate_prime_pair",
]
