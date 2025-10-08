"""Character mapping utilities for the MiniRSA project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .models import EncryptionResult
from .core_types import PowFunction

PUNCTUATION_MAP: Dict[str, int] = {".": 27, ",": 28, "!": 29, "?": 30, ";": 31}
REVERSE_PUNCTUATION_MAP: Dict[int, str] = {value: key for key, value in PUNCTUATION_MAP.items()}


def map_character(character: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Return tuple of (is_valid, numeric value)."""
    if character.isalpha():
        return True, ord(character.lower()) - 96
    if character.isspace():
        return True, 32
    if include_punctuation and character in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[character]
    return False, None


def map_number(number: int, include_punctuation: bool) -> str:
    """Return string representation of numeric block."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def tokenize_cipher_text(cipher_text: str) -> List[int]:
    """Parse space separated cipher numbers, ignoring invalid tokens."""
    numbers: List[int] = []
    for token in cipher_text.split():
        try:
            numbers.append(int(token))
        except ValueError:
            continue
    return numbers


def encrypt_text(
    text: str,
    encrypt_block_fn: PowFunction,
    include_punctuation: bool,
    use_large_numbers: bool,
    n: int,
    e: int,
) -> EncryptionResult:
    from .core import _validate_block  # local import to avoid cycle

    cipher_blocks: List[int] = []
    plain_blocks: List[int] = []
    skipped: List[str] = []
    trace: List[str] = []

    for character in text:
        accepted, mapped = map_character(character, include_punctuation)
        if not accepted:
            skipped.append(character)
            continue
        assert mapped is not None

        _validate_block(mapped, n, "message_block")
        encrypted = encrypt_block_fn(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def decrypt_numbers(
    cipher_numbers: Sequence[int],
    decrypt_block_fn: PowFunction,
    use_large_numbers: bool,
    d: int,
    n: int,
) -> List[int]:
    from .core import _validate_block  # local import to avoid cycle

    numbers: List[int] = []
    for cipher in cipher_numbers:
        _validate_block(cipher, n, "cipher_block")
        numbers.append(decrypt_block_fn(cipher, d, n, use_large_numbers))
    return numbers
