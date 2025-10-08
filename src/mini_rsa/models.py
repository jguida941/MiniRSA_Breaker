from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class EncryptionResult:
    cipher_blocks: List[int]
    plain_blocks: List[int]
    skipped_characters: List[str]
    trace: List[str]
