from __future__ import annotations

from typing import Protocol


class PowFunction(Protocol):
    def __call__(self, base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int: ...
