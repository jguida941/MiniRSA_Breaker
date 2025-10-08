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
from inspect import signature as _mutmut_signature
from typing import Annotated
from typing import Callable
from typing import ClassVar


MutantDict = Annotated[dict[str, Callable], "Mutant"]


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None):
    """Forward call to original or mutated function, depending on the environment"""
    import os
    mutant_under_test = os.environ['MUTANT_UNDER_TEST']
    if mutant_under_test == 'fail':
        from mutmut.__main__ import MutmutProgrammaticFailException
        raise MutmutProgrammaticFailException('Failed programmatically')      
    elif mutant_under_test == 'stats':
        from mutmut.__main__ import record_trampoline_hit
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__)
        result = orig(*call_args, **call_kwargs)
        return result
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_'
    if not mutant_under_test.startswith(prefix):
        result = orig(*call_args, **call_kwargs)
        return result
    mutant_name = mutant_under_test.rpartition('.')[-1]
    if self_arg:
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)
    return result


@dataclass(slots=True)
class EncryptionResult:
    """Structured container returned by encrypt_text."""

    cipher_blocks: List[int]
    plain_blocks: List[int]
    skipped_characters: List[str]
    trace: List[str]


AllowedSymbol = str


def x_gcd__mutmut_orig(a: int, b: int) -> int:
    """Compute the greatest common divisor using Euclid's algorithm."""
    first, second = abs(a), abs(b)
    while second:
        first, second = second, first % second
    return first


def x_gcd__mutmut_1(a: int, b: int) -> int:
    """Compute the greatest common divisor using Euclid's algorithm."""
    first, second = None
    while second:
        first, second = second, first % second
    return first


def x_gcd__mutmut_2(a: int, b: int) -> int:
    """Compute the greatest common divisor using Euclid's algorithm."""
    first, second = abs(None), abs(b)
    while second:
        first, second = second, first % second
    return first


def x_gcd__mutmut_3(a: int, b: int) -> int:
    """Compute the greatest common divisor using Euclid's algorithm."""
    first, second = abs(a), abs(None)
    while second:
        first, second = second, first % second
    return first


def x_gcd__mutmut_4(a: int, b: int) -> int:
    """Compute the greatest common divisor using Euclid's algorithm."""
    first, second = abs(a), abs(b)
    while second:
        first, second = None
    return first


def x_gcd__mutmut_5(a: int, b: int) -> int:
    """Compute the greatest common divisor using Euclid's algorithm."""
    first, second = abs(a), abs(b)
    while second:
        first, second = second, first / second
    return first

x_gcd__mutmut_mutants : ClassVar[MutantDict] = {
'x_gcd__mutmut_1': x_gcd__mutmut_1, 
    'x_gcd__mutmut_2': x_gcd__mutmut_2, 
    'x_gcd__mutmut_3': x_gcd__mutmut_3, 
    'x_gcd__mutmut_4': x_gcd__mutmut_4, 
    'x_gcd__mutmut_5': x_gcd__mutmut_5
}

def gcd(*args, **kwargs):
    result = _mutmut_trampoline(x_gcd__mutmut_orig, x_gcd__mutmut_mutants, args, kwargs)
    return result 

gcd.__signature__ = _mutmut_signature(x_gcd__mutmut_orig)
x_gcd__mutmut_orig.__name__ = 'x_gcd'


def x_mod_inverse__mutmut_orig(e: int, phi: int) -> int:
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


def x_mod_inverse__mutmut_1(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi < 0:
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


def x_mod_inverse__mutmut_2(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 1:
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


def x_mod_inverse__mutmut_3(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError(None)

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


def x_mod_inverse__mutmut_4(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("XXphi must be positiveXX")

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


def x_mod_inverse__mutmut_5(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("PHI MUST BE POSITIVE")

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


def x_mod_inverse__mutmut_6(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE or sympy_mod_inverse is not None:
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


def x_mod_inverse__mutmut_7(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is None:
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


def x_mod_inverse__mutmut_8(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is not None:
        try:
            return int(None)
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


def x_mod_inverse__mutmut_9(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is not None:
        try:
            return int(sympy_mod_inverse(None, phi))
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


def x_mod_inverse__mutmut_10(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is not None:
        try:
            return int(sympy_mod_inverse(e, None))
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


def x_mod_inverse__mutmut_11(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is not None:
        try:
            return int(sympy_mod_inverse(phi))
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


def x_mod_inverse__mutmut_12(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is not None:
        try:
            return int(sympy_mod_inverse(e, ))
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


def x_mod_inverse__mutmut_13(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is not None:
        try:
            return int(sympy_mod_inverse(e, phi))
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(None) from exc

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


def x_mod_inverse__mutmut_14(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is not None:
        try:
            return int(sympy_mod_inverse(e, phi))
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("XXFailed to compute modular inverse via sympyXX") from exc

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


def x_mod_inverse__mutmut_15(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is not None:
        try:
            return int(sympy_mod_inverse(e, phi))
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("failed to compute modular inverse via sympy") from exc

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


def x_mod_inverse__mutmut_16(e: int, phi: int) -> int:
    """Compute modular inverse of e modulo phi."""
    if phi <= 0:
        raise ValueError("phi must be positive")

    if SYMPY_AVAILABLE and sympy_mod_inverse is not None:
        try:
            return int(sympy_mod_inverse(e, phi))
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("FAILED TO COMPUTE MODULAR INVERSE VIA SYMPY") from exc

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


def x_mod_inverse__mutmut_17(e: int, phi: int) -> int:
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
        if a != 0:
            return b, 0, 1
        inner_gcd, inner_x, inner_y = extended_gcd(b % a, a)
        adj_x = inner_y - (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_18(e: int, phi: int) -> int:
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
        if a == 1:
            return b, 0, 1
        inner_gcd, inner_x, inner_y = extended_gcd(b % a, a)
        adj_x = inner_y - (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_19(e: int, phi: int) -> int:
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
            return b, 1, 1
        inner_gcd, inner_x, inner_y = extended_gcd(b % a, a)
        adj_x = inner_y - (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_20(e: int, phi: int) -> int:
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
            return b, 0, 2
        inner_gcd, inner_x, inner_y = extended_gcd(b % a, a)
        adj_x = inner_y - (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_21(e: int, phi: int) -> int:
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
        inner_gcd, inner_x, inner_y = None
        adj_x = inner_y - (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_22(e: int, phi: int) -> int:
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
        inner_gcd, inner_x, inner_y = extended_gcd(None, a)
        adj_x = inner_y - (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_23(e: int, phi: int) -> int:
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
        inner_gcd, inner_x, inner_y = extended_gcd(b % a, None)
        adj_x = inner_y - (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_24(e: int, phi: int) -> int:
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
        inner_gcd, inner_x, inner_y = extended_gcd(a)
        adj_x = inner_y - (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_25(e: int, phi: int) -> int:
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
        inner_gcd, inner_x, inner_y = extended_gcd(b % a, )
        adj_x = inner_y - (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_26(e: int, phi: int) -> int:
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
        inner_gcd, inner_x, inner_y = extended_gcd(b / a, a)
        adj_x = inner_y - (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_27(e: int, phi: int) -> int:
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
        adj_x = None
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_28(e: int, phi: int) -> int:
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
        adj_x = inner_y + (b // a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_29(e: int, phi: int) -> int:
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
        adj_x = inner_y - (b // a) / inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_30(e: int, phi: int) -> int:
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
        adj_x = inner_y - (b / a) * inner_x
        return inner_gcd, adj_x, inner_x

    gcd_val, inv_x, _ = extended_gcd(e % phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_31(e: int, phi: int) -> int:
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

    gcd_val, inv_x, _ = None
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_32(e: int, phi: int) -> int:
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

    gcd_val, inv_x, _ = extended_gcd(None, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_33(e: int, phi: int) -> int:
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

    gcd_val, inv_x, _ = extended_gcd(e % phi, None)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_34(e: int, phi: int) -> int:
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

    gcd_val, inv_x, _ = extended_gcd(phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_35(e: int, phi: int) -> int:
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

    gcd_val, inv_x, _ = extended_gcd(e % phi, )
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_36(e: int, phi: int) -> int:
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

    gcd_val, inv_x, _ = extended_gcd(e / phi, phi)
    if gcd_val != 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_37(e: int, phi: int) -> int:
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
    if gcd_val == 1:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_38(e: int, phi: int) -> int:
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
    if gcd_val != 2:
        raise ValueError("Modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_39(e: int, phi: int) -> int:
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
        raise ValueError(None)
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_40(e: int, phi: int) -> int:
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
        raise ValueError("XXModular inverse does not existXX")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_41(e: int, phi: int) -> int:
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
        raise ValueError("modular inverse does not exist")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_42(e: int, phi: int) -> int:
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
        raise ValueError("MODULAR INVERSE DOES NOT EXIST")
    return (inv_x % phi + phi) % phi


def x_mod_inverse__mutmut_43(e: int, phi: int) -> int:
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
    return (inv_x % phi + phi) / phi


def x_mod_inverse__mutmut_44(e: int, phi: int) -> int:
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
    return (inv_x % phi - phi) % phi


def x_mod_inverse__mutmut_45(e: int, phi: int) -> int:
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
    return (inv_x / phi + phi) % phi

x_mod_inverse__mutmut_mutants : ClassVar[MutantDict] = {
'x_mod_inverse__mutmut_1': x_mod_inverse__mutmut_1, 
    'x_mod_inverse__mutmut_2': x_mod_inverse__mutmut_2, 
    'x_mod_inverse__mutmut_3': x_mod_inverse__mutmut_3, 
    'x_mod_inverse__mutmut_4': x_mod_inverse__mutmut_4, 
    'x_mod_inverse__mutmut_5': x_mod_inverse__mutmut_5, 
    'x_mod_inverse__mutmut_6': x_mod_inverse__mutmut_6, 
    'x_mod_inverse__mutmut_7': x_mod_inverse__mutmut_7, 
    'x_mod_inverse__mutmut_8': x_mod_inverse__mutmut_8, 
    'x_mod_inverse__mutmut_9': x_mod_inverse__mutmut_9, 
    'x_mod_inverse__mutmut_10': x_mod_inverse__mutmut_10, 
    'x_mod_inverse__mutmut_11': x_mod_inverse__mutmut_11, 
    'x_mod_inverse__mutmut_12': x_mod_inverse__mutmut_12, 
    'x_mod_inverse__mutmut_13': x_mod_inverse__mutmut_13, 
    'x_mod_inverse__mutmut_14': x_mod_inverse__mutmut_14, 
    'x_mod_inverse__mutmut_15': x_mod_inverse__mutmut_15, 
    'x_mod_inverse__mutmut_16': x_mod_inverse__mutmut_16, 
    'x_mod_inverse__mutmut_17': x_mod_inverse__mutmut_17, 
    'x_mod_inverse__mutmut_18': x_mod_inverse__mutmut_18, 
    'x_mod_inverse__mutmut_19': x_mod_inverse__mutmut_19, 
    'x_mod_inverse__mutmut_20': x_mod_inverse__mutmut_20, 
    'x_mod_inverse__mutmut_21': x_mod_inverse__mutmut_21, 
    'x_mod_inverse__mutmut_22': x_mod_inverse__mutmut_22, 
    'x_mod_inverse__mutmut_23': x_mod_inverse__mutmut_23, 
    'x_mod_inverse__mutmut_24': x_mod_inverse__mutmut_24, 
    'x_mod_inverse__mutmut_25': x_mod_inverse__mutmut_25, 
    'x_mod_inverse__mutmut_26': x_mod_inverse__mutmut_26, 
    'x_mod_inverse__mutmut_27': x_mod_inverse__mutmut_27, 
    'x_mod_inverse__mutmut_28': x_mod_inverse__mutmut_28, 
    'x_mod_inverse__mutmut_29': x_mod_inverse__mutmut_29, 
    'x_mod_inverse__mutmut_30': x_mod_inverse__mutmut_30, 
    'x_mod_inverse__mutmut_31': x_mod_inverse__mutmut_31, 
    'x_mod_inverse__mutmut_32': x_mod_inverse__mutmut_32, 
    'x_mod_inverse__mutmut_33': x_mod_inverse__mutmut_33, 
    'x_mod_inverse__mutmut_34': x_mod_inverse__mutmut_34, 
    'x_mod_inverse__mutmut_35': x_mod_inverse__mutmut_35, 
    'x_mod_inverse__mutmut_36': x_mod_inverse__mutmut_36, 
    'x_mod_inverse__mutmut_37': x_mod_inverse__mutmut_37, 
    'x_mod_inverse__mutmut_38': x_mod_inverse__mutmut_38, 
    'x_mod_inverse__mutmut_39': x_mod_inverse__mutmut_39, 
    'x_mod_inverse__mutmut_40': x_mod_inverse__mutmut_40, 
    'x_mod_inverse__mutmut_41': x_mod_inverse__mutmut_41, 
    'x_mod_inverse__mutmut_42': x_mod_inverse__mutmut_42, 
    'x_mod_inverse__mutmut_43': x_mod_inverse__mutmut_43, 
    'x_mod_inverse__mutmut_44': x_mod_inverse__mutmut_44, 
    'x_mod_inverse__mutmut_45': x_mod_inverse__mutmut_45
}

def mod_inverse(*args, **kwargs):
    result = _mutmut_trampoline(x_mod_inverse__mutmut_orig, x_mod_inverse__mutmut_mutants, args, kwargs)
    return result 

mod_inverse.__signature__ = _mutmut_signature(x_mod_inverse__mutmut_orig)
x_mod_inverse__mutmut_orig.__name__ = 'x_mod_inverse'


def x_is_prime__mutmut_orig(n: int) -> bool:
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


def x_is_prime__mutmut_1(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n <= 2:
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


def x_is_prime__mutmut_2(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 3:
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


def x_is_prime__mutmut_3(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return True
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


def x_is_prime__mutmut_4(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE or sympy_isprime is not None:
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


def x_is_prime__mutmut_5(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is None:
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


def x_is_prime__mutmut_6(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(None)

    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(math.isqrt(n))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_7(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(None))

    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(math.isqrt(n))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_8(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n != 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(math.isqrt(n))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_9(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n == 3:
        return True
    if n % 2 == 0:
        return False

    limit = int(math.isqrt(n))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_10(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n == 2:
        return False
    if n % 2 == 0:
        return False

    limit = int(math.isqrt(n))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_11(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n == 2:
        return True
    if n / 2 == 0:
        return False

    limit = int(math.isqrt(n))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_12(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n == 2:
        return True
    if n % 3 == 0:
        return False

    limit = int(math.isqrt(n))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_13(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n == 2:
        return True
    if n % 2 != 0:
        return False

    limit = int(math.isqrt(n))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_14(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n == 2:
        return True
    if n % 2 == 1:
        return False

    limit = int(math.isqrt(n))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_15(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n == 2:
        return True
    if n % 2 == 0:
        return True

    limit = int(math.isqrt(n))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_16(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = None
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_17(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(None)
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_18(n: int) -> bool:
    """Check primality via sympy when available, otherwise trial division."""
    if n < 2:
        return False
    if SYMPY_AVAILABLE and sympy_isprime is not None:
        return bool(sympy_isprime(n))

    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(math.isqrt(None))
    for candidate in range(3, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_19(n: int) -> bool:
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
    for candidate in range(None, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_20(n: int) -> bool:
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
    for candidate in range(3, None, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_21(n: int) -> bool:
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
    for candidate in range(3, limit + 1, None):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_22(n: int) -> bool:
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
    for candidate in range(limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_23(n: int) -> bool:
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
    for candidate in range(3, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_24(n: int) -> bool:
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
    for candidate in range(3, limit + 1, ):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_25(n: int) -> bool:
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
    for candidate in range(4, limit + 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_26(n: int) -> bool:
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
    for candidate in range(3, limit - 1, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_27(n: int) -> bool:
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
    for candidate in range(3, limit + 2, 2):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_28(n: int) -> bool:
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
    for candidate in range(3, limit + 1, 3):
        if n % candidate == 0:
            return False
    return True


def x_is_prime__mutmut_29(n: int) -> bool:
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
        if n / candidate == 0:
            return False
    return True


def x_is_prime__mutmut_30(n: int) -> bool:
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
        if n % candidate != 0:
            return False
    return True


def x_is_prime__mutmut_31(n: int) -> bool:
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
        if n % candidate == 1:
            return False
    return True


def x_is_prime__mutmut_32(n: int) -> bool:
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
            return True
    return True


def x_is_prime__mutmut_33(n: int) -> bool:
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
    return False

x_is_prime__mutmut_mutants : ClassVar[MutantDict] = {
'x_is_prime__mutmut_1': x_is_prime__mutmut_1, 
    'x_is_prime__mutmut_2': x_is_prime__mutmut_2, 
    'x_is_prime__mutmut_3': x_is_prime__mutmut_3, 
    'x_is_prime__mutmut_4': x_is_prime__mutmut_4, 
    'x_is_prime__mutmut_5': x_is_prime__mutmut_5, 
    'x_is_prime__mutmut_6': x_is_prime__mutmut_6, 
    'x_is_prime__mutmut_7': x_is_prime__mutmut_7, 
    'x_is_prime__mutmut_8': x_is_prime__mutmut_8, 
    'x_is_prime__mutmut_9': x_is_prime__mutmut_9, 
    'x_is_prime__mutmut_10': x_is_prime__mutmut_10, 
    'x_is_prime__mutmut_11': x_is_prime__mutmut_11, 
    'x_is_prime__mutmut_12': x_is_prime__mutmut_12, 
    'x_is_prime__mutmut_13': x_is_prime__mutmut_13, 
    'x_is_prime__mutmut_14': x_is_prime__mutmut_14, 
    'x_is_prime__mutmut_15': x_is_prime__mutmut_15, 
    'x_is_prime__mutmut_16': x_is_prime__mutmut_16, 
    'x_is_prime__mutmut_17': x_is_prime__mutmut_17, 
    'x_is_prime__mutmut_18': x_is_prime__mutmut_18, 
    'x_is_prime__mutmut_19': x_is_prime__mutmut_19, 
    'x_is_prime__mutmut_20': x_is_prime__mutmut_20, 
    'x_is_prime__mutmut_21': x_is_prime__mutmut_21, 
    'x_is_prime__mutmut_22': x_is_prime__mutmut_22, 
    'x_is_prime__mutmut_23': x_is_prime__mutmut_23, 
    'x_is_prime__mutmut_24': x_is_prime__mutmut_24, 
    'x_is_prime__mutmut_25': x_is_prime__mutmut_25, 
    'x_is_prime__mutmut_26': x_is_prime__mutmut_26, 
    'x_is_prime__mutmut_27': x_is_prime__mutmut_27, 
    'x_is_prime__mutmut_28': x_is_prime__mutmut_28, 
    'x_is_prime__mutmut_29': x_is_prime__mutmut_29, 
    'x_is_prime__mutmut_30': x_is_prime__mutmut_30, 
    'x_is_prime__mutmut_31': x_is_prime__mutmut_31, 
    'x_is_prime__mutmut_32': x_is_prime__mutmut_32, 
    'x_is_prime__mutmut_33': x_is_prime__mutmut_33
}

def is_prime(*args, **kwargs):
    result = _mutmut_trampoline(x_is_prime__mutmut_orig, x_is_prime__mutmut_mutants, args, kwargs)
    return result 

is_prime.__signature__ = _mutmut_signature(x_is_prime__mutmut_orig)
x_is_prime__mutmut_orig.__name__ = 'x_is_prime'


def x_validate_prime_pair__mutmut_orig(p: int, q: int) -> Tuple[List[str], List[str]]:
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


def x_validate_prime_pair__mutmut_1(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = None
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


def x_validate_prime_pair__mutmut_2(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = None

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


def x_validate_prime_pair__mutmut_3(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if is_prime(p):
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


def x_validate_prime_pair__mutmut_4(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(None):
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


def x_validate_prime_pair__mutmut_5(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(p):
        errors.append(None)
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


def x_validate_prime_pair__mutmut_6(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(p):
        errors.append(f"{p} is not prime")
    if is_prime(q):
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


def x_validate_prime_pair__mutmut_7(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(p):
        errors.append(f"{p} is not prime")
    if not is_prime(None):
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


def x_validate_prime_pair__mutmut_8(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(p):
        errors.append(f"{p} is not prime")
    if not is_prime(q):
        errors.append(None)
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


def x_validate_prime_pair__mutmut_9(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(p):
        errors.append(f"{p} is not prime")
    if not is_prime(q):
        errors.append(f"{q} is not prime")
    if p != q:
        errors.append("p and q must be different primes")
    if p < MIN_PRIME_SIZE or q < MIN_PRIME_SIZE:
        errors.append(f"Primes must be at least {MIN_PRIME_SIZE}")
    if p < MIN_SECURE_PRIME or q < MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_10(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(p):
        errors.append(f"{p} is not prime")
    if not is_prime(q):
        errors.append(f"{q} is not prime")
    if p == q:
        errors.append(None)
    if p < MIN_PRIME_SIZE or q < MIN_PRIME_SIZE:
        errors.append(f"Primes must be at least {MIN_PRIME_SIZE}")
    if p < MIN_SECURE_PRIME or q < MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_11(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(p):
        errors.append(f"{p} is not prime")
    if not is_prime(q):
        errors.append(f"{q} is not prime")
    if p == q:
        errors.append("XXp and q must be different primesXX")
    if p < MIN_PRIME_SIZE or q < MIN_PRIME_SIZE:
        errors.append(f"Primes must be at least {MIN_PRIME_SIZE}")
    if p < MIN_SECURE_PRIME or q < MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_12(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(p):
        errors.append(f"{p} is not prime")
    if not is_prime(q):
        errors.append(f"{q} is not prime")
    if p == q:
        errors.append("P AND Q MUST BE DIFFERENT PRIMES")
    if p < MIN_PRIME_SIZE or q < MIN_PRIME_SIZE:
        errors.append(f"Primes must be at least {MIN_PRIME_SIZE}")
    if p < MIN_SECURE_PRIME or q < MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_13(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(p):
        errors.append(f"{p} is not prime")
    if not is_prime(q):
        errors.append(f"{q} is not prime")
    if p == q:
        errors.append("p and q must be different primes")
    if p < MIN_PRIME_SIZE and q < MIN_PRIME_SIZE:
        errors.append(f"Primes must be at least {MIN_PRIME_SIZE}")
    if p < MIN_SECURE_PRIME or q < MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_14(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(p):
        errors.append(f"{p} is not prime")
    if not is_prime(q):
        errors.append(f"{q} is not prime")
    if p == q:
        errors.append("p and q must be different primes")
    if p <= MIN_PRIME_SIZE or q < MIN_PRIME_SIZE:
        errors.append(f"Primes must be at least {MIN_PRIME_SIZE}")
    if p < MIN_SECURE_PRIME or q < MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_15(p: int, q: int) -> Tuple[List[str], List[str]]:
    """Validate whether two integers can serve as an RSA prime pair."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_prime(p):
        errors.append(f"{p} is not prime")
    if not is_prime(q):
        errors.append(f"{q} is not prime")
    if p == q:
        errors.append("p and q must be different primes")
    if p < MIN_PRIME_SIZE or q <= MIN_PRIME_SIZE:
        errors.append(f"Primes must be at least {MIN_PRIME_SIZE}")
    if p < MIN_SECURE_PRIME or q < MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_16(p: int, q: int) -> Tuple[List[str], List[str]]:
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
        errors.append(None)
    if p < MIN_SECURE_PRIME or q < MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_17(p: int, q: int) -> Tuple[List[str], List[str]]:
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
    if p < MIN_SECURE_PRIME and q < MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_18(p: int, q: int) -> Tuple[List[str], List[str]]:
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
    if p <= MIN_SECURE_PRIME or q < MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_19(p: int, q: int) -> Tuple[List[str], List[str]]:
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
    if p < MIN_SECURE_PRIME or q <= MIN_SECURE_PRIME:
        warnings.append(f"For better security, use primes ≥ {MIN_SECURE_PRIME}")

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_20(p: int, q: int) -> Tuple[List[str], List[str]]:
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
        warnings.append(None)

    modulus = p * q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_21(p: int, q: int) -> Tuple[List[str], List[str]]:
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

    modulus = None
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_22(p: int, q: int) -> Tuple[List[str], List[str]]:
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

    modulus = p / q
    if modulus < MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_23(p: int, q: int) -> Tuple[List[str], List[str]]:
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
    if modulus <= MIN_MODULUS:
        errors.append("n = p×q must be at least 33 to handle all characters")

    return errors, warnings


def x_validate_prime_pair__mutmut_24(p: int, q: int) -> Tuple[List[str], List[str]]:
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
        errors.append(None)

    return errors, warnings


def x_validate_prime_pair__mutmut_25(p: int, q: int) -> Tuple[List[str], List[str]]:
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
        errors.append("XXn = p×q must be at least 33 to handle all charactersXX")

    return errors, warnings


def x_validate_prime_pair__mutmut_26(p: int, q: int) -> Tuple[List[str], List[str]]:
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
        errors.append("N = P×Q MUST BE AT LEAST 33 TO HANDLE ALL CHARACTERS")

    return errors, warnings

x_validate_prime_pair__mutmut_mutants : ClassVar[MutantDict] = {
'x_validate_prime_pair__mutmut_1': x_validate_prime_pair__mutmut_1, 
    'x_validate_prime_pair__mutmut_2': x_validate_prime_pair__mutmut_2, 
    'x_validate_prime_pair__mutmut_3': x_validate_prime_pair__mutmut_3, 
    'x_validate_prime_pair__mutmut_4': x_validate_prime_pair__mutmut_4, 
    'x_validate_prime_pair__mutmut_5': x_validate_prime_pair__mutmut_5, 
    'x_validate_prime_pair__mutmut_6': x_validate_prime_pair__mutmut_6, 
    'x_validate_prime_pair__mutmut_7': x_validate_prime_pair__mutmut_7, 
    'x_validate_prime_pair__mutmut_8': x_validate_prime_pair__mutmut_8, 
    'x_validate_prime_pair__mutmut_9': x_validate_prime_pair__mutmut_9, 
    'x_validate_prime_pair__mutmut_10': x_validate_prime_pair__mutmut_10, 
    'x_validate_prime_pair__mutmut_11': x_validate_prime_pair__mutmut_11, 
    'x_validate_prime_pair__mutmut_12': x_validate_prime_pair__mutmut_12, 
    'x_validate_prime_pair__mutmut_13': x_validate_prime_pair__mutmut_13, 
    'x_validate_prime_pair__mutmut_14': x_validate_prime_pair__mutmut_14, 
    'x_validate_prime_pair__mutmut_15': x_validate_prime_pair__mutmut_15, 
    'x_validate_prime_pair__mutmut_16': x_validate_prime_pair__mutmut_16, 
    'x_validate_prime_pair__mutmut_17': x_validate_prime_pair__mutmut_17, 
    'x_validate_prime_pair__mutmut_18': x_validate_prime_pair__mutmut_18, 
    'x_validate_prime_pair__mutmut_19': x_validate_prime_pair__mutmut_19, 
    'x_validate_prime_pair__mutmut_20': x_validate_prime_pair__mutmut_20, 
    'x_validate_prime_pair__mutmut_21': x_validate_prime_pair__mutmut_21, 
    'x_validate_prime_pair__mutmut_22': x_validate_prime_pair__mutmut_22, 
    'x_validate_prime_pair__mutmut_23': x_validate_prime_pair__mutmut_23, 
    'x_validate_prime_pair__mutmut_24': x_validate_prime_pair__mutmut_24, 
    'x_validate_prime_pair__mutmut_25': x_validate_prime_pair__mutmut_25, 
    'x_validate_prime_pair__mutmut_26': x_validate_prime_pair__mutmut_26
}

def validate_prime_pair(*args, **kwargs):
    result = _mutmut_trampoline(x_validate_prime_pair__mutmut_orig, x_validate_prime_pair__mutmut_mutants, args, kwargs)
    return result 

validate_prime_pair.__signature__ = _mutmut_signature(x_validate_prime_pair__mutmut_orig)
x_validate_prime_pair__mutmut_orig.__name__ = 'x_validate_prime_pair'


def x_calculate_entropy__mutmut_orig(n: int) -> float:
    """Return the approximate entropy (log2) of the modulus."""
    if n <= 0:
        raise ValueError("Modulus must be positive")
    return math.log2(n)


def x_calculate_entropy__mutmut_1(n: int) -> float:
    """Return the approximate entropy (log2) of the modulus."""
    if n < 0:
        raise ValueError("Modulus must be positive")
    return math.log2(n)


def x_calculate_entropy__mutmut_2(n: int) -> float:
    """Return the approximate entropy (log2) of the modulus."""
    if n <= 1:
        raise ValueError("Modulus must be positive")
    return math.log2(n)


def x_calculate_entropy__mutmut_3(n: int) -> float:
    """Return the approximate entropy (log2) of the modulus."""
    if n <= 0:
        raise ValueError(None)
    return math.log2(n)


def x_calculate_entropy__mutmut_4(n: int) -> float:
    """Return the approximate entropy (log2) of the modulus."""
    if n <= 0:
        raise ValueError("XXModulus must be positiveXX")
    return math.log2(n)


def x_calculate_entropy__mutmut_5(n: int) -> float:
    """Return the approximate entropy (log2) of the modulus."""
    if n <= 0:
        raise ValueError("modulus must be positive")
    return math.log2(n)


def x_calculate_entropy__mutmut_6(n: int) -> float:
    """Return the approximate entropy (log2) of the modulus."""
    if n <= 0:
        raise ValueError("MODULUS MUST BE POSITIVE")
    return math.log2(n)


def x_calculate_entropy__mutmut_7(n: int) -> float:
    """Return the approximate entropy (log2) of the modulus."""
    if n <= 0:
        raise ValueError("Modulus must be positive")
    return math.log2(None)

x_calculate_entropy__mutmut_mutants : ClassVar[MutantDict] = {
'x_calculate_entropy__mutmut_1': x_calculate_entropy__mutmut_1, 
    'x_calculate_entropy__mutmut_2': x_calculate_entropy__mutmut_2, 
    'x_calculate_entropy__mutmut_3': x_calculate_entropy__mutmut_3, 
    'x_calculate_entropy__mutmut_4': x_calculate_entropy__mutmut_4, 
    'x_calculate_entropy__mutmut_5': x_calculate_entropy__mutmut_5, 
    'x_calculate_entropy__mutmut_6': x_calculate_entropy__mutmut_6, 
    'x_calculate_entropy__mutmut_7': x_calculate_entropy__mutmut_7
}

def calculate_entropy(*args, **kwargs):
    result = _mutmut_trampoline(x_calculate_entropy__mutmut_orig, x_calculate_entropy__mutmut_mutants, args, kwargs)
    return result 

calculate_entropy.__signature__ = _mutmut_signature(x_calculate_entropy__mutmut_orig)
x_calculate_entropy__mutmut_orig.__name__ = 'x_calculate_entropy'


def x__pow_mod__mutmut_orig(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), Integer(modulus)))
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_1(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers or SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), Integer(modulus)))
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_2(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(None)
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_3(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(None, Integer(exponent), Integer(modulus)))
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_4(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), None, Integer(modulus)))
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_5(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), None))
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_6(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(exponent), Integer(modulus)))
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_7(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(modulus)))
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_8(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), ))
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_9(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(None), Integer(exponent), Integer(modulus)))
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_10(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(None), Integer(modulus)))
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_11(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), Integer(None)))
    return pow(base, exponent, modulus)


def x__pow_mod__mutmut_12(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), Integer(modulus)))
    return pow(None, exponent, modulus)


def x__pow_mod__mutmut_13(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), Integer(modulus)))
    return pow(base, None, modulus)


def x__pow_mod__mutmut_14(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), Integer(modulus)))
    return pow(base, exponent, None)


def x__pow_mod__mutmut_15(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), Integer(modulus)))
    return pow(exponent, modulus)


def x__pow_mod__mutmut_16(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), Integer(modulus)))
    return pow(base, modulus)


def x__pow_mod__mutmut_17(base: int, exponent: int, modulus: int, use_large_numbers: bool) -> int:
    """Internal helper that mirrors the PyQt app's pow behaviour."""
    if use_large_numbers and SYMPY_AVAILABLE:
        return int(pow(Integer(base), Integer(exponent), Integer(modulus)))
    return pow(base, exponent, )

x__pow_mod__mutmut_mutants : ClassVar[MutantDict] = {
'x__pow_mod__mutmut_1': x__pow_mod__mutmut_1, 
    'x__pow_mod__mutmut_2': x__pow_mod__mutmut_2, 
    'x__pow_mod__mutmut_3': x__pow_mod__mutmut_3, 
    'x__pow_mod__mutmut_4': x__pow_mod__mutmut_4, 
    'x__pow_mod__mutmut_5': x__pow_mod__mutmut_5, 
    'x__pow_mod__mutmut_6': x__pow_mod__mutmut_6, 
    'x__pow_mod__mutmut_7': x__pow_mod__mutmut_7, 
    'x__pow_mod__mutmut_8': x__pow_mod__mutmut_8, 
    'x__pow_mod__mutmut_9': x__pow_mod__mutmut_9, 
    'x__pow_mod__mutmut_10': x__pow_mod__mutmut_10, 
    'x__pow_mod__mutmut_11': x__pow_mod__mutmut_11, 
    'x__pow_mod__mutmut_12': x__pow_mod__mutmut_12, 
    'x__pow_mod__mutmut_13': x__pow_mod__mutmut_13, 
    'x__pow_mod__mutmut_14': x__pow_mod__mutmut_14, 
    'x__pow_mod__mutmut_15': x__pow_mod__mutmut_15, 
    'x__pow_mod__mutmut_16': x__pow_mod__mutmut_16, 
    'x__pow_mod__mutmut_17': x__pow_mod__mutmut_17
}

def _pow_mod(*args, **kwargs):
    result = _mutmut_trampoline(x__pow_mod__mutmut_orig, x__pow_mod__mutmut_mutants, args, kwargs)
    return result 

_pow_mod.__signature__ = _mutmut_signature(x__pow_mod__mutmut_orig)
x__pow_mod__mutmut_orig.__name__ = 'x__pow_mod'


def x_encrypt_block__mutmut_orig(message_block: int, e: int, n: int, use_large_numbers: bool = False) -> int:
    """Encrypt a numeric message block with RSA."""
    return _pow_mod(message_block, e, n, use_large_numbers)


def x_encrypt_block__mutmut_1(message_block: int, e: int, n: int, use_large_numbers: bool = True) -> int:
    """Encrypt a numeric message block with RSA."""
    return _pow_mod(message_block, e, n, use_large_numbers)


def x_encrypt_block__mutmut_2(message_block: int, e: int, n: int, use_large_numbers: bool = False) -> int:
    """Encrypt a numeric message block with RSA."""
    return _pow_mod(None, e, n, use_large_numbers)


def x_encrypt_block__mutmut_3(message_block: int, e: int, n: int, use_large_numbers: bool = False) -> int:
    """Encrypt a numeric message block with RSA."""
    return _pow_mod(message_block, None, n, use_large_numbers)


def x_encrypt_block__mutmut_4(message_block: int, e: int, n: int, use_large_numbers: bool = False) -> int:
    """Encrypt a numeric message block with RSA."""
    return _pow_mod(message_block, e, None, use_large_numbers)


def x_encrypt_block__mutmut_5(message_block: int, e: int, n: int, use_large_numbers: bool = False) -> int:
    """Encrypt a numeric message block with RSA."""
    return _pow_mod(message_block, e, n, None)


def x_encrypt_block__mutmut_6(message_block: int, e: int, n: int, use_large_numbers: bool = False) -> int:
    """Encrypt a numeric message block with RSA."""
    return _pow_mod(e, n, use_large_numbers)


def x_encrypt_block__mutmut_7(message_block: int, e: int, n: int, use_large_numbers: bool = False) -> int:
    """Encrypt a numeric message block with RSA."""
    return _pow_mod(message_block, n, use_large_numbers)


def x_encrypt_block__mutmut_8(message_block: int, e: int, n: int, use_large_numbers: bool = False) -> int:
    """Encrypt a numeric message block with RSA."""
    return _pow_mod(message_block, e, use_large_numbers)


def x_encrypt_block__mutmut_9(message_block: int, e: int, n: int, use_large_numbers: bool = False) -> int:
    """Encrypt a numeric message block with RSA."""
    return _pow_mod(message_block, e, n, )

x_encrypt_block__mutmut_mutants : ClassVar[MutantDict] = {
'x_encrypt_block__mutmut_1': x_encrypt_block__mutmut_1, 
    'x_encrypt_block__mutmut_2': x_encrypt_block__mutmut_2, 
    'x_encrypt_block__mutmut_3': x_encrypt_block__mutmut_3, 
    'x_encrypt_block__mutmut_4': x_encrypt_block__mutmut_4, 
    'x_encrypt_block__mutmut_5': x_encrypt_block__mutmut_5, 
    'x_encrypt_block__mutmut_6': x_encrypt_block__mutmut_6, 
    'x_encrypt_block__mutmut_7': x_encrypt_block__mutmut_7, 
    'x_encrypt_block__mutmut_8': x_encrypt_block__mutmut_8, 
    'x_encrypt_block__mutmut_9': x_encrypt_block__mutmut_9
}

def encrypt_block(*args, **kwargs):
    result = _mutmut_trampoline(x_encrypt_block__mutmut_orig, x_encrypt_block__mutmut_mutants, args, kwargs)
    return result 

encrypt_block.__signature__ = _mutmut_signature(x_encrypt_block__mutmut_orig)
x_encrypt_block__mutmut_orig.__name__ = 'x_encrypt_block'


def x_decrypt_block__mutmut_orig(cipher_block: int, d: int, n: int, use_large_numbers: bool = False) -> int:
    """Decrypt a numeric cipher block with RSA."""
    return _pow_mod(cipher_block, d, n, use_large_numbers)


def x_decrypt_block__mutmut_1(cipher_block: int, d: int, n: int, use_large_numbers: bool = True) -> int:
    """Decrypt a numeric cipher block with RSA."""
    return _pow_mod(cipher_block, d, n, use_large_numbers)


def x_decrypt_block__mutmut_2(cipher_block: int, d: int, n: int, use_large_numbers: bool = False) -> int:
    """Decrypt a numeric cipher block with RSA."""
    return _pow_mod(None, d, n, use_large_numbers)


def x_decrypt_block__mutmut_3(cipher_block: int, d: int, n: int, use_large_numbers: bool = False) -> int:
    """Decrypt a numeric cipher block with RSA."""
    return _pow_mod(cipher_block, None, n, use_large_numbers)


def x_decrypt_block__mutmut_4(cipher_block: int, d: int, n: int, use_large_numbers: bool = False) -> int:
    """Decrypt a numeric cipher block with RSA."""
    return _pow_mod(cipher_block, d, None, use_large_numbers)


def x_decrypt_block__mutmut_5(cipher_block: int, d: int, n: int, use_large_numbers: bool = False) -> int:
    """Decrypt a numeric cipher block with RSA."""
    return _pow_mod(cipher_block, d, n, None)


def x_decrypt_block__mutmut_6(cipher_block: int, d: int, n: int, use_large_numbers: bool = False) -> int:
    """Decrypt a numeric cipher block with RSA."""
    return _pow_mod(d, n, use_large_numbers)


def x_decrypt_block__mutmut_7(cipher_block: int, d: int, n: int, use_large_numbers: bool = False) -> int:
    """Decrypt a numeric cipher block with RSA."""
    return _pow_mod(cipher_block, n, use_large_numbers)


def x_decrypt_block__mutmut_8(cipher_block: int, d: int, n: int, use_large_numbers: bool = False) -> int:
    """Decrypt a numeric cipher block with RSA."""
    return _pow_mod(cipher_block, d, use_large_numbers)


def x_decrypt_block__mutmut_9(cipher_block: int, d: int, n: int, use_large_numbers: bool = False) -> int:
    """Decrypt a numeric cipher block with RSA."""
    return _pow_mod(cipher_block, d, n, )

x_decrypt_block__mutmut_mutants : ClassVar[MutantDict] = {
'x_decrypt_block__mutmut_1': x_decrypt_block__mutmut_1, 
    'x_decrypt_block__mutmut_2': x_decrypt_block__mutmut_2, 
    'x_decrypt_block__mutmut_3': x_decrypt_block__mutmut_3, 
    'x_decrypt_block__mutmut_4': x_decrypt_block__mutmut_4, 
    'x_decrypt_block__mutmut_5': x_decrypt_block__mutmut_5, 
    'x_decrypt_block__mutmut_6': x_decrypt_block__mutmut_6, 
    'x_decrypt_block__mutmut_7': x_decrypt_block__mutmut_7, 
    'x_decrypt_block__mutmut_8': x_decrypt_block__mutmut_8, 
    'x_decrypt_block__mutmut_9': x_decrypt_block__mutmut_9
}

def decrypt_block(*args, **kwargs):
    result = _mutmut_trampoline(x_decrypt_block__mutmut_orig, x_decrypt_block__mutmut_mutants, args, kwargs)
    return result 

decrypt_block.__signature__ = _mutmut_signature(x_decrypt_block__mutmut_orig)
x_decrypt_block__mutmut_orig.__name__ = 'x_decrypt_block'


def x_map_character__mutmut_orig(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(char.lower()) - 96
    if char.isspace():
        return True, 32
    if include_punctuation and char in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return False, None


def x_map_character__mutmut_1(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return False, ord(char.lower()) - 96
    if char.isspace():
        return True, 32
    if include_punctuation and char in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return False, None


def x_map_character__mutmut_2(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(char.lower()) + 96
    if char.isspace():
        return True, 32
    if include_punctuation and char in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return False, None


def x_map_character__mutmut_3(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(None) - 96
    if char.isspace():
        return True, 32
    if include_punctuation and char in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return False, None


def x_map_character__mutmut_4(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(char.upper()) - 96
    if char.isspace():
        return True, 32
    if include_punctuation and char in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return False, None


def x_map_character__mutmut_5(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(char.lower()) - 97
    if char.isspace():
        return True, 32
    if include_punctuation and char in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return False, None


def x_map_character__mutmut_6(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(char.lower()) - 96
    if char.isspace():
        return False, 32
    if include_punctuation and char in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return False, None


def x_map_character__mutmut_7(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(char.lower()) - 96
    if char.isspace():
        return True, 33
    if include_punctuation and char in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return False, None


def x_map_character__mutmut_8(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(char.lower()) - 96
    if char.isspace():
        return True, 32
    if include_punctuation or char in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return False, None


def x_map_character__mutmut_9(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(char.lower()) - 96
    if char.isspace():
        return True, 32
    if include_punctuation and char not in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return False, None


def x_map_character__mutmut_10(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(char.lower()) - 96
    if char.isspace():
        return True, 32
    if include_punctuation and char in PUNCTUATION_MAP:
        return False, PUNCTUATION_MAP[char]
    return False, None


def x_map_character__mutmut_11(char: str, include_punctuation: bool) -> Tuple[bool, int | None]:
    """Map a single character to its numeric representation."""
    if char.isalpha():
        return True, ord(char.lower()) - 96
    if char.isspace():
        return True, 32
    if include_punctuation and char in PUNCTUATION_MAP:
        return True, PUNCTUATION_MAP[char]
    return True, None

x_map_character__mutmut_mutants : ClassVar[MutantDict] = {
'x_map_character__mutmut_1': x_map_character__mutmut_1, 
    'x_map_character__mutmut_2': x_map_character__mutmut_2, 
    'x_map_character__mutmut_3': x_map_character__mutmut_3, 
    'x_map_character__mutmut_4': x_map_character__mutmut_4, 
    'x_map_character__mutmut_5': x_map_character__mutmut_5, 
    'x_map_character__mutmut_6': x_map_character__mutmut_6, 
    'x_map_character__mutmut_7': x_map_character__mutmut_7, 
    'x_map_character__mutmut_8': x_map_character__mutmut_8, 
    'x_map_character__mutmut_9': x_map_character__mutmut_9, 
    'x_map_character__mutmut_10': x_map_character__mutmut_10, 
    'x_map_character__mutmut_11': x_map_character__mutmut_11
}

def map_character(*args, **kwargs):
    result = _mutmut_trampoline(x_map_character__mutmut_orig, x_map_character__mutmut_mutants, args, kwargs)
    return result 

map_character.__signature__ = _mutmut_signature(x_map_character__mutmut_orig)
x_map_character__mutmut_orig.__name__ = 'x_map_character'


def x_map_number__mutmut_orig(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_1(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 2 <= number <= 26:
        return chr(number - 1 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_2(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 < number <= 26:
        return chr(number - 1 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_3(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number < 26:
        return chr(number - 1 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_4(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 27:
        return chr(number - 1 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_5(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(None)
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_6(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 - ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_7(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number + 1 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_8(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 2 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_9(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord(None))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_10(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord("XXAXX"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_11(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord("a"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_12(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord("A"))
    if number != 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_13(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord("A"))
    if number == 33:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_14(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord("A"))
    if number == 32:
        return "XX XX"
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_15(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation or number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_16(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number not in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return ""


def x_map_number__mutmut_17(number: int, include_punctuation: bool) -> str:
    """Map a numeric block back to a character."""
    if 1 <= number <= 26:
        return chr(number - 1 + ord("A"))
    if number == 32:
        return " "
    if include_punctuation and number in REVERSE_PUNCTUATION_MAP:
        return REVERSE_PUNCTUATION_MAP[number]
    return "XXXX"

x_map_number__mutmut_mutants : ClassVar[MutantDict] = {
'x_map_number__mutmut_1': x_map_number__mutmut_1, 
    'x_map_number__mutmut_2': x_map_number__mutmut_2, 
    'x_map_number__mutmut_3': x_map_number__mutmut_3, 
    'x_map_number__mutmut_4': x_map_number__mutmut_4, 
    'x_map_number__mutmut_5': x_map_number__mutmut_5, 
    'x_map_number__mutmut_6': x_map_number__mutmut_6, 
    'x_map_number__mutmut_7': x_map_number__mutmut_7, 
    'x_map_number__mutmut_8': x_map_number__mutmut_8, 
    'x_map_number__mutmut_9': x_map_number__mutmut_9, 
    'x_map_number__mutmut_10': x_map_number__mutmut_10, 
    'x_map_number__mutmut_11': x_map_number__mutmut_11, 
    'x_map_number__mutmut_12': x_map_number__mutmut_12, 
    'x_map_number__mutmut_13': x_map_number__mutmut_13, 
    'x_map_number__mutmut_14': x_map_number__mutmut_14, 
    'x_map_number__mutmut_15': x_map_number__mutmut_15, 
    'x_map_number__mutmut_16': x_map_number__mutmut_16, 
    'x_map_number__mutmut_17': x_map_number__mutmut_17
}

def map_number(*args, **kwargs):
    result = _mutmut_trampoline(x_map_number__mutmut_orig, x_map_number__mutmut_mutants, args, kwargs)
    return result 

map_number.__signature__ = _mutmut_signature(x_map_number__mutmut_orig)
x_map_number__mutmut_orig.__name__ = 'x_map_number'


def x_encrypt_text__mutmut_orig(
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


def x_encrypt_text__mutmut_1(
    text: str,
    e: int,
    n: int,
    include_punctuation: bool = True,
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


def x_encrypt_text__mutmut_2(
    text: str,
    e: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = True,
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


def x_encrypt_text__mutmut_3(
    text: str,
    e: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> EncryptionResult:
    """Encrypt text and return structured details."""
    cipher_blocks: List[int] = None
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


def x_encrypt_text__mutmut_4(
    text: str,
    e: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> EncryptionResult:
    """Encrypt text and return structured details."""
    cipher_blocks: List[int] = []
    plain_blocks: List[int] = None
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


def x_encrypt_text__mutmut_5(
    text: str,
    e: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> EncryptionResult:
    """Encrypt text and return structured details."""
    cipher_blocks: List[int] = []
    plain_blocks: List[int] = []
    skipped: List[str] = None
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


def x_encrypt_text__mutmut_6(
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
    trace: List[str] = None

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


def x_encrypt_text__mutmut_7(
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
        accepted, mapped = None
        if not accepted or mapped is None:
            skipped.append(character)
            continue

        encrypted = encrypt_block(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_8(
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
        accepted, mapped = map_character(None, include_punctuation)
        if not accepted or mapped is None:
            skipped.append(character)
            continue

        encrypted = encrypt_block(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_9(
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
        accepted, mapped = map_character(character, None)
        if not accepted or mapped is None:
            skipped.append(character)
            continue

        encrypted = encrypt_block(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_10(
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
        accepted, mapped = map_character(include_punctuation)
        if not accepted or mapped is None:
            skipped.append(character)
            continue

        encrypted = encrypt_block(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_11(
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
        accepted, mapped = map_character(character, )
        if not accepted or mapped is None:
            skipped.append(character)
            continue

        encrypted = encrypt_block(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_12(
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
        if not accepted and mapped is None:
            skipped.append(character)
            continue

        encrypted = encrypt_block(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_13(
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
        if accepted or mapped is None:
            skipped.append(character)
            continue

        encrypted = encrypt_block(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_14(
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
        if not accepted or mapped is not None:
            skipped.append(character)
            continue

        encrypted = encrypt_block(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_15(
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
            skipped.append(None)
            continue

        encrypted = encrypt_block(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_16(
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
            break

        encrypted = encrypt_block(mapped, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_17(
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

        encrypted = None
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_18(
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

        encrypted = encrypt_block(None, e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_19(
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

        encrypted = encrypt_block(mapped, None, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_20(
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

        encrypted = encrypt_block(mapped, e, None, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_21(
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

        encrypted = encrypt_block(mapped, e, n, None)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_22(
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

        encrypted = encrypt_block(e, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_23(
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

        encrypted = encrypt_block(mapped, n, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_24(
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

        encrypted = encrypt_block(mapped, e, use_large_numbers)
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_25(
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

        encrypted = encrypt_block(mapped, e, n, )
        cipher_blocks.append(encrypted)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_26(
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
        cipher_blocks.append(None)
        plain_blocks.append(mapped)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_27(
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
        plain_blocks.append(None)
        trace.append(f"Encrypting: {mapped} → {encrypted}")

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_28(
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
        trace.append(None)

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_29(
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

    return EncryptionResult(None, plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_30(
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

    return EncryptionResult(cipher_blocks, None, skipped, trace)


def x_encrypt_text__mutmut_31(
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

    return EncryptionResult(cipher_blocks, plain_blocks, None, trace)


def x_encrypt_text__mutmut_32(
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

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, None)


def x_encrypt_text__mutmut_33(
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

    return EncryptionResult(plain_blocks, skipped, trace)


def x_encrypt_text__mutmut_34(
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

    return EncryptionResult(cipher_blocks, skipped, trace)


def x_encrypt_text__mutmut_35(
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

    return EncryptionResult(cipher_blocks, plain_blocks, trace)


def x_encrypt_text__mutmut_36(
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

    return EncryptionResult(cipher_blocks, plain_blocks, skipped, )

x_encrypt_text__mutmut_mutants : ClassVar[MutantDict] = {
'x_encrypt_text__mutmut_1': x_encrypt_text__mutmut_1, 
    'x_encrypt_text__mutmut_2': x_encrypt_text__mutmut_2, 
    'x_encrypt_text__mutmut_3': x_encrypt_text__mutmut_3, 
    'x_encrypt_text__mutmut_4': x_encrypt_text__mutmut_4, 
    'x_encrypt_text__mutmut_5': x_encrypt_text__mutmut_5, 
    'x_encrypt_text__mutmut_6': x_encrypt_text__mutmut_6, 
    'x_encrypt_text__mutmut_7': x_encrypt_text__mutmut_7, 
    'x_encrypt_text__mutmut_8': x_encrypt_text__mutmut_8, 
    'x_encrypt_text__mutmut_9': x_encrypt_text__mutmut_9, 
    'x_encrypt_text__mutmut_10': x_encrypt_text__mutmut_10, 
    'x_encrypt_text__mutmut_11': x_encrypt_text__mutmut_11, 
    'x_encrypt_text__mutmut_12': x_encrypt_text__mutmut_12, 
    'x_encrypt_text__mutmut_13': x_encrypt_text__mutmut_13, 
    'x_encrypt_text__mutmut_14': x_encrypt_text__mutmut_14, 
    'x_encrypt_text__mutmut_15': x_encrypt_text__mutmut_15, 
    'x_encrypt_text__mutmut_16': x_encrypt_text__mutmut_16, 
    'x_encrypt_text__mutmut_17': x_encrypt_text__mutmut_17, 
    'x_encrypt_text__mutmut_18': x_encrypt_text__mutmut_18, 
    'x_encrypt_text__mutmut_19': x_encrypt_text__mutmut_19, 
    'x_encrypt_text__mutmut_20': x_encrypt_text__mutmut_20, 
    'x_encrypt_text__mutmut_21': x_encrypt_text__mutmut_21, 
    'x_encrypt_text__mutmut_22': x_encrypt_text__mutmut_22, 
    'x_encrypt_text__mutmut_23': x_encrypt_text__mutmut_23, 
    'x_encrypt_text__mutmut_24': x_encrypt_text__mutmut_24, 
    'x_encrypt_text__mutmut_25': x_encrypt_text__mutmut_25, 
    'x_encrypt_text__mutmut_26': x_encrypt_text__mutmut_26, 
    'x_encrypt_text__mutmut_27': x_encrypt_text__mutmut_27, 
    'x_encrypt_text__mutmut_28': x_encrypt_text__mutmut_28, 
    'x_encrypt_text__mutmut_29': x_encrypt_text__mutmut_29, 
    'x_encrypt_text__mutmut_30': x_encrypt_text__mutmut_30, 
    'x_encrypt_text__mutmut_31': x_encrypt_text__mutmut_31, 
    'x_encrypt_text__mutmut_32': x_encrypt_text__mutmut_32, 
    'x_encrypt_text__mutmut_33': x_encrypt_text__mutmut_33, 
    'x_encrypt_text__mutmut_34': x_encrypt_text__mutmut_34, 
    'x_encrypt_text__mutmut_35': x_encrypt_text__mutmut_35, 
    'x_encrypt_text__mutmut_36': x_encrypt_text__mutmut_36
}

def encrypt_text(*args, **kwargs):
    result = _mutmut_trampoline(x_encrypt_text__mutmut_orig, x_encrypt_text__mutmut_mutants, args, kwargs)
    return result 

encrypt_text.__signature__ = _mutmut_signature(x_encrypt_text__mutmut_orig)
x_encrypt_text__mutmut_orig.__name__ = 'x_encrypt_text'


def x_decrypt_numbers__mutmut_orig(
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


def x_decrypt_numbers__mutmut_1(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = True,
    use_large_numbers: bool = False,
) -> List[int]:
    """Decrypt RSA cipher numbers back to their numeric representation."""
    return [
        decrypt_block(cipher, d, n, use_large_numbers)
        for cipher in cipher_numbers
    ]


def x_decrypt_numbers__mutmut_2(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = True,
) -> List[int]:
    """Decrypt RSA cipher numbers back to their numeric representation."""
    return [
        decrypt_block(cipher, d, n, use_large_numbers)
        for cipher in cipher_numbers
    ]


def x_decrypt_numbers__mutmut_3(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> List[int]:
    """Decrypt RSA cipher numbers back to their numeric representation."""
    return [
        decrypt_block(None, d, n, use_large_numbers)
        for cipher in cipher_numbers
    ]


def x_decrypt_numbers__mutmut_4(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> List[int]:
    """Decrypt RSA cipher numbers back to their numeric representation."""
    return [
        decrypt_block(cipher, None, n, use_large_numbers)
        for cipher in cipher_numbers
    ]


def x_decrypt_numbers__mutmut_5(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> List[int]:
    """Decrypt RSA cipher numbers back to their numeric representation."""
    return [
        decrypt_block(cipher, d, None, use_large_numbers)
        for cipher in cipher_numbers
    ]


def x_decrypt_numbers__mutmut_6(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> List[int]:
    """Decrypt RSA cipher numbers back to their numeric representation."""
    return [
        decrypt_block(cipher, d, n, None)
        for cipher in cipher_numbers
    ]


def x_decrypt_numbers__mutmut_7(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> List[int]:
    """Decrypt RSA cipher numbers back to their numeric representation."""
    return [
        decrypt_block(d, n, use_large_numbers)
        for cipher in cipher_numbers
    ]


def x_decrypt_numbers__mutmut_8(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> List[int]:
    """Decrypt RSA cipher numbers back to their numeric representation."""
    return [
        decrypt_block(cipher, n, use_large_numbers)
        for cipher in cipher_numbers
    ]


def x_decrypt_numbers__mutmut_9(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> List[int]:
    """Decrypt RSA cipher numbers back to their numeric representation."""
    return [
        decrypt_block(cipher, d, use_large_numbers)
        for cipher in cipher_numbers
    ]


def x_decrypt_numbers__mutmut_10(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> List[int]:
    """Decrypt RSA cipher numbers back to their numeric representation."""
    return [
        decrypt_block(cipher, d, n, )
        for cipher in cipher_numbers
    ]

x_decrypt_numbers__mutmut_mutants : ClassVar[MutantDict] = {
'x_decrypt_numbers__mutmut_1': x_decrypt_numbers__mutmut_1, 
    'x_decrypt_numbers__mutmut_2': x_decrypt_numbers__mutmut_2, 
    'x_decrypt_numbers__mutmut_3': x_decrypt_numbers__mutmut_3, 
    'x_decrypt_numbers__mutmut_4': x_decrypt_numbers__mutmut_4, 
    'x_decrypt_numbers__mutmut_5': x_decrypt_numbers__mutmut_5, 
    'x_decrypt_numbers__mutmut_6': x_decrypt_numbers__mutmut_6, 
    'x_decrypt_numbers__mutmut_7': x_decrypt_numbers__mutmut_7, 
    'x_decrypt_numbers__mutmut_8': x_decrypt_numbers__mutmut_8, 
    'x_decrypt_numbers__mutmut_9': x_decrypt_numbers__mutmut_9, 
    'x_decrypt_numbers__mutmut_10': x_decrypt_numbers__mutmut_10
}

def decrypt_numbers(*args, **kwargs):
    result = _mutmut_trampoline(x_decrypt_numbers__mutmut_orig, x_decrypt_numbers__mutmut_mutants, args, kwargs)
    return result 

decrypt_numbers.__signature__ = _mutmut_signature(x_decrypt_numbers__mutmut_orig)
x_decrypt_numbers__mutmut_orig.__name__ = 'x_decrypt_numbers'


def x_decrypt_text_blocks__mutmut_orig(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_1(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = True,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_2(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = True,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_3(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = None
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_4(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(None, d, n, include_punctuation, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_5(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, None, n, include_punctuation, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_6(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, None, include_punctuation, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_7(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, None, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_8(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, None)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_9(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(d, n, include_punctuation, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_10(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, n, include_punctuation, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_11(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, include_punctuation, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_12(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, use_large_numbers)
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_13(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, )
    return "".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_14(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, use_large_numbers)
    return "".join(None)


def x_decrypt_text_blocks__mutmut_15(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, use_large_numbers)
    return "XXXX".join(map_number(number, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_16(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, use_large_numbers)
    return "".join(map_number(None, include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_17(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, use_large_numbers)
    return "".join(map_number(number, None) for number in numbers)


def x_decrypt_text_blocks__mutmut_18(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, use_large_numbers)
    return "".join(map_number(include_punctuation) for number in numbers)


def x_decrypt_text_blocks__mutmut_19(
    cipher_numbers: Sequence[int],
    d: int,
    n: int,
    include_punctuation: bool = False,
    use_large_numbers: bool = False,
) -> str:
    """Decrypt numbers and convert to text."""
    numbers = decrypt_numbers(cipher_numbers, d, n, include_punctuation, use_large_numbers)
    return "".join(map_number(number, ) for number in numbers)

x_decrypt_text_blocks__mutmut_mutants : ClassVar[MutantDict] = {
'x_decrypt_text_blocks__mutmut_1': x_decrypt_text_blocks__mutmut_1, 
    'x_decrypt_text_blocks__mutmut_2': x_decrypt_text_blocks__mutmut_2, 
    'x_decrypt_text_blocks__mutmut_3': x_decrypt_text_blocks__mutmut_3, 
    'x_decrypt_text_blocks__mutmut_4': x_decrypt_text_blocks__mutmut_4, 
    'x_decrypt_text_blocks__mutmut_5': x_decrypt_text_blocks__mutmut_5, 
    'x_decrypt_text_blocks__mutmut_6': x_decrypt_text_blocks__mutmut_6, 
    'x_decrypt_text_blocks__mutmut_7': x_decrypt_text_blocks__mutmut_7, 
    'x_decrypt_text_blocks__mutmut_8': x_decrypt_text_blocks__mutmut_8, 
    'x_decrypt_text_blocks__mutmut_9': x_decrypt_text_blocks__mutmut_9, 
    'x_decrypt_text_blocks__mutmut_10': x_decrypt_text_blocks__mutmut_10, 
    'x_decrypt_text_blocks__mutmut_11': x_decrypt_text_blocks__mutmut_11, 
    'x_decrypt_text_blocks__mutmut_12': x_decrypt_text_blocks__mutmut_12, 
    'x_decrypt_text_blocks__mutmut_13': x_decrypt_text_blocks__mutmut_13, 
    'x_decrypt_text_blocks__mutmut_14': x_decrypt_text_blocks__mutmut_14, 
    'x_decrypt_text_blocks__mutmut_15': x_decrypt_text_blocks__mutmut_15, 
    'x_decrypt_text_blocks__mutmut_16': x_decrypt_text_blocks__mutmut_16, 
    'x_decrypt_text_blocks__mutmut_17': x_decrypt_text_blocks__mutmut_17, 
    'x_decrypt_text_blocks__mutmut_18': x_decrypt_text_blocks__mutmut_18, 
    'x_decrypt_text_blocks__mutmut_19': x_decrypt_text_blocks__mutmut_19
}

def decrypt_text_blocks(*args, **kwargs):
    result = _mutmut_trampoline(x_decrypt_text_blocks__mutmut_orig, x_decrypt_text_blocks__mutmut_mutants, args, kwargs)
    return result 

decrypt_text_blocks.__signature__ = _mutmut_signature(x_decrypt_text_blocks__mutmut_orig)
x_decrypt_text_blocks__mutmut_orig.__name__ = 'x_decrypt_text_blocks'


def x_generate_secure_primes__mutmut_orig(bits: int) -> Tuple[int, int]:
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


def x_generate_secure_primes__mutmut_1(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits < 0:
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


def x_generate_secure_primes__mutmut_2(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 1:
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


def x_generate_secure_primes__mutmut_3(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError(None)

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


def x_generate_secure_primes__mutmut_4(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("XXbits must be positiveXX")

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


def x_generate_secure_primes__mutmut_5(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("BITS MUST BE POSITIVE")

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


def x_generate_secure_primes__mutmut_6(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("bits must be positive")

    if CRYPTO_AVAILABLE or number is not None:
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


def x_generate_secure_primes__mutmut_7(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("bits must be positive")

    if CRYPTO_AVAILABLE and number is None:
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


def x_generate_secure_primes__mutmut_8(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("bits must be positive")

    if CRYPTO_AVAILABLE and number is not None:
        p = None
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


def x_generate_secure_primes__mutmut_9(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("bits must be positive")

    if CRYPTO_AVAILABLE and number is not None:
        p = number.getPrime(None)
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


def x_generate_secure_primes__mutmut_10(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("bits must be positive")

    if CRYPTO_AVAILABLE and number is not None:
        p = number.getPrime(bits)
        q = None
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


def x_generate_secure_primes__mutmut_11(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("bits must be positive")

    if CRYPTO_AVAILABLE and number is not None:
        p = number.getPrime(bits)
        q = number.getPrime(None)
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


def x_generate_secure_primes__mutmut_12(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("bits must be positive")

    if CRYPTO_AVAILABLE and number is not None:
        p = number.getPrime(bits)
        q = number.getPrime(bits)
        while p != q:
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


def x_generate_secure_primes__mutmut_13(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("bits must be positive")

    if CRYPTO_AVAILABLE and number is not None:
        p = number.getPrime(bits)
        q = number.getPrime(bits)
        while p == q:
            q = None
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


def x_generate_secure_primes__mutmut_14(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("bits must be positive")

    if CRYPTO_AVAILABLE and number is not None:
        p = number.getPrime(bits)
        q = number.getPrime(bits)
        while p == q:
            q = number.getPrime(None)
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


def x_generate_secure_primes__mutmut_15(bits: int) -> Tuple[int, int]:
    """Return a pair of distinct primes of approximately `bits` length."""
    if bits <= 0:
        raise ValueError("bits must be positive")

    if CRYPTO_AVAILABLE and number is not None:
        p = number.getPrime(bits)
        q = number.getPrime(bits)
        while p == q:
            q = number.getPrime(bits)
        return p, q

    fallback_primes = None
    p = random.choice(fallback_primes)
    remaining = [candidate for candidate in fallback_primes if candidate != p]
    q = random.choice(remaining)
    return p, q


def x_generate_secure_primes__mutmut_16(bits: int) -> Tuple[int, int]:
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
        12,
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


def x_generate_secure_primes__mutmut_17(bits: int) -> Tuple[int, int]:
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
        14,
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


def x_generate_secure_primes__mutmut_18(bits: int) -> Tuple[int, int]:
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
        18,
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


def x_generate_secure_primes__mutmut_19(bits: int) -> Tuple[int, int]:
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
        20,
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


def x_generate_secure_primes__mutmut_20(bits: int) -> Tuple[int, int]:
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
        24,
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


def x_generate_secure_primes__mutmut_21(bits: int) -> Tuple[int, int]:
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
        30,
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


def x_generate_secure_primes__mutmut_22(bits: int) -> Tuple[int, int]:
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
        32,
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


def x_generate_secure_primes__mutmut_23(bits: int) -> Tuple[int, int]:
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
        38,
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


def x_generate_secure_primes__mutmut_24(bits: int) -> Tuple[int, int]:
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
        42,
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


def x_generate_secure_primes__mutmut_25(bits: int) -> Tuple[int, int]:
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
        44,
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


def x_generate_secure_primes__mutmut_26(bits: int) -> Tuple[int, int]:
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
        48,
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


def x_generate_secure_primes__mutmut_27(bits: int) -> Tuple[int, int]:
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
        54,
        59,
        61,
        67,
        71,
    ]
    p = random.choice(fallback_primes)
    remaining = [candidate for candidate in fallback_primes if candidate != p]
    q = random.choice(remaining)
    return p, q


def x_generate_secure_primes__mutmut_28(bits: int) -> Tuple[int, int]:
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
        60,
        61,
        67,
        71,
    ]
    p = random.choice(fallback_primes)
    remaining = [candidate for candidate in fallback_primes if candidate != p]
    q = random.choice(remaining)
    return p, q


def x_generate_secure_primes__mutmut_29(bits: int) -> Tuple[int, int]:
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
        62,
        67,
        71,
    ]
    p = random.choice(fallback_primes)
    remaining = [candidate for candidate in fallback_primes if candidate != p]
    q = random.choice(remaining)
    return p, q


def x_generate_secure_primes__mutmut_30(bits: int) -> Tuple[int, int]:
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
        68,
        71,
    ]
    p = random.choice(fallback_primes)
    remaining = [candidate for candidate in fallback_primes if candidate != p]
    q = random.choice(remaining)
    return p, q


def x_generate_secure_primes__mutmut_31(bits: int) -> Tuple[int, int]:
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
        72,
    ]
    p = random.choice(fallback_primes)
    remaining = [candidate for candidate in fallback_primes if candidate != p]
    q = random.choice(remaining)
    return p, q


def x_generate_secure_primes__mutmut_32(bits: int) -> Tuple[int, int]:
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
    p = None
    remaining = [candidate for candidate in fallback_primes if candidate != p]
    q = random.choice(remaining)
    return p, q


def x_generate_secure_primes__mutmut_33(bits: int) -> Tuple[int, int]:
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
    p = random.choice(None)
    remaining = [candidate for candidate in fallback_primes if candidate != p]
    q = random.choice(remaining)
    return p, q


def x_generate_secure_primes__mutmut_34(bits: int) -> Tuple[int, int]:
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
    remaining = None
    q = random.choice(remaining)
    return p, q


def x_generate_secure_primes__mutmut_35(bits: int) -> Tuple[int, int]:
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
    remaining = [candidate for candidate in fallback_primes if candidate == p]
    q = random.choice(remaining)
    return p, q


def x_generate_secure_primes__mutmut_36(bits: int) -> Tuple[int, int]:
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
    q = None
    return p, q


def x_generate_secure_primes__mutmut_37(bits: int) -> Tuple[int, int]:
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
    q = random.choice(None)
    return p, q

x_generate_secure_primes__mutmut_mutants : ClassVar[MutantDict] = {
'x_generate_secure_primes__mutmut_1': x_generate_secure_primes__mutmut_1, 
    'x_generate_secure_primes__mutmut_2': x_generate_secure_primes__mutmut_2, 
    'x_generate_secure_primes__mutmut_3': x_generate_secure_primes__mutmut_3, 
    'x_generate_secure_primes__mutmut_4': x_generate_secure_primes__mutmut_4, 
    'x_generate_secure_primes__mutmut_5': x_generate_secure_primes__mutmut_5, 
    'x_generate_secure_primes__mutmut_6': x_generate_secure_primes__mutmut_6, 
    'x_generate_secure_primes__mutmut_7': x_generate_secure_primes__mutmut_7, 
    'x_generate_secure_primes__mutmut_8': x_generate_secure_primes__mutmut_8, 
    'x_generate_secure_primes__mutmut_9': x_generate_secure_primes__mutmut_9, 
    'x_generate_secure_primes__mutmut_10': x_generate_secure_primes__mutmut_10, 
    'x_generate_secure_primes__mutmut_11': x_generate_secure_primes__mutmut_11, 
    'x_generate_secure_primes__mutmut_12': x_generate_secure_primes__mutmut_12, 
    'x_generate_secure_primes__mutmut_13': x_generate_secure_primes__mutmut_13, 
    'x_generate_secure_primes__mutmut_14': x_generate_secure_primes__mutmut_14, 
    'x_generate_secure_primes__mutmut_15': x_generate_secure_primes__mutmut_15, 
    'x_generate_secure_primes__mutmut_16': x_generate_secure_primes__mutmut_16, 
    'x_generate_secure_primes__mutmut_17': x_generate_secure_primes__mutmut_17, 
    'x_generate_secure_primes__mutmut_18': x_generate_secure_primes__mutmut_18, 
    'x_generate_secure_primes__mutmut_19': x_generate_secure_primes__mutmut_19, 
    'x_generate_secure_primes__mutmut_20': x_generate_secure_primes__mutmut_20, 
    'x_generate_secure_primes__mutmut_21': x_generate_secure_primes__mutmut_21, 
    'x_generate_secure_primes__mutmut_22': x_generate_secure_primes__mutmut_22, 
    'x_generate_secure_primes__mutmut_23': x_generate_secure_primes__mutmut_23, 
    'x_generate_secure_primes__mutmut_24': x_generate_secure_primes__mutmut_24, 
    'x_generate_secure_primes__mutmut_25': x_generate_secure_primes__mutmut_25, 
    'x_generate_secure_primes__mutmut_26': x_generate_secure_primes__mutmut_26, 
    'x_generate_secure_primes__mutmut_27': x_generate_secure_primes__mutmut_27, 
    'x_generate_secure_primes__mutmut_28': x_generate_secure_primes__mutmut_28, 
    'x_generate_secure_primes__mutmut_29': x_generate_secure_primes__mutmut_29, 
    'x_generate_secure_primes__mutmut_30': x_generate_secure_primes__mutmut_30, 
    'x_generate_secure_primes__mutmut_31': x_generate_secure_primes__mutmut_31, 
    'x_generate_secure_primes__mutmut_32': x_generate_secure_primes__mutmut_32, 
    'x_generate_secure_primes__mutmut_33': x_generate_secure_primes__mutmut_33, 
    'x_generate_secure_primes__mutmut_34': x_generate_secure_primes__mutmut_34, 
    'x_generate_secure_primes__mutmut_35': x_generate_secure_primes__mutmut_35, 
    'x_generate_secure_primes__mutmut_36': x_generate_secure_primes__mutmut_36, 
    'x_generate_secure_primes__mutmut_37': x_generate_secure_primes__mutmut_37
}

def generate_secure_primes(*args, **kwargs):
    result = _mutmut_trampoline(x_generate_secure_primes__mutmut_orig, x_generate_secure_primes__mutmut_mutants, args, kwargs)
    return result 

generate_secure_primes.__signature__ = _mutmut_signature(x_generate_secure_primes__mutmut_orig)
x_generate_secure_primes__mutmut_orig.__name__ = 'x_generate_secure_primes'


def x_tokenize_cipher_text__mutmut_orig(cipher_text: str) -> List[int]:
    """Parse space separated cipher numbers, ignoring invalid tokens."""
    numbers: List[int] = []
    for token in cipher_text.split():
        try:
            numbers.append(int(token))
        except ValueError:
            continue
    return numbers


def x_tokenize_cipher_text__mutmut_1(cipher_text: str) -> List[int]:
    """Parse space separated cipher numbers, ignoring invalid tokens."""
    numbers: List[int] = None
    for token in cipher_text.split():
        try:
            numbers.append(int(token))
        except ValueError:
            continue
    return numbers


def x_tokenize_cipher_text__mutmut_2(cipher_text: str) -> List[int]:
    """Parse space separated cipher numbers, ignoring invalid tokens."""
    numbers: List[int] = []
    for token in cipher_text.split():
        try:
            numbers.append(None)
        except ValueError:
            continue
    return numbers


def x_tokenize_cipher_text__mutmut_3(cipher_text: str) -> List[int]:
    """Parse space separated cipher numbers, ignoring invalid tokens."""
    numbers: List[int] = []
    for token in cipher_text.split():
        try:
            numbers.append(int(None))
        except ValueError:
            continue
    return numbers


def x_tokenize_cipher_text__mutmut_4(cipher_text: str) -> List[int]:
    """Parse space separated cipher numbers, ignoring invalid tokens."""
    numbers: List[int] = []
    for token in cipher_text.split():
        try:
            numbers.append(int(token))
        except ValueError:
            break
    return numbers

x_tokenize_cipher_text__mutmut_mutants : ClassVar[MutantDict] = {
'x_tokenize_cipher_text__mutmut_1': x_tokenize_cipher_text__mutmut_1, 
    'x_tokenize_cipher_text__mutmut_2': x_tokenize_cipher_text__mutmut_2, 
    'x_tokenize_cipher_text__mutmut_3': x_tokenize_cipher_text__mutmut_3, 
    'x_tokenize_cipher_text__mutmut_4': x_tokenize_cipher_text__mutmut_4
}

def tokenize_cipher_text(*args, **kwargs):
    result = _mutmut_trampoline(x_tokenize_cipher_text__mutmut_orig, x_tokenize_cipher_text__mutmut_mutants, args, kwargs)
    return result 

tokenize_cipher_text.__signature__ = _mutmut_signature(x_tokenize_cipher_text__mutmut_orig)
x_tokenize_cipher_text__mutmut_orig.__name__ = 'x_tokenize_cipher_text'


def x_validate_entropy_bounds__mutmut_orig(n: int, expected_min_bits: float) -> None:
    """Raise ValueError when entropy drops below the expected lower bound."""
    entropy = calculate_entropy(n)
    if entropy < expected_min_bits:
        raise ValueError(f"Entropy {entropy:.2f} bits is below expected minimum {expected_min_bits}")


def x_validate_entropy_bounds__mutmut_1(n: int, expected_min_bits: float) -> None:
    """Raise ValueError when entropy drops below the expected lower bound."""
    entropy = None
    if entropy < expected_min_bits:
        raise ValueError(f"Entropy {entropy:.2f} bits is below expected minimum {expected_min_bits}")


def x_validate_entropy_bounds__mutmut_2(n: int, expected_min_bits: float) -> None:
    """Raise ValueError when entropy drops below the expected lower bound."""
    entropy = calculate_entropy(None)
    if entropy < expected_min_bits:
        raise ValueError(f"Entropy {entropy:.2f} bits is below expected minimum {expected_min_bits}")


def x_validate_entropy_bounds__mutmut_3(n: int, expected_min_bits: float) -> None:
    """Raise ValueError when entropy drops below the expected lower bound."""
    entropy = calculate_entropy(n)
    if entropy <= expected_min_bits:
        raise ValueError(f"Entropy {entropy:.2f} bits is below expected minimum {expected_min_bits}")


def x_validate_entropy_bounds__mutmut_4(n: int, expected_min_bits: float) -> None:
    """Raise ValueError when entropy drops below the expected lower bound."""
    entropy = calculate_entropy(n)
    if entropy < expected_min_bits:
        raise ValueError(None)

x_validate_entropy_bounds__mutmut_mutants : ClassVar[MutantDict] = {
'x_validate_entropy_bounds__mutmut_1': x_validate_entropy_bounds__mutmut_1, 
    'x_validate_entropy_bounds__mutmut_2': x_validate_entropy_bounds__mutmut_2, 
    'x_validate_entropy_bounds__mutmut_3': x_validate_entropy_bounds__mutmut_3, 
    'x_validate_entropy_bounds__mutmut_4': x_validate_entropy_bounds__mutmut_4
}

def validate_entropy_bounds(*args, **kwargs):
    result = _mutmut_trampoline(x_validate_entropy_bounds__mutmut_orig, x_validate_entropy_bounds__mutmut_mutants, args, kwargs)
    return result 

validate_entropy_bounds.__signature__ = _mutmut_signature(x_validate_entropy_bounds__mutmut_orig)
x_validate_entropy_bounds__mutmut_orig.__name__ = 'x_validate_entropy_bounds'


def x_ensure_coprime__mutmut_orig(e: int, phi: int) -> None:
    """Validate that e and phi are coprime."""
    if gcd(e, phi) != 1:
        raise ValueError(f"e={e} is not coprime with φ(n)={phi}")


def x_ensure_coprime__mutmut_1(e: int, phi: int) -> None:
    """Validate that e and phi are coprime."""
    if gcd(None, phi) != 1:
        raise ValueError(f"e={e} is not coprime with φ(n)={phi}")


def x_ensure_coprime__mutmut_2(e: int, phi: int) -> None:
    """Validate that e and phi are coprime."""
    if gcd(e, None) != 1:
        raise ValueError(f"e={e} is not coprime with φ(n)={phi}")


def x_ensure_coprime__mutmut_3(e: int, phi: int) -> None:
    """Validate that e and phi are coprime."""
    if gcd(phi) != 1:
        raise ValueError(f"e={e} is not coprime with φ(n)={phi}")


def x_ensure_coprime__mutmut_4(e: int, phi: int) -> None:
    """Validate that e and phi are coprime."""
    if gcd(e, ) != 1:
        raise ValueError(f"e={e} is not coprime with φ(n)={phi}")


def x_ensure_coprime__mutmut_5(e: int, phi: int) -> None:
    """Validate that e and phi are coprime."""
    if gcd(e, phi) == 1:
        raise ValueError(f"e={e} is not coprime with φ(n)={phi}")


def x_ensure_coprime__mutmut_6(e: int, phi: int) -> None:
    """Validate that e and phi are coprime."""
    if gcd(e, phi) != 2:
        raise ValueError(f"e={e} is not coprime with φ(n)={phi}")


def x_ensure_coprime__mutmut_7(e: int, phi: int) -> None:
    """Validate that e and phi are coprime."""
    if gcd(e, phi) != 1:
        raise ValueError(None)

x_ensure_coprime__mutmut_mutants : ClassVar[MutantDict] = {
'x_ensure_coprime__mutmut_1': x_ensure_coprime__mutmut_1, 
    'x_ensure_coprime__mutmut_2': x_ensure_coprime__mutmut_2, 
    'x_ensure_coprime__mutmut_3': x_ensure_coprime__mutmut_3, 
    'x_ensure_coprime__mutmut_4': x_ensure_coprime__mutmut_4, 
    'x_ensure_coprime__mutmut_5': x_ensure_coprime__mutmut_5, 
    'x_ensure_coprime__mutmut_6': x_ensure_coprime__mutmut_6, 
    'x_ensure_coprime__mutmut_7': x_ensure_coprime__mutmut_7
}

def ensure_coprime(*args, **kwargs):
    result = _mutmut_trampoline(x_ensure_coprime__mutmut_orig, x_ensure_coprime__mutmut_mutants, args, kwargs)
    return result 

ensure_coprime.__signature__ = _mutmut_signature(x_ensure_coprime__mutmut_orig)
x_ensure_coprime__mutmut_orig.__name__ = 'x_ensure_coprime'


def x_encrypt_with_reference__mutmut_orig(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, e, n, include_punctuation=True, use_large_numbers=False)
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_1(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = None
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_2(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(None, e, n, include_punctuation=True, use_large_numbers=False)
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_3(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, None, n, include_punctuation=True, use_large_numbers=False)
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_4(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, e, None, include_punctuation=True, use_large_numbers=False)
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_5(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, e, n, include_punctuation=None, use_large_numbers=False)
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_6(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, e, n, include_punctuation=True, use_large_numbers=None)
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_7(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(e, n, include_punctuation=True, use_large_numbers=False)
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_8(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, n, include_punctuation=True, use_large_numbers=False)
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_9(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, e, include_punctuation=True, use_large_numbers=False)
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_10(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, e, n, use_large_numbers=False)
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_11(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, e, n, include_punctuation=True, )
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_12(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, e, n, include_punctuation=False, use_large_numbers=False)
    return result.cipher_blocks


def x_encrypt_with_reference__mutmut_13(text: str, e: int, n: int) -> List[int]:
    """Utility used in differential tests."""
    result = encrypt_text(text, e, n, include_punctuation=True, use_large_numbers=True)
    return result.cipher_blocks

x_encrypt_with_reference__mutmut_mutants : ClassVar[MutantDict] = {
'x_encrypt_with_reference__mutmut_1': x_encrypt_with_reference__mutmut_1, 
    'x_encrypt_with_reference__mutmut_2': x_encrypt_with_reference__mutmut_2, 
    'x_encrypt_with_reference__mutmut_3': x_encrypt_with_reference__mutmut_3, 
    'x_encrypt_with_reference__mutmut_4': x_encrypt_with_reference__mutmut_4, 
    'x_encrypt_with_reference__mutmut_5': x_encrypt_with_reference__mutmut_5, 
    'x_encrypt_with_reference__mutmut_6': x_encrypt_with_reference__mutmut_6, 
    'x_encrypt_with_reference__mutmut_7': x_encrypt_with_reference__mutmut_7, 
    'x_encrypt_with_reference__mutmut_8': x_encrypt_with_reference__mutmut_8, 
    'x_encrypt_with_reference__mutmut_9': x_encrypt_with_reference__mutmut_9, 
    'x_encrypt_with_reference__mutmut_10': x_encrypt_with_reference__mutmut_10, 
    'x_encrypt_with_reference__mutmut_11': x_encrypt_with_reference__mutmut_11, 
    'x_encrypt_with_reference__mutmut_12': x_encrypt_with_reference__mutmut_12, 
    'x_encrypt_with_reference__mutmut_13': x_encrypt_with_reference__mutmut_13
}

def encrypt_with_reference(*args, **kwargs):
    result = _mutmut_trampoline(x_encrypt_with_reference__mutmut_orig, x_encrypt_with_reference__mutmut_mutants, args, kwargs)
    return result 

encrypt_with_reference.__signature__ = _mutmut_signature(x_encrypt_with_reference__mutmut_orig)
x_encrypt_with_reference__mutmut_orig.__name__ = 'x_encrypt_with_reference'
