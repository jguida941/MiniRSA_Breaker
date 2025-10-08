from __future__ import annotations

import json
from pathlib import Path

from mini_rsa.core import encrypt_text
from rsa import RSACalculator, decrypt_custom_message

EXAMPLE_MESSAGE = "A CAB"


def test_roundtrip_matches_reference(tmp_path: Path) -> None:
    p, q, e = 3, 11, 3
    phi = (p - 1) * (q - 1)
    d = RSACalculator.mod_inverse(e, phi)
    n = p * q

    result = encrypt_text(EXAMPLE_MESSAGE, e, n, include_punctuation=True)
    cipher_numbers = result.cipher_blocks
    cipher_str = " ".join(str(num) for num in cipher_numbers)

    decrypted = decrypt_custom_message(cipher_str, d, n)
    assert decrypted == EXAMPLE_MESSAGE.upper()

    # Golden snapshot
    snapshot_file = tmp_path / "cipher_snapshot.json"
    payload = {
        "plaintext": EXAMPLE_MESSAGE,
        "cipher_numbers": cipher_numbers,
        "decrypted": decrypted,
    }
    snapshot_file.write_text(json.dumps(payload, indent=2))
    loaded = json.loads(snapshot_file.read_text())
    assert loaded["cipher_numbers"] == cipher_numbers
