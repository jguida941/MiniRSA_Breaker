import json
from pathlib import Path

import pytest

from mini_rsa.core import encrypt_text


@pytest.mark.parametrize("fixture_name", ["example_encryption"])
def test_example_snapshot_matches_fixture(fixture_name: str) -> None:
    fixture_path = Path(__file__).parent.parent / "resources" / f"{fixture_name}.json"
    payload = json.loads(fixture_path.read_text())

    p = payload["p"]
    q = payload["q"]
    e = payload["e"]
    text = payload["plaintext"]
    n = p * q

    result = encrypt_text(
        text,
        e,
        n,
        include_punctuation=payload.get("include_punctuation", False),
    )
    assert result.cipher_blocks == payload["cipher_numbers"]
