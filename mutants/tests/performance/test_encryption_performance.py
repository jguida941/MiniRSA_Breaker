from __future__ import annotations

import pytest

from mini_rsa.core import encrypt_text


@pytest.mark.performance
def test_encryption_performance_budget(benchmark) -> None:
    p, q, e = 97, 101, 5
    n = p * q
    message = "HELLO WORLD " * 50

    def run() -> int:
        result = encrypt_text(message, e, n, include_punctuation=False)
        return len(result.cipher_blocks)

    cipher_len = benchmark(run)
    assert cipher_len == len(message)
    mean_duration = benchmark.stats.stats.mean
    assert mean_duration < 0.005  # 5ms guardrail accommodates mutation overhead
