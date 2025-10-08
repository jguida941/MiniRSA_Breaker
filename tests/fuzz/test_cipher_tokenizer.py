from __future__ import annotations

from hypothesis import given, strategies as st

from mini_rsa.core import tokenize_cipher_text


@given(
    st.text(
        alphabet=st.characters(
            min_codepoint=0,
            max_codepoint=255,
            blacklist_categories=("Cs",),
        ),
        min_size=0,
        max_size=256,
    )
)
def test_tokenizer_never_raises(data: str) -> None:
    numbers = tokenize_cipher_text(data)
    assert all(isinstance(number, int) for number in numbers)
