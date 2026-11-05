"""Tests for the basic OCR text normalizer."""

import pytest

from alcohol_detector.text import TextNormalizer

N = TextNormalizer()


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("DON'T MISS SMOOTH SMIRNOFF", "dont miss smooth smirnoff"),
        ("Café — bourbon!!", "cafe bourbon"),
        ("  h eine   k-en  ", "h eine k en"),
        ("v0dk4 discount", "v0dk4 discount"),  # leetspeak is left for fuzzy matching
    ],
)
def test_clean(raw: str, clean: str) -> None:
    assert N.clean(raw) == clean


def test_tokens_and_empty_input() -> None:
    assert N.normalize("Try offer, grab now gin").tokens == ("try", "offer", "grab", "now", "gin")
    empty = N.normalize("   ")
    assert empty.clean == "" and empty.tokens == ()
