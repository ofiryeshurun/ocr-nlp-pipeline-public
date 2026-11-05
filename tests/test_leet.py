"""Tests for leetspeak decoding and its guard rules."""

import pytest

from alcohol_detector.text import LeetMapper, TextNormalizer

N = TextNormalizer(leet=LeetMapper())


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Drink g!n", "drink gin"),
        ("ice cold v0dk4", "ice cold vodka"),
        ("HE1NEKEN", "heineken"),
        ("s7ella ar7ois", "stella artois"),
        ("b33r", "beer"),
        ("b€er b@@r", "beer baar"),
    ],
)
def test_leet_inside_words_is_decoded(raw: str, clean: str) -> None:
    assert N.clean(raw) == clean


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("gin!", "gin"),  # trailing punctuation, not a letter
        ("#gin (rum)", "gin rum"),  # leading and wrapping punctuation
        ("700ml 19th 2024", "700ml 19th 2024"),  # numbers, with or without units
        ("#1 deal $5", "1 deal 5"),  # tokens without letters
    ],
)
def test_real_text_is_left_alone(raw: str, clean: str) -> None:
    assert N.clean(raw) == clean


def test_whitespace_is_kept() -> None:
    assert LeetMapper().map("g!n  \tand\nt0nic") == "gin  \tand\ntonic"


def test_basic_normalizer_does_not_decode() -> None:
    assert TextNormalizer().clean("Drink g!n") == "drink g n"


def test_squashed_view() -> None:
    assert N.normalize("h eine k en!").squashed == "heineken"
