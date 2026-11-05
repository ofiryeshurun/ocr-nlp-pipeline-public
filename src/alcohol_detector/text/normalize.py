"""Basic, standard cleaning of OCR text.

Generic preprocessing, plus optional leetspeak decoding. OCR errors and
spacing tricks are left to the classifiers (fuzzy lexicon matching, character n-grams).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from alcohol_detector.text.leet import LeetMapper


@dataclass(frozen=True)
class NormalizedText:
    """One OCR text after basic cleaning."""

    raw: str
    clean: str  # lowercase, accents and punctuation removed, single spaces
    tokens: tuple[str, ...]

    @property
    def squashed(self) -> str:
        """The clean text without spaces ("h eine k en" -> "heineken")."""
        return self.clean.replace(" ", "")


class TextNormalizer:
    """Lowercase, strip accents and punctuation, collapse whitespace. Deterministic.

    With a ``LeetMapper``, leetspeak is decoded after lowercasing and before
    punctuation is removed, so "g!n" becomes "gin" instead of "g n".
    """

    _apostrophes = re.compile(r"[‘’'`´]")
    _non_alnum = re.compile(r"[^a-z0-9]+")

    def __init__(self, leet: LeetMapper | None = None) -> None:
        self._leet: LeetMapper | None = leet

    def normalize(self, text: str) -> NormalizedText:
        """Clean one OCR text. Empty or whitespace-only input gives empty fields."""
        clean = self.clean(text)
        return NormalizedText(raw=text, clean=clean, tokens=tuple(clean.split()))

    def clean(self, text: str) -> str:
        """Return the cleaned text."""
        # NFKD folds look-alike forms (full-width letters, ligatures) to plain ones and splits accents
        # off their letters as separate combining marks, which the next line drops ("é" -> "e").
        text = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text))
        text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
        if self._leet is not None:
            text = self._leet.map(text)
        text = self._apostrophes.sub("", text)  # "don't" -> "dont"
        return self._non_alnum.sub(" ", text).strip()
