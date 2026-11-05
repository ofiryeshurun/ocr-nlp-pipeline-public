"""Find alcohol lexicon terms in OCR text with edit-distance matching.

Two rules, chosen by the term's length (without spaces):
- **Short terms** must equal a whole word of the text (or a run of whole words,
  in order). No edits, and no search inside longer words, so "gin" never matches
  "engine" or "imagine".
- **Long terms** are searched in the squashed text (spaces removed) for the
  best-matching stretch, allowing a few edits. This covers spacing tricks
  ("h eine k en"), glued words ("drinkheineken"), phrases ("jackdaniels") and
  OCR errors ("heinekin").
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from alcohol_detector.lexicon.lexicon import AlcoholLexicon, LexiconTerm
from alcohol_detector.text import NormalizedText, TextNormalizer


@dataclass(frozen=True)
class EditBudget:
    """How many edits a term may differ by, from its length (Elasticsearch-style ``AUTO:low,high``).

    - shorter than ``one_edit_from``: a short term, matched as whole words with no edits
    - from ``one_edit_from``: 1 edit
    - from ``two_edits_from``: 2 edits
    """

    one_edit_from: int
    two_edits_from: int

    def __post_init__(self) -> None:
        if not 1 <= self.one_edit_from <= self.two_edits_from:
            raise ValueError(
                f"Need 1 <= one_edit_from <= two_edits_from, got {self.one_edit_from}, {self.two_edits_from}."
            )

    @classmethod
    def elasticsearch_default(cls) -> EditBudget:
        """Elasticsearch's default fuzziness ``AUTO`` (= ``AUTO:3,6``), the reference point."""
        return cls(one_edit_from=3, two_edits_from=6)

    def is_short(self, length: int) -> bool:
        """True if a term of this length must match whole words exactly."""
        return length < self.one_edit_from

    def max_edits(self, length: int) -> int:
        """Edits allowed for a term of this length."""
        if length < self.one_edit_from:
            return 0
        return 1 if length < self.two_edits_from else 2

    def __str__(self) -> str:
        return f"AUTO:{self.one_edit_from},{self.two_edits_from}"


@dataclass(frozen=True)
class TermMatch:
    """One lexicon term found in a text: the evidence behind a prediction."""

    term: str
    category: str
    matched_text: str  # the word(s) or squashed stretch that matched
    distance: int  # edits between the term and matched_text

    @property
    def similarity(self) -> float:
        """1 - distance / term length (1.0 for an exact match)."""
        return 1.0 - self.distance / len(self.term.replace(" ", ""))


class LexiconMatcher:
    """Return every lexicon term found in a text, with the matched text as evidence."""

    def __init__(self, lexicon: AlcoholLexicon, budget: EditBudget, normalizer: TextNormalizer) -> None:
        self._budget: EditBudget = budget
        self._normalizer: TextNormalizer = normalizer
        self._short: tuple[LexiconTerm, ...] = tuple(t for t in lexicon if budget.is_short(len(t.squashed)))
        self._long: tuple[LexiconTerm, ...] = tuple(t for t in lexicon if not budget.is_short(len(t.squashed)))

    @property
    def budget(self) -> EditBudget:
        """The edit budget in use."""
        return self._budget

    def match(self, text: str) -> list[TermMatch]:
        """Clean ``text`` and match it. Empty text gives an empty list."""
        return self.match_normalized(self._normalizer.normalize(text))

    def match_normalized(self, text: NormalizedText) -> list[TermMatch]:
        """Match already-cleaned text. Results are sorted by distance, then term."""
        matches = [m for t in self._short if (m := self._match_short(t, text.tokens))]
        if text.squashed:
            windows = _Windows(text.squashed)
            matches += [m for t in self._long if (m := self._match_long(t, windows))]
        return sorted(matches, key=lambda m: (m.distance, m.term))

    @staticmethod
    def _match_short(term: LexiconTerm, tokens: tuple[str, ...]) -> TermMatch | None:
        words, n = term.words, len(term.words)
        for start in range(len(tokens) - n + 1):
            if tokens[start : start + n] == words:
                return TermMatch(term.text, term.category, " ".join(words), 0)
        return None

    def _match_long(self, term: LexiconTerm, windows: _Windows) -> TermMatch | None:
        target = term.squashed
        edits = self._budget.max_edits(len(target))
        # A stretch within `edits` of the term differs from it in length by at most `edits`.
        # Same-length stretches come first, so ties prefer them.
        lengths = sorted(range(len(target) - edits, len(target) + edits + 1), key=lambda n: abs(n - len(target)))
        candidates = [w for n in lengths for w in windows.of_length(n)]
        best = process.extractOne(target, candidates, scorer=Levenshtein.distance, score_cutoff=edits)
        if best is None:
            return None
        matched, distance, _ = best
        return TermMatch(term.text, term.category, matched, int(distance))


class _Windows:
    """All stretches of a string, grouped by length and built on demand."""

    def __init__(self, text: str) -> None:
        self._text: str = text
        self._cache: dict[int, list[str]] = {}

    def of_length(self, n: int) -> list[str]:
        if n < 1 or n > len(self._text):
            return []
        if n not in self._cache:
            self._cache[n] = [self._text[i : i + n] for i in range(len(self._text) - n + 1)]
        return self._cache[n]
