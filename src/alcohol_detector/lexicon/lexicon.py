"""The alcohol lexicon: domain-knowledge terms loaded from YAML."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from alcohol_detector.text import TextNormalizer


@dataclass(frozen=True)
class LexiconTerm:
    """One alcohol term, cleaned like OCR text ("Jack Daniel's" -> "jack daniels")."""

    text: str
    category: str

    @property
    def words(self) -> tuple[str, ...]:
        """The term's words, in order."""
        return tuple(self.text.split())

    @property
    def squashed(self) -> str:
        """The term without spaces ("jack daniels" -> "jackdaniels")."""
        return self.text.replace(" ", "")


class AlcoholLexicon:
    """An ordered, duplicate-free collection of strong alcohol terms."""

    def __init__(self, terms: Sequence[LexiconTerm]) -> None:
        if not terms:
            raise ValueError("The lexicon has no terms.")
        seen: dict[str, LexiconTerm] = {}
        for term in terms:
            if not term.squashed:
                raise ValueError(f"Empty term in category {term.category!r}.")
            if term.squashed in seen:
                first = seen[term.squashed]
                raise ValueError(
                    f"Duplicate term {term.text!r} ({term.category}); "
                    f"already listed as {first.text!r} ({first.category})."
                )
            seen[term.squashed] = term
        self._terms: tuple[LexiconTerm, ...] = tuple(terms)

    @classmethod
    def from_yaml(cls, path: str | Path, normalizer: TextNormalizer | None = None) -> AlcoholLexicon:
        """Load ``categories: {name: [term, ...]}`` from a YAML file.

        Terms are cleaned with ``normalizer`` (basic cleaning by default), the same
        way OCR text is, so both sides of a comparison use one alphabet.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Lexicon file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        categories = data.get("categories") if isinstance(data, dict) else None
        if not isinstance(categories, dict) or not categories:
            raise ValueError(f"{path}: expected a non-empty 'categories' mapping.")

        normalizer = normalizer or TextNormalizer()
        terms: list[LexiconTerm] = []
        for category, entries in categories.items():
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"{path}: category {category!r} must be a non-empty list.")
            for entry in entries:
                if not isinstance(entry, str):
                    raise ValueError(f"{path}: non-text entry {entry!r} in {category!r}.")
                terms.append(LexiconTerm(text=normalizer.clean(entry), category=str(category)))
        return cls(terms)

    @property
    def categories(self) -> tuple[str, ...]:
        """Category names, in file order."""
        return tuple(dict.fromkeys(term.category for term in self._terms))

    def __iter__(self) -> Iterator[LexiconTerm]:
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self._terms)
