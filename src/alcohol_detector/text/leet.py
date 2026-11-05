"""Leetspeak decoding: map look-alike characters back to letters.

The table comes from the common leet alphabet (domain knowledge), not from the
training data. Only single-character substitutions are covered; multi-character
forms such as ``|-|`` for h are future work.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Mapping

# Characters that may stand for more than one letter get one fixed letter:
# "1" and "|" become "i" (they also stand for "l"; for long lexicon terms the
# fuzzy matcher's edit allowance absorbs the other reading).
LEET_TABLE: Mapping[str, str] = MappingProxyType(
    {
        "4": "a", "@": "a",
        "8": "b",
        "(": "c",
        "3": "e", "€": "e",
        "9": "g", "6": "g",
        "#": "h",
        "1": "i", "!": "i", "|": "i",
        "0": "o",
        "5": "s", "$": "s",
        "7": "t", "+": "t",
        "2": "z",
    }
)


_WHITESPACE = re.compile(r"(\s+)")  # capturing group keeps the separators


class LeetMapper:
    """Replace leetspeak characters with the letters they imitate, inside words only.

    Guard rules, so real text stays intact:
    - Only tokens that contain a letter are touched ("#1", "2024" and "$5" stay).
    - Digits are mapped only when the token's first letter-or-digit is a letter,
      so numbers with units ("700ml", "19th") stay, while "v0dk4" and "he1neken" map.
    - Symbols are mapped only between letters or digits of the same token, so
      "g!n" maps but the punctuation in "gin!", "#gin" and "(gin)" is left alone.

    Stateless and deterministic. Run it on lowercased text, before punctuation is removed.
    """

    def __init__(self, table: Mapping[str, str] = LEET_TABLE) -> None:
        self._table: dict[str, str] = dict(table)  # a plain copy, so a fitted classifier can be pickled

    def map(self, text: str) -> str:
        """Decode every whitespace-separated token; whitespace is kept as it is."""
        return "".join(self._map_token(part) for part in _WHITESPACE.split(text))

    def _map_token(self, token: str) -> str:
        if not any(ch.isalpha() for ch in token):
            return token
        alnum_positions = [i for i, ch in enumerate(token) if ch.isalnum()]
        first, last = alnum_positions[0], alnum_positions[-1]
        map_digits = token[first].isalpha()
        out: list[str] = []
        for i, ch in enumerate(token):
            letter = self._table.get(ch)
            if letter is None:
                out.append(ch)
            elif ch.isdigit():
                out.append(letter if map_digits else ch)
            else:
                out.append(letter if first < i < last else ch)
        return "".join(out)

