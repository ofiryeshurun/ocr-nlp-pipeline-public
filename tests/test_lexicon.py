"""Tests for the alcohol lexicon and the fuzzy matcher."""

from pathlib import Path

import pytest

from alcohol_detector.lexicon import AlcoholLexicon, EditBudget, LexiconMatcher, LexiconTerm
from alcohol_detector.text import LeetMapper, TextNormalizer

LEXICON_YAML = Path(__file__).resolve().parents[1] / "data" / "lexicon.yaml"

# A test budget only (5 letters -> 1 edit, 8 -> 2); the real one is picked from train evidence.
BUDGET = EditBudget(one_edit_from=5, two_edits_from=8)
TERMS = [("gin", "spirits"), ("port", "wine"), ("vodka", "spirits"), ("heineken", "beer_brands"),
         ("jack daniels", "spirit_brands")]
LEXICON = AlcoholLexicon([LexiconTerm(text, category) for text, category in TERMS])
MATCHER = LexiconMatcher(LEXICON, BUDGET, TextNormalizer(leet=LeetMapper()))


def found(text: str, matcher: LexiconMatcher = MATCHER) -> set[str]:
    return {m.term for m in matcher.match(text)}


# --- lexicon file -----------------------------------------------------------------

def test_lexicon_file_loads_clean_unique_terms() -> None:
    lexicon = AlcoholLexicon.from_yaml(LEXICON_YAML)
    assert len(lexicon) > 100
    clean = TextNormalizer().clean
    assert all(term.text == clean(term.text) for term in lexicon)
    texts = {term.text for term in lexicon}
    assert {"jack daniels", "moet", "jagermeister"} <= texts  # punctuation and accents cleaned
    assert not texts & {"brew", "bar", "corona", "patron", "spirits"}  # left out on purpose


def test_lexicon_rejects_duplicates_and_bad_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        AlcoholLexicon([LexiconTerm("jack daniels", "a"), LexiconTerm("jackdaniels", "b")])
    with pytest.raises(FileNotFoundError):
        AlcoholLexicon.from_yaml(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("terms: [gin]", encoding="utf-8")
    with pytest.raises(ValueError, match="categories"):
        AlcoholLexicon.from_yaml(bad)


# --- edit budget --------------------------------------------------------------------

def test_edit_budget_bands() -> None:
    assert [BUDGET.max_edits(n) for n in (3, 4, 5, 7, 8, 12)] == [0, 0, 1, 1, 2, 2]
    assert BUDGET.is_short(4) and not BUDGET.is_short(5)
    assert str(EditBudget.elasticsearch_default()) == "AUTO:3,6"
    with pytest.raises(ValueError):
        EditBudget(one_edit_from=6, two_edits_from=3)


# --- short terms: whole words, exact -------------------------------------------------

@pytest.mark.parametrize("text", ["engine", "imagine the original", "ginger", "g in"])
def test_short_term_never_matches_inside_words(text: str) -> None:
    assert "gin" not in found(text)


@pytest.mark.parametrize("text", ["support the airport sport", "passport"])
def test_port_is_whole_word_only(text: str) -> None:
    assert "port" not in found(text)


def test_short_terms_match_whole_words() -> None:
    assert found("Drink GIN tonight, port check") == {"gin", "port"}


# --- long terms: fuzzy, in the squashed text -----------------------------------------

@pytest.mark.parametrize(
    "text",
    ["heinekin", "h eine k en", "drinkheineken now", "HE1NEKEN", "#eineken"],
)
def test_long_term_survives_ocr_errors_spacing_and_leet(text: str) -> None:
    assert found(text) == {"heineken"}


def test_phrase_matches_spaced_glued_and_in_order_only() -> None:
    assert found("try jack daniels") == {"jack daniels"}
    assert found("menujackdaniels") == {"jack daniels"}
    assert found("daniels jack") == set()


def test_match_reports_evidence() -> None:
    (match,) = MATCHER.match("cold h eine kin")
    assert (match.term, match.category, match.matched_text, match.distance) == (
        "heineken", "beer_brands", "heinekin", 1)
    assert match.similarity == pytest.approx(1 - 1 / 8)


# --- leetspeak decoding ------------------------------------------------------------------

def test_leet_decoding_is_what_catches_short_and_heavy_leet() -> None:
    basic = LexiconMatcher(LEXICON, BUDGET, TextNormalizer())
    assert found("Drink g!n v0dk4") == {"gin", "vodka"}
    # Without decoding, g!n splits into "g n", and v0dk4 is 2 edits from a 5-letter term.
    assert found("Drink g!n v0dk4", basic) == set()


@pytest.mark.parametrize("text", ["", "   ", "!!!", "2024"])
def test_empty_or_textless_input(text: str) -> None:
    assert MATCHER.match(text) == []
