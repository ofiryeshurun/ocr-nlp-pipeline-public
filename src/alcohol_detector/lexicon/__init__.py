"""The alcohol lexicon and the fuzzy matcher that finds its terms in OCR text."""

from alcohol_detector.lexicon.lexicon import AlcoholLexicon, LexiconTerm
from alcohol_detector.lexicon.matcher import EditBudget, LexiconMatcher, TermMatch

__all__ = ["AlcoholLexicon", "EditBudget", "LexiconMatcher", "LexiconTerm", "TermMatch"]
