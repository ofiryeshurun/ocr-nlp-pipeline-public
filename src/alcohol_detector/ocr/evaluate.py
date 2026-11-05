"""OCR quality metrics against human transcriptions (the annotated set)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


def squash(text: str) -> str:
    """Lowercase and keep only letters and digits.

    Spaces are removed on purpose: overlays use spacing tricks ("h eine k en")
    and OCR drops spaces ("bourbonevent"), so spacing is not an OCR error here.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass(frozen=True)
class OCRScore:
    """How well one OCR read matches the human transcription."""

    text_found: bool
    key_term_hit: bool
    overlay_cer: float


class OCREvaluator:
    """Compare OCR output with the annotated overlay text and key term."""

    def key_term_hit(self, ocr_text: str, key_term: str) -> bool:
        """True if the key term appears in the OCR text (case- and space-insensitive)."""
        key = squash(key_term)
        return bool(key) and key in squash(ocr_text)

    def overlay_cer(self, ocr_text: str, reference_text: str) -> float:
        """Character error rate of the overlay, ignoring extra background text.

        The reference overlay is aligned to its best-matching stretch of the OCR text,
        so text read from the photo's background is not counted as an error.
        1.0 means nothing of the overlay was recovered.
        """
        ref, ocr = squash(reference_text), squash(ocr_text)
        if not ref:
            return 0.0
        if not ocr:
            return 1.0
        if len(ocr) > len(ref):
            align = fuzz.partial_ratio_alignment(ref, ocr)
            ocr = ocr[align.dest_start : align.dest_end]
        return min(1.0, Levenshtein.distance(ref, ocr) / len(ref))

    def score(self, ocr_text: str, reference_text: str, key_term: str) -> OCRScore:
        """All metrics for one image."""
        return OCRScore(
            text_found=bool(squash(ocr_text)),
            key_term_hit=self.key_term_hit(ocr_text, key_term),
            overlay_cer=self.overlay_cer(ocr_text, reference_text),
        )
