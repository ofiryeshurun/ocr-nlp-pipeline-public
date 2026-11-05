"""Run-level statistics of cached OCR output: speed, retries, text found."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from alcohol_detector.ocr.corpus import OCRDocument


@dataclass(frozen=True)
class OCRRunStats:
    """How one OCR setup behaved on a set of images (inputs to the engine trade-off)."""

    setup: str
    n_images: int
    sec_per_img: float  # mean wall time per image, retries included
    retry_rate: float  # share of images where the mirror retry ran
    text_found_rate: float  # share of images with any recognised text
    error_rate: float  # share of images that failed to load or read

    @classmethod
    def from_documents(cls, setup: str, docs: Sequence[OCRDocument]) -> OCRRunStats:
        """Summarise cached documents of one setup."""
        if not docs:
            raise ValueError(f"No documents for setup {setup!r}.")
        results = [d.result for d in docs]
        return cls(
            setup=setup,
            n_images=len(results),
            sec_per_img=float(np.mean([r.elapsed_s for r in results])),
            retry_rate=float(np.mean([len(r.variants_tried) > 1 for r in results])),
            text_found_rate=float(np.mean([bool(r.text.strip()) for r in results])),
            error_rate=float(np.mean([not r.ok for r in results])),
        )
