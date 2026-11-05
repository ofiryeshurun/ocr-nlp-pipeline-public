"""Test-time augmentation (TTA) for OCR: retry flipped and rotated copies of an image.

EDA showed 22% of overlays are mirrored and 32%
rotated, often diagonally. No OCR model reads mirrored text, and diagonal text
is often missed, so the engine is also run on transformed copies of the image.

All decisions use only the OCR output itself (no labels), so the same logic
runs at inference time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from alcohol_detector.ocr.base import OCREngine, OCRResult, TextBox
from alcohol_detector.ocr.preprocess import ImagePreprocessor, NoPreprocessor


@dataclass(frozen=True)
class Variant:
    """A geometric transform applied to the image before OCR."""

    name: str
    mirror: bool = False
    angle: float = 0.0  # degrees, counter-clockwise

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Return the transformed image (rotation expands the canvas so nothing is cut off)."""
        out = cv2.flip(image, 1) if self.mirror else image  # 1 = around the vertical axis (left-right)
        if self.angle:
            h, w = out.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), self.angle, 1.0)
            cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
            new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
            matrix[0, 2] += new_w / 2 - w / 2
            matrix[1, 2] += new_h / 2 - h / 2
            out = cv2.warpAffine(out, matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
        return out


ORIGINAL = Variant("original")
ALL_VARIANTS: tuple[Variant, ...] = (
    ORIGINAL,
    Variant("mirror", mirror=True),
    Variant("rot+20", angle=20),
    Variant("rot-20", angle=-20),
    Variant("rot+40", angle=40),
    Variant("rot-40", angle=-40),
    Variant("mirror_rot+20", mirror=True, angle=20),
    Variant("mirror_rot-20", mirror=True, angle=-20),
)


class QualityScorer:
    """Label-free measures of how good an OCR read looks."""

    @staticmethod
    def mean_confidence(boxes: list[TextBox]) -> float:
        """Confidence averaged over characters (long reads count more than one-letter reads).

        Returns 0.0 when nothing was read.
        """
        total = sum(len(b.text) for b in boxes)
        if total == 0:
            return 0.0
        return sum(b.confidence * len(b.text) for b in boxes) / total

    @staticmethod
    def confident_chars(boxes: list[TextBox]) -> float:
        """Sum of confidence x length: rewards reading more text, confidently."""
        return sum(b.confidence * len(b.text) for b in boxes)


class VariantSelector:
    """Turn per-variant OCR reads into one final read.

    Kept separate from the engine so strategies can be replayed offline from
    cached per-variant results.
    """

    def __init__(
        self,
        strategy: str = "always",
        trigger_threshold: float | None = None,
        combine: str = "best",
        variants: tuple[str, ...] = tuple(v.name for v in ALL_VARIANTS),
    ) -> None:
        """Args:
        strategy: "always" (try every variant) or "conditional" (only if the original read is weak).
        trigger_threshold: For "conditional": retry when the original's mean confidence is below this.
        combine: "best" (keep the variant with the most confident characters) or
            "union" (merge distinct boxes from every variant tried).
        variants: Variant names allowed (must include "original").
        """
        if strategy not in {"always", "conditional"}:
            raise ValueError(f"Unknown strategy: {strategy}")
        if strategy == "conditional" and trigger_threshold is None:
            raise ValueError("A conditional strategy needs a trigger_threshold.")
        if combine not in {"best", "union"}:
            raise ValueError(f"Unknown combine rule: {combine}")
        if "original" not in variants:
            raise ValueError("Variants must include 'original'.")
        self.strategy: str = strategy
        self.trigger_threshold: float | None = trigger_threshold
        self.combine: str = combine
        self.variants: tuple[str, ...] = variants

    def needs_retry(self, original_boxes: list[TextBox]) -> bool:
        """Whether the other variants should be tried."""
        if self.strategy == "always":
            return True
        assert self.trigger_threshold is not None
        return QualityScorer.mean_confidence(original_boxes) < self.trigger_threshold

    def select(self, reads: dict[str, list[TextBox]]) -> tuple[str, list[TextBox]]:
        """Combine reads (keyed by variant name) into (chosen variant label, boxes)."""
        if self.combine == "best":
            # On a tie in confident characters, the original read wins (True sorts above False).
            name = max(reads, key=lambda n: (QualityScorer.confident_chars(reads[n]), n == "original"))
            return name, reads[name]
        seen: set[str] = set()
        merged: list[TextBox] = []
        for name in reads:
            for box in reads[name]:
                key = box.text.strip().lower()
                if key not in seen:
                    seen.add(key)
                    merged.append(box)
        return "union", merged


class OrientationTTA:
    """Run an OCR engine on the original image and, when needed, on flipped/rotated copies."""

    def __init__(
        self, engine: OCREngine, selector: VariantSelector, preprocessor: ImagePreprocessor | None = None
    ) -> None:
        """Args:
        engine: The OCR backend.
        selector: Decides whether to retry and how to combine the reads.
        preprocessor: Applied once to the image before any variant (default: none).
        """
        self.engine: OCREngine = engine
        self.selector: VariantSelector = selector
        self.preprocessor: ImagePreprocessor = preprocessor or NoPreprocessor()
        self._variants: dict[str, Variant] = {v.name: v for v in ALL_VARIANTS if v.name in selector.variants}

    def run(self, image: np.ndarray, file_name: str) -> OCRResult:
        """OCR one image with TTA. Engine errors are captured in the result, never raised."""
        start = time.perf_counter()
        reads: dict[str, list[TextBox]] = {}
        try:
            image = self.preprocessor.apply(image)
            reads["original"] = self.engine.read(image)
            if self.selector.needs_retry(reads["original"]):
                for name, variant in self._variants.items():
                    if name != "original":
                        reads[name] = self.engine.read(variant.apply(image))
            chosen, boxes = self.selector.select(reads)
            error = None
        except Exception as exc:  # one bad image must never stop a batch
            # Keep the original read if it finished before the failure, so the error report shows it.
            chosen, boxes, error = "original", reads.get("original", []), f"{type(exc).__name__}: {exc}"
        return OCRResult(
            file_name=file_name,
            boxes=boxes,
            variant=chosen,
            engine_id=self.engine.engine_id,
            elapsed_s=time.perf_counter() - start,
            variants_tried=list(reads),
            error=error,
        )
