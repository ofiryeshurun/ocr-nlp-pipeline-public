"""OCR layer: engine interface, PaddleOCR backend, TTA, caching and evaluation."""

from alcohol_detector.ocr.base import OCREngine, OCRResult, TextBox
from alcohol_detector.ocr.cache import OCRCache
from alcohol_detector.ocr.evaluate import OCREvaluator, OCRScore, squash
from alcohol_detector.ocr.image_loader import ImageLoader, ImageLoadError
from alcohol_detector.ocr.tta import ALL_VARIANTS, OrientationTTA, QualityScorer, Variant, VariantSelector

__all__ = [
    "ALL_VARIANTS",
    "ImageLoadError",
    "ImageLoader",
    "OCRCache",
    "OCREngine",
    "OCREvaluator",
    "OCRResult",
    "OCRScore",
    "OrientationTTA",
    "QualityScorer",
    "TextBox",
    "Variant",
    "VariantSelector",
    "squash",
]
