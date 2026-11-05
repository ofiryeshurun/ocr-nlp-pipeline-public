"""Run OCR over many images with caching and per-image error isolation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from alcohol_detector.ocr.base import OCREngine, OCRResult
from alcohol_detector.ocr.cache import OCRCache
from alcohol_detector.ocr.image_loader import ImageLoader, ImageLoadError
from alcohol_detector.ocr.preprocess import ImagePreprocessor, NoPreprocessor
from alcohol_detector.ocr.tta import ALL_VARIANTS, OrientationTTA, Variant, VariantSelector

logger = logging.getLogger(__name__)


def tta_cache_config(engine_id: str, selector: VariantSelector, preprocess: str = "none") -> dict[str, object]:
    """Cache key of a TTA run. Shared by the runner and by readers of cached results.

    The preprocessor is omitted for "none", so raw-image cache keys stay stable.
    """
    config: dict[str, object] = {
        "engine": engine_id,
        "mode": "tta",
        "strategy": selector.strategy,
        "threshold": selector.trigger_threshold,
        "combine": selector.combine,
        "variants": list(selector.variants),
    }
    if preprocess != "none":
        config["preprocess"] = preprocess
    return config


@dataclass(frozen=True)
class ImageJob:
    """One image to OCR."""

    file_name: str
    path: Path


class OCRBatchRunner:
    """OCR a list of images, reusing cached results and never stopping on a bad image."""

    def __init__(
        self,
        engine: OCREngine,
        cache_root: Path,
        loader: ImageLoader | None = None,
        preprocessor: ImagePreprocessor | None = None,
    ) -> None:
        """Args:
        engine: OCR backend.
        cache_root: Root folder of the OCR cache.
        loader: Image loader (default: :class:`ImageLoader`).
        preprocessor: Applied to each image before OCR (default: none).
        """
        self.engine: OCREngine = engine
        self.cache_root: Path = Path(cache_root)
        self.loader: ImageLoader = loader or ImageLoader()
        self.preprocessor: ImagePreprocessor = preprocessor or NoPreprocessor()

    def _with_preprocess(self, config: dict[str, object]) -> dict[str, object]:
        """Add the preprocessor to a cache config (omitted for "none", so raw-image caches stay valid)."""
        if self.preprocessor.name != "none":
            config["preprocess"] = self.preprocessor.name
        return config

    def variant_cache(self, variant: Variant) -> OCRCache:
        """Cache holding raw reads of a single variant (used for benchmarking)."""
        return OCRCache(
            self.cache_root,
            self._with_preprocess({"engine": self.engine.engine_id, "mode": "single", "variant": variant.name}),
        )

    def tta_cache(self, selector: VariantSelector) -> OCRCache:
        """Cache holding TTA results for a given selector configuration."""
        return OCRCache(self.cache_root, tta_cache_config(self.engine.engine_id, selector, self.preprocessor.name))

    def run_per_variant(self, jobs: list[ImageJob], variants: tuple[Variant, ...] = ALL_VARIANTS) -> None:
        """OCR every image under every variant separately, caching each read."""
        caches = {v.name: self.variant_cache(v) for v in variants}
        for i, job in enumerate(jobs, start=1):
            todo = [v for v in variants if caches[v.name].get(job.file_name) is None]
            if not todo:
                continue
            try:
                image = self.preprocessor.apply(self.loader.load(job.path))
            except ImageLoadError as exc:
                for v in todo:
                    caches[v.name].put(OCRResult(job.file_name, [], v.name, self.engine.engine_id, error=str(exc)))
                continue
            for v in todo:
                start = time.perf_counter()
                try:
                    boxes, error = self.engine.read(v.apply(image)), None
                except Exception as exc:  # isolate backend failures per image
                    boxes, error = [], f"{type(exc).__name__}: {exc}"
                caches[v.name].put(
                    OCRResult(job.file_name, boxes, v.name, self.engine.engine_id, time.perf_counter() - start, [v.name], error)
                )
            logger.info("[%d/%d] %s", i, len(jobs), job.file_name)

    def run_tta(self, jobs: list[ImageJob], selector: VariantSelector) -> list[OCRResult]:
        """OCR every image with TTA, using and filling the cache. Returns results in job order."""
        cache = self.tta_cache(selector)
        tta = OrientationTTA(self.engine, selector, self.preprocessor)
        results: list[OCRResult] = []
        for i, job in enumerate(jobs, start=1):
            cached = cache.get(job.file_name)
            if cached is not None:
                results.append(cached)
                continue
            try:
                result = tta.run(self.loader.load(job.path), job.file_name)
            except ImageLoadError as exc:
                result = OCRResult(job.file_name, [], engine_id=self.engine.engine_id, error=str(exc))
            cache.put(result)
            results.append(result)
            logger.info("[%d/%d] %s variant=%s %.1fs", i, len(jobs), job.file_name, result.variant, result.elapsed_s)
        return results
