"""Read cached OCR results back as a labelled text corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alcohol_detector.data import SplitManifest
from alcohol_detector.ocr.base import OCRResult
from alcohol_detector.ocr.batch import tta_cache_config
from alcohol_detector.ocr.cache import OCRCache
from alcohol_detector.ocr.preprocess import make_preprocessor
from alcohol_detector.ocr.tta import VariantSelector

#: The locked OCR retry rule. If the original read's character-weighted
#: mean confidence is below 0.9, the mirrored image is read too and the read with more confident
#: characters is kept. 0.9 is mid-plateau: results barely changed for any trigger from 0.85 to 0.95.
CONFIG_A = VariantSelector("conditional", 0.9, "best", ("original", "mirror"))


@dataclass(frozen=True)
class OCRDocument:
    """OCR output of one labelled image."""

    file_name: str
    label: str
    split: str
    result: OCRResult

    @property
    def text(self) -> str:
        """The recognised text."""
        return self.result.text


class OCRCorpus:
    """Load cached OCR results for chosen splits, engine and retry rule."""

    def __init__(
        self,
        manifest_path: Path,
        cache_root: Path,
        engine_id: str,
        selector: VariantSelector = CONFIG_A,
        preprocess: str = "none",
    ) -> None:
        """Args:
        manifest_path: The ``file_name,label,split`` CSV.
        cache_root: Root of the OCR cache.
        engine_id: Engine id of the cached run (``PaddleOCREngine.make_engine_id``).
        selector: Retry rule of the cached run.
        preprocess: Preprocessor key of the cached run (``make_preprocessor``); "none" = raw image.
        """
        self.manifest = SplitManifest.read(Path(manifest_path))
        # The runner keys caches by the preprocessor's full name, parameters included.
        self.cache = OCRCache(
            Path(cache_root), tta_cache_config(engine_id, selector, make_preprocessor(preprocess).name)
        )

    def load(self, splits: tuple[str, ...]) -> list[OCRDocument]:
        """Return the cached documents of the given splits.

        Raises:
            FileNotFoundError: If any image of those splits has no cached result,
                so a partial corpus is never used by accident.
        """
        docs: list[OCRDocument] = []
        missing: list[str] = []
        for a in self.manifest:
            if a.split not in splits:
                continue
            result = self.cache.get(a.file_name)
            if result is None:
                missing.append(a.file_name)
            else:
                docs.append(OCRDocument(a.file_name, a.label, a.split, result))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images of splits {splits} have no cached OCR in {self.cache.folder} "
                f"(e.g. {missing[:3]}); run the OCR over these splits first."
            )
        return docs
