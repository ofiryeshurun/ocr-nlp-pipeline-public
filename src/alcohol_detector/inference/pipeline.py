"""The end-to-end classifier: image -> OCR -> text classifier -> label."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from alcohol_detector.config import UNREADABLE_LABEL, OCRSetup, ocr_setup
from alcohol_detector.inference.images import ImageDirectory
from alcohol_detector.models.base import LABELS, TextClassifier
from alcohol_detector.models.bundle import ModelBundle, ModelCard
from alcohol_detector.ocr.base import OCREngine
from alcohol_detector.ocr.batch import tta_cache_config
from alcohol_detector.ocr.image_loader import ImageLoader
from alcohol_detector.ocr.paddle import PaddleOCREngine
from alcohol_detector.ocr.preprocess import make_preprocessor
from alcohol_detector.ocr.tta import OrientationTTA, VariantSelector

logger = logging.getLogger(__name__)

#: Builds the OCR engine of a setup; swapped for a fake in tests.
EngineFactory = Callable[[OCRSetup], OCREngine]


def paddle_engine(setup: OCRSetup) -> OCREngine:
    """The PaddleOCR engine of an OCR setup, at native input size like every cached run."""
    return PaddleOCREngine(setup.det_model, setup.rec_model)


@dataclass(frozen=True)
class ImagePrediction:
    """The label of one image, with what it was based on."""

    file_name: str
    label: str
    score: float | None = None  # None when the image could not be processed
    evidence: tuple[str, ...] = ()  # the classifier's own evidence, unchanged
    ocr_text: str = ""
    ocr_variant: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True if the image was read, OCR'd and scored without an error."""
        return self.error is None


class AlcoholImagePipeline:
    """Label images ``alcohol`` / ``non_alcohol`` from their superimposed text.

    Each image is loaded, OCR'd with the same setup the classifier was trained on
    (preprocessing plus the mirror retry), and its text is scored by the classifier.
    OCR always runs fresh: there is no cache, because cached results are keyed by file
    name and new images may reuse earlier file names. An image that cannot be
    processed gets ``unreadable_label`` and its error; it never stops the batch.
    """

    def __init__(
        self,
        ocr: OrientationTTA,
        classifier: TextClassifier,
        loader: ImageLoader | None = None,
        unreadable_label: str = UNREADABLE_LABEL,
    ) -> None:
        """Args:
        ocr: OCR engine with its preprocessing and retry rule.
        classifier: Fitted text classifier, with its threshold set.
        loader: Image loader (default: :class:`ImageLoader`).
        unreadable_label: Label of an image that cannot be processed (default: alcohol).
        """
        if unreadable_label not in LABELS:
            raise ValueError(f"unreadable_label must be one of {LABELS}, got {unreadable_label!r}")
        self.ocr: OrientationTTA = ocr
        self.classifier: TextClassifier = classifier
        self.loader: ImageLoader = loader or ImageLoader()
        self.unreadable_label: str = unreadable_label

    @classmethod
    def from_model_dir(
        cls, model_dir: Path, engine_factory: EngineFactory = paddle_engine, loader: ImageLoader | None = None
    ) -> AlcoholImagePipeline:
        """Load a saved model bundle and rebuild the OCR its card describes.

        Raises:
            FileNotFoundError, TypeError, ValueError: From :meth:`ModelBundle.load`, or (ValueError)
                if the card's OCR configuration cannot be rebuilt exactly.
        """
        bundle = ModelBundle.load(Path(model_dir))
        return cls(cls.build_ocr(bundle.card, engine_factory), bundle.classifier, loader)

    @staticmethod
    def build_ocr(card: ModelCard, engine_factory: EngineFactory = paddle_engine) -> OrientationTTA:
        """Rebuild the OCR a classifier was trained on from its model card.

        The configuration is checked against the card before the engine is loaded, and
        the loaded engine's id is checked after, so inference cannot silently run a
        different OCR than training did.

        Raises:
            ValueError: If the card names an unknown setup or the rebuilt OCR differs from the card's.
        """
        setup = ocr_setup(card.ocr_setup)
        try:
            selector = VariantSelector(
                str(card.ocr["strategy"]), card.ocr["threshold"], str(card.ocr["combine"]), tuple(card.ocr["variants"])
            )
        except KeyError as exc:
            raise ValueError(f"The model card's OCR entry lacks {exc}.") from None
        preprocessor = make_preprocessor(setup.preprocess)
        expected = tta_cache_config(setup.engine_id, selector, preprocessor.name)
        if expected != card.ocr:
            raise ValueError(
                f"OCR setup {setup.name!r} rebuilds as {expected}, but the model card says {card.ocr}."
            )
        engine = engine_factory(setup)
        if engine.engine_id != setup.engine_id:
            raise ValueError(f"The engine factory built {engine.engine_id!r}, expected {setup.engine_id!r}.")
        return OrientationTTA(engine, selector, preprocessor)

    def predict_file(self, path: Path) -> ImagePrediction:
        """Label one image file. Never raises: any failure gives ``unreadable_label`` and the error."""
        path = Path(path)
        try:
            result = self.ocr.run(self.loader.load(path), path.name)
            # OCR errors come back inside the result; a load error raises into the except below.
            if not result.ok:
                return self._unreadable(path.name, f"OCR failed: {result.error}", result.text, result.variant)
            # Empty text is not a special case: it has no known n-gram, so it gets the model's baseline
            # score (the card's no_text_score) and the threshold decides its label.
            prediction = self.classifier.predict([result.text])[0]
            return ImagePrediction(
                path.name, prediction.label, prediction.score, prediction.evidence, result.text, result.variant
            )
        except Exception as exc:  # one bad image must never stop a batch
            return self._unreadable(path.name, f"{type(exc).__name__}: {exc}")

    def predict_directory(self, directory: Path) -> list[ImagePrediction]:
        """Label every image directly inside ``directory``, in name order."""
        images, _ = ImageDirectory(directory).scan()
        predictions: list[ImagePrediction] = []
        for i, path in enumerate(images, start=1):
            prediction = self.predict_file(path)
            predictions.append(prediction)
            score = "n/a" if prediction.score is None else f"{prediction.score:.3f}"
            logger.info("[%d/%d] %s -> %s (score %s)", i, len(images), path.name, prediction.label, score)
        return predictions

    def _unreadable(self, file_name: str, error: str, text: str = "", variant: str = "") -> ImagePrediction:
        """The fallback prediction for an image that could not be processed."""
        logger.warning("%s: %s; labelled %s", file_name, error, self.unreadable_label)
        return ImagePrediction(file_name, self.unreadable_label, None, (), text, variant, error)
