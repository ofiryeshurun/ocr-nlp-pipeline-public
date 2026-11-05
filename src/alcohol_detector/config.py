"""The locked settings of the pipeline, in one place."""

from __future__ import annotations

from dataclasses import dataclass, field

from alcohol_detector.lexicon.matcher import EditBudget
from alcohol_detector.models.base import ALCOHOL
from alcohol_detector.ocr.paddle import PaddleOCREngine
from alcohol_detector.seed import DEFAULT_SEED
from alcohol_detector.text import LeetMapper, TextNormalizer


@dataclass(frozen=True)
class OCRSetup:
    """One OCR configuration whose train+val output is cached.

    All setups use the mirror retry on weak reads (``ocr.corpus.CONFIG_A``).
    """

    name: str
    det_model: str
    rec_model: str
    preprocess: str = "none"

    @property
    def engine_id(self) -> str:
        """Engine id of the cached run."""
        return PaddleOCREngine.make_engine_id(self.det_model, self.rec_model)


#: The OCR setups compared during model selection.
OCR_SETUPS: tuple[OCRSetup, ...] = (
    OCRSetup("v6_medium", "PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
    OCRSetup("v6_small", "PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
    OCRSetup("v6_small_lab_clahe", "PP-OCRv6_small_det", "PP-OCRv6_small_rec", "lab_clahe"),
)


#: The shipped pipeline: OCR setup name and classifier arm id.
SHIPPED_OCR_SETUP: str = "v6_small_lab_clahe"
SHIPPED_ARM: str = "v1_char_ngram"

#: The shipped classifier's threshold and logistic-regression C (scikit-learn's default).
SHIPPED_THRESHOLD: float = 0.5
SHIPPED_C: float = 1.0


#: Label of an image the pipeline cannot process (unreadable file, OCR failure): we could not check
#: it, so it is avoided, in line with the recall-first framing.
UNREADABLE_LABEL: str = ALCOHOL


def ocr_setup(name: str) -> OCRSetup:
    """The OCR setup called ``name``.

    Raises:
        ValueError: If no setup has that name.
    """
    for setup in OCR_SETUPS:
        if setup.name == name:
            return setup
    raise ValueError(f"Unknown OCR setup {name!r}; choose from {[s.name for s in OCR_SETUPS]}")


def shipped_ocr_setup() -> OCRSetup:
    """The OCR setup the pipeline ships with."""
    return ocr_setup(SHIPPED_OCR_SETUP)


@dataclass(frozen=True)
class ModelConfig:
    """Text-side settings shared by every classifier arm.

    Defaults are the locked choices: edit budget AUTO:5,7 with leetspeak decoding,
    the frozen MiniLM encoder at a pinned revision, seed 1998.
    """

    budget: EditBudget = field(default_factory=lambda: EditBudget(one_edit_from=5, two_edits_from=7))
    leet: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    seed: int = DEFAULT_SEED

    def normalizer(self) -> TextNormalizer:
        """The text normalizer every arm uses, so all arms read the same cleaned text."""
        return TextNormalizer(leet=LeetMapper() if self.leet else None)
