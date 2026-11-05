"""Read and write the prediction CSV files."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from alcohol_detector.inference.pipeline import ImagePrediction
from alcohol_detector.models.base import LABELS

#: Joins a prediction's evidence items in the details file.
EVIDENCE_SEPARATOR: str = " | "


def _check_predictions(predictions: Sequence[ImagePrediction]) -> None:
    """Reject unknown labels and repeated file names before anything is written."""
    bad = sorted({p.label for p in predictions} - set(LABELS))
    if bad:
        raise ValueError(f"Unknown labels {bad}; expected {LABELS}.")
    names = [p.file_name for p in predictions]
    if len(names) != len(set(names)):
        raise ValueError("A file name appears more than once in the predictions.")


class ResultsCSV:
    """The results file: ``file_name,prediction``, one row per image."""

    FIELDS: tuple[str, str] = ("file_name", "prediction")

    @classmethod
    def write(cls, predictions: Sequence[ImagePrediction], path: Path) -> None:
        """Write the predictions, creating parent folders if needed.

        Raises:
            ValueError: If a label is not ``alcohol`` / ``non_alcohol`` or a file name repeats.
        """
        _check_predictions(predictions)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cls.FIELDS)
            writer.writerows((p.file_name, p.label) for p in predictions)

    @classmethod
    def read(cls, path: Path) -> dict[str, str]:
        """Read a results file as {file name: prediction}.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the header, a label or a repeated file name is invalid.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Results file not found: {path}")
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != cls.FIELDS:
                raise ValueError(f"Unexpected header in {path}: {reader.fieldnames}; expected {cls.FIELDS}")
            rows = [(r["file_name"], r["prediction"]) for r in reader]
        bad = sorted({label for _, label in rows} - set(LABELS))
        if bad:
            raise ValueError(f"Unknown labels in {path}: {bad}")
        predictions = dict(rows)
        if len(predictions) != len(rows):
            raise ValueError(f"A file name appears more than once in {path}.")
        return predictions


class DetailsCSV:
    """Optional per-image details: score, evidence, OCR text and any error.

    ``score`` is empty for an image that could not be processed; ``evidence`` is the
    classifier's own evidence, empty when the text has no known n-gram.
    """

    FIELDS: tuple[str, ...] = ("file_name", "prediction", "score", "evidence", "ocr_text", "ocr_variant", "error")

    @classmethod
    def write(cls, predictions: Sequence[ImagePrediction], path: Path) -> None:
        """Write the details, creating parent folders if needed.

        Raises:
            ValueError: If a label is not ``alcohol`` / ``non_alcohol`` or a file name repeats.
        """
        _check_predictions(predictions)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cls.FIELDS)
            for p in predictions:
                # repr gives the shortest text that reads back as the same float, so scores are not rounded.
                writer.writerow((
                    p.file_name, p.label, "" if p.score is None else repr(p.score),
                    EVIDENCE_SEPARATOR.join(p.evidence), p.ocr_text, p.ocr_variant, p.error or "",
                ))
