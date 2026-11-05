"""A persistent log of OCR configuration experiments (one CSV row per configuration)."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class OCRExperiment:
    """Summary metrics of one OCR configuration on one evaluation set."""

    experiment_id: str
    engine: str
    preprocess: str
    preprocess_role: str  # "none" (raw image) or "replace" (OCR sees only the processed image)
    variants: str  # "+"-joined variant names
    strategy: str  # "original only", "always" or "conditional@<T>"
    combine: str
    eval_set: str  # name of the annotated image set
    n_images: int
    key_term_hit: float
    key_term_sim: float
    overlay_cer: float
    sec_per_img: float
    ocr_passes: float
    notes: str = ""


class ExperimentLog:
    """Load, update and save the experiment CSV. Rows are keyed by ``experiment_id``."""

    def __init__(self, path: Path) -> None:
        """Args:
        path: CSV file; created on first save.
        """
        self.path: Path = Path(path)
        self._rows: dict[str, OCRExperiment] = {}
        if self.path.is_file():
            self._load()

    def _load(self) -> None:
        types = {f.name: f.type for f in fields(OCRExperiment)}
        with self.path.open(newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                row = {
                    k: (int(v) if types[k] == "int" else float(v) if types[k] == "float" else v)
                    for k, v in raw.items()
                }
                exp = OCRExperiment(**row)
                self._rows[exp.experiment_id] = exp

    def upsert(self, experiment: OCRExperiment) -> None:
        """Add an experiment, replacing any earlier row with the same id."""
        self._rows[experiment.experiment_id] = experiment

    @property
    def experiments(self) -> list[OCRExperiment]:
        """All experiments, sorted by evaluation set then runtime."""
        return sorted(self._rows.values(), key=lambda e: (e.eval_set, e.sec_per_img))

    def save(self) -> None:
        """Write all rows to the CSV."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        names = [f.name for f in fields(OCRExperiment)]
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=names)
            writer.writeheader()
            for exp in self.experiments:
                writer.writerow(asdict(exp))
