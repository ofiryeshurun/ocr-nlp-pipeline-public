"""Score a predictions file against the labels of one split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from alcohol_detector.data import SplitManifest
from alcohol_detector.models.base import ALCOHOL, NON_ALCOHOL
from alcohol_detector.models.metrics import EvaluationResult, ModelEvaluator

#: How many missing or extra file names an error message lists.
_MAX_NAMES_SHOWN: int = 10


@dataclass(frozen=True)
class SplitScore:
    """Metrics of one predictions file on one split. Alcohol is the positive class."""

    split: str
    result: EvaluationResult

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict: split, n, confusion counts and each metric with its interval."""
        r = self.result
        return {
            "split": self.split,
            "n": r.n,
            "confusion": {"tp": r.tp, "fp": r.fp, "fn": r.fn, "tn": r.tn},
            "metrics": {name: asdict(ci) for name, ci in r.metrics.items()},
        }


class SplitScorer:
    """Join predictions to the true labels of one split and compute the metrics.

    Labels come from the split manifest, never from file names. The predictions must
    cover exactly the split's images: a missing or extra file is an error, so a partial
    run can never be scored by mistake.
    """

    def __init__(self, manifest: Path, evaluator: ModelEvaluator) -> None:
        """Args:
        manifest: The ``file_name,label,split`` CSV.
        evaluator: Computes the metrics and their bootstrap intervals.
        """
        self.manifest: Path = Path(manifest)
        self.evaluator: ModelEvaluator = evaluator

    def score(self, predictions: dict[str, str], split: str) -> SplitScore:
        """Score {file name: prediction} against ``split``.

        Raises:
            ValueError: If the split is empty or the predictions miss or add files.
        """
        truth = {r.file_name: r.label for r in SplitManifest.read(self.manifest) if r.split == split}
        if not truth:
            raise ValueError(f"No images in split {split!r} of {self.manifest}.")
        missing = sorted(truth.keys() - predictions.keys())
        extra = sorted(predictions.keys() - truth.keys())
        if missing or extra:
            raise ValueError(
                f"Predictions do not match split {split!r}: {len(missing)} missing "
                f"{missing[:_MAX_NAMES_SHOWN]}, {len(extra)} extra {extra[:_MAX_NAMES_SHOWN]}."
            )
        names = sorted(truth)
        result = self.evaluator.evaluate([truth[n] for n in names], [predictions[n] for n in names])
        return SplitScore(split, result)


class ConfusionMatrixPlot:
    """A 2x2 confusion matrix with counts and the share of each true class, as a PNG."""

    #: One-hue sequential ramp (light -> dark blue): darker = more images.
    CMAP: LinearSegmentedColormap = LinearSegmentedColormap.from_list("counts", ["#eef5fd", "#184f95"])

    @classmethod
    def save(cls, result: EvaluationResult, path: Path, title: str) -> None:
        """Draw the matrix (rows = true label, columns = prediction) and save it to ``path``."""
        labels = (ALCOHOL, NON_ALCOHOL)
        counts = [[result.tp, result.fn], [result.fp, result.tn]]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = Figure(figsize=(4.6, 3.9), dpi=200)  # no pyplot: no global backend or figure state
        ax = fig.subplots()
        peak = max(max(row) for row in counts) or 1
        ax.imshow(counts, cmap=cls.CMAP, vmin=0, vmax=peak)
        for i, row in enumerate(counts):
            row_total = sum(row) or 1
            for j, count in enumerate(row):
                ink = "white" if count > peak / 2 else "#1f1f1f"
                ax.text(j, i - 0.08, f"{count}", ha="center", va="center", fontsize=16, color=ink, weight="bold")
                ax.text(j, i + 0.22, f"{count / row_total:.1%} of {labels[i]}", ha="center", va="center",
                        fontsize=8, color=ink)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color="white", linewidth=2)  # surface gap between cells
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(which="both", length=0)
        fig.tight_layout()
        fig.savefig(path)
