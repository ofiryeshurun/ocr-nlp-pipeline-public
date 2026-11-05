"""Score a results CSV against the true labels of one split.

Labels come from data/splits.csv, never from file names. The results must cover exactly
the split's images. Reports recall first (the primary metric), then precision, F1
and accuracy, each with a 95% paired bootstrap interval, and the confusion matrix.

Writes:
- <output-dir>/metrics.json      n, confusion counts, metrics with intervals, inputs
- --figure (PNG)                 confusion matrix (default: <output-dir>/confusion_matrix.png)

Example:
    python scripts/evaluate.py --predictions test_results.csv --split test --output-dir test_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alcohol_detector.data import SPLITS
from alcohol_detector.inference import ConfusionMatrixPlot, ResultsCSV, SplitScorer
from alcohol_detector.models.metrics import METRIC_NAMES, ModelEvaluator
from alcohol_detector.seed import DEFAULT_SEED

#: Bootstrap settings, the same as for the cross-validation metrics.
CONFIDENCE: float = 0.95
N_RESAMPLES: int = 9999


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True, help="Results CSV (file_name,prediction)")
    p.add_argument("--split", choices=SPLITS, required=True, help="Split whose labels to score against")
    p.add_argument("--manifest", type=Path, default=Path("data/splits.csv"), help="file_name,label,split CSV")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write metrics.json")
    p.add_argument("--figure", type=Path, default=None, help="Confusion-matrix PNG path")
    return p.parse_args()


def main() -> None:
    """Score the predictions and write the metrics and the figure."""
    args = parse_args()
    try:
        predictions = ResultsCSV.read(args.predictions)
        scored = SplitScorer(args.manifest, ModelEvaluator(DEFAULT_SEED, CONFIDENCE, N_RESAMPLES)).score(
            predictions, args.split
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        **scored.to_dict(),
        "bootstrap": {"method": "paired percentile", "confidence": CONFIDENCE, "n_resamples": N_RESAMPLES,
                      "seed": DEFAULT_SEED},
        "inputs": {"predictions": args.predictions.as_posix(), "manifest": args.manifest.as_posix()},
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figure = args.figure or args.output_dir / "confusion_matrix.png"
    ConfusionMatrixPlot.save(scored.result, figure, f"Confusion matrix, {args.split} split (n={scored.result.n})")

    r = scored.result
    print(f"{args.split}: n={r.n}  TP={r.tp}  FN={r.fn}  FP={r.fp}  TN={r.tn}")
    for name in METRIC_NAMES:
        ci = r[name]
        print(f"  {name:<9} {ci.value:.3f}  [{ci.low:.3f}, {ci.high:.3f}]")
    print(f"Wrote {metrics_path} and {figure}")


if __name__ == "__main__":
    main()
