"""Cross-validate the classifier arms on train+val OCR text, for each cached OCR setup.

Stratified 5-fold CV (seed 1998), the same folds for every setup and arm. Test is never used.

Writes, under --output-dir:
- model_runs.csv        one row per (OCR setup, arm): pooled out-of-fold metrics with 95% bootstrap CIs,
                        spread across folds, timings
- ocr_run_stats.csv     per OCR setup: seconds per image, retry rate, text-found rate
- cv_predictions.csv    per (OCR setup, arm, image): fold, label, out-of-fold prediction, score, evidence, OCR text
- c_search.csv          tuned arms only: inner-CV ROC-AUC per (OCR setup, arm, outer fold, C)

Example:
    python scripts/compare_models.py
    python scripts/compare_models.py --setups v6_medium --arms v0_lexicon v1_char_ngram
    python scripts/compare_models.py --setups v6_small_lab_clahe --arms v1_char_ngram_tuned v2_minilm_tuned
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from alcohol_detector.config import OCR_SETUPS, ModelConfig
from alcohol_detector.lexicon import AlcoholLexicon
from alcohol_detector.models import ModelEvaluator, ModelRunLog
from alcohol_detector.models.comparison import ARM_NAMES, TUNED_ARM_NAMES, ArmFactory, ModelComparison
from alcohol_detector.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=Path("data/splits.csv"), help="file_name,label,split CSV")
    p.add_argument("--cache-root", type=Path, default=Path("cache/ocr"), help="Root of the OCR cache")
    p.add_argument("--lexicon", type=Path, default=Path("data/lexicon.yaml"), help="Alcohol lexicon YAML")
    p.add_argument("--output-dir", type=Path, default=Path("results"), help="Where to write the CSVs")
    p.add_argument("--setups", nargs="+", choices=[s.name for s in OCR_SETUPS], default=[s.name for s in OCR_SETUPS])
    p.add_argument("--arms", nargs="+", choices=ARM_NAMES + TUNED_ARM_NAMES, default=list(ARM_NAMES),
                   help="Default: the plain arms. The *_tuned arms pick C by nested CV.")
    p.add_argument("--n-folds", type=int, default=5, help="Stratified folds (5 = scikit-learn's default cv)")
    return p.parse_args()


def merge_csv(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    """Write ``new`` to ``path``, replacing only the rows whose key groups ``new`` covers.

    A run on a subset of setups or arms then keeps the other rows already on disk.
    Floats are read back exactly (``round_trip``), so the kept rows are unchanged.
    """
    if path.is_file():
        old = pd.read_csv(path, float_precision="round_trip")
        covered = old[keys].apply(tuple, axis=1).isin(set(new[keys].apply(tuple, axis=1)))
        new = pd.concat([old[~covered], new], ignore_index=True)
    new.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    config = ModelConfig()
    set_global_seed(config.seed)
    lexicon = AlcoholLexicon.from_yaml(args.lexicon)
    comparison = ModelComparison(
        args.splits, args.cache_root, ArmFactory(config, lexicon), ModelEvaluator(config.seed), config.seed, args.n_folds
    )
    setups = [s for s in OCR_SETUPS if s.name in args.setups]
    result = comparison.run(setups, args.arms)  # train+val only; test is scored once, by evaluate.py

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = ModelRunLog(args.output_dir / "model_runs.csv")
    for run in result.runs:
        log.upsert(run)
    log.save()
    merge_csv(args.output_dir / "ocr_run_stats.csv", pd.DataFrame([asdict(s) for s in result.ocr_stats]), ["setup"])
    merge_csv(args.output_dir / "cv_predictions.csv", result.predictions, ["ocr_setup", "arm"])
    if not result.c_search.empty:
        merge_csv(args.output_dir / "c_search.csv", result.c_search, ["ocr_setup", "arm"])
    print(f"Wrote {len(result.runs)} runs to {log.path}")


if __name__ == "__main__":
    main()
