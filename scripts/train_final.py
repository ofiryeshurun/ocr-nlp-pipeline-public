"""Fit the shipped classifier on all train+val OCR text and save it with its model card.

The shipped setup is v1 (char n-gram TF-IDF + logistic regression, C=1) on v6 small + LAB CLAHE
OCR, threshold 0.5. Training reads the cached OCR text of train and val only;
test is never used. Seed 1998, so a rerun writes the same model.

Writes, under --output-dir:
- classifier.joblib    the fitted classifier (a pickle: load only files you created)
- model_card.json      OCR configuration, threshold, C, training data, no-text score, versions

Example:
    python scripts/train_final.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path

import sklearn

from alcohol_detector.config import (
    SHIPPED_ARM,
    SHIPPED_C,
    SHIPPED_THRESHOLD,
    ModelConfig,
    shipped_ocr_setup,
)
from alcohol_detector.models import ALCOHOL, NON_ALCOHOL, CharNgramClassifier
from alcohol_detector.models.bundle import CARD_FILE, MODEL_FILE, ModelBundle, ModelCard
from alcohol_detector.ocr.batch import tta_cache_config
from alcohol_detector.ocr.corpus import CONFIG_A, OCRCorpus
from alcohol_detector.ocr.preprocess import make_preprocessor
from alcohol_detector.seed import set_global_seed

TRAIN_SPLITS: tuple[str, ...] = ("train", "val")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=Path("data/splits.csv"), help="file_name,label,split CSV")
    p.add_argument("--cache-root", type=Path, default=Path("cache/ocr"), help="Root of the OCR cache")
    p.add_argument("--output-dir", type=Path, default=Path("models"), help="Where to write the model bundle")
    return p.parse_args()


def git_commit(repo: Path) -> str:
    """The commit of ``repo``, suffixed "-dirty" if tracked files have uncommitted changes, or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"], cwd=repo, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    if SHIPPED_ARM != CharNgramClassifier.name:
        raise SystemExit(f"This script trains {CharNgramClassifier.name}, but the shipped arm is {SHIPPED_ARM}.")
    config = ModelConfig()
    set_global_seed(config.seed)
    setup = shipped_ocr_setup()
    docs = OCRCorpus(args.splits, args.cache_root, setup.engine_id, preprocess=setup.preprocess).load(TRAIN_SPLITS)
    texts, labels = [d.text for d in docs], [d.label for d in docs]

    classifier = CharNgramClassifier(config.normalizer(), config.seed, c=SHIPPED_C, threshold=SHIPPED_THRESHOLD)
    classifier.fit(texts, labels)
    # Empty text has no n-gram, so this is the intercept-only baseline: the score, and hence the label,
    # of every image without usable text. Inference also uses it to check the bundle.
    no_text_score = float(classifier.scores([""])[0])
    card = ModelCard(
        arm=classifier.name, threshold=SHIPPED_THRESHOLD, c=SHIPPED_C, ocr_setup=setup.name,
        ocr=tta_cache_config(setup.engine_id, CONFIG_A, make_preprocessor(setup.preprocess).name),
        leet=config.leet, seed=config.seed, train_splits=list(TRAIN_SPLITS), n_train=len(docs),
        n_train_alcohol=sum(label == ALCOHOL for label in labels), no_text_score=no_text_score,
        no_text_label=ALCOHOL if no_text_score >= SHIPPED_THRESHOLD else NON_ALCOHOL,
        sklearn_version=sklearn.__version__, created=dt.date.today().isoformat(), git_commit=git_commit(Path(__file__).resolve().parents[1]),
    )
    ModelBundle(classifier, card).save(args.output_dir)

    size_kb = (args.output_dir / MODEL_FILE).stat().st_size / 1024
    print(f"Fit {card.arm} on {card.n_train} {'+'.join(TRAIN_SPLITS)} texts ({card.n_train_alcohol} alcohol), "
          f"OCR setup {card.ocr_setup}, C={card.c}, threshold {card.threshold}")
    print(f"No-text score {no_text_score:.4f} -> {card.no_text_label}")
    print(f"Wrote {args.output_dir / MODEL_FILE} ({size_kb:.0f} KB) and {args.output_dir / CARD_FILE}")


if __name__ == "__main__":
    main()
