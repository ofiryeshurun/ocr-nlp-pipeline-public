"""Run OCR over labelled images and cache the results (one JSON per image).

Modes:
  per-variant  OCR each image under every flip/rotation variant separately (for benchmarking).
  tta          OCR with test-time augmentation using a chosen strategy (for the pipeline).

Examples:
  # The shipped setup over train + val (the cache scripts/train_final.py reads)
  python scripts/run_ocr.py --mode tta --splits train val --det-model PP-OCRv6_small_det --rec-model PP-OCRv6_small_rec --preprocess lab_clahe --strategy conditional --threshold 0.9 --combine best --variants original mirror

  # Benchmark: every variant separately, for the images listed in a CSV with a file_name column
  python scripts/run_ocr.py --mode per-variant --files-from subset.csv --det-model PP-OCRv5_mobile_det --rec-model en_PP-OCRv5_mobile_rec
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from alcohol_detector.data import SplitManifest
from alcohol_detector.ocr.batch import ImageJob, OCRBatchRunner
from alcohol_detector.ocr.paddle import PaddleOCREngine
from alcohol_detector.ocr.preprocess import PREPROCESSORS, make_preprocessor
from alcohol_detector.ocr.tta import ALL_VARIANTS, VariantSelector


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["per-variant", "tta"], required=True)
    p.add_argument("--image-root", type=Path, default=Path("images"))
    p.add_argument("--manifest", type=Path, default=Path("data/splits.csv"))
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    p.add_argument("--files-from", type=Path, help="CSV with a file_name column; restricts the run to those images")
    p.add_argument("--cache-dir", type=Path, default=Path("cache/ocr"))
    p.add_argument("--det-model", default=None, help="PaddleOCR detection model (default: library default)")
    p.add_argument("--rec-model", default=None, help="PaddleOCR recognition model (default: library default)")
    p.add_argument("--det-limit-side-len", type=int, default=None, help="Detection input size limit (px); default: native")
    p.add_argument("--det-limit-type", choices=["max", "min"], default=None, help="'max' shrinks long side, 'min' enlarges short side")
    p.add_argument("--preprocess", choices=sorted(PREPROCESSORS), default="none", help="Image preprocessing before OCR")
    p.add_argument("--strategy", choices=["always", "conditional"], default="always")
    p.add_argument("--threshold", type=float, default=None, help="Trigger threshold for --strategy conditional")
    p.add_argument("--combine", choices=["best", "union"], default="best")
    p.add_argument("--variants", nargs="+", default=[v.name for v in ALL_VARIANTS])
    return p.parse_args()


def build_jobs(args: argparse.Namespace) -> list[ImageJob]:
    """Select images from the manifest by split (and optionally by an explicit file list)."""
    rows = [a for a in SplitManifest.read(args.manifest) if a.split in args.splits]
    if args.files_from:
        with args.files_from.open(newline="", encoding="utf-8") as f:
            wanted = {r["file_name"] for r in csv.DictReader(f)}
        rows = [a for a in rows if a.file_name in wanted]
    return [ImageJob(a.file_name, args.image_root / a.label / a.file_name) for a in rows]


def main() -> None:
    """Build the engine and run the requested mode."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    jobs = build_jobs(args)
    logging.info("%d images (%s)", len(jobs), ", ".join(args.splits))
    runner = OCRBatchRunner(
        PaddleOCREngine(args.det_model, args.rec_model, det_limit_side_len=args.det_limit_side_len, det_limit_type=args.det_limit_type),
        args.cache_dir, preprocessor=make_preprocessor(args.preprocess)
    )
    if args.mode == "per-variant":
        variants = tuple(v for v in ALL_VARIANTS if v.name in args.variants)
        runner.run_per_variant(jobs, variants)
    else:
        selector = VariantSelector(args.strategy, args.threshold, args.combine, tuple(args.variants))
        results = runner.run_tta(jobs, selector)
        failed = [r.file_name for r in results if not r.ok]
        logging.info("done: %d results, %d errors %s", len(results), len(failed), failed[:10])


if __name__ == "__main__":
    main()
