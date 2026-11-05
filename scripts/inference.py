"""Label every image in a folder as alcohol / non_alcohol from its superimposed text.

Reads the images directly inside --input-dir (jpg, jpeg, png, bmp, webp; sub-folders are
not entered), runs the shipped pipeline (OCR with the model card's settings, then the
text classifier) and writes one row per image to --output:

    file_name,prediction
    img_001.jpg,alcohol

File and folder names are never used to decide the label. OCR runs fresh on every image
(no cache). An image that cannot be read or OCR'd is labelled alcohol and its error
is logged; it never stops the run. --details also writes each image's score, the
classifier's evidence, the OCR text and any error.

The first run downloads the PaddleOCR models (about 40 MB), so it needs internet access.

Example:
    python scripts/inference.py --input-dir path/to/images --output results.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

from alcohol_detector.inference import AlcoholImagePipeline, DetailsCSV, ResultsCSV
from alcohol_detector.models.base import ALCOHOL, NON_ALCOHOL
from alcohol_detector.seed import set_global_seed

#: The shipped model bundle, found relative to this script so the CLI works from any folder.
DEFAULT_MODEL_DIR: Path = Path(__file__).resolve().parents[1] / "models"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True, help="Folder of images to label")
    p.add_argument("--output", type=Path, default=Path("results.csv"), help="Results CSV (file_name,prediction)")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Model bundle folder (default: models/)")
    p.add_argument("--details", type=Path, default=None, help="Optional CSV with score, evidence, OCR text and errors")
    return p.parse_args()


def main() -> None:
    """Load the pipeline, label the folder and write the CSV file(s)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    set_global_seed()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {args.input_dir}")
    try:
        pipeline = AlcoholImagePipeline.from_model_dir(args.model_dir)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise SystemExit(f"Cannot load the model from {args.model_dir}: {exc}") from exc

    start = time.perf_counter()
    predictions = pipeline.predict_directory(args.input_dir)
    elapsed = time.perf_counter() - start
    if not predictions:
        logging.warning("No images found in %s; writing a header-only file.", args.input_dir)

    ResultsCSV.write(predictions, args.output)
    if args.details is not None:
        DetailsCSV.write(predictions, args.details)

    counts = Counter(p.label for p in predictions)
    failed = [p.file_name for p in predictions if not p.ok]
    per_image = elapsed / len(predictions) if predictions else 0.0
    print(f"{len(predictions)} images in {elapsed:.0f} s ({per_image:.2f} s/img): "
          f"{counts[ALCOHOL]} {ALCOHOL}, {counts[NON_ALCOHOL]} {NON_ALCOHOL}")
    if failed:
        print(f"{len(failed)} could not be processed and were labelled {pipeline.unreadable_label}: {failed[:10]}")
    print(f"Wrote {args.output}" + (f" and {args.details}" if args.details is not None else ""))


if __name__ == "__main__":
    main()
