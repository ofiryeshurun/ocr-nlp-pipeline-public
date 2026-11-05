"""Create the stratified 70/15/15 train/val/test split manifest (seed 1998).

The split manifest ``data/splits.csv`` is exactly this script's output, and
every other script reads it. The script refuses to overwrite an existing
manifest; to reproduce the split, write it elsewhere and compare.

Usage:
    python scripts/make_split.py --image-root images --output data/splits.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from alcohol_detector.data import LabeledImageIndex, SplitManifest, StratifiedSplitter
from alcohol_detector.seed import DEFAULT_SEED


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-root", type=Path, default=Path("images"), help="Folder with alcohol/ and non_alcohol/")
    parser.add_argument(
        "--output", type=Path, default=Path("data/splits.csv"), help="Where to write the manifest (must not exist)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Index the images, split them and write the manifest; return the exit code."""
    args = parse_args(argv)
    if args.output.exists():
        print(f"{args.output} already exists; refusing to overwrite it.", file=sys.stderr)
        return 1
    records = LabeledImageIndex(args.image_root).scan()
    assignments = StratifiedSplitter(seed=args.seed).split(records)
    SplitManifest.write(assignments, args.output)

    counts = Counter((a.split, a.label) for a in assignments)
    print(f"Wrote {len(assignments)} rows to {args.output}")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: alcohol={counts[(split, 'alcohol')]:3d}  non_alcohol={counts[(split, 'non_alcohol')]:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
