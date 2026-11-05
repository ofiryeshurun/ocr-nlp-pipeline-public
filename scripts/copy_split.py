"""Copy the images of one split into a flat folder, to run scripts/inference.py on them.

Used to score the test split through the real inference CLI. Files are
copied without being opened, so no image is displayed. The destination must be empty.

Example:
    python scripts/copy_split.py --split test --dest cache/inference/test_images
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alcohol_detector.data import SPLITS, SplitCopier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--dest", type=Path, required=True, help="Empty folder to copy the images into")
    p.add_argument("--manifest", type=Path, default=Path("data/splits.csv"), help="file_name,label,split CSV")
    p.add_argument("--image-root", type=Path, default=Path("images"), help="Folder with one sub-folder per label")
    return p.parse_args()


def main() -> None:
    """Copy the split's images."""
    args = parse_args()
    try:
        n = SplitCopier(args.manifest, args.image_root).copy(args.split, args.dest)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Copied {n} {args.split} images to {args.dest}")


if __name__ == "__main__":
    main()
