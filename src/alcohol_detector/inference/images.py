"""List the images of an inference input folder."""

from __future__ import annotations

import logging
from pathlib import Path

from alcohol_detector.data import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


class ImageDirectory:
    """The image files directly inside one folder, in name order.

    Sub-folders are not entered, and only the extension decides what counts as an
    image. File and folder names are never used as a signal for the label.
    """

    def __init__(self, root: Path, extensions: frozenset[str] = IMAGE_EXTENSIONS) -> None:
        """Args:
        root: The input folder.
        extensions: Lower-case suffixes treated as images (default: jpg, jpeg, png, bmp, webp).

        Raises:
            FileNotFoundError: If ``root`` does not exist.
            NotADirectoryError: If ``root`` is not a folder.
        """
        self.root: Path = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Input folder not found: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Not a folder: {self.root}")
        self.extensions: frozenset[str] = extensions

    def scan(self) -> tuple[list[Path], list[Path]]:
        """Return (image files, other entries skipped), both sorted by name.

        Extensions are compared case-insensitively, so ``IMG.JPG`` counts.
        """
        images: list[Path] = []
        skipped: list[Path] = []
        for path in sorted(self.root.iterdir(), key=lambda p: p.name):
            (images if path.is_file() and path.suffix.lower() in self.extensions else skipped).append(path)
        for path in skipped:
            logger.warning("Skipping %s: not an image file (%s)", path.name, ", ".join(sorted(self.extensions)))
        return images, skipped
