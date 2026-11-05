"""Safe image loading."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageLoadError(Exception):
    """Raised when an image file is missing, empty or cannot be decoded."""


class ImageLoader:
    """Load images as BGR ``uint8`` arrays, rejecting anything unreadable."""

    def load(self, path: Path) -> np.ndarray:
        """Load one image.

        ``np.fromfile`` + ``cv2.imdecode`` is used instead of ``cv2.imread`` so
        non-ASCII Windows paths work.

        Raises:
            ImageLoadError: If the file is missing, empty or not a decodable image.
        """
        path = Path(path)
        if not path.is_file():
            raise ImageLoadError(f"File not found: {path}")
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            raise ImageLoadError(f"Empty file: {path}")
        # IMREAD_COLOR always gives 3-channel BGR (grayscale expanded, alpha dropped), as the preprocessors expect.
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ImageLoadError(f"Cannot decode image: {path}")
        return image
