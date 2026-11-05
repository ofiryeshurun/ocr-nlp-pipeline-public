"""Image preprocessing applied before OCR."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class ImagePreprocessor(ABC):
    """Swappable image transform run before OCR (and before any TTA variant)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier (part of the OCR cache key)."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Return a 3-channel BGR image ready for OCR."""


class NoPreprocessor(ImagePreprocessor):
    """Pass the image through unchanged."""

    @property
    def name(self) -> str:
        return "none"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image


class GrayCLAHEPreprocessor(ImagePreprocessor):
    """Grayscale, then CLAHE (local contrast equalisation), aimed at faint or low-contrast text.

    Defaults are OpenCV's commonly used values, not tuned on this data.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid: int = 8) -> None:
        """Args:
        clip_limit: CLAHE contrast limit.
        tile_grid: Number of tiles per side for local equalisation.
        """
        self.clip_limit: float = clip_limit
        self.tile_grid: int = tile_grid
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))

    @property
    def name(self) -> str:
        return f"gray_clahe(clip={self.clip_limit},tile={self.tile_grid})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.cvtColor(self._clahe.apply(gray), cv2.COLOR_GRAY2BGR)


class ColorCLAHEPreprocessor(ImagePreprocessor):
    """CLAHE on the lightness channel only (LAB colour space), so colours are kept.

    Unlike :class:`GrayCLAHEPreprocessor`, overlays that differ from their
    background only by colour (e.g. orange text on white) stay distinguishable.
    Defaults are the same commonly used OpenCV values as the grayscale version, not tuned on this data.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid: int = 8) -> None:
        """Args:
        clip_limit: CLAHE contrast limit.
        tile_grid: Number of tiles per side for local equalisation.
        """
        self.clip_limit: float = clip_limit
        self.tile_grid: int = tile_grid
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))

    @property
    def name(self) -> str:
        return f"lab_clahe(clip={self.clip_limit},tile={self.tile_grid})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge((self._clahe.apply(lightness), a, b)), cv2.COLOR_LAB2BGR)


class UpscalePreprocessor(ImagePreprocessor):
    """Enlarge the image (bicubic) so small text gets more pixels per character."""

    def __init__(self, factor: float = 2.0) -> None:
        """Args:
        factor: Scale factor applied to both sides.
        """
        self.factor: float = factor

    @property
    def name(self) -> str:
        return f"upscale(x{self.factor})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, None, fx=self.factor, fy=self.factor, interpolation=cv2.INTER_CUBIC)


class AutoContrastPreprocessor(ImagePreprocessor):
    """Stretch each colour channel so its low/high percentiles map to 0/255.

    A global stretch that keeps colour, aimed at washed-out or semi-transparent overlays.
    """

    def __init__(self, low_pct: float = 1.0, high_pct: float = 99.0) -> None:
        """Args:
        low_pct, high_pct: Percentiles mapped to black and white.
        """
        self.low_pct: float = low_pct
        self.high_pct: float = high_pct

    @property
    def name(self) -> str:
        return f"autocontrast(p{self.low_pct}-{self.high_pct})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        out = np.empty_like(image)
        for c in range(image.shape[2]):
            lo, hi = np.percentile(image[..., c], [self.low_pct, self.high_pct])
            scale = 255.0 / max(hi - lo, 1.0)
            out[..., c] = np.clip((image[..., c].astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)
        return out


class BilateralCLAHEPreprocessor(ImagePreprocessor):
    """Edge-preserving smoothing, then colour CLAHE.

    The bilateral filter flattens background texture while keeping strong edges,
    so the contrast boost that follows amplifies text more than clutter.
    Defaults are the canonical OpenCV example values, not tuned on this data.
    """

    def __init__(self, diameter: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> None:
        """Args:
        diameter: Pixel neighbourhood diameter of the bilateral filter.
        sigma_color: How different in colour neighbours may be and still be averaged.
        sigma_space: How far apart neighbours may be and still be averaged.
        """
        self.diameter: int = diameter
        self.sigma_color: float = sigma_color
        self.sigma_space: float = sigma_space
        self._clahe = ColorCLAHEPreprocessor()

    @property
    def name(self) -> str:
        return f"bilateral(d={self.diameter},sc={self.sigma_color},ss={self.sigma_space})+{self._clahe.name}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        smoothed = cv2.bilateralFilter(image, self.diameter, self.sigma_color, self.sigma_space)
        return self._clahe.apply(smoothed)


class TopHatPreprocessor(ImagePreprocessor):
    """Morphological contrast boost on lightness: L' = L + top-hat(L) - black-hat(L).

    Top-hat keeps thin bright structures (strokes) and black-hat thin dark ones,
    each relative to the local background, so faint strokes are strengthened
    while broad background shapes are not. Applied in LAB so colour is kept.
    The kernel should be larger than a letter stroke; 15 px is a common
    default, not tuned on this data.
    """

    def __init__(self, kernel_size: int = 15) -> None:
        """Args:
        kernel_size: Side of the square structuring element (pixels).
        """
        self.kernel_size: int = kernel_size
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    @property
    def name(self) -> str:
        return f"tophat(k={self.kernel_size})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        top = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, self._kernel)
        black = cv2.morphologyEx(lightness, cv2.MORPH_BLACKHAT, self._kernel)
        boosted = cv2.subtract(cv2.add(lightness, top), black)
        return cv2.cvtColor(cv2.merge((boosted, a, b)), cv2.COLOR_LAB2BGR)


class ChannelViewPreprocessor(ImagePreprocessor):
    """Show OCR a single derived channel (replicated to 3 channels).

    - ``saturation``: HSV saturation. Translucent white tints lower saturation
      over colourful photos, which brightness alone can hide.
    - ``min_channel``: per-pixel min(B, G, R). Bright tints raise it, while
      saturated photo colours keep it low.
    """

    VIEWS: tuple[str, ...] = ("saturation", "min_channel")

    def __init__(self, view: str) -> None:
        """Args:
        view: One of ``VIEWS``.
        """
        if view not in self.VIEWS:
            raise ValueError(f"Unknown channel view {view!r}; choose from {self.VIEWS}")
        self.view: str = view

    @property
    def name(self) -> str:
        return f"channel({self.view})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.view == "saturation":
            channel = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1]
        else:
            channel = image.min(axis=2)
        return cv2.cvtColor(np.ascontiguousarray(channel), cv2.COLOR_GRAY2BGR)


class HybridGrayTopHatCLAHEPreprocessor(ImagePreprocessor):
    """Grayscale, then top-hat/black-hat stroke boost, then strong CLAHE.

    A stacked contrast pipeline for faint text: grayscale drops colour,
    top-hat strengthens thin strokes against the local background, and CLAHE
    (clip 4.0) equalises local contrast. Common, untuned defaults.
    """

    def __init__(self, kernel_size: int = 15, clip_limit: float = 4.0, tile_grid: int = 8) -> None:
        """Args:
        kernel_size: Top-hat structuring element side (pixels).
        clip_limit: CLAHE contrast limit.
        tile_grid: CLAHE tiles per side.
        """
        self.kernel_size: int = kernel_size
        self.clip_limit: float = clip_limit
        self.tile_grid: int = tile_grid
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))

    @property
    def name(self) -> str:
        return f"hybrid_gray_tophat{self.kernel_size}_clahe{self.clip_limit}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self._kernel)
        black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self._kernel)
        boosted = cv2.subtract(cv2.add(gray, top), black)
        return cv2.cvtColor(self._clahe.apply(boosted), cv2.COLOR_GRAY2BGR)


#: CLI key -> preprocessor. A class is built with its defaults; an instance carries non-default settings.
PREPROCESSORS: dict[str, ImagePreprocessor | type[ImagePreprocessor]] = {
    "hybrid": HybridGrayTopHatCLAHEPreprocessor,
    "tophat": TopHatPreprocessor,
    "saturation": ChannelViewPreprocessor("saturation"),
    "min_channel": ChannelViewPreprocessor("min_channel"),
    "bilateral_clahe": BilateralCLAHEPreprocessor,
    "none": NoPreprocessor,
    "gray_clahe": GrayCLAHEPreprocessor,
    "lab_clahe": ColorCLAHEPreprocessor,
    "lab_clahe_strong": ColorCLAHEPreprocessor(clip_limit=4.0),
    "upscale2": UpscalePreprocessor,
    "autocontrast": AutoContrastPreprocessor,
}


def make_preprocessor(key: str) -> ImagePreprocessor:
    """Build a preprocessor from its CLI key (see ``PREPROCESSORS``)."""
    if key not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocessor {key!r}; choose from {sorted(PREPROCESSORS)}")
    entry = PREPROCESSORS[key]
    return entry if isinstance(entry, ImagePreprocessor) else entry()
