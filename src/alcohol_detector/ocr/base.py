"""Core OCR data types and the engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TextBox:
    """One piece of text read by OCR."""

    text: str
    confidence: float
    polygon: list[list[float]] = field(default_factory=list)


@dataclass
class OCRResult:
    """OCR output for one image (possibly after trying several image variants)."""

    file_name: str
    boxes: list[TextBox]
    variant: str = "original"
    engine_id: str = ""
    elapsed_s: float = 0.0
    variants_tried: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True if the image was read without an error (it may still contain no text)."""
        return self.error is None

    @property
    def text(self) -> str:
        """All recognised text, joined with spaces in reading order."""
        return " ".join(b.text for b in self.boxes)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCRResult:
        """Rebuild from :meth:`to_dict` output."""
        boxes = [TextBox(**b) for b in data.get("boxes", [])]
        return cls(**{**data, "boxes": boxes})


class OCREngine(ABC):
    """Swappable OCR backend: image array in, text boxes out."""

    @property
    @abstractmethod
    def engine_id(self) -> str:
        """Stable identifier of the engine and its configuration (used for caching)."""

    @abstractmethod
    def read(self, image: np.ndarray) -> list[TextBox]:
        """Read text from a BGR image array.

        Raises:
            Exception: Backend failures propagate; callers decide how to handle them.
        """
