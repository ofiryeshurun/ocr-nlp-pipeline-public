"""On-disk cache of OCR results: one JSON file per image, per configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from alcohol_detector.ocr.base import OCRResult


class OCRCache:
    """Store and load :class:`OCRResult` objects under ``<root>/<config_id>/``.

    The config id is a hash of the full OCR configuration, so results from
    different engines, variants or settings are never mixed.
    """

    def __init__(self, root: Path, config: dict[str, Any]) -> None:
        """Args:
        root: Cache root folder.
        config: Everything that affects OCR output (engine id, variant, options).
        """
        self.config: dict[str, Any] = config
        self.config_id: str = self.hash_config(config)
        self.folder: Path = Path(root) / self.config_id

    @staticmethod
    def hash_config(config: dict[str, Any]) -> str:
        """Short stable hash of a JSON-serialisable config."""
        payload = json.dumps(config, sort_keys=True).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:12]

    def path_for(self, file_name: str) -> Path:
        """JSON path for an image."""
        return self.folder / f"{file_name}.json"

    def get(self, file_name: str) -> OCRResult | None:
        """Return the cached result, or None if missing or unreadable."""
        path = self.path_for(file_name)
        if not path.is_file():
            return None
        try:
            return OCRResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, KeyError):
            return None

    def put(self, result: OCRResult) -> None:
        """Write a result (and the config, once) to disk."""
        self.folder.mkdir(parents=True, exist_ok=True)
        config_path = self.folder / "_config.json"
        if not config_path.exists():
            config_path.write_text(json.dumps(self.config, indent=2, sort_keys=True), encoding="utf-8")
        self.path_for(result.file_name).write_text(json.dumps(result.to_dict()), encoding="utf-8")
