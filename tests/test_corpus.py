"""Tests for loading cached OCR results as a corpus."""

from pathlib import Path

import pytest

from alcohol_detector.data import SplitAssignment, SplitManifest
from alcohol_detector.ocr import OCRCache, OCRResult, TextBox
from alcohol_detector.ocr.batch import tta_cache_config
from alcohol_detector.ocr.corpus import CONFIG_A, OCRCorpus
from alcohol_detector.ocr.preprocess import make_preprocessor


def _setup(tmp_path: Path, cached: list[str]) -> tuple[Path, Path]:
    manifest = tmp_path / "splits.csv"
    SplitManifest.write(
        [SplitAssignment("1a.jpg", "alcohol", "train"), SplitAssignment("2na.jpg", "non_alcohol", "train"),
         SplitAssignment("3a.jpg", "alcohol", "test")],
        manifest,
    )
    cache = OCRCache(tmp_path / "cache", tta_cache_config("eng", CONFIG_A))
    for f in cached:
        cache.put(OCRResult(f, [TextBox("vodka", 0.9)]))
    return manifest, tmp_path / "cache"


def test_load_returns_only_requested_splits(tmp_path: Path) -> None:
    manifest, root = _setup(tmp_path, ["1a.jpg", "2na.jpg"])
    docs = OCRCorpus(manifest, root, "eng").load(("train",))
    assert [(d.file_name, d.label, d.text) for d in docs] == [("1a.jpg", "alcohol", "vodka"), ("2na.jpg", "non_alcohol", "vodka")]


def test_load_refuses_partial_corpus(tmp_path: Path) -> None:
    manifest, root = _setup(tmp_path, ["1a.jpg"])
    with pytest.raises(FileNotFoundError):
        OCRCorpus(manifest, root, "eng").load(("train",))


def test_preprocess_selects_its_own_cache(tmp_path: Path) -> None:
    manifest, root = _setup(tmp_path, ["1a.jpg", "2na.jpg"])
    # Written the way OCRBatchRunner keys it: by the preprocessor's full name.
    clahe = OCRCache(root, tta_cache_config("eng", CONFIG_A, make_preprocessor("lab_clahe").name))
    for f in ["1a.jpg", "2na.jpg"]:
        clahe.put(OCRResult(f, [TextBox("gin", 0.9)]))
    docs = OCRCorpus(manifest, root, "eng", preprocess="lab_clahe").load(("train",))
    assert [d.text for d in docs] == ["gin", "gin"]
