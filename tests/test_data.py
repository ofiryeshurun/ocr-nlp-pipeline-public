"""Tests for dataset indexing and splitting."""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from alcohol_detector.data import LabeledImageIndex, SplitManifest, StratifiedSplitter

REPO = Path(__file__).resolve().parents[1]


def _make_tree(root: Path, n_per_label: int = 100) -> None:
    for label, suffix in (("alcohol", "a"), ("non_alcohol", "na")):
        folder = root / label
        folder.mkdir(parents=True)
        for i in range(n_per_label):
            (folder / f"{i:04d}{suffix}.jpg").write_bytes(b"x")
        (folder / "notes.txt").write_text("ignored")


def test_scan_finds_images_only(tmp_path: Path) -> None:
    _make_tree(tmp_path, 5)
    records = LabeledImageIndex(tmp_path).scan()
    assert len(records) == 10
    assert {r.label for r in records} == {"alcohol", "non_alcohol"}


def test_scan_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LabeledImageIndex(tmp_path / "missing").scan()


def test_split_sizes_stratified_and_disjoint(tmp_path: Path) -> None:
    _make_tree(tmp_path, 100)
    records = LabeledImageIndex(tmp_path).scan()
    assignments = StratifiedSplitter().split(records)
    sizes = Counter(a.split for a in assignments)
    assert sizes == {"train": 140, "val": 30, "test": 30}
    per_label = Counter((a.split, a.label) for a in assignments)
    assert per_label[("val", "alcohol")] == per_label[("val", "non_alcohol")] == 15
    assert len({a.file_name for a in assignments}) == len(assignments)


def test_split_is_reproducible(tmp_path: Path) -> None:
    _make_tree(tmp_path, 50)
    records = LabeledImageIndex(tmp_path).scan()
    assert StratifiedSplitter(seed=1998).split(records) == StratifiedSplitter(seed=1998).split(records)


def test_manifest_round_trip(tmp_path: Path) -> None:
    _make_tree(tmp_path / "imgs", 10)
    assignments = StratifiedSplitter().split(LabeledImageIndex(tmp_path / "imgs").scan())
    out = tmp_path / "data" / "splits.csv"
    SplitManifest.write(assignments, out)
    assert SplitManifest.read(out) == assignments


def test_make_split_cli_writes_once_and_never_overwrites(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("make_split_cli", REPO / "scripts" / "make_split.py")
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    _make_tree(tmp_path / "imgs", 20)
    out = tmp_path / "splits.csv"
    argv = ["--image-root", str(tmp_path / "imgs"), "--output", str(out)]
    assert cli.main(argv) == 0
    first = out.read_bytes()
    assert Counter(a.split for a in SplitManifest.read(out)) == {"train": 28, "val": 6, "test": 6}
    assert cli.main(argv) == 1  # an existing manifest is never overwritten
    assert out.read_bytes() == first


@pytest.mark.skipif(not (REPO / "images").is_dir() or not (REPO / "data" / "splits.csv").is_file(),
                    reason="the dataset and its split are not versioned")
def test_committed_split_matches_make_split() -> None:
    fresh = StratifiedSplitter().split(LabeledImageIndex(REPO / "images").scan())
    assert SplitManifest.read(REPO / "data" / "splits.csv") == fresh
