"""Dataset indexing and train/val/test splitting.

The label of a training image is the name of the folder it sits in. File names
are never used as a signal anywhere in the pipeline.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from alcohol_detector.seed import DEFAULT_SEED

LABELS: tuple[str, str] = ("alcohol", "non_alcohol")
SPLITS: tuple[str, str, str] = ("train", "val", "test")
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@dataclass(frozen=True)
class ImageRecord:
    """One labelled image on disk."""

    file_name: str
    label: str
    path: Path


@dataclass(frozen=True)
class SplitAssignment:
    """The split an image belongs to."""

    file_name: str
    label: str
    split: str


class LabeledImageIndex:
    """Index a directory laid out as ``<root>/<label>/<image>``."""

    def __init__(self, root: Path, labels: tuple[str, ...] = LABELS) -> None:
        """Args:
        root: Directory that contains one sub-folder per label.
        labels: The label folder names to look for.
        """
        self.root: Path = Path(root)
        self.labels: tuple[str, ...] = labels

    def scan(self) -> list[ImageRecord]:
        """Return every image under the label folders, sorted by label then name.

        Raises:
            FileNotFoundError: If the root or a label folder is missing.
            ValueError: If a file name appears under more than one label.
        """
        if not self.root.is_dir():
            raise FileNotFoundError(f"Image root not found: {self.root}")
        records: list[ImageRecord] = []
        for label in self.labels:
            folder = self.root / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Label folder not found: {folder}")
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(ImageRecord(file_name=path.name, label=label, path=path))
        names = [r.file_name for r in records]
        if len(names) != len(set(names)):
            raise ValueError("The same file name appears under more than one label folder.")
        return records


class StratifiedSplitter:
    """Assign images to train/val/test, stratified by label, with a fixed seed."""

    def __init__(
        self,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Args:
        train_frac, val_frac, test_frac: Split fractions; must sum to 1.
        seed: Random seed for reproducible assignment.
        """
        total = train_frac + val_frac + test_frac
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1, got {total}.")
        self.train_frac: float = train_frac
        self.val_frac: float = val_frac
        self.test_frac: float = test_frac
        self.seed: int = seed

    def split(self, records: list[ImageRecord]) -> list[SplitAssignment]:
        """Split records into train/val/test.

        Two stratified cuts: train vs held-out, then held-out into val vs test.
        """
        names = [r.file_name for r in records]
        labels = [r.label for r in records]
        held_frac = self.val_frac + self.test_frac
        train_n, held_n, _, held_y = train_test_split(
            names, labels, test_size=held_frac, stratify=labels, random_state=self.seed
        )
        val_n, test_n = train_test_split(
            held_n, test_size=self.test_frac / held_frac, stratify=held_y, random_state=self.seed
        )
        split_of = {**{n: "train" for n in train_n}, **{n: "val" for n in val_n}, **{n: "test" for n in test_n}}
        return [SplitAssignment(r.file_name, r.label, split_of[r.file_name]) for r in records]


class SplitManifest:
    """Read and write the ``file_name,label,split`` CSV."""

    FIELDS: tuple[str, str, str] = ("file_name", "label", "split")

    @classmethod
    def write(cls, assignments: list[SplitAssignment], path: Path) -> None:
        """Write assignments to CSV, creating parent folders if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cls.FIELDS)
            for a in assignments:
                writer.writerow((a.file_name, a.label, a.split))

    @classmethod
    def read(cls, path: Path) -> list[SplitAssignment]:
        """Read assignments from CSV.

        Raises:
            FileNotFoundError: If the manifest does not exist.
            ValueError: If the header or a split name is invalid.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {path}")
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != cls.FIELDS:
                raise ValueError(f"Unexpected header in {path}: {reader.fieldnames}")
            rows = [SplitAssignment(r["file_name"], r["label"], r["split"]) for r in reader]
        bad = {r.split for r in rows} - set(SPLITS)
        if bad:
            raise ValueError(f"Unknown split names in {path}: {sorted(bad)}")
        return rows


class SplitCopier:
    """Copy the images of one split into a flat folder, e.g. to run the inference CLI on test.

    Files are copied, never opened or decoded, so nothing is displayed.
    """

    def __init__(self, manifest: Path, image_root: Path, labels: tuple[str, ...] = LABELS) -> None:
        """Args:
        manifest: The ``file_name,label,split`` CSV.
        image_root: Directory laid out as ``<root>/<label>/<image>``.
        labels: The label folder names.
        """
        self.manifest: Path = Path(manifest)
        self.image_root: Path = Path(image_root)
        self.labels: tuple[str, ...] = labels

    def copy(self, split: str, dest: Path) -> int:
        """Copy every image of ``split`` into ``dest`` (created if needed). Returns the number copied.

        Raises:
            ValueError: If the split is unknown or empty, or ``dest`` already holds files.
            FileNotFoundError: If an image listed in the manifest is not on disk.
        """
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; choose from {SPLITS}")
        dest = Path(dest)
        if dest.exists() and any(dest.iterdir()):
            raise ValueError(f"{dest} is not empty; copy into an empty folder so no other image is mixed in.")
        rows = [r for r in SplitManifest.read(self.manifest) if r.split == split]
        if not rows:
            raise ValueError(f"No images in split {split!r} of {self.manifest}.")
        sources = [self.image_root / r.label / r.file_name for r in rows]
        missing = [str(p) for p in sources if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images of split {split!r} not found, e.g. {missing[:3]}")
        dest.mkdir(parents=True, exist_ok=True)
        for source in sources:
            shutil.copy2(source, dest / source.name)
        return len(sources)
