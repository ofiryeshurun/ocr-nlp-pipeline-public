"""Tests for the inference pipeline, the CSV files, split scoring and split copying.

A fake OCR engine stands in for PaddleOCR, so no OCR model is loaded.
"""

import csv
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import sklearn

from alcohol_detector.config import ModelConfig, shipped_ocr_setup
from alcohol_detector.data import SplitAssignment, SplitCopier, SplitManifest
from alcohol_detector.inference import (
    AlcoholImagePipeline,
    ConfusionMatrixPlot,
    DetailsCSV,
    ImageDirectory,
    ImagePrediction,
    ResultsCSV,
    SplitScorer,
)
from alcohol_detector.models import ALCOHOL, LABELS, NON_ALCOHOL, CharNgramClassifier, ModelCard, ModelEvaluator
from alcohol_detector.ocr import OCREngine, OrientationTTA, TextBox
from alcohol_detector.ocr.batch import tta_cache_config
from alcohol_detector.ocr.corpus import CONFIG_A
from alcohol_detector.ocr.preprocess import make_preprocessor

REPO = Path(__file__).resolve().parents[1]
CONFIG = ModelConfig()
TRAIN_TEXTS = ["try our vodka tonight", "ice cold heineken", "fine whiskey on the rocks", "gin and tonic hour",
               "fresh orange juice", "hot coffee to go", "green tea latte", "sparkling water deal"]
TRAIN_LABELS = [ALCOHOL] * 4 + [NON_ALCOHOL] * 4
SETUP = shipped_ocr_setup()

#: The fake engine reads an image's text from its height, so each test image says something different.
TEXT_BY_HEIGHT = {40: "ice cold vodka tonight", 50: "fresh orange juice deal", 70: ""}
FAILING_HEIGHT = 60


class FakeEngine(OCREngine):
    """Returns the text of ``TEXT_BY_HEIGHT`` for an image's height; raises for ``FAILING_HEIGHT``."""

    def __init__(self, engine_id: str = SETUP.engine_id) -> None:
        self._engine_id = engine_id

    @property
    def engine_id(self) -> str:
        return self._engine_id

    def read(self, image: np.ndarray) -> list[TextBox]:
        height = image.shape[0]
        if height == FAILING_HEIGHT:
            raise RuntimeError("engine crashed")
        text = TEXT_BY_HEIGHT.get(height, "")
        return [TextBox(text, 0.99)] if text else []


def _classifier() -> CharNgramClassifier:
    return CharNgramClassifier(CONFIG.normalizer(), CONFIG.seed).fit(TRAIN_TEXTS, TRAIN_LABELS)


def _pipeline(**kwargs: object) -> AlcoholImagePipeline:
    return AlcoholImagePipeline(OrientationTTA(FakeEngine(), CONFIG_A), _classifier(), **kwargs)  # type: ignore[arg-type]


def _write_image(path: Path, height: int) -> None:
    ok, data = cv2.imencode(path.suffix, np.full((height, 80, 3), 200, dtype=np.uint8))
    assert ok
    path.write_bytes(data.tobytes())


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """A folder mixing readable images, a PNG, an upper-case extension, broken files and non-images."""
    root = tmp_path / "input"
    root.mkdir()
    _write_image(root / "b_vodka.jpg", 40)
    _write_image(root / "c_juice.PNG", 50)
    _write_image(root / "d_crash.jpeg", FAILING_HEIGHT)
    _write_image(root / "e_blank.bmp", 70)
    (root / "a_corrupt.jpg").write_bytes(b"not an image")
    (root / "f_empty.webp").write_bytes(b"")
    (root / "notes.txt").write_text("vodka", encoding="utf-8")
    (root / "sub").mkdir()
    _write_image(root / "sub" / "nested.jpg", 40)
    return root


def _card(clf: CharNgramClassifier, **overrides: object) -> ModelCard:
    values: dict[str, object] = dict(
        arm="v1_char_ngram", threshold=0.5, c=1.0, ocr_setup=SETUP.name,
        ocr=tta_cache_config(SETUP.engine_id, CONFIG_A, make_preprocessor(SETUP.preprocess).name),
        leet=True, seed=1998, train_splits=["train", "val"], n_train=8, n_train_alcohol=4,
        no_text_score=float(clf.scores([""])[0]), no_text_label=ALCOHOL,
        sklearn_version=sklearn.__version__, created="2026-10-03", git_commit="abc1234",
    )
    values.update(overrides)
    return ModelCard(**values)  # type: ignore[arg-type]


# --- image folder -----------------------------------------------------------------

def test_image_directory_lists_images_only_in_name_order(image_dir: Path) -> None:
    images, skipped = ImageDirectory(image_dir).scan()
    assert [p.name for p in images] == ["a_corrupt.jpg", "b_vodka.jpg", "c_juice.PNG", "d_crash.jpeg",
                                        "e_blank.bmp", "f_empty.webp"]
    assert sorted(p.name for p in skipped) == ["notes.txt", "sub"]


def test_image_directory_rejects_missing_or_non_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ImageDirectory(tmp_path / "nope")
    (tmp_path / "file.jpg").write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        ImageDirectory(tmp_path / "file.jpg")


# --- pipeline ---------------------------------------------------------------------

def test_pipeline_gives_one_valid_row_per_image_and_never_stops(image_dir: Path) -> None:
    preds = {p.file_name: p for p in _pipeline().predict_directory(image_dir)}
    assert len(preds) == 6
    assert {p.label for p in preds.values()} <= set(LABELS)
    assert preds["b_vodka.jpg"].label == ALCOHOL and preds["b_vodka.jpg"].ok
    assert preds["b_vodka.jpg"].evidence and preds["b_vodka.jpg"].ocr_text == TEXT_BY_HEIGHT[40]
    assert preds["c_juice.PNG"].label == NON_ALCOHOL and preds["c_juice.PNG"].score < 0.5


def test_unreadable_and_failed_images_get_the_fallback_label_and_an_error(image_dir: Path) -> None:
    preds = {p.file_name: p for p in _pipeline().predict_directory(image_dir)}
    for name in ("a_corrupt.jpg", "f_empty.webp", "d_crash.jpeg"):
        assert preds[name].label == ALCOHOL and preds[name].score is None and not preds[name].ok
    assert "OCR failed" in (preds["d_crash.jpeg"].error or "")
    fallback = {p.file_name: p for p in _pipeline(unreadable_label=NON_ALCOHOL).predict_directory(image_dir)}
    assert fallback["a_corrupt.jpg"].label == NON_ALCOHOL
    with pytest.raises(ValueError):
        _pipeline(unreadable_label="maybe")


def test_image_without_text_is_scored_by_the_classifier_with_empty_evidence(image_dir: Path) -> None:
    pipeline = _pipeline()
    blank = pipeline.predict_file(image_dir / "e_blank.bmp")
    assert blank.ok and blank.ocr_text == "" and blank.evidence == ()
    assert blank.score == pytest.approx(float(pipeline.classifier.scores([""])[0]))


def test_classifier_errors_are_isolated_per_image(image_dir: Path) -> None:
    pipeline = _pipeline()

    def broken(texts: object) -> object:
        raise RuntimeError("classifier bug")

    pipeline.classifier.predict = broken  # type: ignore[method-assign]
    pred = pipeline.predict_file(image_dir / "b_vodka.jpg")
    assert pred.label == ALCOHOL and "classifier bug" in (pred.error or "")


# --- rebuilding OCR from the model card ---------------------------------------------

def test_build_ocr_matches_the_card() -> None:
    tta = AlcoholImagePipeline.build_ocr(_card(_classifier()), lambda setup: FakeEngine(setup.engine_id))
    assert tta.selector.trigger_threshold == CONFIG_A.trigger_threshold
    assert tta.preprocessor.name == make_preprocessor(SETUP.preprocess).name


@pytest.mark.parametrize("change", [
    {"ocr_setup": "v6_small"},  # card's OCR entry says LAB CLAHE, this setup has none
    {"ocr_setup": "unknown"},
])
def test_build_ocr_rejects_cards_it_cannot_rebuild(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AlcoholImagePipeline.build_ocr(_card(_classifier(), **change), lambda setup: FakeEngine(setup.engine_id))


def test_build_ocr_takes_the_retry_rule_from_the_card() -> None:
    card = _card(_classifier())
    tta = AlcoholImagePipeline.build_ocr(replace(card, ocr={**card.ocr, "threshold": 0.8}),
                                         lambda setup: FakeEngine(setup.engine_id))
    assert tta.selector.trigger_threshold == 0.8  # the card is the record of the training OCR


def test_build_ocr_rejects_an_incomplete_or_unknown_ocr_entry() -> None:
    card = _card(_classifier())
    for ocr in ({k: v for k, v in card.ocr.items() if k != "combine"},  # missing field
                {**card.ocr, "engine": card.ocr["engine"] + "|detsize=max960"},  # engine the setup doesn't build
                {**card.ocr, "preprocess": "lab_clahe(clip=4.0,tile=8)"}):  # other preprocessing parameters
        with pytest.raises(ValueError):
            AlcoholImagePipeline.build_ocr(replace(card, ocr=ocr), lambda setup: FakeEngine(setup.engine_id))


def test_build_ocr_rejects_an_engine_other_than_the_setup() -> None:
    with pytest.raises(ValueError):
        AlcoholImagePipeline.build_ocr(_card(_classifier()), lambda setup: FakeEngine("other-engine"))


@pytest.mark.skipif(not (REPO / "models" / "model_card.json").is_file(),
                    reason="no trained model; build one with scripts/train_final.py")
def test_shipped_model_bundle_loads_with_its_own_ocr() -> None:
    pipeline = AlcoholImagePipeline.from_model_dir(REPO / "models", lambda setup: FakeEngine(setup.engine_id))
    assert pipeline.classifier.threshold == 0.5
    assert pipeline.ocr.engine.engine_id == SETUP.engine_id


# --- CSV files --------------------------------------------------------------------

PREDS = [
    ImagePrediction("1a.jpg", ALCOHOL, 0.9, ("'vodk' +1.20", "'odka' +0.80"), "vodka, now", "original"),
    ImagePrediction("2b.png", NON_ALCOHOL, 0.1, (), "juice", "mirror"),
    ImagePrediction("3c.jpg", ALCOHOL, None, (), "", "", "Cannot decode image"),
]


def test_results_csv_matches_the_required_format(tmp_path: Path) -> None:
    path = tmp_path / "out" / "results.csv"
    ResultsCSV.write(PREDS, path)
    assert path.read_bytes().startswith(b"file_name,prediction\r\n1a.jpg,alcohol\r\n")
    assert ResultsCSV.read(path) == {"1a.jpg": ALCOHOL, "2b.png": NON_ALCOHOL, "3c.jpg": ALCOHOL}


def test_results_csv_rejects_bad_labels_duplicates_and_headers(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ResultsCSV.write([replace(PREDS[0], label="beer")], tmp_path / "r.csv")
    with pytest.raises(ValueError):
        ResultsCSV.write([PREDS[0], PREDS[0]], tmp_path / "r.csv")
    (tmp_path / "bad.csv").write_text("name,label\n1a.jpg,alcohol\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ResultsCSV.read(tmp_path / "bad.csv")
    (tmp_path / "dup.csv").write_text("file_name,prediction\n1a.jpg,alcohol\n1a.jpg,alcohol\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ResultsCSV.read(tmp_path / "dup.csv")
    with pytest.raises(FileNotFoundError):
        ResultsCSV.read(tmp_path / "missing.csv")


def test_details_csv_keeps_text_evidence_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "details.csv"
    DetailsCSV.write(PREDS, path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == DetailsCSV.FIELDS
    assert rows[0]["ocr_text"] == "vodka, now" and rows[0]["evidence"] == "'vodk' +1.20 | 'odka' +0.80"
    assert float(rows[0]["score"]) == 0.9
    assert rows[2]["score"] == "" and rows[2]["error"] == "Cannot decode image" and rows[2]["prediction"] == ALCOHOL


# --- scoring against a split ------------------------------------------------------

@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "splits.csv"
    SplitManifest.write([
        SplitAssignment("1a.jpg", ALCOHOL, "test"), SplitAssignment("2na.jpg", NON_ALCOHOL, "test"),
        SplitAssignment("3a.jpg", ALCOHOL, "test"), SplitAssignment("4na.jpg", NON_ALCOHOL, "test"),
        SplitAssignment("5a.jpg", ALCOHOL, "train"),
    ], path)
    return path


def test_split_scorer_uses_manifest_labels(manifest: Path, tmp_path: Path) -> None:
    preds = {"1a.jpg": ALCOHOL, "2na.jpg": ALCOHOL, "3a.jpg": NON_ALCOHOL, "4na.jpg": NON_ALCOHOL}
    scored = SplitScorer(manifest, ModelEvaluator(1998, n_resamples=99)).score(preds, "test")
    r = scored.result
    assert (r.n, r.tp, r.fp, r.fn, r.tn) == (4, 1, 1, 1, 1)
    assert scored.to_dict()["metrics"]["recall"]["value"] == 0.5
    ConfusionMatrixPlot.save(r, tmp_path / "fig" / "cm.png", "test")
    assert (tmp_path / "fig" / "cm.png").stat().st_size > 0


def test_split_scorer_rejects_missing_extra_or_empty(manifest: Path) -> None:
    scorer = SplitScorer(manifest, ModelEvaluator(1998, n_resamples=99))
    full = {"1a.jpg": ALCOHOL, "2na.jpg": ALCOHOL, "3a.jpg": NON_ALCOHOL, "4na.jpg": NON_ALCOHOL}
    with pytest.raises(ValueError, match="1 missing"):
        scorer.score({k: v for k, v in full.items() if k != "1a.jpg"}, "test")
    with pytest.raises(ValueError, match="1 extra"):
        scorer.score({**full, "5a.jpg": ALCOHOL}, "test")
    with pytest.raises(ValueError):
        scorer.score(full, "val")


# --- copying a split ---------------------------------------------------------------

def test_split_copier_copies_one_split_flat(manifest: Path, tmp_path: Path) -> None:
    root = tmp_path / "images"
    for name, label in [("1a.jpg", ALCOHOL), ("2na.jpg", NON_ALCOHOL), ("3a.jpg", ALCOHOL),
                        ("4na.jpg", NON_ALCOHOL), ("5a.jpg", ALCOHOL)]:
        (root / label).mkdir(parents=True, exist_ok=True)
        (root / label / name).write_bytes(name.encode())
    copier = SplitCopier(manifest, root)
    assert copier.copy("test", tmp_path / "flat") == 4
    assert sorted(p.name for p in (tmp_path / "flat").iterdir()) == ["1a.jpg", "2na.jpg", "3a.jpg", "4na.jpg"]
    with pytest.raises(ValueError):  # destination not empty
        copier.copy("test", tmp_path / "flat")
    (root / ALCOHOL / "5a.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        copier.copy("train", tmp_path / "flat_train")
    with pytest.raises(ValueError):
        copier.copy("holdout", tmp_path / "x")


# --- the inference CLI, end to end with the fake engine ----------------------------

def test_inference_cli_writes_results_and_details(image_dir: Path, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("inference_cli", REPO / "scripts" / "inference.py")
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli.AlcoholImagePipeline, "from_model_dir", classmethod(lambda cls, model_dir: _pipeline()))
    out, details = tmp_path / "results.csv", tmp_path / "details.csv"
    monkeypatch.setattr(sys, "argv", ["inference.py", "--input-dir", str(image_dir), "--output", str(out),
                                      "--details", str(details)])
    cli.main()
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["file_name", "prediction"]
    assert [r[0] for r in rows[1:]] == ["a_corrupt.jpg", "b_vodka.jpg", "c_juice.PNG", "d_crash.jpeg",
                                        "e_blank.bmp", "f_empty.webp"]
    assert {r[1] for r in rows[1:]} <= set(LABELS)
    assert details.is_file()
