"""Tests for the OCR layer (a fake engine stands in for PaddleOCR)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from alcohol_detector.ocr import (
    ImageLoader,
    ImageLoadError,
    OCRCache,
    OCREngine,
    OCREvaluator,
    OCRResult,
    OrientationTTA,
    QualityScorer,
    TextBox,
    Variant,
    VariantSelector,
    squash,
)


class FakeEngine(OCREngine):
    """Returns a fixed read for unmodified images and another for transformed ones."""

    def __init__(self, original: list[TextBox], other: list[TextBox], fail: bool = False) -> None:
        self.original, self.other, self.fail = original, other, fail
        self.calls = 0
        self._shape: tuple[int, ...] | None = None

    @property
    def engine_id(self) -> str:
        return "fake"

    def read(self, image: np.ndarray) -> list[TextBox]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("backend exploded")
        if self._shape is None:
            self._shape = image.shape
            return self.original
        return self.other


GOOD = [TextBox("heineken", 0.99)]
GARBAGE = [TextBox("uaxauiau", 0.55)]


def test_variant_mirror_keeps_shape_and_rotation_expands() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    assert Variant("m", mirror=True).apply(img).shape == img.shape
    rotated = Variant("r", angle=30).apply(img)
    assert rotated.shape[0] > 100 and rotated.shape[1] > 200


def test_loader_rejects_missing_and_corrupt(tmp_path: Path) -> None:
    loader = ImageLoader()
    with pytest.raises(ImageLoadError):
        loader.load(tmp_path / "nope.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError):
        loader.load(bad)
    good = tmp_path / "ok.png"
    cv2.imwrite(str(good), np.full((10, 10, 3), 255, dtype=np.uint8))
    assert loader.load(good).shape == (10, 10, 3)


def test_result_and_cache_round_trip(tmp_path: Path) -> None:
    result = OCRResult("x.jpg", [TextBox("gin", 0.9, [[0, 0], [1, 1]])], variant="mirror", variants_tried=["original", "mirror"])
    assert OCRResult.from_dict(result.to_dict()) == result
    cache_a = OCRCache(tmp_path, {"engine": "a"})
    cache_b = OCRCache(tmp_path, {"engine": "b"})
    cache_a.put(result)
    assert cache_a.get("x.jpg") == result
    assert cache_b.get("x.jpg") is None


def test_quality_scorer_weights_by_length() -> None:
    boxes = [TextBox("B", 1.0), TextBox("garbage text", 0.5)]
    assert QualityScorer.mean_confidence(boxes) == pytest.approx((1.0 + 0.5 * 12) / 13)
    assert QualityScorer.mean_confidence([]) == 0.0


def test_selector_best_and_union() -> None:
    reads = {"original": GARBAGE, "mirror": GOOD}
    assert VariantSelector(combine="best").select(reads) == ("mirror", GOOD)
    name, merged = VariantSelector(combine="union").select(reads)
    assert name == "union" and {b.text for b in merged} == {"uaxauiau", "heineken"}


def test_conditional_skips_retry_when_original_is_confident() -> None:
    engine = FakeEngine(original=GOOD, other=GARBAGE)
    tta = OrientationTTA(engine, VariantSelector(strategy="conditional", trigger_threshold=0.9))
    result = tta.run(np.zeros((50, 50, 3), dtype=np.uint8), "a.jpg")
    assert engine.calls == 1 and result.variants_tried == ["original"] and result.text == "heineken"


def test_conditional_retries_weak_read_and_picks_better_variant() -> None:
    engine = FakeEngine(original=GARBAGE, other=GOOD)
    tta = OrientationTTA(engine, VariantSelector(strategy="conditional", trigger_threshold=0.9))
    result = tta.run(np.zeros((50, 50, 3), dtype=np.uint8), "a.jpg")
    assert engine.calls > 1 and result.text == "heineken"


def test_engine_failure_is_captured_not_raised() -> None:
    tta = OrientationTTA(FakeEngine(GOOD, GOOD, fail=True), VariantSelector())
    result = tta.run(np.zeros((50, 50, 3), dtype=np.uint8), "a.jpg")
    assert not result.ok and "backend exploded" in (result.error or "")


def test_evaluator_ignores_spacing_and_background() -> None:
    ev = OCREvaluator()
    assert squash("H eine K-en!") == "heineken"
    assert ev.key_term_hit("B 886 h eine k en", "heineken")
    assert not ev.key_term_hit("cannabis", "heineken")
    assert ev.overlay_cer("STOP sign Try event imported lager premium", "Try event imported lager premium") == 0.0
    assert ev.overlay_cer("", "gin") == 1.0


def test_batch_runner_survives_corrupt_image_and_uses_cache(tmp_path: Path) -> None:
    from alcohol_detector.ocr.batch import ImageJob, OCRBatchRunner

    good, bad = tmp_path / "good.png", tmp_path / "bad.jpg"
    cv2.imwrite(str(good), np.full((20, 20, 3), 255, dtype=np.uint8))
    bad.write_bytes(b"\x00\x01")
    engine = FakeEngine(original=GOOD, other=GOOD)
    runner = OCRBatchRunner(engine, tmp_path / "cache")
    jobs = [ImageJob("bad.jpg", bad), ImageJob("good.png", good)]
    results = runner.run_tta(jobs, VariantSelector(strategy="conditional", trigger_threshold=0.5))
    assert [r.ok for r in results] == [False, True]
    calls = engine.calls
    runner.run_tta(jobs, VariantSelector(strategy="conditional", trigger_threshold=0.5))
    assert engine.calls == calls  # second run served from cache


def test_gray_clahe_returns_three_channel_image_of_same_size() -> None:
    from alcohol_detector.ocr.preprocess import GrayCLAHEPreprocessor, make_preprocessor

    img = np.random.default_rng(1998).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    out = GrayCLAHEPreprocessor().apply(img)
    assert out.shape == img.shape and (out[..., 0] == out[..., 1]).all()
    assert make_preprocessor("none").apply(img) is img
    with pytest.raises(ValueError):
        make_preprocessor("sharpen")


def test_preprocessor_changes_cache_key_but_none_keeps_old_key(tmp_path: Path) -> None:
    from alcohol_detector.ocr.batch import OCRBatchRunner
    from alcohol_detector.ocr.preprocess import GrayCLAHEPreprocessor

    engine = FakeEngine(GOOD, GOOD)
    raw = OCRBatchRunner(engine, tmp_path).variant_cache(Variant("original"))
    clahe = OCRBatchRunner(engine, tmp_path, preprocessor=GrayCLAHEPreprocessor()).variant_cache(Variant("original"))
    legacy = OCRCache(tmp_path, {"engine": "fake", "mode": "single", "variant": "original"})
    assert raw.config_id == legacy.config_id != clahe.config_id


def test_lab_clahe_keeps_colour_and_shape() -> None:
    from alcohol_detector.ocr.preprocess import make_preprocessor

    img = np.zeros((30, 30, 3), dtype=np.uint8)
    img[:, :, 2] = 200  # pure red image
    out = make_preprocessor("lab_clahe").apply(img)
    assert out.shape == img.shape
    assert out[..., 2].mean() > out[..., 0].mean() + 50  # still red, not gray


def test_upscale_and_autocontrast() -> None:
    from alcohol_detector.ocr.preprocess import make_preprocessor

    img = np.full((20, 30, 3), 100, dtype=np.uint8)
    img[5:10, 5:10] = 140  # faint patch
    assert make_preprocessor("upscale2").apply(img).shape == (40, 60, 3)
    stretched = make_preprocessor("autocontrast").apply(img)
    assert stretched.shape == img.shape and int(stretched.max()) - int(stretched.min()) > 200
    assert make_preprocessor("lab_clahe_strong").name.startswith("lab_clahe(clip=4.0")


def test_bilateral_clahe_keeps_shape_and_colour() -> None:
    from alcohol_detector.ocr.preprocess import make_preprocessor

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 2] = 180
    out = make_preprocessor("bilateral_clahe").apply(img)
    assert out.shape == img.shape and out[..., 2].mean() > out[..., 0].mean() + 50


def test_tophat_and_channel_views() -> None:
    from alcohol_detector.ocr.preprocess import ChannelViewPreprocessor, make_preprocessor

    img = np.full((40, 40, 3), 120, dtype=np.uint8)
    img[18:22, 5:35] = 150  # faint thin bright stroke
    out = make_preprocessor("tophat").apply(img)
    assert out.shape == img.shape
    assert int(out[20, 20].mean()) - int(out[5, 5].mean()) > 150 - 120  # stroke contrast increased
    for key in ("saturation", "min_channel"):
        view = make_preprocessor(key).apply(img)
        assert view.shape == img.shape and (view[..., 0] == view[..., 2]).all()
    with pytest.raises(ValueError):
        ChannelViewPreprocessor("hue")


def test_experiment_log_upsert_and_round_trip(tmp_path: Path) -> None:
    from alcohol_detector.ocr.experiments import ExperimentLog, OCRExperiment

    exp = OCRExperiment("v6|raw|A", "v6", "none", "none", "original+mirror", "conditional@0.9", "best",
                        "sample", 60, 0.867, 0.923, 0.087, 3.98, 1.35)
    log = ExperimentLog(tmp_path / "ocr_experiments.csv")
    log.upsert(exp)
    log.upsert(OCRExperiment(**{**exp.__dict__, "key_term_hit": 0.9}))  # same id replaces
    log.save()
    reloaded = ExperimentLog(tmp_path / "ocr_experiments.csv").experiments
    assert len(reloaded) == 1 and reloaded[0].key_term_hit == 0.9 and reloaded[0].n_images == 60


def test_hybrid_preprocessor_and_engine_size_id() -> None:
    from alcohol_detector.ocr.paddle import PaddleOCREngine
    from alcohol_detector.ocr.preprocess import make_preprocessor

    img = np.random.default_rng(1998).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    out = make_preprocessor("hybrid").apply(img)
    assert out.shape == img.shape and (out[..., 0] == out[..., 1]).all()
    native = PaddleOCREngine.make_engine_id("d", "r")
    assert "detsize" not in native  # native-size ids unchanged, so existing caches stay valid
    assert PaddleOCREngine.make_engine_id("d", "r", True, 480, "max").endswith("|detsize=max480")
