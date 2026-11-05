"""Tests for the classifier arms, metrics, run log, comparison runner, C tuning and ranking metrics."""

from collections.abc import Sequence
from pathlib import Path

import json
import warnings

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score

from alcohol_detector.config import (
    SHIPPED_ARM,
    SHIPPED_C,
    SHIPPED_THRESHOLD,
    ModelConfig,
    OCRSetup,
    shipped_ocr_setup,
)
from alcohol_detector.data import SplitAssignment, SplitManifest
from alcohol_detector.lexicon import AlcoholLexicon, EditBudget, LexiconTerm
from alcohol_detector.models import (
    ALCOHOL,
    NON_ALCOHOL,
    CharNgramClassifier,
    EmbeddingClassifier,
    LexiconClassifier,
    ModelBundle,
    ModelCard,
    ModelEvaluator,
    ModelRun,
    ModelRunLog,
    OperatingPointTable,
    RankingEvaluator,
    SentenceTransformerEncoder,
    TextClassifier,
    TextEncoder,
    TunedC,
    to_binary,
)
from alcohol_detector.models.comparison import ArmFactory, ModelComparison
from alcohol_detector.models.ranking import roc_auc
from alcohol_detector.ocr import OCRCache, OCRResult, TextBox
from alcohol_detector.ocr.batch import tta_cache_config
from alcohol_detector.ocr.corpus import CONFIG_A, OCRDocument
from alcohol_detector.ocr.stats import OCRRunStats

CONFIG = ModelConfig()
LEXICON = AlcoholLexicon([LexiconTerm(t, c) for t, c in [("gin", "spirits"), ("vodka", "spirits"),
                                                         ("heineken", "beer_brands"), ("whiskey", "spirits")]])
TRAIN_TEXTS = ["try our vodka tonight", "ice cold heineken", "fine whiskey on the rocks", "gin and tonic hour",
               "fresh orange juice", "hot coffee to go", "green tea latte", "sparkling water deal"]
TRAIN_LABELS = [ALCOHOL] * 4 + [NON_ALCOHOL] * 4
CV_ALCOHOL = ["try our vodka tonight", "ice cold heineken", "fine whiskey on the rocks", "gin and tonic hour",
              "vodka shots", "heineken deal", "whiskey tasting", "gin bar", "cold vodka", "heineken time"]
CV_OTHER = ["fresh orange juice", "hot coffee to go", "green tea latte", "sparkling water deal", "iced tea",
            "lemon soda", "matcha bowl", "apple juice", "espresso shot", "berry smoothie"]
CV_TEXTS = CV_ALCOHOL + CV_OTHER
CV_LABELS = [ALCOHOL] * 10 + [NON_ALCOHOL] * 10


class FakeEncoder(TextEncoder):
    """Deterministic bag-of-letters vectors: enough to test the v2 plumbing without torch."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(texts), 26), dtype=np.float32)
        for i, text in enumerate(texts):
            for ch in text:
                if "a" <= ch <= "z":
                    rows[i, ord(ch) - 97] += 1
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.where(norms > 0, norms, 1)


# --- config and labels ------------------------------------------------------------

def test_config_holds_the_locked_choices() -> None:
    assert CONFIG.budget == EditBudget(5, 7)
    assert CONFIG.leet and CONFIG.seed == 1998
    assert CONFIG.normalizer().clean("V0DK4!") == "vodka"


def test_shipped_setup_is_v1_on_small_lab_clahe() -> None:
    setup = shipped_ocr_setup()
    assert (setup.det_model, setup.preprocess) == ("PP-OCRv6_small_det", "lab_clahe")
    assert SHIPPED_ARM == CharNgramClassifier.name
    assert (SHIPPED_THRESHOLD, SHIPPED_C) == (0.5, 1.0)


def test_to_binary_rejects_unknown_labels() -> None:
    assert to_binary([ALCOHOL, NON_ALCOHOL]).tolist() == [1, 0]
    with pytest.raises(ValueError, match="Unknown"):
        to_binary(["beer"])


# --- v0 ---------------------------------------------------------------------------

def test_lexicon_classifier_flags_any_allowed_match() -> None:
    clf = LexiconClassifier(LEXICON, CONFIG.budget, CONFIG.normalizer()).fit([], [])
    # 7-letter "whiskey" with 2 edits is the loosest allowed match.
    assert clf.threshold == pytest.approx(1 - 2 / 7)
    preds = clf.predict(["Grab a v0dk4 now", "fix the engine", "whisky bar", ""])
    assert [p.label for p in preds] == [ALCOHOL, NON_ALCOHOL, ALCOHOL, NON_ALCOHOL]
    assert preds[0].score == 1.0 and preds[0].evidence == ("vodka <- vodka (0)",)
    assert preds[1].score == 0.0 and preds[1].evidence == ()


# --- v1 ---------------------------------------------------------------------------

def test_char_ngram_classifier_learns_and_explains() -> None:
    clf = CharNgramClassifier(CONFIG.normalizer(), CONFIG.seed)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.scores(["vodka"])
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    scores = clf.scores(["vodka special", "orange coffee"])
    assert ((scores >= 0) & (scores <= 1)).all() and scores[0] > scores[1]
    assert clf.threshold == 0.5
    assert len(clf.explain("vodka special")) == 3
    assert clf.explain("") == ()


# --- v2 ---------------------------------------------------------------------------

def test_embedding_classifier_fits_head_and_lists_nearest_terms() -> None:
    clf = EmbeddingClassifier(FakeEncoder(), CONFIG.normalizer(), LEXICON, CONFIG.seed)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.scores(["vodka"])
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    scores = clf.scores(["vodka", "coffee"])
    assert scores.shape == (2,) and ((scores >= 0) & (scores <= 1)).all()
    assert clf.explain("v0dka")[0].startswith("vodka ")  # leet decoded, then nearest term
    assert clf.explain("   ") == ()
    assert clf.scores([]).shape == (0,)


def test_sentence_transformer_encoder_caches_per_text() -> None:
    calls: list[list[str]] = []

    class FakeModel:
        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            calls.append(list(texts))
            return np.ones((len(texts), 4))

    enc = SentenceTransformerEncoder("unused", "unused")
    enc._model = FakeModel()
    assert enc.encode(["a", "b", "a"]).shape == (3, 4)
    enc.encode(["b", "c"])
    assert calls == [["a", "b"], ["c"]]


# --- metrics ----------------------------------------------------------------------

def test_evaluator_counts_and_intervals() -> None:
    y_true = [ALCOHOL] * 4 + [NON_ALCOHOL] * 4
    y_pred = [ALCOHOL, ALCOHOL, ALCOHOL, NON_ALCOHOL, ALCOHOL, NON_ALCOHOL, NON_ALCOHOL, NON_ALCOHOL]
    result = ModelEvaluator(seed=1, n_resamples=999).evaluate(y_true, y_pred)
    assert (result.tp, result.fn, result.fp, result.tn) == (3, 1, 1, 3)
    for name in ("recall", "precision", "f1", "accuracy"):
        m = result[name]
        assert m.value == pytest.approx(0.75) and m.low <= m.value <= m.high


def test_evaluator_perfect_and_invalid_input() -> None:
    perfect = ModelEvaluator(seed=1, n_resamples=999).evaluate([ALCOHOL, NON_ALCOHOL] * 5, [ALCOHOL, NON_ALCOHOL] * 5)
    assert (perfect["recall"].low, perfect["recall"].high) == (1.0, 1.0)
    with pytest.raises(ValueError):
        ModelEvaluator(seed=1).evaluate([ALCOHOL], [])


# --- run log and OCR stats --------------------------------------------------------

def test_model_run_log_round_trip(tmp_path: Path) -> None:
    result = ModelEvaluator(seed=1, n_resamples=999).evaluate([ALCOHOL, NON_ALCOHOL] * 3, [ALCOHOL] * 6)
    run = ModelRun.from_evaluation(result, ocr_setup="s", arm="a", eval_set="val", train_set="train",
                                   threshold=0.5, fit_s=0.1, ms_per_text=2.0)
    log = ModelRunLog(tmp_path / "runs.csv")
    log.upsert(run)
    log.upsert(run)
    log.save()
    assert ModelRunLog(tmp_path / "runs.csv").runs == [run]
    assert run.run_id == "s|a|val" and run.precision == pytest.approx(0.5)


def test_ocr_run_stats() -> None:
    docs = [
        OCRDocument("1a.jpg", ALCOHOL, "train", OCRResult("1a.jpg", [TextBox("gin", 0.9)], elapsed_s=1.0,
                                                          variants_tried=["original"])),
        OCRDocument("2na.jpg", NON_ALCOHOL, "train", OCRResult("2na.jpg", [], elapsed_s=3.0,
                                                               variants_tried=["original", "mirror"])),
    ]
    stats = OCRRunStats.from_documents("s", docs)
    assert (stats.n_images, stats.sec_per_img, stats.retry_rate, stats.text_found_rate) == (2, 2.0, 0.5, 0.5)


# --- comparison runner ------------------------------------------------------------

def _cached_corpus(tmp_path: Path, setup: OCRSetup) -> Path:
    """10 train+val texts per class (cross-validated) and one test text (must never be used)."""
    manifest = tmp_path / "splits.csv"
    rows, cache = [], OCRCache(tmp_path / "cache", tta_cache_config(setup.engine_id, CONFIG_A))
    items = [(t, ALCOHOL) for t in CV_ALCOHOL] + [(t, NON_ALCOHOL) for t in CV_OTHER] + [("gin test", ALCOHOL)]
    for i, (text, label) in enumerate(items):
        name = f"{i}{'a' if label == ALCOHOL else 'na'}.jpg"
        split = "test" if i == len(items) - 1 else ("val" if i % 5 == 0 else "train")
        rows.append(SplitAssignment(name, label, split))
        cache.put(OCRResult(name, [TextBox(text, 0.95)], elapsed_s=1.0, variants_tried=["original"]))
    SplitManifest.write(rows, manifest)
    return manifest


def test_cross_validation_scores_every_train_val_image_once(tmp_path: Path) -> None:
    setup = OCRSetup("toy", "det", "rec")
    manifest = _cached_corpus(tmp_path, setup)
    factory = ArmFactory(CONFIG, LEXICON, encoder=FakeEncoder())
    comparison = ModelComparison(manifest, tmp_path / "cache", factory, ModelEvaluator(1, n_resamples=999),
                                 seed=CONFIG.seed, log=lambda _: None)
    result = comparison.run([setup])
    assert [r.arm for r in result.runs] == ["v0_lexicon", "v1_char_ngram", "v2_minilm"]
    assert all(r.n == 20 and r.n_folds == 5 and r.eval_set == "cv5(train+val)" for r in result.runs)
    assert result.runs[0].recall == 1.0 and result.runs[0].recall_fold_std == 0.0
    preds = result.predictions
    for _, group in preds.groupby("arm"):
        assert len(group) == 20 and group["file_name"].is_unique  # one out-of-fold prediction per image
    assert "20na.jpg" not in set(preds["file_name"]) and "test" not in set(preds["split"])
    # Stratified: every fold holds 2 alcohol and 2 non-alcohol images.
    v0 = preds[preds["arm"] == "v0_lexicon"]
    assert (v0.groupby(["fold", "label"]).size() == 2).all()
    assert result.ocr_stats[0].n_images == 20


def test_cross_validation_refuses_test_and_unknown_arms(tmp_path: Path) -> None:
    setup = OCRSetup("toy", "det", "rec")
    manifest = _cached_corpus(tmp_path, setup)
    factory = ArmFactory(CONFIG, LEXICON, encoder=FakeEncoder())
    comparison = ModelComparison(manifest, tmp_path / "cache", factory, ModelEvaluator(1), seed=1, log=lambda _: None)
    with pytest.raises(ValueError, match="test"):
        comparison.run([setup], splits=("train", "test"))
    with pytest.raises(ValueError, match="Unknown arm"):
        factory.build("v9")


def test_classifiers_pass_c_to_the_logistic_regression() -> None:
    assert CharNgramClassifier(CONFIG.normalizer(), CONFIG.seed)._model.C == 1.0  # scikit-learn's default
    assert CharNgramClassifier(CONFIG.normalizer(), CONFIG.seed, c=0.01)._model.C == 0.01
    assert EmbeddingClassifier(FakeEncoder(), CONFIG.normalizer(), LEXICON, CONFIG.seed, c=100.0)._model.C == 100.0


# --- C tuning ---------------------------------------------------------------------

ALCOHOL_WORDS = ("vodka", "gin", "whiskey", "heineken")


class RecordingArm(TextClassifier):
    """Scores 1 for a known alcohol word, else 0, unless C < 1 (then a constant).

    Records what it was fit on and scored, and raises a ConvergenceWarning when C > 5.
    """

    name = "fake"

    def __init__(self, c: float, log: list["RecordingArm"]) -> None:
        super().__init__(0.5)
        self.c = c
        self.seen: set[str] = set()
        self.scored: set[str] = set()
        self._log = log

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "RecordingArm":
        if self.c > 5:
            warnings.warn("did not converge", ConvergenceWarning)
        self.seen = set(texts)
        self._log.append(self)
        return self

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        self.scored |= set(texts)
        if self.c < 1:
            return np.full(len(texts), 0.5)
        return np.array([float(any(w in t for w in ALCOHOL_WORDS)) for t in texts])

    def explain(self, text: str) -> tuple[str, ...]:
        return ()


def test_tuned_c_is_nested_and_ties_go_to_the_smallest_c() -> None:
    log: list[RecordingArm] = []
    tuned = TunedC(lambda c: RecordingArm(c, log), seed=1, grid=(1.0, 2.0, 3.0))
    assert tuned.name == "fake_tuned"
    with pytest.raises(RuntimeError, match="not fitted"):
        tuned.scores(["vodka"])
    tuned.fit(CV_TEXTS, CV_LABELS)
    assert len(log) == 3 * 5 + 1  # 3 C values x 5 inner folds, then one refit
    for arm in log[:-1]:  # inner fits never score a text they were fit on
        assert len(arm.seen) == 16 and arm.seen.isdisjoint(arm.scored) and len(arm.scored) == 4
    assert [p.mean_auc for p in tuned.search] == [1.0, 1.0, 1.0]
    assert tuned.best_c == 1.0 and log[-1].c == 1.0 and log[-1].seen == set(CV_TEXTS)


def test_tuned_c_picks_the_best_mean_auc_and_counts_non_convergence() -> None:
    log: list[RecordingArm] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)  # must be captured, not raised
        tuned = TunedC(lambda c: RecordingArm(c, log), seed=1, grid=(0.1, 2.0, 10.0)).fit(CV_TEXTS, CV_LABELS)
    by_c = {p.c: p for p in tuned.search}
    assert by_c[0.1].mean_auc == 0.5 and by_c[2.0].mean_auc == 1.0
    assert (by_c[0.1].n_not_converged, by_c[2.0].n_not_converged, by_c[10.0].n_not_converged) == (0, 0, 5)
    assert tuned.best_c == 2.0 and not tuned.final_not_converged
    assert tuned.predict(["vodka night"])[0].label == ALCOHOL


def test_tuned_arms_in_cross_validation_record_the_search(tmp_path: Path) -> None:
    setup = OCRSetup("toy", "det", "rec")
    manifest = _cached_corpus(tmp_path, setup)
    factory = ArmFactory(CONFIG, LEXICON, encoder=FakeEncoder())
    comparison = ModelComparison(manifest, tmp_path / "cache", factory, ModelEvaluator(1, n_resamples=999),
                                 seed=CONFIG.seed, log=lambda _: None)
    result = comparison.run([setup], arms=["v1_char_ngram_tuned", "v2_minilm_tuned"])
    assert [r.arm for r in result.runs] == ["v1_char_ngram_tuned", "v2_minilm_tuned"]
    assert all(r.notes.startswith("C per fold: ") and r.n == 20 for r in result.runs)
    search = result.c_search
    assert len(search) == 2 * 5 * 10  # arms x outer folds x default grid
    assert (search.groupby(["arm", "fold"])["chosen"].sum() == 1).all()
    assert set(result.predictions["arm"]) == {"v1_char_ngram_tuned", "v2_minilm_tuned"}


# --- ranking metrics --------------------------------------------------------------

def test_rank_formula_auc_matches_scikit_learn_with_ties() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(4, 60))
    s = rng.integers(0, 5, size=(4, 60)).astype(float)  # many ties
    expected = [roc_auc_score(y[i], s[i]) for i in range(4)]
    assert roc_auc(y, s, axis=-1) == pytest.approx(expected)
    assert np.isnan(roc_auc(np.ones(5), np.arange(5.0)))  # one class only


def test_ranking_evaluator_values_intervals_and_folds() -> None:
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, 200)
    scores = t * 0.3 + rng.random(200)
    labels = [ALCOHOL if v else NON_ALCOHOL for v in t]
    folds = np.arange(200) % 4
    result = RankingEvaluator(seed=1, n_resamples=999).evaluate(labels, scores, folds)
    assert result.roc_auc.value == pytest.approx(roc_auc_score(t, scores))
    assert result.average_precision.value == pytest.approx(average_precision_score(t, scores))
    for m in (result.roc_auc, result.average_precision):
        assert m.low <= m.value <= m.high
    per_fold = [roc_auc_score(t[folds == k], scores[folds == k]) for k in range(4)]
    assert result.roc_auc_fold_mean == pytest.approx(np.mean(per_fold))
    assert result.roc_auc_fold_std == pytest.approx(np.std(per_fold, ddof=1))
    with pytest.raises(ValueError):
        RankingEvaluator(seed=1).evaluate([ALCOHOL], [])


# --- operating points -------------------------------------------------------------

OP_LABELS = [ALCOHOL, ALCOHOL, ALCOHOL, NON_ALCOHOL, NON_ALCOHOL, NON_ALCOHOL]
OP_SCORES = [0.9, 0.6, 0.3, 0.7, 0.2, 0.1]


def test_operating_point_counts_by_hand() -> None:
    table = OperatingPointTable(OP_LABELS, OP_SCORES)
    p = table.at(0.5)  # flags 0.9, 0.6 (alcohol) and 0.7 (not)
    assert (p.tp, p.fn, p.fp, p.tn, p.flagged) == (2, 1, 1, 2, 3)
    assert (p.recall, p.precision, p.f1, p.accuracy) == pytest.approx((2 / 3, 2 / 3, 2 / 3, 4 / 6))
    assert table.at(0.6).tp == 2  # a score equal to the threshold is flagged
    everything, nothing = table.at(0.0), table.at(0.95)
    assert (everything.recall, everything.precision) == (1.0, 0.5)
    assert (nothing.recall, nothing.precision, nothing.flagged) == (0.0, 0.0, 0)  # zero_division=0
    expected = ModelEvaluator.point_metrics(OP_LABELS, [ALCOHOL if s >= 0.5 else NON_ALCOHOL for s in OP_SCORES])
    assert (p.recall, p.precision, p.f1, p.accuracy) == pytest.approx(tuple(expected.values()))


def test_operating_point_fold_spread_and_curve() -> None:
    table = OperatingPointTable(OP_LABELS, OP_SCORES, folds=[0, 1, 0, 1, 0, 1])
    p = table.at(0.5)  # fold 0 recall 1/2, precision 1; fold 1 recall 1, precision 1/2
    assert p.recall_fold_std == pytest.approx(np.std([0.5, 1.0], ddof=1))
    assert p.precision_fold_std == pytest.approx(np.std([1.0, 0.5], ddof=1))
    curve = table.curve()
    assert curve.threshold.tolist() == sorted(OP_SCORES)
    assert curve.flagged.tolist() == [6, 5, 4, 3, 2, 1]
    assert np.isnan(OperatingPointTable(OP_LABELS, OP_SCORES).at(0.5).recall_fold_std)
    with pytest.raises(ValueError):
        OperatingPointTable(OP_LABELS, OP_SCORES[:3])


# --- model bundle -----------------------------------------------------------------

def _card(clf: CharNgramClassifier, **overrides: object) -> ModelCard:
    values: dict[str, object] = dict(
        arm="v1_char_ngram", threshold=0.4, c=1.0, ocr_setup="toy", ocr={"engine": "det+rec"}, leet=True, seed=1998,
        train_splits=["train", "val"], n_train=8, n_train_alcohol=4, no_text_score=float(clf.scores([""])[0]),
        no_text_label=ALCOHOL,
        sklearn_version=sklearn.__version__, created="2026-10-03", git_commit="abc1234",
    )
    values.update(overrides)
    return ModelCard(**values)  # type: ignore[arg-type]


def test_model_bundle_round_trip_keeps_scores_and_applies_card_threshold(tmp_path: Path) -> None:
    clf = CharNgramClassifier(CONFIG.normalizer(), CONFIG.seed).fit(TRAIN_TEXTS, TRAIN_LABELS)
    ModelBundle(clf, _card(clf)).save(tmp_path / "m")
    loaded = ModelBundle.load(tmp_path / "m")
    texts = ["vodka special", "orange coffee", ""]
    assert np.array_equal(loaded.classifier.scores(texts), clf.scores(texts))  # bit-identical
    assert loaded.classifier.threshold == 0.4 and loaded.card == _card(clf)


def test_model_bundle_rejects_bad_input(tmp_path: Path) -> None:
    clf = CharNgramClassifier(CONFIG.normalizer(), CONFIG.seed).fit(TRAIN_TEXTS, TRAIN_LABELS)
    with pytest.raises(ValueError, match="does not match"):
        ModelBundle(clf, _card(clf, arm="v2_minilm"))
    with pytest.raises(FileNotFoundError, match="is missing"):
        ModelBundle.load(tmp_path / "empty")
    ModelBundle(clf, _card(clf, sklearn_version="0.0.1")).save(tmp_path / "old")
    with pytest.warns(UserWarning, match="scikit-learn 0.0.1"):
        ModelBundle.load(tmp_path / "old")
    ModelBundle(clf, _card(clf)).save(tmp_path / "wrong")
    joblib.dump({"not": "a classifier"}, tmp_path / "wrong" / "classifier.joblib")
    with pytest.raises(TypeError, match="not a TextClassifier"):
        ModelBundle.load(tmp_path / "wrong")


def test_model_bundle_rejects_a_card_that_is_incomplete_or_from_another_model(tmp_path: Path) -> None:
    clf = CharNgramClassifier(CONFIG.normalizer(), CONFIG.seed).fit(TRAIN_TEXTS, TRAIN_LABELS)
    card_path = tmp_path / "m" / "model_card.json"
    ModelBundle(clf, _card(clf)).save(tmp_path / "m")
    raw = json.loads(card_path.read_text(encoding="utf-8"))
    card_path.write_text(json.dumps({k: v for k, v in raw.items() if k != "no_text_label"}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"lacks \['no_text_label'\]"):
        ModelBundle.load(tmp_path / "m")
    other = CharNgramClassifier(CONFIG.normalizer(), CONFIG.seed, c=0.01).fit(TRAIN_TEXTS, TRAIN_LABELS)
    ModelBundle(other, _card(clf)).save(tmp_path / "other")  # a card copied from another fit
    with pytest.raises(ValueError, match="does not describe the stored classifier"):
        ModelBundle.load(tmp_path / "other")
