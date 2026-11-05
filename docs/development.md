# Development guide

How to work on the repository: the environment, the full command sequence, the project layout and reproducibility. The reasons behind each choice are in [design.md](design.md).

## Setup

Use Python 3.12 (tested with 3.12.10 on Windows 11, CPU-only):
- PaddlePaddle, which the OCR engine depends on, ships wheels only up to Python 3.13.
- PaddleOCR documents 3.8–3.12 as its tested range.
- 3.12 is the most reliable version across the whole dependency stack, including PyTorch for the v2 arm.

```bash
py -3.12 -m venv .venv            # Linux/macOS: python3.12 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
```

- `requirements.txt` pins the third-party packages the code imports, plus `paddlepaddle`, the OCR backend.
- `requirements-dev.txt` adds pytest and the v2 text encoder: CPU PyTorch from the PyTorch index, and sentence-transformers. The default model does not use v2.
- **OCR models download automatically** the first time OCR runs, into `~/.paddlex/official_models/`. The default setup needs about 40 MB; PP-OCRv6 medium needs more.
- **oneDNN is disabled on purpose.** PaddlePaddle 3.3.1 crashes on Windows CPU when oneDNN is on, so the code always turns it off.
- **Paddle and PyTorch in one process (Windows):** importing torch after paddle fails with `WinError 127`. `PaddleOCREngine` therefore imports torch first, when it is installed.

## Data placement

No dataset is versioned. Place your labelled images like this; file names must be unique across both folders:

```
images/
  alcohol/
  non_alcohol/
```

The folder layout is used **only** for training and evaluation. The inference script reads a flat directory of images and never uses file names or folders as a signal.

## Commands

```bash
# Stratified 70/15/15 split (fixed seed). Every other script reads data/splits.csv.
# The script refuses to overwrite an existing file
python scripts/make_split.py --image-root images --output data/splits.csv

# OCR train + val with the default setup: v6 small + LAB CLAHE, mirror retry on weak reads.
# Results are cached as one JSON per image under cache/ocr/<config hash>/, so reruns are instant.
# The cache is keyed by file name: clear cache/ocr/ when you switch datasets
python scripts/run_ocr.py --mode tta --splits train val \
    --det-model PP-OCRv6_small_det --rec-model PP-OCRv6_small_rec --preprocess lab_clahe \
    --strategy conditional --threshold 0.9 --combine best --variants original mirror
# Other OCR setups compared by compare_models.py: v6_medium (PP-OCRv6_medium_det/_rec, no --preprocess)
# and v6_small (no --preprocess)

# OCR benchmark against hand transcriptions: every flip/rotation variant cached separately,
# for the images listed in a CSV with a file_name column (see ocr/evaluate.py for the metrics)
python scripts/run_ocr.py --mode per-variant --files-from subset.csv \
    --det-model PP-OCRv6_small_det --rec-model PP-OCRv6_small_rec [--preprocess lab_clahe]

# Model comparison: stratified 5-fold CV of v0/v1/v2 on train+val OCR text, for each cached OCR setup
# (test is never used). Writes results/model_runs.csv, results/ocr_run_stats.csv, results/cv_predictions.csv
python scripts/compare_models.py [--setups v6_medium v6_small v6_small_lab_clahe] [--arms v0_lexicon v1_char_ngram v2_minilm]

# Tuning check: the same CV with C picked by nested CV; also writes results/c_search.csv
python scripts/compare_models.py --arms v1_char_ngram_tuned v2_minilm_tuned

# Fit the default classifier (v1, C=1, threshold 0.5) on all train+val OCR text -> models/ (seconds, deterministic)
python scripts/train_final.py

# Score the test split once: copy it flat, label it with the real CLI, score it against data/splits.csv
python scripts/copy_split.py --split test --dest cache/inference/test_images
python scripts/inference.py --input-dir cache/inference/test_images --output results/test/test_predictions.csv --details results/test/test_details.csv
python scripts/evaluate.py --predictions results/test/test_predictions.csv --split test --output-dir results/test

# Tests
pytest
```

Matching OCR text against the alcohol lexicon (the v0 arm):

```python
from alcohol_detector.lexicon import AlcoholLexicon, EditBudget, LexiconMatcher
from alcohol_detector.text import LeetMapper, TextNormalizer

matcher = LexiconMatcher(AlcoholLexicon.from_yaml("data/lexicon.yaml"),
                         EditBudget(one_edit_from=5, two_edits_from=7),
                         TextNormalizer(leet=LeetMapper()))
matcher.match("IMAGINE a cold h eine kin")  # [TermMatch(term='heineken', matched_text='heinekin', distance=1, ...)]
```

## Project layout

```
src/alcohol_detector/             library code
src/alcohol_detector/config.py    default settings in one place (OCR setups, default arm, C, threshold, edit budget, encoder revision)
src/alcohol_detector/data.py      image indexing, stratified split, split manifest, split copying
src/alcohol_detector/ocr/         OCR engine interface + PaddleOCR wrapper, preprocessing, mirror/rotation TTA,
                                  OCR cache, batch runner, OCR-quality metrics, run stats
src/alcohol_detector/text/        text normalizer and leetspeak decoder
src/alcohol_detector/lexicon/     lexicon loader and fuzzy matcher with a length-based edit budget
src/alcohol_detector/models/      classifier arms v0/v1/v2, evaluator with bootstrap CIs, cross-validation runner,
                                  C tuning, ROC-AUC/AP, threshold table, model bundle (save/load with its card)
src/alcohol_detector/inference/   image -> OCR -> classifier pipeline, image-folder listing, results/details CSV,
                                  scoring a results file against one split, confusion-matrix figure
scripts/                          command-line entry points (split, OCR, model comparison, final fit, inference, evaluate, split copy)
data/lexicon.yaml                 strong alcohol terms from domain knowledge
docs/design.md                    design decisions and their reasons
docs/ocr_benchmark.md             the 43 OCR configurations compared on the hand-annotated images
docs/report/                      the report: report.tex, its figures and the compiled ocr_nlp_report.pdf
tests/                            pytest suite (fake OCR engine; no data or OCR models needed)
```

Generated and git-ignored: `images/`, `data/splits.csv`, `cache/`, `results/`, `models/`.

## Reproducibility

- All randomness is seeded through `alcohol_detector.seed.set_global_seed` (`DEFAULT_SEED`).
- `requirements.txt` pins every directly used package to an exact version.
- `train_final.py` is deterministic: the same OCR cache and split give the same model.
- `models/model_card.json` records how the model was built: OCR configuration, `C`, threshold, training splits, scikit-learn version and git commit. Inference checks the card and rebuilds the OCR to match. `classifier.joblib` is a pickle, so load only bundles you created.
