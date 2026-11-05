# ocr-nlp-pipeline

Classify images as `alcohol` or `non_alcohol` from the **text that appears in them**: captions, overlays, signs. The pipeline reads the text with OCR, cleans it, and classifies it with a character n-gram model. Everything runs on CPU.

## The problem

Content-moderation systems often need to know whether an image *talks about* alcohol, e.g. a street photo with "craft beer happy hour" written across it, even when no bottle is visible. Text in images is hard to read reliably:
- it can be faint, low-contrast, mirrored or rotated;
- it can be deliberately obfuscated (`v0dk4`, `g!n`, `h eine ken`);
- OCR adds its own errors (`heinekin`).

The classifier has to tolerate all of that.

The filter is **recall-first**. Missing an alcohol image is assumed to cost more than flagging a safe one. Precision, F1, accuracy and the confusion matrix are always reported alongside recall, so recall can't be bought by flagging everything.

## Pipeline

```
image ─► preprocessing ─► OCR (+ mirror retry) ─► text cleaning ─► classifier ─► threshold ─► results.csv
```

| Stage | What it does | Code |
|---|---|---|
| 1. Load | Reads `.jpg/.jpeg/.png/.bmp/.webp`. Unreadable files are logged and labelled `alcohol`, and the batch continues | `ocr/image_loader.py`, `inference/images.py` |
| 2. Preprocess | CLAHE on the lightness channel of LAB colour space: lifts faint text and keeps colour contrast | `ocr/preprocess.py` |
| 3. OCR | PaddleOCR PP-OCRv6 small at native size. If the mean character confidence is below 0.9, it reads the mirrored image too and keeps the more confident read | `ocr/paddle.py`, `ocr/tta.py` |
| 4. Clean | Unicode NFKC, accent stripping, lowercasing, leetspeak decoding (`v0dk4` → `vodka`), punctuation removal | `text/normalize.py`, `text/leet.py` |
| 5. Classify (**v1**) | TF-IDF over character n-grams (2–5, within words) + logistic regression | `models/char_ngram.py` |
| 6. Decide | Score ≥ threshold (0.5) → `alcohol`; writes `file_name,prediction`, plus optional per-image evidence | `inference/pipeline.py`, `inference/writers.py` |

The shipped classifier, **v1**, was chosen over two other arms by 5-fold cross-validation (`scripts/compare_models.py`; scores under [Results](#results)):
- **v0:** a domain lexicon of drink types and brands with length-aware fuzzy matching (`lexicon/`)
- **v2:** frozen `all-MiniLM-L6-v2` sentence embeddings + logistic regression (`models/embedding.py`)

The reasoning behind each choice is in [docs/design.md](docs/design.md).

## Results

Measured on a private set of 1,100 labelled images (550 per class), with a stratified 70/15/15 train/val/test split. The images and the trained model are not included here. Every choice was made by cross-validation on train+val, and the test split was scored once, at the end. The [report](docs/report/ocr_nlp_report.pdf) covers data exploration, experiments and failure cases, with sample images.

**Held-out test split** (165 images, scored once):

| Metric | Test [95% CI] |
|---|---|
| Recall (`alcohol`) | 0.916 [0.852, 0.973] |
| Precision | 0.916 [0.853, 0.974] |
| F1 | 0.916 [0.867, 0.956] |
| Accuracy | 0.915 [0.873, 0.958] |

**Classifier comparison** (5-fold cross-validation on the 935 train+val images; every arm reads the same OCR text):

| Arm | Recall [95% CI] | Precision | F1 |
|---|---|---|---|
| v0: lexicon + fuzzy matching | 0.839 [0.807, 0.872] | 0.891 | 0.864 |
| **v1: char n-gram TF-IDF + LR (shipped)** | **0.953** [0.933, 0.971] | **0.927** | **0.940** |
| v2: frozen MiniLM embeddings + LR | 0.878 [0.847, 0.907] | 0.878 | 0.878 |

These scores were used to pick the model, so they run higher than the test scores above.

**OCR benchmark:** [43 configurations](docs/ocr_benchmark.md) were compared on 60 hand-annotated training images: 3 PaddleOCR models, 8 preprocessing options and several mirror/rotation retry setups. The shipped setup reads the key term in 85% of them (overlay character error rate 0.103). PP-OCRv6 medium reads 87% but is about 4x slower: 3.31 s vs 0.83 s per image over the 935 train+val images.

To get these metrics on your own data, follow the steps under [Run](#run); `scripts/evaluate.py` reports them with a confusion matrix.

## How this was built

Developed with Claude Code, in planned stages. Each stage had its own branch and a review checkpoint before merging, and every non-trivial choice was recorded with its reason ([docs/design.md](docs/design.md)). The working agreements Claude Code followed (workflow, evaluation rules, code standards) are in [CLAUDE.md](CLAUDE.md).

## Install

Python 3.12 is recommended; 3.11–3.13 are supported, and PaddlePaddle has no wheels for 3.14. No GPU is needed.

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt      # add -r requirements-dev.txt for tests and the v2 arm
```

The first OCR run downloads the PaddleOCR models (about 40 MB) to `~/.paddlex/official_models/`.

## Run

### 1. Prepare data

Put your labelled images in two folders, named by label. File names must be unique across both folders.

```
images/
  alcohol/
  non_alcohol/
```

### 2. Train

```bash
# Stratified 70/15/15 train/val/test split -> data/splits.csv
python scripts/make_split.py

# OCR train + val with the default setup (cached as one JSON per image under cache/ocr/)
python scripts/run_ocr.py --mode tta --splits train val \
    --det-model PP-OCRv6_small_det --rec-model PP-OCRv6_small_rec --preprocess lab_clahe \
    --strategy conditional --threshold 0.9 --combine best --variants original mirror

# Optional: compare the classifier arms by 5-fold cross-validation (writes results/*.csv)
python scripts/compare_models.py --setups v6_small_lab_clahe

# Fit the classifier on train + val -> models/classifier.joblib + models/model_card.json
python scripts/train_final.py
```

### 3. Evaluate on the test split (once)

```bash
python scripts/copy_split.py --split test --dest cache/inference/test_images
python scripts/inference.py --input-dir cache/inference/test_images \
    --output results/test/test_predictions.csv --details results/test/test_details.csv
python scripts/evaluate.py --predictions results/test/test_predictions.csv --split test --output-dir results/test
```

### 4. Label new images

```bash
python scripts/inference.py --input-dir path/to/images --output results.csv
```

| Option | Default | Description |
|---|---|---|
| `--input-dir` | required | Folder of images to label. Sub-folders and other files are skipped with a warning |
| `--output` | `results.csv` | Results file: `file_name,prediction`, with `prediction` either `alcohol` or `non_alcohol` |
| `--details` | off | Also write each image's score, the n-grams that drove it, the OCR text, whether the mirrored read was used, and any error |
| `--model-dir` | `models/` | Model folder to load |

Inference always runs OCR fresh (about 0.8 s per image on an 8-core CPU) and never uses file or folder names as a signal. `classifier.joblib` is a pickle: load only model files you trust.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake OCR engine and temporary files, so they need no data and no OCR models.

## Project layout

```
src/alcohol_detector/   library: OCR engine wrapper, preprocessing, OCR cache, text cleaning,
                        lexicon matcher, classifier arms, metrics and CV, inference pipeline
scripts/                command-line entry points (split, OCR, compare, train, inference, evaluate)
data/lexicon.yaml       alcohol terms from domain knowledge (used by the v0 arm)
docs/design.md          design decisions and their reasons
docs/ocr_benchmark.md   the 43 OCR configurations compared, one row each
docs/development.md     developer guide: all commands, layout, reproducibility
docs/report/            the report (ocr_nlp_report.pdf, its LaTeX source and figures)
tests/                  pytest suite
```

## Troubleshooting

- **The OCR model download fails:** set the environment variable `PADDLE_PDX_MODEL_SOURCE` to `bos`, `modelscope` or `aistudio` to download from another host. The default is Hugging Face.
- **`python3.12 -m venv` fails with "ensurepip is not available" (Ubuntu / Debian):** run `sudo apt-get install -y python3.12-venv`.
- **`libGL.so.1` or `libgthread` errors on Linux (e.g. in Docker):** run `apt-get install -y libgl1 libglib2.0-0`.
- **PowerShell refuses to run `Activate.ps1`:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window first.

## License

MIT, see [LICENSE](LICENSE). © 2026 Ofir Yeshurun.
