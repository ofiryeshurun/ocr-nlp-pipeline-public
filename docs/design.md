# Design notes

Why the pipeline is built the way it is. The defaults live in `src/alcohol_detector/config.py`. They were picked during development by cross-validation on training data only. They are starting points, not universal constants: re-select them on your own data with `scripts/compare_models.py` (see `docs/development.md`).

## Problem framing and metric

- **Task:** label an image `alcohol` or `non_alcohol` from the text that appears in it (captions, overlays, signs). The label comes from the text, not from the objects in the photo.
- **Use case:** a content-moderation filter that flags images with alcohol references, so they can be kept away from audiences or contexts where alcohol is unwanted.
- **Recall on `alcohol` is the primary metric.** This is an assumption, stated openly: a missed alcohol image costs more than a false flag, which only removes a safe image.
- **Guard against the degenerate case:** flagging everything gives recall = 1. Precision, F1, accuracy and the confusion matrix are therefore always reported too, with 95% bootstrap confidence intervals. The threshold is picked from an out-of-fold operating-point table, not by maximizing recall alone.

## OCR

- **Engine: PaddleOCR 3.x, PP-OCRv6 small, on CPU.**
  - It is actively maintained and strong on scene text.
  - It includes a text-line orientation classifier.
  - It runs at well under a second per image on a laptop CPU.
  - Vision-language OCR models were left out: they take seconds per image on CPU, which doesn't suit high-volume classification.
  - The larger PP-OCRv6 medium is available as an alternative setup. It is about 4x slower.
- **Native image size.** The detector runs at the image's own resolution. Upscaling adds no detail and costs time.
- **LAB CLAHE preprocessing.** This is contrast-limited adaptive histogram equalization (local contrast enhancement), applied only to the lightness channel of the LAB colour space.
  - It lifts faint, low-contrast text.
  - It keeps colour, which matters because overlay text often differs from its background more by colour than by brightness.
  - Grayscale CLAHE throws that colour difference away.
- **Mirror retry on weak reads.** OCR models don't read horizontally mirrored text.
  - When the first read's character-weighted mean confidence is below a trigger (0.9), the image is read again mirrored, and the more confident read is kept.
  - Running the retry only on weak reads keeps the average cost low.
  - Rotation retries were tried and dropped: they cost much more time and added no recall.
- **oneDNN is disabled.** With oneDNN on, PaddlePaddle 3.3.1 crashes at the first CPU inference on Windows. Turning it off on every platform keeps results identical, at some cost in speed.
- **Detector family kept.** PP-OCRv6's detector is the current DBNet generation. Switching to CRAFT or FCENet would add a second deep-learning framework for a small share of curved or diagonal text. This is left as future work.

## Text cleaning

- **Standard cleaning** (`TextNormalizer`): Unicode NFKC, accent stripping, lowercasing, apostrophe removal, punctuation turned into spaces, whitespace collapsed.
- **Leetspeak decoding** (`LeetMapper`) runs before punctuation is removed, so `g!n` becomes `gin` instead of `g n`.
  - It uses a small table from the common leet alphabet: `4 @`→a, `3`→e, `1 ! |`→i, `0`→o, `5 $`→s, and so on.
  - **Guard rules keep real text intact:** only tokens that contain a letter are touched; digits are mapped only when the token starts with a letter (`700ml` and `19th` stay); symbols are mapped only inside a word (`#gin` and `gin!` keep their punctuation).
  - No maintained Python library decodes leetspeak, and a dozen characters don't justify a dependency.

## Classifiers

Three single-model arms. They share the same normalized text and are compared by cross-validation. One wins; there is no ensemble.

| Arm | Model | Strength |
|---|---|---|
| v0 | Domain lexicon + fuzzy matching | Rule-based, fully explainable, covers brands never seen in training |
| **v1 (default)** | TF-IDF over character n-grams (`char_wb`, 2–5) + logistic regression | Robust to OCR typos |
| v2 | Frozen `all-MiniLM-L6-v2` sentence embeddings + logistic regression | Tests whether semantic knowledge generalizes to unseen words |

- **Why character n-grams tolerate OCR noise.** A misread word keeps most of its short pieces: `heinekin` still shares `hei`, `ine` and `ei` with `heineken`. `char_wb` keeps n-grams inside word boundaries, and an upper length of 5 covers a padded three-letter word (` gin `).
- **Lexicon (v0).**
  - `data/lexicon.yaml` holds strong alcohol terms only, from general domain knowledge: drink types, cocktails and major brands. It is not built from training data.
  - Everyday words that are also drink names (corona, patron, rose, shot) are left out, because they would flag ordinary text.
  - **Matching:** short terms must match a whole word exactly, because `gin` hides in `imagine`. Long terms are matched fuzzily in the text with spaces removed, with a length-based edit budget (`AUTO:5,7`, in the style of Elasticsearch fuzziness). This also handles spacing tricks such as `h eine k en`.
- **Embeddings (v2)** are frozen rather than fine-tuned. Fine-tuning a transformer on a small dataset adds many hyperparameters, seed variance and hundreds of MB of weights. Frozen vectors are computed once, and the head trains in seconds.
- **Hyperparameters.**
  - Logistic regression keeps scikit-learn's default `C=1`.
  - Nested cross-validation (inner 5-fold, scored by ROC-AUC) is available as a check through the `*_tuned` arms.
  - ROC-AUC is used because `C` has to be chosen before the threshold.

## Evaluation protocol

- **Split:** a stratified 70/15/15 train/val/test split with a fixed seed (`scripts/make_split.py`).
- **Model selection:** stratified 5-fold cross-validation on train+val pooled. The folds are shared by every arm and OCR setup, so the comparison is paired. Each image gets one out-of-fold prediction.
- **The test split is scored once,** at the end, through the real inference CLI, and nothing is tuned afterwards. Cross-validation numbers are selection numbers and are optimistic; the test score is the unbiased estimate.
- **Threshold:** chosen from the out-of-fold operating-point table, which supported keeping scikit-learn's default of 0.5. Lowering it to 0.45 would have caught 14 more alcohol images for 27 more false flags, and each further step down cost more. Raising it lost recall for almost no gain in F1.

## Inference behaviour

- **No OCR cache at inference.** The development cache is keyed by file name, and new images can reuse old names. Inference therefore always runs OCR fresh.
- **Unreadable images are labelled `alcohol`.** This covers missing, corrupt and non-image files, and OCR failures. The image could not be checked, so under the recall-first framing it is flagged. Its error is logged, and one bad image never stops a batch.
- **Images without text** get no special rule. The classifier's baseline score decides them, like any other image.
- **Explainability:** `--details` writes each image's score, the character n-grams that drove it, the OCR text, whether the mirrored read was used, and any error. File and folder names are never used as a signal.
- **The model card** (`models/model_card.json`) records the OCR configuration, `C`, the threshold, the training splits and the scikit-learn version. At load time, inference rebuilds the OCR exactly as the model was trained.

## Future work

- **Text augmentation** of the training OCR text: OCR-style noise, spacing splits, leetspeak, and swaps within a class.
- **A preprocessing retry** for faint text, e.g. a saturation view, used only on weak reads.
- **Detection and correction of curved or rotated text.**
- **Joint search** of the n-gram range and `C` by nested cross-validation.
- **A shortcut test:** remove the alcohol term from alcohol texts and count how many the learned arms still flag.
