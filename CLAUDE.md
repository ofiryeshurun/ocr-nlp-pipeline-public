# CLAUDE.md

## Project
Classify images as `alcohol` or `non_alcohol` from the **text that appears in them**. Pipeline = OCR -> text cleaning -> text classification. Overview: `README.md`; reasons for each choice: `docs/design.md`.

- Data: none is versioned. Training and evaluation read `images/alcohol/` and `images/non_alcohol/`; the label is the folder, never the file name or image content.
- Constraint: **CPU-only**. No GPU-dependent steps.
- Priorities: robustness to noisy, distorted and obfuscated text; explainability (every choice documented); honest evaluation.
- Inference: `scripts/inference.py` takes an image folder and writes `results.csv` with header `file_name,prediction`, where `prediction` is exactly `alcohol` or `non_alcohol`. It must not depend on the labelled folder layout.

## Key design choices (reasons in `docs/design.md`)
- **Framing:** a recall-first content-moderation filter. Recall on `alcohol` is the primary metric; the "a missed alcohol image is costlier" assumption is stated openly. Precision, F1, accuracy and the confusion matrix are always reported too.
- **OCR:** PaddleOCR 3.7.0, PP-OCRv6 small at native size, LAB CLAHE preprocessing, mirror retry when char-weighted mean confidence < 0.9, `enable_mkldnn=False`.
- **Classifier:** v1, char_wb 2–5 TF-IDF + logistic regression, `C=1`, threshold 0.5. Compared against v0 (lexicon + fuzzy) and v2 (frozen sentence embeddings + LR) by 5-fold CV on train+val. No hybrid; the single winner ships.
- **Evaluation:** 70/15/15 stratified split with a fixed seed. Model selection uses CV on train+val only; the test split is scored once and never tuned on.
- **Inference:** fresh OCR with no cache (file-name keys can collide). Unreadable or failed images are labelled `alcohol` (`config.UNREADABLE_LABEL`). Optional `--details` writes the classifier's own evidence.
- **Hard negatives:** evidence-triggered only. **Augmentation:** none; text augmentation is future work.
- **Report:** LaTeX source and figures in `docs/report/`; the compiled `ocr_nlp_report.pdf` is committed next to it.
- **Environment:** Python 3.12 in `.venv`. `requirements.txt` pins the imported runtime packages; `requirements-dev.txt` adds tests and the v2 stack (PyTorch CPU, sentence-transformers).

## Workflow
- Work in numbered buckets (e.g. B0 environment, B1 EDA, B2 OCR, …), each planned before coding.
- Each bucket gets its own branch `bucket/bN-<name>`, a checkpoint where the user reviews, and then `git merge --no-ff` into `main` after approval.
- Keep `README.md`, `docs/development.md` and `requirements*.txt` updated in every bucket.

## Working agreements
- **Explain choices.** For every non-trivial decision (OCR engine, thresholds, preprocessing, lexicon entries, metrics), record the *why* in `docs/design.md`. Explainability matters as much as the score.
- **Ask before big steps.** Confirm first before installing heavy dependencies (PaddlePaddle, PyTorch), downloading models, or committing large files.
- **Evaluate honestly.** Split train/validation before tuning. Never tune the lexicon or thresholds against the final holdout. Report precision, recall, F1 and the confusion matrix. Consider OCR-level metrics (text found or not) as intermediate metrics.
- **Keep results reproducible.** Cache OCR output (e.g. JSON per image) so experiments don't re-run slow OCR. Do not commit the cache if it's large.

## Code quality
Scripts must be as robust as possible, using solid object-oriented design.
- Use classes wherever a concept has state or behavior (e.g. an OCR engine wrapper, a text normalizer, a lexicon matcher, a classifier, a pipeline). Create a class when it's needed; don't leave logic as loose functions in scripts.
- Use abstract base classes or protocols for swappable components (e.g. the `OCREngine` and `ImagePreprocessor` interfaces) so engines and classifiers can be compared and replaced easily.
- Type hints everywhere: all function signatures, return types and class attributes. Use `dataclass`es for structured data (e.g. OCR results, predictions).
- Robust error handling: handle unreadable or corrupt images, empty OCR results, and missing paths. One bad image must never crash a batch run.
- Docstrings on public classes and methods. Keep modules small and single-purpose. Avoid global state and hard-coded paths; take config as arguments.
- The inference script is a thin CLI (`argparse`) over the library classes.

## Asking questions
Ask questions often to reduce uncertainty. Don't guess when a wrong assumption would cost real work.
- If requirements, data, metrics or a design choice are ambiguous, ask before implementing.
- When there are several reasonable options, briefly present them with a recommendation and ask which to take.
- Check assumptions about the data and the task early, rather than after building on them.

## Git workflow
Commit early and often without waiting to be asked. Push when a milestone is done.
- Commit when one logical task is complete (a bug fix, a feature, a docs update). Never mix a bug fix and a feature in one commit.
- Commit before starting a new task or refactor, so there is a rollback point.
- Commit when a test passes.
- Commit before ending a session or taking a break.
- Messages: short imperative subject, body explains *why* when non-obvious. Never skip hooks or force-push.
- The public repository is a cleaned single-commit copy; the development history is private.

## Layout
See "Project layout" in `docs/development.md`. Keep it current when files are added.

## Environment notes
- Developed on Windows 11 (PowerShell and Git Bash); the code is cross-platform.
- PaddlePaddle has no wheels for Python 3.14. Always use the Python 3.12 `.venv`.
