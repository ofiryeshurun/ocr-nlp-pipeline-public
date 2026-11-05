# OCR benchmark

43 OCR configurations, compared on 60 hand-annotated training images (30 per class) whose overlay text and key term were transcribed by hand. Each row is one configuration. Row B26 is the setup the pipeline ships with. These numbers come from 60 images after comparing many configurations, so they are optimistic; the classifier's cross-validation and test scores in the [README](../README.md) are the real check.

- **Engines:** PaddleOCR PP-OCRv5 mobile, PP-OCRv6 medium and PP-OCRv6 small, at native image size.
- **Retry setup:** `original only` reads the image once. `orig+mirror` also reads the horizontally mirrored image; `rot±20` adds ±20° rotations; `all 8` reads every flip/rotation variant. `cond@0.9` runs the extra reads only when the first read's character-weighted mean confidence is below 0.9, `always` runs them on every image. The most confident read is kept.
- **Key-term hit:** share of images whose key term appears in the OCR text (case- and space-insensitive).
- **Key-term sim:** 1 minus the edit distance between the key term and its best-matching OCR stretch, divided by the key-term length.
- **Overlay CER:** character error rate of the overlay text, ignoring extra background text in the photo.
- **s/img:** OCR seconds per image, preprocessing included (8-core CPU, oneDNN off).

## A. Retry setups on the raw image

| # | Engine | Preprocessing | Retry setup | n | Key-term hit | Key-term sim | Overlay CER | s/img |
|---|---|---|---|---|---|---|---|---|
| 1 | v5 mobile | none | original only | 60 | 0.500 | 0.661 | 0.343 | 1.07 |
| 2 | v5 mobile | none | orig+mirror cond@0.9 | 60 | 0.700 | 0.828 | 0.179 | 1.41 |
| 3 | v5 mobile | none | orig+mirror always | 60 | 0.683 | 0.809 | 0.198 | 2.21 |
| 4 | v5 mobile | none | orig+mirror+rot±20+mirror_rot±20 cond@0.9 | 60 | 0.717 | 0.846 | 0.164 | 3.01 |
| 5 | v5 mobile | none | all 8 always | 60 | 0.733 | 0.872 | 0.134 | 9.33 |
| 6 | v6 medium | none | original only | 60 | 0.667 | 0.776 | 0.245 | 3.01 |
| 7 | v6 medium | none | orig+mirror cond@0.9 | 60 | 0.867 | 0.923 | 0.087 | 3.98 |
| 8 | v6 medium | none | orig+mirror always | 60 | 0.867 | 0.923 | 0.087 | 6.00 |
| 9 | v6 medium | none | orig+mirror+rot±20+mirror_rot±20 cond@0.9 | 60 | 0.883 | 0.956 | 0.059 | 9.75 |
| 10 | v6 medium | none | all 8 always | 60 | 0.900 | 0.973 | 0.066 | 33.07 |
| 11 | v6 small | none | original only | 60 | 0.583 | 0.739 | 0.275 | 0.79 |
| 12 | v6 small | none | orig+mirror cond@0.9 | 60 | 0.817 | 0.921 | 0.101 | 1.04 |
| 13 | v6 small | none | orig+mirror always | 60 | 0.817 | 0.921 | 0.101 | 1.57 |
| 14 | v6 small | none | orig+mirror+rot±20+mirror_rot±20 cond@0.9 | 60 | 0.817 | 0.926 | 0.092 | 2.37 |
| 15 | v6 small | none | all 8 always | 60 | 0.833 | 0.929 | 0.114 | 8.11 |

## B. Preprocessing replaces the raw image

| # | Engine | Preprocessing | Retry setup | n | Key-term hit | Key-term sim | Overlay CER | s/img |
|---|---|---|---|---|---|---|---|---|
| 1 | v6 medium | saturation | original only | 60 | 0.517 | 0.643 | 0.365 | 2.58 |
| 2 | v6 medium | strong CLAHE | original only | 60 | 0.617 | 0.729 | 0.286 | 2.58 |
| 3 | v6 medium | min channel | original only | 60 | 0.533 | 0.691 | 0.315 | 2.81 |
| 4 | v6 medium | top-hat | original only | 60 | 0.617 | 0.775 | 0.240 | 2.86 |
| 5 | v6 medium | gray+CLAHE | original only | 60 | 0.533 | 0.710 | 0.298 | 2.94 |
| 6 | v6 medium | LAB CLAHE | original only | 60 | 0.633 | 0.757 | 0.249 | 2.94 |
| 7 | v6 medium | bilateral+CLAHE | original only | 60 | 0.617 | 0.778 | 0.242 | 2.96 |
| 8 | v6 medium | strong CLAHE | orig+mirror cond@0.9 | 60 | 0.833 | 0.892 | 0.121 | 3.38 |
| 9 | v6 medium | saturation | orig+mirror cond@0.9 | 60 | 0.700 | 0.781 | 0.236 | 3.60 |
| 10 | v6 medium | bilateral+CLAHE | orig+mirror cond@0.9 | 60 | 0.817 | 0.911 | 0.101 | 3.79 |
| 11 | v6 medium | top-hat | orig+mirror cond@0.9 | 60 | 0.817 | 0.918 | 0.094 | 3.82 |
| 12 | v6 medium | LAB CLAHE | orig+mirror cond@0.9 | 60 | 0.817 | 0.892 | 0.115 | 3.88 |
| 13 | v6 medium | gray+CLAHE | orig+mirror cond@0.9 | 60 | 0.733 | 0.837 | 0.174 | 3.89 |
| 14 | v6 medium | min channel | orig+mirror cond@0.9 | 60 | 0.717 | 0.804 | 0.198 | 3.91 |
| 15 | v6 small | saturation | original only | 60 | 0.500 | 0.608 | 0.398 | 0.64 |
| 16 | v6 small | top-hat | original only | 60 | 0.567 | 0.698 | 0.300 | 0.72 |
| 17 | v6 small | min channel | original only | 60 | 0.450 | 0.622 | 0.377 | 0.73 |
| 18 | v6 small | gray+CLAHE | original only | 60 | 0.550 | 0.701 | 0.307 | 0.75 |
| 19 | v6 small | LAB CLAHE | original only | 60 | 0.617 | 0.742 | 0.285 | 0.83 |
| 20 | v6 small | strong CLAHE | original only | 60 | 0.583 | 0.745 | 0.280 | 0.84 |
| 21 | v6 small | saturation | orig+mirror cond@0.9 | 60 | 0.683 | 0.748 | 0.271 | 0.91 |
| 22 | v6 small | bilateral+CLAHE | original only | 60 | 0.600 | 0.716 | 0.299 | 0.92 |
| 23 | v6 small | gray+CLAHE | orig+mirror cond@0.9 | 60 | 0.733 | 0.831 | 0.173 | 0.99 |
| 24 | v6 small | top-hat | orig+mirror cond@0.9 | 60 | 0.783 | 0.880 | 0.128 | 1.00 |
| 25 | v6 small | min channel | orig+mirror cond@0.9 | 60 | 0.583 | 0.747 | 0.230 | 1.02 |
| 26 | **v6 small (shipped)** | **LAB CLAHE** | **orig+mirror cond@0.9** | 60 | 0.850 | 0.934 | 0.103 | 1.09 |
| 27 | v6 small | strong CLAHE | orig+mirror cond@0.9 | 60 | 0.800 | 0.905 | 0.116 | 1.15 |
| 28 | v6 small | bilateral+CLAHE | orig+mirror cond@0.9 | 60 | 0.783 | 0.877 | 0.130 | 1.19 |
