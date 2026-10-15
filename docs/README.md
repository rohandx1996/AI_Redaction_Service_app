# OCR Redaction Pipeline

## Primary integrated flow

Use `run_integrated_redaction.py` for the finalized original-PDF pipeline. It renders each source page; extracts sensitive values from the page image with Opus; runs legacy PaddleOCR DB/SVTR and PP-DocLayoutV3/OpenCV; maps extracted values to OCR spans with Opus; classifies image-tag crops with Haiku; applies selective Opus material-text decisions; combines PDF-coordinate redaction regions; and audits the output. The original PDF is never edited in place.

Release is fail-closed. Any review region, unmatched sensitive value, failed stage, partial-page run, or audit failure leaves the PDF in `quarantine/`; only a complete document with a passing audit and no blockers moves to `release/`.

## Runtime separation

The pipeline intentionally isolates two incompatible Paddle generations:

- PaddleOCR 2.7/PaddlePaddle 2.6 runs as a persistent OCR worker process (`src/redaction_flow/ocr_worker.py`). Select its interpreter with `OCR_LEGACY_PYTHON` or `--ocr-python`.
- PP-DocLayoutV3 uses PaddleX 3/PaddlePaddle 3 packages loaded from `PADDLE_LAYOUT_PACKAGES` or `--layout-packages`. Install the optional layout requirements from `requirements-ppdoclayout3.txt` into a dedicated environment/package target. Do not overlay Paddle 3 into the legacy OCR environment.
- The PP-DocLayoutV3 weights must already be cached at the path supplied by `--layout-cache` (defaults to `~/.paddlex/official_models/PP-DocLayoutV3`).
- Pass `--layout-packages` (or set `PADDLE_LAYOUT_PACKAGES`) to the isolated PaddleX package directory; set `--ocr-python` (or `OCR_LEGACY_PYTHON`) to the PaddleOCR 2.x worker interpreter.
- Bosch Gateway credentials come from `.env`/environment: `BOSCH_GATEWAY_URL` and `BOSCH_GATEWAY_TOKEN`. Do not commit credentials.

## Run

Install core dependencies and the package with `pip install -e .`.

Point `--input-root` at a folder containing original PDFs. For matched-counterpart directories, use `--original-only` so only `original_01.pdf` files are ingested. `--smoke` limits processing to one PDF and its first page; such a partial run is intentionally blocked from release.

A normal run creates a timestamped batch under the selected output root. Each document contains page renders, OCR recognition JSON, redaction-information JSON, span mappings, image classifications, material decisions, a quarantined PDF, and `pipeline_result.json`. Inspect the blockers and audit before considering any output releasable.

## Tests

Run the offline regression suite with the project environment:

- `python -m unittest discover -s tests -v`

The tests cover coordinate conversion (including rotated pages), PaddleOCR worker and PP-DocLayoutV3 output contracts, model request formatting, invalid span handling, classification/review release gates, QR whole-image-tag redaction, orchestration order, and removal of searchable text beneath PDF redactions. They do not replace a live model smoke test.

The repository is focused on the integrated runner and its required modules. Superseded batch runners, data-preparation/report scripts, and model-comparison prototypes have been removed from the working tree; their history remains available in Git.
