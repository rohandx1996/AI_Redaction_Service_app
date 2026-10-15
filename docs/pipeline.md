# Integrated pipeline

1. Render each original PDF page, then send the page image to the configured best vision model (default Opus) for redaction-information JSON.
2. Run PaddleOCR DB/SVTR for OCR spans and PP-DocLayoutV3 plus OpenCV for image-tag and QR/barcode regions.
3. Map redaction JSON values to OCR spans using Opus; unmatched values block release.
4. Classify image-tag crops with Haiku. A confirmed QR/barcode crop is redacted as a whole image tag; material-sample crops go to Opus for selective material-text decisions.
5. Convert all regions to PDF coordinates, redact the original PDF, remove hidden PDF objects, and audit.
6. Release only if the audit passes and there are no review, unmatched-value, stage-error, or partial-document blockers. Otherwise keep the result in quarantine.

The original PDF is never modified in place. Smoke or page-limited runs are explicitly release-blocked.

## Runtime separation

PaddleOCR 2.7/PaddlePaddle 2.6 runs in a persistent worker process. PP-DocLayoutV3 uses PaddleX 3/PaddlePaddle 3 in a separate package environment. Do not install both Paddle generations into the same environment.

Set `OCR_LEGACY_PYTHON` and `PADDLE_LAYOUT_PACKAGES` in the environment or pass `--ocr-python` and `--layout-packages`. PP-DocLayoutV3 weights must be cached at the path supplied through `--layout-cache`. Bosch Gateway credentials are supplied with `BOSCH_GATEWAY_URL` and `BOSCH_GATEWAY_TOKEN`.

Install the package with `pip install -e .`. The `ocr-redact` console command is the primary entry point; `run_integrated_redaction.py` remains as a source-checkout launcher. See `docs/README.md` for run and smoke-test instructions.
