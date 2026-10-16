# Integrated pipeline

1. Render each original PDF page, then send the page image to the configured best vision model (default Opus) for redaction-information JSON.
2. Run PaddleOCR DB/SVTR for OCR spans and PP-DocLayoutV3 plus OpenCV for image-tag and QR/barcode regions.
3. Map redaction JSON values to OCR spans using Opus; unmatched values block release.
4. Classify both layout image crops and detected QR/barcode candidate crops visually with Haiku, without decoding code payloads. Redact QR/barcode-classified boxes; send material-sample crops to Opus for selective material-text decisions; preserve noise.
5. Convert all regions to PDF coordinates, redact the original PDF, remove hidden PDF objects, and audit.
6. Release only if the audit passes and there are no review, unmatched-value, stage-error, or partial-document blockers. Otherwise keep the result in quarantine.

The searchable-text audit is page-scoped for OCR-mapped values. Text found only in a material-photo crop is validated against its redaction region rather than searched document-wide. OCR-dense false code detections are discarded. Remaining candidates go through the same visual classifier as layout image crops; a confident QR/barcode classification redacts that candidate's bounding box without inspecting its payload. Low-confidence or failed classifications remain review blockers. Decoding is not part of this flow.

Only pages already carrying a visual review flag receive one additional fast-model comparison of the original and redacted page. The model may approve only those supplied flags, with confidence at or above `--review-threshold` (default 0.95); it cannot add redactions or waive unmatched extracted values or a failed PDF audit. Pages without review flags incur no second-model call. Review remarks and decisions are saved under `targeted_review/` in each document package. A separate compatible review endpoint can be supplied with `REDACTION_REVIEW_GATEWAY_URL` and `REDACTION_REVIEW_GATEWAY_TOKEN`.

The original PDF is never modified in place. Smoke or page-limited runs are explicitly release-blocked.

## Runtime separation

PaddleOCR 2.7/PaddlePaddle 2.6 runs in a persistent worker process. PP-DocLayoutV3 uses PaddleX 3/PaddlePaddle 3 in a separate package environment. Do not install both Paddle generations into the same environment.

Set `OCR_LEGACY_PYTHON` and `PADDLE_LAYOUT_PACKAGES` in the environment or pass `--ocr-python` and `--layout-packages`. PP-DocLayoutV3 weights must be cached at the path supplied through `--layout-cache`. Bosch Gateway credentials are supplied with `BOSCH_GATEWAY_URL` and `BOSCH_GATEWAY_TOKEN`.

Install the package with `pip install -e .`. The `ocr-redact` console command is the primary entry point; `run_integrated_redaction.py` remains as a source-checkout launcher. See `docs/README.md` for run and smoke-test instructions.
