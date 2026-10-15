"""Command-line entry point for the integrated original-PDF redaction flow."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

from .analyzers import PaddlePageAnalyzer
from .gateway import VisionGateway
from .runner import IntegratedRedactionRunner
from .stages import RedactionStages


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Original PDFs; recursively searched")
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.getenv("REDACTION_OUTPUT_ROOT", REPO_ROOT / "output" / "integrated_redaction")))
    parser.add_argument("--document", help="Exact PDF file name (with or without .pdf)")
    parser.add_argument("--original-only", action="store_true", help="Only ingest files named original_01.pdf")
    parser.add_argument("--limit", type=int, help="Process at most N PDFs")
    parser.add_argument("--smoke", action="store_true", help="Process first PDF and its first page only")
    parser.add_argument("--page-limit", type=int, help="Process first N pages of each PDF")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--layout-threshold", type=float, default=0.5)
    parser.add_argument("--classification-threshold", type=float, default=0.75)
    parser.add_argument("--extraction-model", default="claude-opus-5-5")
    parser.add_argument("--mapping-model", default="claude-opus-5-5")
    parser.add_argument("--classifier-model", default="claude-haiku-4-5@20251001")
    parser.add_argument("--material-model", default="claude-opus-5-5")
    parser.add_argument("--device", default="cpu", help="PP-DocLayoutV3 device, normally cpu")
    parser.add_argument("--ocr-python", type=Path,
                        default=Path(os.getenv("OCR_LEGACY_PYTHON", sys.executable)),
                        help="Python 2.x environment for PaddleOCR worker subprocess")
    parser.add_argument("--layout-packages", type=Path,
                        default=Path(os.environ["PADDLE_LAYOUT_PACKAGES"]) if os.getenv("PADDLE_LAYOUT_PACKAGES") else None,
                        help="Isolated PaddleX/Paddle 3 package directory")
    parser.add_argument("--layout-cache", type=Path,
                        default=Path.home() / ".paddlex" / "official_models" / "PP-DocLayoutV3",
                        help="Local PP-DocLayoutV3 model directory")
    args = parser.parse_args()

    if not args.input_root.exists():
        parser.error(f"Input root does not exist: {args.input_root}")
    if (args.dpi <= 0 or not 0.0 <= args.layout_threshold <= 1.0
            or not 0.0 <= args.classification_threshold <= 1.0):
        parser.error("dpi must be positive and both thresholds must be in [0, 1]")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be at least 1")
    if args.page_limit is not None and args.page_limit < 1:
        parser.error("page-limit must be at least 1")
    pdfs = sorted(args.input_root.rglob("*.pdf"))
    if args.original_only:
        pdfs = [path for path in pdfs if path.name.casefold() == "original_01.pdf"]
    output_resolved = args.output_root.resolve()
    pdfs = [path for path in pdfs if not path.resolve().is_relative_to(output_resolved)]
    if args.document:
        requested = args.document.casefold()
        if not requested.endswith(".pdf"):
            requested += ".pdf"
        pdfs = [path for path in pdfs if path.name.casefold() == requested]
    if args.smoke:
        pdfs = pdfs[:1]
        args.page_limit = 1
    if args.limit:
        pdfs = pdfs[:args.limit]
    if not pdfs:
        parser.error("No matching PDFs found")
    if args.layout_packages is None:
        parser.error("Set PADDLE_LAYOUT_PACKAGES or pass --layout-packages to the isolated PaddleX package directory")
    if not args.layout_cache.exists():
        parser.error(f"PP-DocLayoutV3 cache not found: {args.layout_cache}")

    batch_root = args.output_root / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    gateway = VisionGateway()
    analyzer = PaddlePageAnalyzer(
        dpi=args.dpi, layout_threshold=args.layout_threshold, device=args.device,
        ocr_python=args.ocr_python, layout_packages=args.layout_packages,
    )
    stages = RedactionStages(
        gateway, args.extraction_model, args.mapping_model,
        args.classifier_model, args.material_model,
        classification_threshold=args.classification_threshold,
    )
    runner = IntegratedRedactionRunner(analyzer, stages, text_redaction_mode=0)
    manifest = runner.run(pdfs, batch_root, page_limit=args.page_limit)
    print(f"Batch output: {batch_root}")
    print(f"Statuses: {manifest['statuses']}")
    print(f"Manifest: {batch_root / 'manifest.json'}")
    if manifest["statuses"]["ERROR"]:
        raise SystemExit(1)
