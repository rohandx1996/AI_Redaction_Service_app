"""Fresh OCR + multilingual review/repair + redaction batch for original PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

import run_timestamped_batch as pipeline

SOURCE_ROOT = Path(r"C:\Users\DHR1RT\Downloads\OneDrive_1_9-7-2026\02_Original")
BATCH_ROOT = Path(__file__).resolve().parent / "output" / "timestamped_batches"
console = Console()

POLICY_VERSION = "published-evidence-v1"


def long_path(path: Path) -> Path:
    return pipeline.long_path(path)


def document_name(pdf_path: Path) -> str:
    relative = str(pdf_path.relative_to(SOURCE_ROOT)).replace("\\", "__")
    safe = "".join(char if char.isalnum() else "_" for char in relative)
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:180]}_{digest}"


def build_ocr_pages(ocr, pdf_path: Path, output_root: Path, logger: logging.Logger) -> tuple[str, int]:
    name = document_name(pdf_path)
    document_dir = long_path(output_root / name)
    document_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(str(long_path(pdf_path))) as document:
        for page_number in range(len(document)):
            label = f"page_{page_number + 1:04d}"
            page_dir = long_path(document_dir / label)
            page_dir.mkdir(parents=True, exist_ok=True)
            json_path = long_path(page_dir / f"{label}.json")
            png_path = long_path(page_dir / f"{label}.png")
            if json_path.exists() and png_path.exists():
                continue
            page = document[page_number]
            pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            result = ocr.ocr(np.asarray(image), cls=False)
            detections = result[0] if result and result[0] else []
            draw = ImageDraw.Draw(image)
            spans = []
            for index, detection in enumerate(detections):
                polygon, recognition = detection
                text, confidence = recognition
                polygon = [[float(point[0]), float(point[1])] for point in polygon]
                draw.polygon([tuple(point) for point in polygon], outline="red", width=2)
                draw.text(tuple(polygon[0]), str(index), fill="blue")
                spans.append({
                    "index": index,
                    "text": text,
                    "confidence": float(confidence),
                    "polygon": polygon,
                })
            json_path.write_text(json.dumps({
                "model": "PaddleOCR DBNet + SVTR_LCNet",
                "ocr_language": "en",
                "source_pdf": str(pdf_path),
                "page_number": page_number,
                "render_dpi": 150,
                "image_width": pix.width,
                "image_height": pix.height,
                "spans": spans,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            image.save(png_path)
    logger.info("OCR_OK | %s | pages=%d", name, len(fitz.open(str(long_path(pdf_path)))))
    return name, len(fitz.open(str(long_path(pdf_path))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--document-limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--match-root", type=Path, help="Process original_01.pdf files under a matched-counterparts folder.")
    args = parser.parse_args()
    global SOURCE_ROOT
    source_root = args.match_root.resolve() if args.match_root else SOURCE_ROOT
    if not source_root.exists():
        raise SystemExit(f"Source root not found: {source_root}")
    SOURCE_ROOT = source_root
    if not pipeline.settings.bosch_gateway_token or not pipeline.settings.bosch_gateway_url:
        raise SystemExit("Bosch Gateway credentials not configured (.env).")
    pipeline.patch_tiktoken()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = BATCH_ROOT / f"batch_{run_id}"
    ocr_root = batch_dir / "ocr_pages"
    batch_dir.mkdir(parents=True, exist_ok=True)
    pipeline.OCR_ROOT = ocr_root.resolve()
    log_path = batch_dir / "batch.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(log_path, encoding="utf-8")])
    logger = logging.getLogger("original_redaction_batch")

    pdfs = sorted(SOURCE_ROOT.rglob("original_01.pdf")) if args.match_root else sorted(SOURCE_ROOT.rglob("*.pdf"))
    if args.document_limit:
        pdfs = pdfs[:args.document_limit]
    if args.smoke:
        pdfs = pdfs[:1]
    page_counts = []
    for pdf_path in pdfs:
        with fitz.open(str(long_path(pdf_path))) as document:
            page_counts.append(len(document))
    total_pages = sum(page_counts)
    logger.info("START | source=%s | documents=%d | pages=%d | policy=%s", SOURCE_ROOT, len(pdfs), total_pages, POLICY_VERSION)

    import paddle
    from paddleocr import PaddleOCR
    try:
        paddle.set_flags({"FLAGS_use_mkldnn": False, "FLAGS_use_onednn": False})
    except ValueError:
        paddle.set_flags({"FLAGS_use_mkldnn": False})
    ocr = PaddleOCR(det_algorithm="DB", rec_algorithm="SVTR_LCNet", use_gpu=False, lang="en", use_angle_cls=False)

    stats = {"pages_done": 0, "total_pages": total_pages, "ocr_total": 0.0, "llm_total": 0.0, "llm_calls": 0, "review_total": 0.0, "repair_total": 0.0, "suspicious_total": 0}
    lock = threading.Lock()
    results = []
    with Progress(TextColumn("Original redaction batch"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        doc_task = progress.add_task("documents", total=len(pdfs), status="starting")
        page_task = progress.add_task("pages", total=total_pages, status=f"pages=0/{total_pages}")
        for index, pdf_path in enumerate(pdfs):
            try:
                name, _ = build_ocr_pages(ocr, pdf_path, ocr_root, logger)
                result = pipeline.process_document(name, batch_dir, args.workers, logger, progress, page_task, stats, lock)
                result["source_pdf"] = str(pdf_path)
                result["policy_version"] = POLICY_VERSION
                results.append(result)
            except Exception as error:
                logger.exception("DOCUMENT_FAIL | %s | %s", pdf_path, error)
                results.append({"document": document_name(pdf_path), "source_pdf": str(pdf_path), "status": "ERROR", "error": str(error), "pages": 0, "boxes": 0, "redacted_boxes": 0})
            progress.advance(doc_task)
            progress.update(doc_task, description=f"documents | done={index + 1}/{len(pdfs)}")

    manifest = {
        "batch_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(SOURCE_ROOT), "document_count": len(pdfs), "page_count": total_pages,
        "ocr_language": "en", "render_dpi": 150, "workers": args.workers,
        "policy_version": POLICY_VERSION, "documents": results,
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Batch complete: {batch_dir}")
    console.print(f"Manifest: {batch_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
