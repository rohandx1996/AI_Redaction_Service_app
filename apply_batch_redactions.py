"""Apply completed page-level redaction bboxes to source PDFs.

This stage does not call the LLM. It reads *_redactions.json files created by
identify_redactions.py, converts rendered-image coordinates to PDF coordinates,
and writes one *_redacted.pdf per source document.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import fitz
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from redact import redact_pdf
from release_audit import write_release_audit

REPO_ROOT = Path(__file__).resolve().parent
OCR_ROOT = REPO_ROOT / "output" / "batch" / "paddle_dbnet_svtr_pages"
OUTPUT_ROOT = REPO_ROOT / "output" / "redacted_documents"
LOG_ROOT = REPO_ROOT / "logs"
REPORT_ROOT = REPO_ROOT / "output" / "reports"
console = Console()


def to_windows_long_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    resolved = path.resolve(strict=False)
    path_text = str(resolved)
    if path_text.startswith("\\\\?\\") or len(path_text) < 260:
        return resolved
    return Path("\\\\?\\" + path_text)


def page_dirs(doc_dir: Path) -> list[Path]:
    return sorted(
        (path for path in doc_dir.iterdir() if path.is_dir() and path.name.startswith("page_")),
        key=lambda path: path.name,
    )


def bbox_regions(record: dict, image_path: Path, pdf_page: fitz.Page) -> tuple[list[dict], int]:
    with Image.open(to_windows_long_path(image_path)) as image:
        image_width, image_height = image.size
    if not image_width or not image_height:
        return [], 0

    scale_x = pdf_page.rect.width / image_width
    scale_y = pdf_page.rect.height / image_height
    regions = []
    entity_count = 0
    for entries in record.get("entities", {}).values():
        for entity in entries or []:
            bbox = entity.get("bbox")
            if not bbox:
                continue
            try:
                rect = fitz.Rect(
                    float(bbox["x0"]) * scale_x,
                    float(bbox["y0"]) * scale_y,
                    float(bbox["x1"]) * scale_x,
                    float(bbox["y1"]) * scale_y,
                ) & pdf_page.rect
            except (KeyError, TypeError, ValueError):
                continue
            if rect.is_empty or rect.width <= 0 or rect.height <= 0:
                continue
            regions.append({
                "page_number": int(record.get("page_number", 0)),
                "x0": rect.x0,
                "y0": rect.y0,
                "x1": rect.x1,
                "y1": rect.y1,
            })
            entity_count += 1
    return regions, entity_count


def process_document(doc_dir: Path, output_dir: Path, logger: logging.Logger) -> dict:
    records = []
    source_pdf = None
    for page_dir in page_dirs(doc_dir):
        ocr_path = to_windows_long_path(page_dir / f"{page_dir.name}.json")
        if ocr_path.exists() and source_pdf is None:
            source_pdf = json.loads(ocr_path.read_text(encoding="utf-8")).get("source_pdf")
        redaction_path = to_windows_long_path(page_dir / f"{page_dir.name}_redactions.json")
        if redaction_path.exists():
            records.append((page_dir, json.loads(redaction_path.read_text(encoding="utf-8"))))

    if not source_pdf:
        raise FileNotFoundError(f"No source_pdf found for {doc_dir.name}")
    source_path = to_windows_long_path(Path(source_pdf))
    if not source_path.exists():
        raise FileNotFoundError(f"Source PDF not found: {source_pdf}")

    output_path = to_windows_long_path(output_dir / f"{doc_dir.name}_redacted.pdf")
    if output_path.exists():
        output_path.unlink()

    all_regions = []
    pages_with_redactions = 0
    entities = 0
    with fitz.open(str(source_path)) as document:
        for page_dir, record in records:
            page_number = int(record.get("page_number", 0))
            if page_number < 0 or page_number >= len(document):
                logger.warning("Invalid page number %s in %s", page_number, page_dir)
                continue
            png_path = to_windows_long_path(page_dir / f"{page_dir.name}.png")
            if not png_path.exists():
                logger.warning("Missing page image: %s", png_path)
                continue
            regions, count = bbox_regions(record, png_path, document[page_number])
            all_regions.extend(regions)
            if regions:
                pages_with_redactions += 1
                entities += count

    audit = redact_pdf(source_path, output_path, all_regions)
    logger.info("OK | %s | pages=%d | entities=%d | output=%s | audit=%s", doc_dir.name, pages_with_redactions, entities, output_path, audit.get("status"))
    return {
        "document": doc_dir.name,
        "source_pdf": str(source_path),
        "output_pdf": str(output_path),
        "pages_with_redactions": pages_with_redactions,
        "entities": entities,
        "status": "OK",
        "error": "",
        "release_file": str(output_path),
        "release_audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", help="Process one OCR document folder.")
    args = parser.parse_args()

    if not OCR_ROOT.exists():
        raise SystemExit(f"OCR output root not found: {OCR_ROOT}")
    doc_dirs = sorted(path for path in OCR_ROOT.iterdir() if path.is_dir())
    if args.doc:
        doc_dirs = [path for path in doc_dirs if path.name == args.doc]
        if not doc_dirs:
            raise SystemExit(f"Document folder not found: {args.doc}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"apply_batch_redactions_{run_id}.log"
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ])
    logger = logging.getLogger("apply_batch_redactions")
    rows = []

    with Progress(TextColumn("Applying PDF redactions"), BarColumn(), MofNCompleteColumn(), console=console) as progress:
        task = progress.add_task("documents", total=len(doc_dirs))
        for doc_dir in doc_dirs:
            try:
                rows.append(process_document(doc_dir, OUTPUT_ROOT, logger))
            except Exception as error:
                logger.exception("FAIL | %s | %s", doc_dir.name, error)
                rows.append({
                    "document": doc_dir.name, "source_pdf": "", "output_pdf": "",
                    "pages_with_redactions": 0, "entities": 0,
                    "status": "ERROR", "error": str(error),
                })
            progress.advance(task)

    report_path = REPORT_ROOT / "pdf_redaction_summary.csv"
    csv_fields = ["document", "source_pdf", "output_pdf", "pages_with_redactions", "entities", "status", "error"]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in csv_fields} for row in rows)

    audit_documents = [row for row in rows if row.get("release_audit")]
    audit_path = REPORT_ROOT / f"release_audit_{run_id}.xlsx"
    if audit_documents:
        write_release_audit(audit_path, audit_documents)

    ok = sum(row["status"] == "OK" for row in rows)
    errors = len(rows) - ok
    console.print(f"Completed: {ok}/{len(rows)} documents | errors={errors}")
    console.print(f"Redacted PDFs: {OUTPUT_ROOT}")
    console.print(f"Summary: {report_path}")
    if audit_documents:
        console.print(f"Release audit: {audit_path}")
    console.print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
