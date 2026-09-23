"""Build timestamped full-document bbox reviews from the PaddleOCR corpus."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import fitz
from PIL import Image

from redact import MAX_REDACTION_PAGE_HEIGHT_RATIO, filter_redaction_regions


REPO_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REPO_ROOT / "output" / "batch" / "paddle_dbnet_svtr_pages"
BATCH_ROOT = REPO_ROOT / "output" / "timestamped_batches"


def page_records(document_dir: Path) -> list[dict]:
    records = []
    for page_dir in sorted(path for path in document_dir.iterdir() if path.is_dir() and path.name.startswith("page_")):
        label = page_dir.name
        page_json = page_dir / f"{label}.json"
        redaction_json = page_dir / f"{label}_redactions.json"
        image_path = page_dir / f"{label}.png"
        if not page_json.exists() or not redaction_json.exists() or not image_path.exists():
            continue
        try:
            page_data = json.loads(page_json.read_text(encoding="utf-8"))
            redaction_data = json.loads(redaction_json.read_text(encoding="utf-8"))
            with Image.open(image_path) as image:
                image_size = image.size
        except (OSError, json.JSONDecodeError):
            continue
        records.append({
            "label": label,
            "page_number": int(page_data.get("page_number", 0)),
            "source_pdf": page_data.get("source_pdf"),
            "image_size": image_size,
            "redactions": redaction_data,
        })
    return records


def image_regions(record: dict) -> list[dict]:
    regions = []
    for entities in record["redactions"].get("entities", {}).values():
        for entity in entities or []:
            bbox = entity.get("bbox") if isinstance(entity, dict) else None
            if not bbox:
                continue
            try:
                regions.append({
                    "page_number": record["page_number"],
                    "x0": float(bbox["x0"]),
                    "y0": float(bbox["y0"]),
                    "x1": float(bbox["x1"]),
                    "y1": float(bbox["y1"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return regions


def scaled_regions(record: dict, page_rect: fitz.Rect) -> list[dict]:
    image_width, image_height = record["image_size"]
    scale_x = page_rect.width / image_width
    scale_y = page_rect.height / image_height
    regions = []
    for region in image_regions(record):
        rect = fitz.Rect(
            region["x0"] * scale_x,
            region["y0"] * scale_y,
            region["x1"] * scale_x,
            region["y1"] * scale_y,
        ) & page_rect
        regions.append({
            "page_number": record["page_number"],
            "x0": rect.x0,
            "y0": rect.y0,
            "x1": rect.x1,
            "y1": rect.y1,
        })
    return regions


def region_key(region: dict) -> tuple[float, float, float, float]:
    return tuple(round(float(region[key]), 4) for key in ("x0", "y0", "x1", "y1"))


def count_removed(records: list[dict]) -> int:
    removed = 0
    for record in records:
        width, height = record["image_size"]
        regions = image_regions(record)
        kept = filter_redaction_regions(
            regions,
            {record["page_number"]: fitz.Rect(0, 0, width, height)},
        )
        removed += len(regions) - len(kept)
    return removed


def build_document_review(document_dir: Path, records: list[dict], output_dir: Path) -> dict:
    source_pdf = next((record["source_pdf"] for record in records if record["source_pdf"]), None)
    if not source_pdf or not Path(source_pdf).exists():
        raise FileNotFoundError(f"Source PDF unavailable for {document_dir.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "whole_document_bbox_height_review.pdf"
    record_by_page = {record["page_number"]: record for record in records}
    review = fitz.open()
    raw_total = 0
    kept_total = 0
    page_results = []

    with fitz.open(source_pdf) as source:
        for page_number, source_page in enumerate(source):
            page_rect = source_page.rect
            review_page = review.new_page(width=page_rect.width, height=page_rect.height)
            review_page.show_pdf_page(review_page.rect, source, page_number)
            record = record_by_page.get(page_number)
            regions = scaled_regions(record, page_rect) if record else []
            kept = filter_redaction_regions(regions, {page_number: page_rect})
            kept_keys = {region_key(region) for region in kept}
            page_kept = 0
            page_removed = 0
            for region in regions:
                is_kept = region_key(region) in kept_keys
                review_page.draw_rect(
                    fitz.Rect(region["x0"], region["y0"], region["x1"], region["y1"]),
                    color=(0, 0.65, 0) if is_kept else (1, 0, 0),
                    width=2,
                )
                if is_kept:
                    page_kept += 1
                else:
                    page_removed += 1
            review_page.insert_text(
                (18, 20),
                f"GREEN KEPT: {page_kept} | RED REMOVED BY HEIGHT: {page_removed}",
                fontsize=10,
                color=(0, 0, 0),
            )
            raw_total += len(regions)
            kept_total += page_kept
            page_results.append({
                "page_number": page_number + 1,
                "raw_boxes": len(regions),
                "kept_boxes": page_kept,
                "removed_boxes": page_removed,
            })

    review.save(str(output_pdf), garbage=4, deflate=True)
    review.close()
    result = {
        "document": document_dir.name,
        "source_pdf": source_pdf,
        "page_count": len(page_results),
        "raw_boxes": raw_total,
        "kept_boxes": kept_total,
        "removed_boxes": raw_total - kept_total,
        "review_pdf": str(output_pdf),
        "pages": page_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=5, help="Number of documents with outliers to review.")
    args = parser.parse_args()
    if args.documents < 1:
        raise SystemExit("--documents must be at least 1")

    candidates = []
    for document_dir in sorted(path for path in CORPUS_ROOT.iterdir() if path.is_dir()):
        records = page_records(document_dir)
        removed = count_removed(records)
        if removed:
            candidates.append((removed, document_dir, records))
    candidates.sort(key=lambda item: (-item[0], item[1].name))
    selected = candidates[:args.documents]
    if len(selected) < args.documents:
        raise SystemExit(f"Only {len(selected)} documents contain height outliers")

    batch_id = datetime.now().strftime("bbox_height_review_%Y%m%d_%H%M%S")
    batch_dir = BATCH_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    results = [
        build_document_review(document_dir, records, batch_dir / document_dir.name)
        for _, document_dir, records in selected
    ]
    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(CORPUS_ROOT),
        "filter": "bbox height only",
        "max_height_percent": MAX_REDACTION_PAGE_HEIGHT_RATIO * 100,
        "document_count": len(results),
        "page_count": sum(result["page_count"] for result in results),
        "raw_boxes": sum(result["raw_boxes"] for result in results),
        "kept_boxes": sum(result["kept_boxes"] for result in results),
        "removed_boxes": sum(result["removed_boxes"] for result in results),
        "documents": results,
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()