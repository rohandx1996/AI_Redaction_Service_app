"""Build raw PaddleOCR bbox review PDFs with no LLM data or bbox filtering."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import fitz
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REPO_ROOT / "output" / "batch" / "paddle_dbnet_svtr_pages"
BATCH_ROOT = REPO_ROOT / "output" / "timestamped_batches"
CONTAINMENT_TOLERANCE = 2.0
HUGE_PARENT_AREA_RATIO = 4.0


def latest_source_manifest() -> Path:
    manifests = sorted(BATCH_ROOT.glob("bbox_height_review_*/manifest.json"))
    if not manifests:
        raise SystemExit("No prior bbox height review manifest found")
    return manifests[-1]


def load_page_records(document_dir: Path) -> dict[int, dict]:
    records = {}
    for page_dir in sorted(path for path in document_dir.iterdir() if path.is_dir() and path.name.startswith("page_")):
        label = page_dir.name
        page_json = page_dir / f"{label}.json"
        image_path = page_dir / f"{label}.png"
        if not page_json.exists() or not image_path.exists():
            continue
        try:
            data = json.loads(page_json.read_text(encoding="utf-8"))
            with Image.open(image_path) as image:
                image_size = image.size
        except (OSError, json.JSONDecodeError):
            continue
        page_number = int(data.get("page_number", 0))
        records[page_number] = {
            "label": label,
            "source_pdf": data.get("source_pdf"),
            "image_size": image_size,
            "spans": data.get("spans", []),
        }
    return records


def scaled_ocr_rects(record: dict, page_rect: fitz.Rect) -> list[fitz.Rect]:
    image_width, image_height = record["image_size"]
    scale_x = page_rect.width / image_width
    scale_y = page_rect.height / image_height
    rectangles = []
    for span in record["spans"]:
        polygon = span.get("polygon") or []
        if not polygon:
            continue
        try:
            x_values = [float(point[0]) for point in polygon]
            y_values = [float(point[1]) for point in polygon]
        except (IndexError, TypeError, ValueError):
            continue
        rect = fitz.Rect(
            min(x_values) * scale_x,
            min(y_values) * scale_y,
            max(x_values) * scale_x,
            max(y_values) * scale_y,
        ) & page_rect
        if not rect.is_empty and rect.width > 0 and rect.height > 0:
            rectangles.append(rect)
    return rectangles


def filter_rectangles(rectangles: list[fitz.Rect]) -> tuple[list[fitz.Rect], list[dict]]:
    """Apply only exact duplicate, nested, and huge-parent rules to OCR boxes."""
    unique = []
    removed = []
    seen = set()
    for rect in rectangles:
        rect_key = tuple(round(value, 2) for value in (rect.x0, rect.y0, rect.x1, rect.y1))
        if rect_key in seen:
            removed.append({"rect": rect, "removal_reason": "EXACT_DUPLICATE"})
        else:
            seen.add(rect_key)
            unique.append(rect)

    kept = []
    for rect in unique:
        contained = [
            other for other in unique
            if other is not rect
            and rect.x0 <= other.x0 + CONTAINMENT_TOLERANCE
            and rect.y0 <= other.y0 + CONTAINMENT_TOLERANCE
            and rect.x1 >= other.x1 - CONTAINMENT_TOLERANCE
            and rect.y1 >= other.y1 - CONTAINMENT_TOLERANCE
        ]
        if not contained:
            kept.append(rect)
            continue
        largest_child_area = max(other.get_area() for other in contained)
        reason = "HUGE_PARENT_BOX" if rect.get_area() >= largest_child_area * HUGE_PARENT_AREA_RATIO else "NESTED_BOX"
        removed.append({"rect": rect, "removal_reason": reason, "contained_box_count": len(contained)})
    return kept, removed


def build_review(document_name: str, output_dir: Path) -> dict:
    document_dir = CORPUS_ROOT / document_name
    records = load_page_records(document_dir)
    source_pdf = next((record["source_pdf"] for record in records.values() if record["source_pdf"]), None)
    if not source_pdf or not Path(source_pdf).exists():
        raise FileNotFoundError(f"Source PDF unavailable for {document_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "whole_document_paddleocr_bboxes.pdf"
    review = fitz.open()
    page_results = []
    raw_box_total = 0
    box_total = 0
    removed_by_reason = {"EXACT_DUPLICATE": 0, "NESTED_BOX": 0, "HUGE_PARENT_BOX": 0}
    with fitz.open(source_pdf) as source:
        for page_number, source_page in enumerate(source):
            page_rect = source_page.rect
            review_page = review.new_page(width=page_rect.width, height=page_rect.height)
            review_page.show_pdf_page(review_page.rect, source, page_number)
            record = records.get(page_number)
            raw_rectangles = scaled_ocr_rects(record, page_rect) if record else []
            rectangles, removed = filter_rectangles(raw_rectangles)
            for item in removed:
                review_page.draw_rect(item["rect"], color=(1, 0, 0), width=1.5)
                removed_by_reason[item["removal_reason"]] += 1
            for rect in rectangles:
                review_page.draw_rect(rect, color=(0, 0.55, 0), width=1.5)
            review_page.insert_text(
                (18, 20),
                f"OCR RAW: {len(raw_rectangles)} | GREEN KEPT: {len(rectangles)} | RED REMOVED: {len(removed)}",
                fontsize=10,
                color=(0, 0, 0),
            )
            raw_box_total += len(raw_rectangles)
            box_total += len(rectangles)
            page_results.append({"page_number": page_number + 1, "raw_boxes": len(raw_rectangles), "kept_boxes": len(rectangles), "removed_boxes": len(removed)})
    review.save(str(output_pdf), garbage=4, deflate=True)
    review.close()

    result = {
        "document": document_name,
        "source_pdf": source_pdf,
        "page_count": len(page_results),
        "raw_paddleocr_boxes": raw_box_total,
        "paddleocr_boxes_after_post_processing": box_total,
        "bbox_filter_used": "exact_duplicate_nested_huge_parent_only",
        "removed_by_reason": removed_by_reason,
        "litellm_used": False,
        "review_pdf": str(output_pdf),
        "pages": page_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-batch", type=Path, help="Use completed document names from this timestamped batch.")
    args = parser.parse_args()
    if bool(args.source_manifest) == bool(args.source_batch):
        raise SystemExit("Specify exactly one of --source-manifest or --source-batch")
    if args.source_batch:
        document_names = sorted(path.name for path in args.source_batch.iterdir() if (path / "recognition.json").exists())
    else:
        source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        document_names = [item["document"] for item in source_manifest.get("documents", [])]
    if not document_names:
        raise SystemExit(f"No documents listed in {args.source_manifest}")

    batch_id = datetime.now().strftime("paddle_ocr_structure_review_%Y%m%d_%H%M%S")
    batch_dir = BATCH_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    results = [build_review(name, batch_dir / name) for name in document_names]
    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bbox_source": "PaddleOCR page_NNNN.json spans[].polygon",
        "bbox_filter_used": "exact_duplicate_nested_huge_parent_only",
        "litellm_used": False,
        "document_count": len(results),
        "page_count": sum(result["page_count"] for result in results),
        "raw_paddleocr_boxes": sum(result["raw_paddleocr_boxes"] for result in results),
        "paddleocr_boxes_after_post_processing": sum(result["paddleocr_boxes_after_post_processing"] for result in results),
        "documents": results,
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()