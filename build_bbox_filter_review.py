"""Build per-page visual evidence for the shared redaction bbox filter."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from redact import MAX_REDACTION_PAGE_HEIGHT_RATIO, filter_redaction_regions


REPO_ROOT = Path(__file__).resolve().parent
BATCH_ROOT = REPO_ROOT / "output" / "timestamped_batches"
REVIEW_FOLDER_NAME = "bbox_filter_review"


def latest_batch() -> Path:
    batches = sorted(path for path in BATCH_ROOT.glob("batch_*") if path.is_dir())
    if not batches:
        raise SystemExit(f"No timestamped batches found under {BATCH_ROOT}")
    return batches[-1]


def raw_boxes(page: dict) -> list[dict]:
    boxes = []
    for category, entities in page.get("entities", {}).items():
        for entity in entities or []:
            bbox = entity.get("bbox") if isinstance(entity, dict) else None
            if not bbox:
                continue
            try:
                boxes.append({
                    "page_number": int(page["page_number"]),
                    "category": category,
                    "value": str(entity.get("value", "")),
                    "x0": float(bbox["x0"]),
                    "y0": float(bbox["y0"]),
                    "x1": float(bbox["x1"]),
                    "y1": float(bbox["y1"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return boxes


def coordinates(box: dict) -> tuple[float, float, float, float]:
    return tuple(round(float(box[key]), 4) for key in ("x0", "y0", "x1", "y1"))


def clipped_coordinates(box: dict, page_rect: fitz.Rect) -> tuple[float, float, float, float]:
    rect = fitz.Rect(box["x0"], box["y0"], box["x1"], box["y1"]) & page_rect
    return tuple(round(value, 4) for value in (rect.x0, rect.y0, rect.x1, rect.y1))


def draw_boxes(image: Image.Image, boxes: list[dict], color: str) -> None:
    draw = ImageDraw.Draw(image)
    line_width = max(3, round(min(image.size) / 500))
    for box in boxes:
        rectangle = coordinates(box)
        draw.rectangle(rectangle, outline=color, width=line_width)


def draw_label(image: Image.Image, label: str, color: str) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 410), 34), fill="white")
    draw.text((8, 9), label, fill=color)


def images_to_pdf(image_paths: list[Path], output_path: Path) -> None:
    document = fitz.open()
    try:
        for image_path in image_paths:
            with Image.open(image_path) as image:
                width, height = image.size
            page = document.new_page(width=width, height=height)
            page.insert_image(page.rect, filename=str(image_path))
        document.save(str(output_path), garbage=4, deflate=True)
    finally:
        document.close()


def build_review(batch_dir: Path) -> dict:
    review_dir = batch_dir / REVIEW_FOLDER_NAME
    review_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    total_raw = 0
    total_kept = 0

    package_dirs = sorted(
        path for path in batch_dir.iterdir()
        if path.is_dir() and (path / "redaction_information.json").exists()
    )
    if not package_dirs:
        raise SystemExit(f"No redaction_information.json packages found in {batch_dir}")

    for package_dir in package_dirs:
        redaction_data = json.loads((package_dir / "redaction_information.json").read_text(encoding="utf-8"))
        document_name = redaction_data.get("document", package_dir.name)
        document_review_dir = review_dir / document_name
        document_review_dir.mkdir(parents=True, exist_ok=True)
        page_results = []
        source_pdf = Path(redaction_data["source_pdf"])

        with fitz.open(str(source_pdf)) as source_document:
            for page in sorted(redaction_data.get("pages", []), key=lambda item: item["page_label"]):
                page_label = page["page_label"]
                image_path = batch_dir / "ocr_pages" / package_dir.name / page_label / f"{page_label}.png"
                if not image_path.exists():
                    continue

                with Image.open(image_path) as source:
                    image_width, image_height = source.size
                page_number = int(page["page_number"])
                page_raw = raw_boxes(page)
                source_page = source_document[page_number]
                page_rect = source_page.rect
                scale_x = page_rect.width / image_width
                scale_y = page_rect.height / image_height
                pixmap = source_page.get_pixmap(
                    matrix=fitz.Matrix(image_width / page_rect.width, image_height / page_rect.height),
                    alpha=False,
                )
                source_image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                pdf_boxes = [{
                    **box,
                    "x0": box["x0"] * scale_x,
                    "y0": box["y0"] * scale_y,
                    "x1": box["x1"] * scale_x,
                    "y1": box["y1"] * scale_y,
                } for box in page_raw]
                page_kept_pdf = filter_redaction_regions(pdf_boxes, {page_number: page_rect})
                kept_coordinates = {coordinates(box) for box in page_kept_pdf}
                page_kept = [
                    box for box, pdf_box in zip(page_raw, pdf_boxes)
                    if clipped_coordinates(pdf_box, page_rect) in kept_coordinates
                ]
                page_removed = [box for box in page_raw if box not in page_kept]

                before_image = source_image.copy()
                draw_boxes(before_image, page_raw, "#d00000")
                draw_label(before_image, f"BEFORE FILTER: {len(page_raw)} boxes", "#d00000")
                before_name = f"{page_label}_before_all_boxes.png"
                before_image.save(document_review_dir / before_name)

                after_image = source_image.copy()
                draw_boxes(after_image, page_kept, "#008000")
                draw_label(after_image, f"AFTER FILTER: {len(page_kept)} boxes", "#008000")
                after_name = f"{page_label}_after_kept_boxes.png"
                after_image.save(document_review_dir / after_name)

                comparison_image = source_image.copy()
                draw_boxes(comparison_image, page_removed, "#d00000")
                draw_boxes(comparison_image, page_kept, "#008000")
                draw_label(
                    comparison_image,
                    f"GREEN KEPT: {len(page_kept)} | RED REMOVED: {len(page_removed)}",
                    "#202124",
                )
                comparison_name = f"{page_label}_comparison.png"
                comparison_image.save(document_review_dir / comparison_name)

                page_results.append({
                    "page_label": page_label,
                    "raw_boxes": len(page_raw),
                    "kept_boxes": len(page_kept),
                    "removed_boxes": len(page_removed),
                    "before_image": before_name,
                    "after_image": after_name,
                    "comparison_image": comparison_name,
                })
                total_raw += len(page_raw)
                total_kept += len(page_kept)

            after_pdf = document_review_dir / "whole_document_after_filter.pdf"
            comparison_pdf = document_review_dir / "whole_document_filter_comparison.pdf"
            images_to_pdf([document_review_dir / page["after_image"] for page in page_results], after_pdf)
            images_to_pdf([document_review_dir / page["comparison_image"] for page in page_results], comparison_pdf)
            documents.append({
                "document": document_name,
                "pages": page_results,
                "after_filter_pdf": after_pdf.name,
                "comparison_pdf": comparison_pdf.name,
            })

    summary = {
        "batch_id": batch_dir.name,
        "coordinate_space": "OCR image boxes scaled to source PDF coordinates before filtering",
        "filter": {
            "method": "corpus-derived extreme page-height outlier",
            "max_height_percent": MAX_REDACTION_PAGE_HEIGHT_RATIO * 100,
            "remove_enclosing_boxes": False,
        },
        "raw_boxes": total_raw,
        "kept_boxes": total_kept,
        "removed_boxes": total_raw - total_kept,
        "documents": documents,
    }
    (review_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_index(review_dir, summary)
    return summary


def write_index(review_dir: Path, summary: dict) -> None:
    sections = []
    for document in summary["documents"]:
        rows = []
        document_name = html.escape(document["document"])
        folder = html.escape(document["document"], quote=True)
        for page in document["pages"]:
            comparison = html.escape(page["comparison_image"], quote=True)
            before = html.escape(page["before_image"], quote=True)
            after = html.escape(page["after_image"], quote=True)
            rows.append(
                f'<article><h3>{html.escape(page["page_label"])}: '
                f'{page["raw_boxes"]} raw, {page["kept_boxes"]} kept, {page["removed_boxes"]} removed</h3>'
                f'<a href="{folder}/{comparison}"><img src="{folder}/{comparison}" alt="Filter comparison"></a>'
                f'<p><a href="{folder}/{before}">Before</a> | <a href="{folder}/{after}">After</a></p></article>'
            )
        after_pdf = html.escape(document["after_filter_pdf"], quote=True)
        comparison_pdf = html.escape(document["comparison_pdf"], quote=True)
        sections.append(
            f'<section><h2>{document_name}</h2><p><a href="{folder}/{after_pdf}">Whole document after filter</a> | '
            f'<a href="{folder}/{comparison_pdf}">Whole document comparison</a></p>{"".join(rows)}</section>'
        )

    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bbox filter review</title><style>
body {{ font-family: Segoe UI, sans-serif; margin: 24px; color: #202124; }}
.summary {{ padding: 12px; border: 2px solid #202124; }}
article {{ margin: 28px 0; border-top: 1px solid #bbb; }}
img {{ display: block; width: min(100%, 1000px); height: auto; border: 1px solid #777; }}
a {{ color: #005ea8; }}
</style></head><body><h1>Bbox filter review: {html.escape(summary["batch_id"])}</h1>
<p class="summary"><b>{summary["raw_boxes"]} raw</b> | <b style="color:#008000">{summary["kept_boxes"]} kept</b> | <b style="color:#d00000">{summary["removed_boxes"]} removed</b><br>
Comparison images use green for retained boxes and red for filtered boxes.</p>{''.join(sections)}</body></html>"""
    (review_dir / "index.html").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", nargs="?", type=Path, help="Timestamped batch folder (defaults to latest).")
    args = parser.parse_args()
    batch_dir = args.batch.resolve() if args.batch else latest_batch()
    summary = build_review(batch_dir)
    print(json.dumps({
        "review_folder": str(batch_dir / REVIEW_FOLDER_NAME),
        "raw_boxes": summary["raw_boxes"],
        "kept_boxes": summary["kept_boxes"],
        "removed_boxes": summary["removed_boxes"],
    }, indent=2))


if __name__ == "__main__":
    main()