"""Review exact duplicates, nested boxes, and huge parent boxes in an existing batch."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from aggregate_ocr_json import OCR_ROOT


REPO_ROOT = Path(__file__).resolve().parent
BATCH_ROOT = REPO_ROOT / "output" / "timestamped_batches"
REVIEW_FOLDER_NAME = "bbox_structure_review"
CONTAINMENT_TOLERANCE = 2.0
HUGE_PARENT_AREA_RATIO = 4.0


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
                    "x0": float(bbox["x0"]), "y0": float(bbox["y0"]),
                    "x1": float(bbox["x1"]), "y1": float(bbox["y1"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return boxes


def key(box: dict) -> tuple[float, float, float, float]:
    return tuple(round(box[axis], 2) for axis in ("x0", "y0", "x1", "y1"))


def area(box: dict) -> float:
    return max(0.0, box["x1"] - box["x0"]) * max(0.0, box["y1"] - box["y0"])


def contains(parent: dict, child: dict) -> bool:
    return (
        parent["x0"] <= child["x0"] + CONTAINMENT_TOLERANCE
        and parent["y0"] <= child["y0"] + CONTAINMENT_TOLERANCE
        and parent["x1"] >= child["x1"] - CONTAINMENT_TOLERANCE
        and parent["y1"] >= child["y1"] - CONTAINMENT_TOLERANCE
    )


def filter_boxes(boxes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep one exact box and reject enclosing candidates only when clearly oversized."""
    kept = []
    removed = []
    seen = set()
    for box in boxes:
        box_key = key(box)
        if box_key in seen:
            removed.append({**box, "removal_reason": "EXACT_DUPLICATE"})
        else:
            seen.add(box_key)
            kept.append(box)

    final_kept = []
    for candidate in kept:
        contained = [box for box in kept if box is not candidate and contains(candidate, box)]
        if not contained:
            final_kept.append(candidate)
            continue
        largest_child_area = max(area(box) for box in contained)
        candidate_area = area(candidate)
        if candidate_area >= largest_child_area * HUGE_PARENT_AREA_RATIO:
            removed.append({
                **candidate,
                "removal_reason": "HUGE_PARENT_BOX",
                "contained_box_count": len(contained),
                "area_ratio_to_largest_child": round(candidate_area / largest_child_area, 2),
            })
        else:
            removed.append({
                **candidate,
                "removal_reason": "NESTED_BOX",
                "contained_box_count": len(contained),
            })
    return final_kept, removed


def draw_boxes(image: Image.Image, boxes: list[dict], color: str) -> None:
    draw = ImageDraw.Draw(image)
    width = max(3, round(min(image.size) / 500))
    for box in boxes:
        draw.rectangle((box["x0"], box["y0"], box["x1"], box["y1"]), outline=color, width=width)


def write_pdf(image_paths: list[Path], output_path: Path) -> None:
    output = fitz.open()
    try:
        for image_path in image_paths:
            with Image.open(image_path) as image:
                page = output.new_page(width=image.width, height=image.height)
            page.insert_image(page.rect, filename=str(image_path))
        output.save(str(output_path), garbage=4, deflate=True)
    finally:
        output.close()


def build_review(batch_dir: Path) -> dict:
    review_dir = batch_dir / REVIEW_FOLDER_NAME
    review_dir.mkdir(parents=True, exist_ok=True)
    package_dirs = sorted(path for path in batch_dir.iterdir() if (path / "redaction_information.json").exists())
    if not package_dirs:
        raise SystemExit(f"No completed redaction packages found in {batch_dir}")

    documents = []
    all_removed = []
    total_raw = total_kept = 0
    for package_dir in package_dirs:
        data = json.loads((package_dir / "redaction_information.json").read_text(encoding="utf-8"))
        document_name = data.get("document", package_dir.name)
        document_dir = review_dir / document_name
        document_dir.mkdir(parents=True, exist_ok=True)
        page_results = []
        with fitz.open(data["source_pdf"]) as source_pdf:
            for page in sorted(data.get("pages", []), key=lambda item: item["page_label"]):
                boxes = raw_boxes(page)
                kept, removed = filter_boxes(boxes)
                page_number = int(page["page_number"])
                source_page = source_pdf[page_number]
                ocr_image_path = OCR_ROOT / document_name / page["page_label"] / f"{page['page_label']}.png"
                with Image.open(ocr_image_path) as ocr_image:
                    image_width, image_height = ocr_image.size
                rendered = source_page.get_pixmap(
                    matrix=fitz.Matrix(image_width / source_page.rect.width, image_height / source_page.rect.height),
                    alpha=False,
                )
                image = Image.frombytes("RGB", (rendered.width, rendered.height), rendered.samples)
                draw_boxes(image, kept, "#008000")
                draw_boxes(image, removed, "#d00000")
                image_name = f"{page['page_label']}_comparison.png"
                image.save(document_dir / image_name)
                after_image = Image.frombytes("RGB", (rendered.width, rendered.height), rendered.samples)
                draw_boxes(after_image, kept, "#008000")
                after_image_name = f"{page['page_label']}_after_post_processing.png"
                after_image.save(document_dir / after_image_name)
                page_results.append({
                    "page_label": page["page_label"], "raw_boxes": len(boxes),
                    "kept_boxes": len(kept), "removed_boxes": len(removed), "comparison_image": image_name,
                    "after_image": after_image_name,
                    "removed": removed,
                })
                all_removed.extend({"document": document_name, "page_label": page["page_label"], **box} for box in removed)
                total_raw += len(boxes)
                total_kept += len(kept)
        pdf_path = document_dir / "whole_document_comparison.pdf"
        write_pdf([document_dir / page["comparison_image"] for page in page_results], pdf_path)
        after_pdf_path = document_dir / "whole_document_after_post_processing.pdf"
        write_pdf([document_dir / page["after_image"] for page in page_results], after_pdf_path)
        documents.append({
            "document": document_name,
            "pages": page_results,
            "comparison_pdf": pdf_path.name,
            "after_post_processing_pdf": after_pdf_path.name,
        })

    reasons = Counter(box["removal_reason"] for box in all_removed)
    summary = {
        "batch_id": batch_dir.name,
        "purpose": "non-destructive candidate review; source redaction JSON is unchanged",
        "rules": {
            "exact_duplicate": "same page and rounded coordinates; retain the first occurrence",
            "nested_box": "candidate contains another candidate within 2 pixels",
            "huge_parent_box": f"nested candidate area is at least {HUGE_PARENT_AREA_RATIO:g}x its largest contained candidate",
        },
        "raw_boxes": total_raw,
        "kept_boxes": total_kept,
        "removed_boxes": len(all_removed),
        "removed_by_reason": dict(sorted(reasons.items())),
        "documents": documents,
    }
    (review_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (review_dir / "removed_boxes.json").write_text(json.dumps(all_removed, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="Timestamped batch folder to review.")
    args = parser.parse_args()
    summary = build_review(args.batch.resolve())
    print(json.dumps({key: summary[key] for key in ("raw_boxes", "kept_boxes", "removed_boxes", "removed_by_reason")}, indent=2))


if __name__ == "__main__":
    main()