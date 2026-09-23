"""Create one consolidated red-box review PDF per selected document."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
from PIL import Image

from aggregate_ocr_json import OCR_ROOT

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output" / "reports" / "bbox_review_documents"
DOCUMENTS = [
    "01_OSAT_Files__ASE__BMM350__Updated__ACC_SAC405_95_5SN_4_0AG_0_5CU_20260829__Security_C__pdf_049d60f47f",
    "01_OSAT_Files__ASE__BMM350__Updated__FUJ_LTC9320_LTC9310_E07_20260411__Security_C__pdf_367ebfca71",
    "01_OSAT_Files__ASE__BMM350__Updated__DOW_Purified_CuSO4_20240616__Security_C__pdf_94936babda",
]


def make_review_pdf(document_name: str) -> tuple[Path, int, int]:
    output_path = OUTPUT_DIR / f"{document_name}_bbox_review.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = fitz.open()
    page_count = 0
    bbox_count = 0
    document_dir = OCR_ROOT / document_name

    for page_dir in sorted(document_dir.glob("page_*")):
        if not page_dir.is_dir():
            continue
        image_path = page_dir / f"{page_dir.name}.png"
        redaction_path = page_dir / f"{page_dir.name}_redactions.json"
        if not image_path.exists() or not redaction_path.exists():
            continue

        with Image.open(image_path) as image:
            width, height = image.size
        page = output.new_page(width=width, height=height)
        page.insert_image(fitz.Rect(0, 0, width, height), filename=str(image_path))
        record = json.loads(redaction_path.read_text(encoding="utf-8"))
        for entries in record.get("entities", {}).values():
            for entity in entries or []:
                bbox = entity.get("bbox")
                if not bbox:
                    continue
                rect = fitz.Rect(
                    float(bbox["x0"]), float(bbox["y0"]),
                    float(bbox["x1"]), float(bbox["y1"]),
                ) & page.rect
                if rect.is_empty or rect.width <= 0 or rect.height <= 0:
                    continue
                page.draw_rect(rect, color=(1, 0, 0), width=3)
                bbox_count += 1
        page_count += 1

    if not page_count:
        output.close()
        raise RuntimeError(f"No page PNG/redaction pairs found for {document_name}")
    output.save(str(output_path), garbage=4, deflate=True)
    output.close()
    return output_path, page_count, bbox_count


def main() -> None:
    for document_name in DOCUMENTS:
        path, pages, bboxes = make_review_pdf(document_name)
        print(f"Wrote {pages} pages and {bboxes} red boxes: {path}")


if __name__ == "__main__":
    main()
