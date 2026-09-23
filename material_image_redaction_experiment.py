"""Detect embedded material images, OCR enhanced variants, and redact all detected text."""

from __future__ import annotations

import argparse
import json
import re
from io import BytesIO
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from config import settings
from identify_redactions import call_llm
from redact import redact_pdf


MATERIAL_IMAGE_SCHEMA_KEYS = [
    "SUPPLIER_NAME", "MANUFACTURER_NAME", "CUSTOMER_NAME", "LABORATORY_NAME", "COMPANY",
    "PART_NUMBER", "PRODUCT_CODE", "PRODUCT_NAME", "SAMPLE_NAME", "LOT_NUMBER", "BATCH_NUMBER",
    "SERIAL_NUMBER", "MODEL_NUMBER", "INTERNAL_REFERENCE", "DOCUMENT_ID", "REPORT_NUMBER",
    "SUPPLIER_IDENTIFIER", "CUSTOMER_IDENTIFIER", "TRACKING_NUMBER", "URL", "QR_CODE_TEXT",
    "BARCODE_TEXT",
]

MATERIAL_IMAGE_SYSTEM_PROMPT = """You are a material-image information extraction engine.

The input OCR text comes from a cropped product, material, packaging, label, sticker, sample, or specimen photograph.
Identify information that may directly or indirectly identify a supplier, manufacturer, customer, laboratory, product owner, or business partner. Use only text present in OCR spans. Do not translate, normalize, repair OCR, summarize, paraphrase, or infer missing text. Return values exactly as recognized.

Extract only: SUPPLIER_NAME, MANUFACTURER_NAME, CUSTOMER_NAME, LABORATORY_NAME, COMPANY, PART_NUMBER, PRODUCT_CODE, PRODUCT_NAME, SAMPLE_NAME, LOT_NUMBER, BATCH_NUMBER, SERIAL_NUMBER, MODEL_NUMBER, INTERNAL_REFERENCE, DOCUMENT_ID, REPORT_NUMBER, SUPPLIER_IDENTIFIER, CUSTOMER_IDENTIFIER, TRACKING_NUMBER, URL, QR_CODE_TEXT, BARCODE_TEXT.

Do not extract measurement scales, scale bars, magnification values, microscope settings, image dimensions, units, chemical names, CAS numbers, RoHS, REACH, PFAS, PFOS, PFOA, IEC or ISO standards, scientific labels, technical annotations, or generic sample descriptions. Keep examples such as 500 um, 100 um, 200x, 500x, 20 mm, Scale Bar, Magnification, IEC62321, RoHS, and PFOS.

Examples: MITSUBISHI GAS CHEMICAL is SUPPLIER_NAME. LOT: MGC 2024-016 is LOT_NUMBER. P/N: HL830NSF is PART_NUMBER. SN123456 is SERIAL_NUMBER. ETR24B02841 is REPORT_NUMBER. PIN CODE: 494B2797 is DOCUMENT_ID.

Return JSON only. Include every schema key. Each entity must be {"value":"","span_indices":[0],"confidence":0.95}."""

MATERIAL_SELECTION_CONFIDENCE = 0.50


def preprocess_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    """Produce complementary local-contrast variants for dark material surfaces."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    inverted = cv2.bitwise_not(clahe)
    gamma = np.uint8(np.power(clahe / 255.0, 0.55) * 255)
    threshold = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7,
    )
    cleaned = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return {"clahe": clahe, "inverted": inverted, "gamma": gamma, "threshold": cleaned}


def _overlaps_existing(rect: fitz.Rect, existing: list[fitz.Rect]) -> bool:
    for candidate in existing:
        intersection = rect & candidate
        if not intersection.is_empty and intersection.get_area() / min(rect.get_area(), candidate.get_area()) >= 0.70:
            return True
    return False


def build_material_user_prompt(spans: list[dict]) -> str:
    numbered = "\n".join(f"{span['index']}. {span['text']}" for span in spans)
    return f"Material-image OCR spans:\n{numbered}\n\nReturn the JSON per schema."


def is_excluded_material_text(value: str) -> bool:
    """Keep measurements and scientific/compliance labels even if a model misclassifies them."""
    normalized = value.casefold().strip()
    excluded_terms = (
        "scale bar", "magnification", "rohs", "reach", "pfas", "pfos", "pfoa", "iec", "iso",
        "cas", "standard",
    )
    return any(term in normalized for term in excluded_terms) or bool(
        re.fullmatch(r"\d+(?:\.\d+)?\s*(?:um|µm|mm|nm|x)", normalized)
    )


def parse_material_selections(raw: str, spans_by_index: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    """Validate model selections and retain one mapped box per selected OCR span."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return [], [{"reason": "invalid_json"}]
    selected = []
    rejected = []
    for category in MATERIAL_IMAGE_SCHEMA_KEYS:
        entities = parsed.get(category, [])
        if not isinstance(entities, list):
            continue
        for entity in entities:
            try:
                value = str(entity["value"])
                confidence = float(entity.get("confidence", 0.0))
                span_indices = [int(index) for index in entity.get("span_indices", [])]
            except (KeyError, TypeError, ValueError):
                rejected.append({"category": category, "reason": "malformed_entity"})
                continue
            for span_index in dict.fromkeys(span_indices):
                span = spans_by_index.get(span_index)
                if (
                    not span or not value or span["text"].strip() != value.strip()
                    or confidence < MATERIAL_SELECTION_CONFIDENCE or is_excluded_material_text(value)
                ):
                    rejected.append({"category": category, "span_index": span_index, "reason": "ungrounded_or_low_confidence"})
                    continue
                selected.append({**span, "category": category, "llm_confidence": confidence})
    unique = []
    seen = set()
    for entity in selected:
        key = (entity["page_number"], entity["image_xref"], entity["index"])
        if key not in seen:
            seen.add(key)
            unique.append(entity)
    return unique, rejected


def select_material_redactions(candidates: list[dict]) -> tuple[list[dict], dict]:
    """Ask LiteLLM which grounded OCR candidates from one material image need redaction."""
    spans = [{**candidate, "index": index} for index, candidate in enumerate(candidates)]
    spans_by_index = {span["index"]: span for span in spans}
    raw = call_llm(build_material_user_prompt(spans), system_prompt=MATERIAL_IMAGE_SYSTEM_PROMPT)
    selected, rejected = parse_material_selections(raw, spans_by_index)
    return selected, {"raw_response": raw, "rejected_selections": rejected}


def material_images(page: fitz.Page, document: fitz.Document) -> list[tuple[int, fitz.Rect, bytes]]:
    """Return native image blocks that do not simply cover an entire rendered page."""
    found = []
    page_area = page.rect.get_area()
    for image in page.get_images(full=True):
        xref = image[0]
        extracted = document.extract_image(xref)
        if extracted["width"] < 100 or extracted["height"] < 100:
            continue
        for rect in page.get_image_rects(xref):
            if rect.get_area() / page_area >= 0.85:
                continue
            found.append((xref, rect, extracted["image"]))
    return found


def ocr_material_image(ocr: PaddleOCR, image_bytes: bytes, image_rect: fitz.Rect, page_number: int, xref: int) -> list[dict]:
    image = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
    candidates = []
    for technique, variant in preprocess_variants(image).items():
        result = ocr.ocr(variant, cls=False)
        for polygon, recognition in (result[0] if result and result[0] else []):
            text, confidence = recognition
            points = np.asarray(polygon, dtype=float)
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            rect = fitz.Rect(
                image_rect.x0 + x0 * image_rect.width / image.shape[1],
                image_rect.y0 + y0 * image_rect.height / image.shape[0],
                image_rect.x0 + x1 * image_rect.width / image.shape[1],
                image_rect.y0 + y1 * image_rect.height / image.shape[0],
            )
            candidates.append({
                "page_number": page_number, "image_xref": xref, "technique": technique,
                "text": str(text), "confidence": float(confidence),
                "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1,
            })
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    selected = []
    selected_rects = []
    for candidate in candidates:
        rect = fitz.Rect(candidate["x0"], candidate["y0"], candidate["x1"], candidate["y1"])
        if candidate["text"].strip() and not _overlaps_existing(rect, selected_rects):
            selected.append(candidate)
            selected_rects.append(rect)
    return selected


def write_review_pdf(source_pdf: Path, output_pdf: Path, candidates: list[dict], selections: list[dict]) -> None:
    by_page: dict[int, list[dict]] = {}
    for detection in candidates:
        by_page.setdefault(detection["page_number"], []).append(detection)
    with fitz.open(source_pdf) as source:
        review = fitz.open()
        review.insert_pdf(source)
        for page_number, page_detections in by_page.items():
            page = review[page_number]
            for detection in page_detections:
                page.draw_rect(fitz.Rect(detection["x0"], detection["y0"], detection["x1"], detection["y1"]), color=(1, 0.55, 0), width=0.6)
        for selection in selections:
            page = review[selection["page_number"]]
            page.draw_rect(fitz.Rect(selection["x0"], selection["y0"], selection["x1"], selection["y1"]), color=(1, 0, 0), width=2)
        review.save(output_pdf, garbage=4, deflate=True)
        review.close()


def verify_redacted_material_regions(ocr: PaddleOCR, pdf_path: Path, detections: list[dict]) -> list[dict]:
    """OCR each blacked-out material text rectangle to detect any remaining visible text."""
    leaks = []
    with fitz.open(pdf_path) as document:
        for detection in detections:
            page = document[detection["page_number"]]
            rect = fitz.Rect(detection["x0"], detection["y0"], detection["x1"], detection["y1"])
            pixmap = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=rect, alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
            result = ocr.ocr(image, cls=False)
            recognized = [
                str(recognition[0])
                for _, recognition in (result[0] if result and result[0] else [])
                if str(recognition[0]).strip()
            ]
            if recognized:
                leaks.append({"page_number": detection["page_number"], "image_xref": detection["image_xref"], "recognized_count": len(recognized)})
    return leaks


def process_pdf(ocr: PaddleOCR, source_pdf: Path, output_dir: Path, page_mode: str = "all") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    detections = []
    selections = []
    selection_audits = []
    images_found = 0
    with fitz.open(source_pdf) as document:
        page_numbers = range(len(document))
        if page_mode == "second-and-last":
            page_numbers = sorted({min(1, len(document) - 1), len(document) - 1})
        for page_number in page_numbers:
            page = document[page_number]
            for xref, image_rect, image_bytes in material_images(page, document):
                images_found += 1
                candidates = ocr_material_image(ocr, image_bytes, image_rect, page_number, xref)
                detections.extend(candidates)
                selected, audit = select_material_redactions(candidates)
                selections.extend(selected)
                selection_audits.append({"page_number": page_number, "image_xref": xref, "candidate_count": len(candidates), **audit})
    regions = [{key: selection[key] for key in ("page_number", "x0", "y0", "x1", "y1")} for selection in selections]
    redacted_pdf = output_dir / "material_image_redacted.pdf"
    audit = redact_pdf(source_pdf, redacted_pdf, regions, apply_bbox_filter=False)
    remaining_material_text = verify_redacted_material_regions(ocr, redacted_pdf, selections)
    audit["material_image_ocr_after_redaction"] = remaining_material_text
    if remaining_material_text:
        audit["status"] = "FAIL"
    write_review_pdf(source_pdf, output_dir / "material_image_review.pdf", detections, selections)
    report = {
        "source_pdf": str(source_pdf), "native_material_images": images_found,
        "page_mode": page_mode,
        "detected_text_regions": len(detections), "selected_redaction_regions": len(selections),
        "llm_model": settings.llm_model, "detections": detections, "selections": selections,
        "selection_audits": selection_audits, "pdf_audit": audit,
        "redacted_pdf": str(redacted_pdf), "review_pdf": str(output_dir / "material_image_review.pdf"),
    }
    (output_dir / "material_image_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="+", type=Path, help="Original PDF(s) to process.")
    parser.add_argument("--output-root", type=Path, default=Path("output/material_image_experiments"))
    parser.add_argument("--page-mode", choices=("all", "second-and-last"), default="all")
    args = parser.parse_args()
    ocr = PaddleOCR(det_algorithm="DB", rec_algorithm="SVTR_LCNet", use_gpu=False, lang="en", use_angle_cls=False)
    for index, pdf in enumerate(args.pdf, 1):
        report = process_pdf(ocr, pdf, args.output_root / f"document_{index:02d}", args.page_mode)
        print(json.dumps({key: report[key] for key in ("source_pdf", "native_material_images", "detected_text_regions", "redacted_pdf", "review_pdf")}, indent=2))


if __name__ == "__main__":
    main()