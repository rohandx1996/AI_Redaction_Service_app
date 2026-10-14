"""Classify PaddleOCR-VL image crops with the configured vision API, isolated from redaction."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

from PIL import Image

from run_timestamped_batch import call_vision
from config import settings


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "output" / "paddleocr_vl_region_experiment"
DEFAULT_CLASSIFICATION_ROOT = REPO_ROOT / "output" / "paddleocr_vl_image_classification"
CLASSIFICATION_PROMPT = """Classify this cropped image region into exactly one category:

- MATERIAL_SAMPLE: a physical material, product, part, packaging/bottle/label, product surface,
  specimen, component, or sample photo. Include visible product/material photos even if they contain
  printed text, unless the crop is actually a QR/barcode symbol.
- QR_BARCODE: a QR code, barcode, DataMatrix, or other machine-readable optical code.
- NOISE: anything else, such as a technical flowchart, generic diagram, page decoration, logo,
  chart, table, text-only excerpt, or an irrelevant/non-material image.

Decide from the visible crop only. Do not infer from the document filename. Return JSON only:
{"category":"MATERIAL_SAMPLE|QR_BARCODE|NOISE","confidence":0.0,"reason":"short visual reason"}
"""


def parse_response(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].casefold() == "json":
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    value = json.loads(text)
    category = str(value.get("category", "")).upper().strip()
    if category not in {"MATERIAL_SAMPLE", "QR_BARCODE", "NOISE"}:
        raise ValueError(f"Unexpected classification: {category!r}")
    confidence = float(value.get("confidence", 0.0))
    return {
        "category": category,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(value.get("reason", ""))[:500],
    }


def box_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def overlaps_known_code(image_box: list[int], page_regions: list[dict]) -> list[dict]:
    x0, y0, x1, y1 = image_box
    matched = []
    for region in page_regions:
        if region.get("type") not in {"qr_code", "barcode"}:
            continue
        code_box = region.get("bbox", [])
        if len(code_box) != 4:
            continue
        a, b, c, d = code_box
        overlap = max(0, min(x1, c) - max(x0, a)) * max(0, min(y1, d) - max(y0, b))
        if overlap / max(1, box_area(code_box)) >= 0.50:
            matched.append(region)
    return matched


def classify_one(crop_path: Path) -> dict:
    with Image.open(crop_path) as image:
        rgb = image.convert("RGB")
        raw, latency = call_vision(CLASSIFICATION_PROMPT, rgb)
    parsed = parse_response(raw)
    return {**parsed, "latency_sec": round(latency, 3), "source": "vision_api"}


def process_document(
    report_path: Path,
    classification_root: Path,
    max_images: int | None = None,
    resume: bool = False,
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_dir = report_path.parent
    document = report.get("document", source_dir.name)
    output_dir = classification_root / document
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "image_classification.json"
    previous = {}
    if resume and result_path.exists():
        try:
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            previous = {item["crop"]: item for item in saved.get("classifications", [])}
        except (OSError, json.JSONDecodeError, KeyError):
            previous = {}

    classifications = []
    api_calls = 0
    for page in report.get("pages", []):
        page_regions = page.get("regions", [])
        for region in page_regions:
            if region.get("type") != "image":
                continue
            crop_name = str(region.get("crop", ""))
            crop_path = source_dir / crop_name
            if not crop_path.is_file():
                classifications.append({
                    "crop": crop_name,
                    "page_label": page.get("page_label", ""),
                    "status": "ERROR",
                    "error": "Image crop file is missing",
                })
                continue
            known_codes = overlaps_known_code(region.get("bbox", []), page_regions)
            if known_codes:
                code_types = sorted({
                    "QR_BARCODE" for code in known_codes
                })
                result = {
                    "category": "QR_BARCODE",
                    "confidence": max(float(item.get("score", 0.0)) for item in known_codes),
                    "reason": "Overlaps a QR/barcode region already detected on the page",
                    "source": "page_detector_overlap",
                    "known_code_types": [item.get("type") for item in known_codes],
                }
            elif crop_name in previous and previous[crop_name].get("status") == "OK":
                result = previous[crop_name]["result"]
            elif max_images is not None and api_calls >= max_images:
                result = {
                    "category": "NOT_CLASSIFIED",
                    "confidence": 0.0,
                    "reason": "Skipped because the smoke-test API call limit was reached",
                    "source": "not_run",
                }
            else:
                try:
                    result = classify_one(crop_path)
                    api_calls += 1
                except Exception as error:
                    api_calls += 1
                    classifications.append({
                        "crop": crop_name,
                        "page_label": page.get("page_label", ""),
                        "bbox": region.get("bbox"),
                        "layout_score": region.get("score"),
                        "status": "ERROR",
                        "error": str(error),
                    })
                    continue

            category = result["category"]
            category_dir = output_dir / {
                "MATERIAL_SAMPLE": "material_sample",
                "QR_BARCODE": "qr_barcode",
                "NOISE": "noise",
                "NOT_CLASSIFIED": "not_classified",
            }.get(category, "not_classified")
            category_dir.mkdir(parents=True, exist_ok=True)
            copied_path = category_dir / crop_path.name
            if crop_path.resolve() != copied_path.resolve():
                shutil.copy2(crop_path, copied_path)
            classifications.append({
                "crop": crop_name,
                "page_label": page.get("page_label", ""),
                "bbox": region.get("bbox"),
                "layout_score": region.get("score"),
                "status": "OK",
                "result": result,
                "classified_crop": str(copied_path),
            })
            result_path.write_text(json.dumps({
                "document": document,
                "method": "OpenCV page-code overlap + configured vision API classification of remaining image crops",
                "vision_model": settings.llm_model,
                "api_calls_this_run": api_calls,
                "classifications": classifications,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = {category: sum(
        item.get("status") == "OK" and item.get("result", {}).get("category") == category
        for item in classifications
    ) for category in ("MATERIAL_SAMPLE", "QR_BARCODE", "NOISE", "NOT_CLASSIFIED")}
    errors = sum(item.get("status") == "ERROR" for item in classifications)
    result = {
        "document": document,
        "vision_model": settings.llm_model,
        "image_crops": len(classifications),
        "api_calls_this_run": api_calls,
        "material_sample": counts["MATERIAL_SAMPLE"],
        "qr_barcode": counts["QR_BARCODE"],
        "noise": counts["NOISE"],
        "not_classified": counts["NOT_CLASSIFIED"],
        "errors": errors,
        "output_folder": str(output_dir),
        "classifications": classifications,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CLASSIFICATION_ROOT)
    parser.add_argument("--document")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-images", type=int, help="Limit vision API calls per document (smoke testing)")
    parser.add_argument("--model", default="claude-haiku-4-5@20251001", help="Vision model ID configured on the Bosch gateway")
    parser.add_argument("--image-only-documents", action="store_true", help="Only classify folders with image regions and no detected QR/barcode regions")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings.llm_model = args.model

    reports = sorted(args.experiment_root.glob("*/regions.json"))
    if args.image_only_documents:
        selected_reports = []
        for report_path in reports:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            region_types = {
                region.get("type")
                for page in report.get("pages", [])
                for region in page.get("regions", [])
            }
            if region_types == {"image"}:
                selected_reports.append(report_path)
        reports = selected_reports
    if args.document:
        reports = [path for path in reports if path.parent.name == args.document]
    if args.limit:
        reports = reports[:args.limit]
    if not reports:
        raise SystemExit("No matching PaddleOCR-VL regions.json reports found")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, report_path in enumerate(reports, 1):
        summary = process_document(report_path, args.output_root, args.max_images, args.resume)
        summaries.append({key: value for key, value in summary.items() if key != "classifications"})
        print(
            f"[{index}/{len(reports)}] {summary['document']} crops={summary['image_crops']} "
            f"api_calls={summary['api_calls_this_run']} material={summary['material_sample']} "
            f"qr_barcode={summary['qr_barcode']} noise={summary['noise']} "
            f"unclassified={summary['not_classified']} errors={summary['errors']}",
            flush=True,
        )

    totals = {
        "scope": "image_only_document_folders" if args.image_only_documents else "all_document_folders",
        "vision_model": settings.llm_model,
        "documents": len(summaries),
        "image_crops": sum(row["image_crops"] for row in summaries),
        "api_calls": sum(row["api_calls_this_run"] for row in summaries),
        "material_sample": sum(row["material_sample"] for row in summaries),
        "qr_barcode": sum(row["qr_barcode"] for row in summaries),
        "noise": sum(row["noise"] for row in summaries),
        "not_classified": sum(row["not_classified"] for row in summaries),
        "errors": sum(row["errors"] for row in summaries),
        "document_summaries": summaries,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (args.output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in summaries[0] if key != "output_folder"] + ["output_folder"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({key: value for key, value in totals.items() if key != "document_summaries"}, indent=2))


if __name__ == "__main__":
    main()
