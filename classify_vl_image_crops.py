"""Classify PaddleOCR-VL image crops as QR/barcode or non-code, without altering PDFs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "output" / "paddleocr_vl_region_experiment"


def image_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants = [("original", image)]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    variants.append(("grayscale", gray))
    height, width = gray.shape[:2]
    scale = min(4.0, max(1.0, 600.0 / max(1, min(height, width))))
    if scale > 1.0:
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(("grayscale_upscaled", enlarged))
        _, otsu = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu_upscaled", otsu))
        variants.append(("otsu_inverted_upscaled", cv2.bitwise_not(otsu)))
    else:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu", otsu))
        variants.append(("otsu_inverted", cv2.bitwise_not(otsu)))
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7,
    )
    variants.append(("adaptive_threshold", adaptive))
    return variants


def _has_qr(detector: cv2.QRCodeDetector, image: np.ndarray) -> tuple[bool, bool]:
    """Return (detected, decoded), using detection as well as successful decoding."""
    try:
        result = detector.detectAndDecode(image)
        if len(result) >= 2 and result[1] is not None:
            return True, bool(result[0])
    except cv2.error:
        pass
    try:
        detected, points = detector.detect(image)
        if detected and points is not None:
            return True, False
    except cv2.error:
        pass
    try:
        multi_result = detector.detectAndDecodeMulti(image)
        if multi_result and multi_result[0] and multi_result[2] is not None:
            decoded_values = multi_result[1] or []
            return True, any(bool(value) for value in decoded_values)
    except (cv2.error, ValueError, TypeError):
        pass
    return False, False


def _barcode_detected(detector: cv2.barcode_BarcodeDetector, image: np.ndarray) -> tuple[bool, bool]:
    try:
        result = detector.detectAndDecode(image)
    except (cv2.error, ValueError, TypeError):
        return False, False
    if len(result) == 3:
        value, barcode_type, points = result
        return points is not None, bool(value)
    if len(result) >= 4:
        _, values, types, corners = result[:4]
        found = corners is not None and len(corners) > 0
        return bool(found), any(bool(value) for value in (values or []))
    return False, False


def classify_crop(crop_path: Path) -> dict:
    image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "classification": "ERROR",
            "crop_path": str(crop_path),
            "error": "OpenCV could not read crop",
        }
    qr_detector = cv2.QRCodeDetector()
    barcode_detector = cv2.barcode_BarcodeDetector()
    qr_evidence = []
    for variant_name, variant in image_variants(image):
        qr_found, qr_decoded = _has_qr(qr_detector, variant)
        if qr_found:
            qr_evidence.append({
                "variant": variant_name,
                "decoded": qr_decoded,
            })
    barcode_evidence = []
    if not qr_evidence:
        for variant_name, variant in image_variants(image):
            barcode_found, barcode_decoded = _barcode_detected(barcode_detector, variant)
            if barcode_found:
                barcode_evidence.append({
                    "variant": variant_name,
                    "decoded": barcode_decoded,
                })
                break
    evidence = qr_evidence if qr_evidence else barcode_evidence
    code_types = ["QR"] if qr_evidence else ["BARCODE"] if barcode_evidence else []
    return {
        "classification": "QR" if qr_evidence else "BARCODE" if barcode_evidence else "NO_QR",
        "code_types": code_types,
        "evidence": evidence,
        "crop_width": int(image.shape[1]),
        "crop_height": int(image.shape[0]),
        "crop_path": str(crop_path),
    }


def overlapping_page_code_types(image_bbox: list[int], regions: list[dict]) -> list[str]:
    """Reuse same-page QR/barcode detections when the code lies inside an image-tag crop."""
    x0, y0, x1, y1 = image_bbox
    matches = set()
    for region in regions:
        region_type = region.get("type")
        if region_type not in {"qr_code", "barcode"}:
            continue
        a, b, c, d = region.get("bbox", [])
        if len([a, b, c, d]) != 4:
            continue
        intersection = max(0, min(x1, c) - max(x0, a)) * max(0, min(y1, d) - max(y0, b))
        code_area = max(1, (c - a) * (d - b))
        if intersection / code_area >= 0.50:
            matches.add("QR" if region_type == "qr_code" else "BARCODE")
    return sorted(matches)


def process_document(report_path: Path) -> tuple[list[dict], dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    document_dir = report_path.parent
    non_qr_dir = document_dir / "non_qr_images"
    if non_qr_dir.exists():
        shutil.rmtree(non_qr_dir)
    non_qr_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for page in report.get("pages", []):
        page_regions = page.get("regions", [])
        for region in page_regions:
            if region.get("type") != "image":
                continue
            crop_name = region.get("crop")
            crop_path = document_dir / str(crop_name)
            result = classify_crop(crop_path)
            contextual_codes = overlapping_page_code_types(region.get("bbox", []), page_regions)
            if contextual_codes:
                result["classification"] = contextual_codes[0]
                result["code_types"] = contextual_codes
                result["evidence"] = [
                    *result.get("evidence", []),
                    {
                        "source": "same_page_code_bbox_overlap",
                        "types": contextual_codes,
                        "overlap_threshold": 0.50,
                    },
                ]
            result.update({
                "document": report.get("document", document_dir.name),
                "page_label": page.get("page_label", ""),
                "source_region_label": region.get("type", "image"),
                "layout_score": region.get("score"),
                "bbox": region.get("bbox"),
            })
            if result["classification"] == "NO_QR" and crop_path.exists():
                shutil.copy2(crop_path, non_qr_dir / crop_path.name)
                result["non_qr_copy"] = str(non_qr_dir / crop_path.name)
            results.append(result)

    counts = Counter(result["classification"] for result in results)
    qr_types = Counter(
        code_type
        for result in results
        for code_type in result.get("code_types", [])
    )
    summary = {
        "document": report.get("document", document_dir.name),
        "image_crops_classified": len(results),
        "qr_crops": counts["QR"],
        "barcode_crops": counts["BARCODE"],
        "no_qr_crops": counts["NO_QR"],
        "classification_errors": counts["ERROR"],
        "qr_detections": qr_types["QR"],
        "barcode_detections": qr_types["BARCODE"],
        "non_qr_images_folder": str(non_qr_dir),
        "classifications": results,
    }
    (document_dir / "image_crop_classification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--document", help="Classify one exact document folder name")
    parser.add_argument("--limit", type=int, help="Process only the first N report folders")
    args = parser.parse_args()

    reports = sorted(args.experiment_root.glob("*/regions.json"))
    if args.document:
        reports = [path for path in reports if path.parent.name == args.document]
    if args.limit:
        reports = reports[:args.limit]
    if not reports:
        raise SystemExit("No matching document regions.json files found")

    all_results = []
    document_summaries = []
    for index, report_path in enumerate(reports, 1):
        results, summary = process_document(report_path)
        all_results.extend(results)
        document_summaries.append({key: value for key, value in summary.items() if key != "classifications"})
        print(
            f"[{index}/{len(reports)}] {report_path.parent.name}: "
            f"image_crops={summary['image_crops_classified']} "
            f"qr={summary['qr_crops']} barcode={summary['barcode_crops']} "
            f"no_qr={summary['no_qr_crops']} "
            f"errors={summary['classification_errors']}",
            flush=True,
        )

    counts = Counter(result["classification"] for result in all_results)
    summary = {
        "experiment": "OpenCV image-crop QR/barcode classification",
        "documents_classified": len(document_summaries),
        "image_crops_classified": len(all_results),
        "qr_crops": counts["QR"],
        "barcode_crops": counts["BARCODE"],
        "no_qr_crops": counts["NO_QR"],
        "classification_errors": counts["ERROR"],
        "qr_detections": sum(item["qr_detections"] for item in document_summaries),
        "barcode_detections": sum(item["barcode_detections"] for item in document_summaries),
        "documents": document_summaries,
    }
    args.experiment_root.joinpath("image_crop_classification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    csv_path = args.experiment_root / "image_crop_classification_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "document", "image_crops_classified", "qr_crops", "barcode_crops",
            "no_qr_crops", "classification_errors", "qr_detections",
            "barcode_detections", "non_qr_images_folder",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(document_summaries)
    print(json.dumps({key: value for key, value in summary.items() if key != "documents"}, indent=2))


if __name__ == "__main__":
    main()
