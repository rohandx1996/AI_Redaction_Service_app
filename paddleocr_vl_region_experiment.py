"""Isolated PaddleOCR-VL layout plus QR/barcode region-cropping experiment."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from paddlex import create_model


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OCR_ROOT = REPO_ROOT / "output" / "batch" / "paddle_dbnet_svtr_pages"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "paddleocr_vl_region_experiment"
IMAGE_LABELS = {"image"}


def qr_barcode_regions(image: np.ndarray) -> list[dict]:
    detections = []
    qr_detector = cv2.QRCodeDetector()
    decoded, points, _ = qr_detector.detectAndDecode(image)
    if points is not None:
        points = np.asarray(points).reshape(-1, 2)
        detections.append({
            "type": "qr_code",
            "score": 1.0 if decoded else 0.8,
            "decoded": bool(decoded),
            "bbox": [
                int(points[:, 0].min()), int(points[:, 1].min()),
                int(points[:, 0].max()), int(points[:, 1].max()),
            ],
        })
    barcode_detector = cv2.barcode_BarcodeDetector()
    barcode_result = barcode_detector.detectAndDecode(image)
    if len(barcode_result) == 3:
        value, barcode_type, points = barcode_result
        decoded_info = [value] if points is not None else []
        barcode_types = [barcode_type] if points is not None else []
        corners = [points] if points is not None else []
    else:
        _, decoded_info, barcode_types, corners = barcode_result
    for value, barcode_type, points in zip(decoded_info or [], barcode_types or [], corners or []):
        points = np.asarray(points).reshape(-1, 2)
        detections.append({
            "type": "barcode",
            "barcode_type": str(barcode_type),
            "score": 1.0 if value else 0.8,
            "decoded": bool(value),
            "bbox": [
                int(points[:, 0].min()), int(points[:, 1].min()),
                int(points[:, 0].max()), int(points[:, 1].max()),
            ],
        })
    return detections


def layout_image_regions(layout_model, image_path: Path, threshold: float) -> list[dict]:
    predictions = list(layout_model.predict(str(image_path), threshold=threshold))
    if not predictions:
        return []
    payload = predictions[0].json
    result = payload.get("res", payload)
    boxes = result.get("boxes", [])
    regions = []
    for box in boxes:
        label = str(box.get("label", "")).casefold()
        if label not in IMAGE_LABELS:
            continue
        coordinate = [int(round(value)) for value in box.get("coordinate", [])]
        if len(coordinate) != 4:
            continue
        regions.append({
            "type": "image",
            "score": float(box.get("score", 0.0)),
            "bbox": coordinate,
        })
    return regions


def save_regions(
    image: np.ndarray,
    regions: list[dict],
    output_dir: Path,
    page_label: str,
) -> list[dict]:
    height, width = image.shape[:2]
    annotated = image.copy()
    saved = []
    colors = {"image": (0, 0, 255), "qr_code": (0, 180, 0), "barcode": (255, 0, 0)}
    for index, region in enumerate(regions, 1):
        x0, y0, x1, y1 = region["bbox"]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        crop_name = f"{page_label}_{region['type']}_{index:03d}.png"
        cv2.imwrite(str(output_dir / crop_name), image[y0:y1, x0:x1])
        color = colors[region["type"]]
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 3)
        cv2.putText(
            annotated, f"{region['type']} {region['score']:.2f}",
            (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
        saved.append({**region, "bbox": [x0, y0, x1, y1], "crop": crop_name})
    if saved:
        cv2.imwrite(str(output_dir / f"{page_label}_annotated.png"), annotated)
    return saved


def process_document(layout_model, document_dir: Path, output_root: Path, threshold: float) -> dict:
    output_dir = output_root / document_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    errors = []
    started = time.perf_counter()
    for page_dir in sorted(path for path in document_dir.iterdir() if path.is_dir()):
        image_path = page_dir / f"{page_dir.name}.png"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        page_started = time.perf_counter()
        try:
            layout_regions = layout_image_regions(layout_model, image_path, threshold)
            symbol_regions = qr_barcode_regions(image)
            saved = save_regions(image, [*layout_regions, *symbol_regions], output_dir, page_dir.name)
        except Exception as error:
            errors.append({"page_label": page_dir.name, "error": str(error)})
            saved = []
        pages.append({
            "page_label": page_dir.name,
            "image": str(image_path),
            "elapsed_sec": round(time.perf_counter() - page_started, 3),
            "regions": saved,
        })
    report = {
        "document": document_dir.name,
        "method": "PaddleOCR-VL PP-DocLayoutV3 image detection + OpenCV QR/barcode detection",
        "pages_processed": len(pages),
        "pages_with_regions": sum(bool(page["regions"]) for page in pages),
        "image_regions": sum(region["type"] == "image" for page in pages for region in page["regions"]),
        "qr_regions": sum(region["type"] == "qr_code" for page in pages for region in page["regions"]),
        "barcode_regions": sum(region["type"] == "barcode" for page in pages for region in page["regions"]),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "errors": errors,
        "pages": pages,
    }
    (output_dir / "regions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--document")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    documents = sorted(path for path in args.ocr_root.iterdir() if path.is_dir())
    if args.document:
        documents = [path for path in documents if path.name == args.document]
    if args.limit:
        documents = documents[:args.limit]
    if not documents:
        raise SystemExit("No matching OCR document folders found")

    args.output_root.mkdir(parents=True, exist_ok=True)
    layout_model = create_model("PP-DocLayoutV3", device="cpu")
    reports = []
    for index, document_dir in enumerate(documents, 1):
        report_path = args.output_root / document_dir.name / "regions.json"
        if args.resume and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = process_document(layout_model, document_dir, args.output_root, args.threshold)
        reports.append(report)
        print(
            f"[{index}/{len(documents)}] {document_dir.name} pages={report['pages_processed']} "
            f"images={report['image_regions']} qr={report['qr_regions']} "
            f"barcodes={report['barcode_regions']} errors={len(report.get('errors', []))} "
            f"elapsed={report['elapsed_sec']:.1f}s",
            flush=True,
        )

    summary = {
        "method": "isolated PaddleOCR-VL layout + OpenCV QR/barcode experiment",
        "document_count": len(reports),
        "page_count": sum(report["pages_processed"] for report in reports),
        "documents_with_regions": sum(report["pages_with_regions"] > 0 for report in reports),
        "image_regions": sum(report["image_regions"] for report in reports),
        "qr_regions": sum(report["qr_regions"] for report in reports),
        "barcode_regions": sum(report["barcode_regions"] for report in reports),
        "elapsed_sec": round(sum(report["elapsed_sec"] for report in reports), 3),
        "error_count": sum(len(report.get("errors", [])) for report in reports),
        "documents": [
            {key: report[key] for key in (
                "document", "pages_processed", "pages_with_regions", "image_regions",
                "qr_regions", "barcode_regions", "elapsed_sec",
            )}
            for report in reports
        ],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["documents"][0]))
        writer.writeheader()
        writer.writerows(summary["documents"])
    print(json.dumps({key: summary[key] for key in (
        "document_count", "page_count", "documents_with_regions", "image_regions",
        "qr_regions", "barcode_regions", "elapsed_sec",
    )}, indent=2))


if __name__ == "__main__":
    main()