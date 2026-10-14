"""Detect likely material-image regions using only structural image heuristics.

This experiment is isolated from the redaction pipeline. It reads existing OCR page
PNGs/JSON, writes candidate crops and annotated previews, and never modifies PDFs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
OCR_ROOT = REPO_ROOT / "output" / "batch" / "paddle_dbnet_svtr_pages"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "structural_material_image_experiment"

MIN_PAGE_AREA_RATIO = 0.015
MAX_PAGE_AREA_RATIO = 0.70
MIN_SIDE_PIXELS = 100


def shannon_entropy(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram[histogram > 0] / gray.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def intersection_area(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    merged = []
    for box in sorted(boxes, key=lambda value: (value[0], value[1])):
        changed = True
        while changed:
            changed = False
            for index, existing in enumerate(merged):
                overlap = intersection_area(box, existing)
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                existing_area = (existing[2] - existing[0]) * (existing[3] - existing[1])
                gap_x = max(existing[0] - box[2], box[0] - existing[2], 0)
                gap_y = max(existing[1] - box[3], box[1] - existing[3], 0)
                if overlap / max(1, min(box_area, existing_area)) >= 0.30 or (gap_x <= 16 and gap_y <= 16):
                    box = (
                        min(box[0], existing[0]), min(box[1], existing[1]),
                        max(box[2], existing[2]), max(box[3], existing[3]),
                    )
                    merged.pop(index)
                    changed = True
                    break
        merged.append(box)
    return merged


def tile_candidates(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = gray.shape
    tile = max(24, round(min(height, width) / 45))
    rows = math.ceil(height / tile)
    columns = math.ceil(width / tile)
    mask = np.zeros((rows, columns), np.uint8)
    edges = cv2.Canny(gray, 60, 160)
    for row in range(rows):
        for column in range(columns):
            y0, y1 = row * tile, min(height, (row + 1) * tile)
            x0, x1 = column * tile, min(width, (column + 1) * tile)
            crop = gray[y0:y1, x0:x1]
            foreground = float(np.mean(crop < 242))
            entropy = shannon_entropy(crop)
            edge_density = float(np.mean(edges[y0:y1, x0:x1] > 0))
            if (entropy >= 3.4 and edge_density >= 0.045 and foreground >= 0.08) or (
                entropy >= 2.7 and edge_density >= 0.075 and foreground >= 0.18
            ):
                mask[row, column] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [
        (x * tile, y * tile, min(width, (x + w) * tile), min(height, (y + h) * tile))
        for contour in contours
        for x, y, w, h in [cv2.boundingRect(contour)]
    ]


def framed_candidates(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 140)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    page_area = width * height
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area_ratio = box_width * box_height / page_area
        rectangularity = cv2.contourArea(contour) / max(1, box_width * box_height)
        if (
            MIN_PAGE_AREA_RATIO <= area_ratio <= MAX_PAGE_AREA_RATIO
            and min(box_width, box_height) >= MIN_SIDE_PIXELS
            and rectangularity >= 0.45
        ):
            boxes.append((x, y, x + box_width, y + box_height))
    return boxes


def ocr_occupancy(box: tuple[int, int, int, int], spans: list[dict]) -> float:
    box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    covered = 0
    for span in spans:
        polygon = span.get("polygon", [])
        if not polygon:
            continue
        span_box = (
            int(min(point[0] for point in polygon)), int(min(point[1] for point in polygon)),
            int(max(point[0] for point in polygon)), int(max(point[1] for point in polygon)),
        )
        covered += intersection_area(box, span_box)
    return min(1.0, covered / box_area)


def score_candidate(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    spans: list[dict],
) -> dict | None:
    height, width = gray.shape
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    box_width, box_height = x1 - x0, y1 - y0
    page_ratio = box_width * box_height / (width * height)
    if (
        page_ratio < MIN_PAGE_AREA_RATIO or page_ratio > MAX_PAGE_AREA_RATIO
        or min(box_width, box_height) < MIN_SIDE_PIXELS
    ):
        return None
    crop = gray[y0:y1, x0:x1]
    foreground_mask = crop < 242
    foreground = float(np.mean(foreground_mask))
    entropy = shannon_entropy(crop)
    edge_density = float(np.mean(cv2.Canny(crop, 60, 160) > 0))
    row_density = foreground_mask.mean(axis=1)
    column_density = foreground_mask.mean(axis=0)
    row_coverage = float(np.mean(row_density > 0.08))
    column_coverage = float(np.mean(column_density > 0.08))
    occupancy = ocr_occupancy((x0, y0, x1, y1), spans)
    score = (
        min(entropy / 6.0, 1.0) * 0.30
        + min(edge_density / 0.15, 1.0) * 0.25
        + min(row_coverage / 0.70, 1.0) * 0.15
        + min(column_coverage / 0.70, 1.0) * 0.15
        + min(foreground / 0.50, 1.0) * 0.15
        - min(occupancy / 0.45, 1.0) * 0.25
    )
    if entropy < 2.2 or edge_density < 0.018 or row_coverage < 0.12 or column_coverage < 0.12:
        return None
    if entropy < 3.0 and occupancy > 0.25:
        return None
    if occupancy > 0.55:
        return None
    if score < 0.43:
        return None
    return {
        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "width": box_width,
        "height": box_height,
        "page_area_ratio": round(page_ratio, 6),
        "entropy": round(entropy, 4),
        "edge_density": round(edge_density, 6),
        "foreground_density": round(foreground, 6),
        "row_coverage": round(row_coverage, 6),
        "column_coverage": round(column_coverage, 6),
        "ocr_occupancy": round(occupancy, 6),
        "score": round(score, 6),
    }


def detect_page(image: np.ndarray, spans: list[dict]) -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = merge_boxes([*tile_candidates(gray), *framed_candidates(gray)])
    detections = [score_candidate(gray, box, spans) for box in candidates]
    detections = [detection for detection in detections if detection]
    detections.sort(key=lambda item: (item["score"], item["page_area_ratio"]), reverse=True)
    return detections


def process_document(document_dir: Path, output_root: Path, save_all_pages: bool = False) -> dict:
    output_dir = output_root / document_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    total_candidates = 0
    for page_dir in sorted(path for path in document_dir.iterdir() if path.is_dir() and path.name.startswith("page_")):
        image_path = page_dir / f"{page_dir.name}.png"
        json_path = page_dir / f"{page_dir.name}.json"
        if not image_path.exists() or not json_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        detections = detect_page(image, data.get("spans", []))
        page_record = {
            "page_label": page_dir.name,
            "page_number": int(data.get("page_number", 0)),
            "image": str(image_path),
            "candidate_count": len(detections),
            "candidates": [],
        }
        annotated = image.copy()
        for index, detection in enumerate(detections, 1):
            bbox = detection["bbox"]
            crop_name = f"{page_dir.name}_candidate_{index:03d}.png"
            crop = image[bbox["y0"]:bbox["y1"], bbox["x0"]:bbox["x1"]]
            cv2.imwrite(str(output_dir / crop_name), crop)
            cv2.rectangle(
                annotated, (bbox["x0"], bbox["y0"]), (bbox["x1"], bbox["y1"]),
                (0, 0, 255), max(2, round(min(image.shape[:2]) / 600)),
            )
            cv2.putText(
                annotated, f"{index} {detection['score']:.2f}",
                (bbox["x0"], max(20, bbox["y0"] - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 2, cv2.LINE_AA,
            )
            page_record["candidates"].append({**detection, "crop": crop_name})
        if detections or save_all_pages:
            annotated_name = f"{page_dir.name}_annotated.png"
            cv2.imwrite(str(output_dir / annotated_name), annotated)
            page_record["annotated_page"] = annotated_name
        pages.append(page_record)
        total_candidates += len(detections)
    report = {
        "document": document_dir.name,
        "method": "rule-based contours + whitespace/density + OCR occupancy; no LLM/VLM",
        "pages_processed": len(pages),
        "pages_with_candidates": sum(page["candidate_count"] > 0 for page in pages),
        "candidate_count": total_candidates,
        "pages": pages,
    }
    (output_dir / "structural_material_candidates.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-root", type=Path, default=OCR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--document")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-all-pages", action="store_true")
    args = parser.parse_args()

    documents = sorted(path for path in args.ocr_root.iterdir() if path.is_dir())
    if args.document:
        documents = [path for path in documents if path.name == args.document]
    if args.limit:
        documents = documents[:args.limit]
    if not documents:
        raise SystemExit("No matching OCR document folders found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, document_dir in enumerate(documents, 1):
        report = process_document(document_dir, args.output_root, args.save_all_pages)
        reports.append(report)
        print(
            f"[{index}/{len(documents)}] {document_dir.name} | pages={report['pages_processed']} "
            f"candidate_pages={report['pages_with_candidates']} candidates={report['candidate_count']}"
        )
    summary = {
        "method": "isolated rule-based material-image detection experiment",
        "document_count": len(reports),
        "page_count": sum(report["pages_processed"] for report in reports),
        "documents_with_candidates": sum(report["candidate_count"] > 0 for report in reports),
        "pages_with_candidates": sum(report["pages_with_candidates"] for report in reports),
        "candidate_count": sum(report["candidate_count"] for report in reports),
        "documents": [{key: report[key] for key in ("document", "pages_processed", "pages_with_candidates", "candidate_count")} for report in reports],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["document", "pages_processed", "pages_with_candidates", "candidate_count"])
        writer.writeheader()
        writer.writerows(summary["documents"])
    print(json.dumps({key: summary[key] for key in ("document_count", "page_count", "documents_with_candidates", "pages_with_candidates", "candidate_count")}, indent=2))


if __name__ == "__main__":
    main()