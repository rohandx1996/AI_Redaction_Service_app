"""Isolated CRAFT CPU text detection/masking on classified material-photo crops.

This is a detection experiment, not a release-approved sensitive-text redactor.
It masks ALL detected text within each crop, including generic technical markings.
No recognition, API calls, PDF modification, or production integration is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from easyocr.detection import get_detector, test_net

ROOT = Path(__file__).resolve().parent
WEIGHT_MD5 = "2f8227d2def4037cdb3b34389dcf9ec1"


def read_image(path: Path) -> np.ndarray:
    """Unicode-safe OpenCV image read on Windows."""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG encoding failed: {path}")
    path.write_bytes(encoded.tobytes())


def material_inputs(input_root: Path, document: str | None) -> list[dict]:
    inputs = []
    for path in sorted(input_root.glob("*/image_classification.json")):
        if document and path.parent.name != document:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("classifications", []):
            if item.get("status") != "OK" or item.get("result", {}).get("category") != "MATERIAL_SAMPLE":
                continue
            crop_name = item["crop"]
            # Use this selected tree, not stale absolute paths copied from a previous run.
            crop_path = path.parent / "material_sample" / crop_name
            inputs.append({
                "document": path.parent.name, "crop": crop_name,
                "path": crop_path, "page_label": item.get("page_label", ""),
                "page_bbox": item.get("bbox"),
                "classification": item.get("result", {}),
            })
    return inputs


def process_crop(net, item: dict, output_root: Path, args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    image = read_image(item["path"])
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, polygons_batch = test_net(
        args.canvas_size, args.mag_ratio, net, rgb,
        args.text_threshold, args.link_threshold, args.low_text,
        poly=False, device="cpu",
    )
    detections = []
    mask = np.zeros((height, width), dtype=np.uint8)
    preview = image.copy()
    for polygon in polygons_batch[0]:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if not np.isfinite(points).all():
            continue
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        integer_points = np.rint(points).astype(np.int32)
        if cv2.contourArea(integer_points) < 1:
            continue
        cv2.fillPoly(mask, [integer_points], 255)
        cv2.polylines(preview, [integer_points], True, (0, 0, 255), 2)
        coordinates = [
            float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()),
        ]
        page_polygon = None
        page_bbox = item.get("page_bbox")
        if isinstance(page_bbox, list) and len(page_bbox) == 4:
            # Layout crop coordinates are in rendered-page pixels, NOT PDF points.
            scale_x = (page_bbox[2] - page_bbox[0]) / width
            scale_y = (page_bbox[3] - page_bbox[1]) / height
            page_polygon = [
                [float(x * scale_x + page_bbox[0]), float(y * scale_y + page_bbox[1])]
                for x, y in points
            ]
        detections.append({
            "label": "SCENE_TEXT", "polygon_crop_px": points.tolist(),
            "bbox_crop_px": coordinates, "polygon_page_image_px": page_polygon,
            "recognized_text": None,
        })
    if args.padding:
        size = 2 * args.padding + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
    redacted = image.copy()
    redacted[mask > 0] = 0
    preserved = bool(np.array_equal(redacted[mask == 0], image[mask == 0]))
    masked_black = bool(np.all(redacted[mask > 0] == 0))
    destination = output_root / item["document"] / Path(item["crop"]).stem
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("original.png", image), ("text_regions.png", preview),
        ("mask.png", mask), ("redacted.png", redacted),
    ):
        write_image(destination / name, value)
    saved = read_image(destination / "redacted.png")
    roundtrip = bool(np.array_equal(saved, redacted))
    result = {
        "status": "OK", "document": item["document"], "crop": item["crop"],
        "input_path": str(item["path"]), "output_folder": str(destination),
        "page_label": item["page_label"], "material_bbox_page_image_px": item["page_bbox"],
        "width": width, "height": height, "detected_text_regions": len(detections),
        "masked_pixels": int(np.count_nonzero(mask)),
        "masked_area_percent": round(np.count_nonzero(mask) / mask.size * 100, 4),
        "outside_mask_unchanged": preserved, "mask_is_black": masked_black,
        "saved_png_matches": roundtrip,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "policy": "Mask all detected text within material crop; sensitive identity not classified",
        "geometry": "CRAFT quadrilateral boxes; curved polygon output disabled for NumPy compatibility",
        "limitations": [
            "CRAFT can miss faint, tiny, curved, or low-contrast text",
            "Non-text textures can be false detections",
            "Text outside the material crop is not scanned",
            "On-object versus surrounding text is not semantically classified",
            "Existing OCR outlines in crop inputs may affect detection",
            "No security/release approval or recall/precision claim",
        ],
        "detections": detections,
    }
    (destination / "detections.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "output" / "paddleocr_vl_image_only_haiku")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "craft" / "craft_mlt_25k.pth")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--document")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--canvas-size", type=int, default=1600)
    parser.add_argument("--mag-ratio", type=float, default=1.5)
    parser.add_argument("--text-threshold", type=float, default=0.7)
    parser.add_argument("--link-threshold", type=float, default=0.4)
    parser.add_argument("--low-text", type=float, default=0.4)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.padding < 0 or args.threads < 1 or args.canvas_size < 32 or args.mag_ratio <= 0:
        parser.error("Invalid padding, threads, canvas size or magnification")
    inputs = material_inputs(args.input_root, args.document)
    if args.limit:
        inputs = inputs[:args.limit]
    if not inputs:
        raise SystemExit("No matching MATERIAL_SAMPLE crops found")
    digest = hashlib.md5(args.model.read_bytes()).hexdigest()
    if digest != WEIGHT_MD5:
        raise RuntimeError("CRAFT weight checksum mismatch")
    output_root = args.output_root or ROOT / "output" / f"craft_material_text_{datetime.now():%Y%m%d_%H%M%S}"
    output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    net = get_detector(str(args.model), device="cpu", quantize=False)
    results = []
    for index, item in enumerate(inputs, 1):
        try:
            result = process_crop(net, item, output_root, args)
        except Exception as error:
            result = {"status": "ERROR", "document": item["document"], "crop": item["crop"], "error": str(error)}
        results.append(result)
        print(f"[{index}/{len(inputs)}] {item['document']} / {item['crop']} status={result['status']} text_regions={result.get('detected_text_regions', 0)}", flush=True)
    documents = sorted({result["document"] for result in results})
    for document in documents:
        folder = output_root / document
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "document_results.json").write_text(json.dumps({
            "document": document, "crops": [r for r in results if r["document"] == document],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    good = [r for r in results if r["status"] == "OK"]
    summary = {
        "experiment": "CRAFT material-crop all-text masking",
        "backend": "EasyOCR 1.7.2 CRAFT PyTorch CPU (not TFLite)",
        "weight_md5": digest, "input_root": str(args.input_root),
        "parameters": {key: getattr(args, key) for key in ("canvas_size", "mag_ratio", "text_threshold", "link_threshold", "low_text", "padding", "threads")},
        "documents": len(documents), "crops_attempted": len(results),
        "crops_processed": len(good), "errors": len(results) - len(good),
        "crops_with_text": sum(r["detected_text_regions"] > 0 for r in good),
        "detected_text_regions": sum(r["detected_text_regions"] for r in good),
        "pixel_integrity_pass": all(r["outside_mask_unchanged"] and r["mask_is_black"] and r["saved_png_matches"] for r in good),
        "results": results,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        columns = ["document", "crop", "status", "detected_text_regions", "masked_area_percent", "elapsed_seconds", "output_folder", "error"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"results", "parameters"}}, indent=2), flush=True)
    if summary["errors"] or not summary["pixel_integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()