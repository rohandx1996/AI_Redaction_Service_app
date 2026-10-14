"""Independent DBNet++/PAN material-crop all-text masking experiment (CPU)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mmocr.apis import TextDetInferencer

ROOT = Path(__file__).resolve().parent
MODELS = {
    "dbnetpp_r50": ("dbnetpp_resnet50_fpnc_1200e_icdar2015", "dbnetpp_r50.pth"),
    "panet_r18": ("panet_resnet18_fpem-ffm_600e_ctw1500", "panet_r18.pth"),
}


def inputs(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.glob("*/image_classification.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("classifications", []):
            if item.get("status") == "OK" and item.get("result", {}).get("category") == "MATERIAL_SAMPLE":
                records.append({
                    "document": path.parent.name, "crop": item["crop"],
                    "path": path.parent / "material_sample" / item["crop"],
                    "page_label": item.get("page_label"), "page_bbox": item.get("bbox"),
                })
    return records


def save_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("PNG encoding failed")
    path.write_bytes(encoded.tobytes())


def process(inferencer, item: dict, root: Path, threshold: float, padding: int) -> dict:
    start = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(item["path"].read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable crop")
    height, width = image.shape[:2]
    prediction = inferencer(image, return_vis=False, progress_bar=False)["predictions"][0]
    mask = np.zeros((height, width), np.uint8)
    preview = image.copy()
    regions = []
    for polygon, score in zip(prediction.get("polygons", []), prediction.get("scores", [])):
        if float(score) < threshold:
            continue
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if not np.isfinite(points).all():
            continue
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        rounded = np.rint(points).astype(np.int32)
        if cv2.contourArea(rounded) < 1:
            continue
        cv2.fillPoly(mask, [rounded], 255)
        cv2.polylines(preview, [rounded], True, (0, 0, 255), 2)
        page_points = None
        box = item["page_bbox"]
        if box and len(box) == 4:
            page_points = [[float(x * (box[2]-box[0])/width+box[0]), float(y * (box[3]-box[1])/height+box[1])] for x, y in points]
        regions.append({"label": "SCENE_TEXT", "score": float(score), "polygon_crop_px": points.tolist(), "polygon_page_image_px": page_points, "recognized_text": None})
    if padding:
        mask = cv2.dilate(mask, np.ones((2*padding+1, 2*padding+1), np.uint8))
    masked = image.copy()
    masked[mask > 0] = 0
    folder = root / item["document"] / Path(item["crop"]).stem
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in (("original.png", image), ("text_regions.png", preview), ("mask.png", mask), ("redacted.png", masked)):
        save_png(folder/name, value)
    saved = cv2.imdecode(np.frombuffer((folder/"redacted.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    result = {
        "status": "OK", "document": item["document"], "crop": item["crop"],
        "page_label": item["page_label"], "material_bbox_page_image_px": item["page_bbox"],
        "width": width, "height": height, "detected_text_regions": len(regions),
        "masked_area_percent": float(np.count_nonzero(mask)/mask.size*100),
        "pixel_integrity_pass": bool(np.array_equal(image[mask == 0], masked[mask == 0]) and np.all(masked[mask > 0] == 0) and np.array_equal(saved, masked)),
        "elapsed_seconds": round(time.perf_counter()-start, 3),
        "output_folder": str(folder), "detections": regions,
        "limitations": "Masks all detected text including technical/ruler markings; no sensitivity classification, no full-page scan, no security guarantee",
    }
    (folder/"detections.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT/"output"/"paddleocr_vl_image_only_haiku")
    parser.add_argument("--output-root", type=Path, default=ROOT/"output"/"mmocr_material_text_experiment")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--padding", type=int, default=2)
    args = parser.parse_args()
    records = inputs(args.input_root)
    if args.limit:
        records = records[:args.limit]
    if not records or args.padding < 0 or not 0 <= args.threshold <= 1:
        parser.error("Missing inputs or invalid masking parameters")
    torch.set_num_threads(4)
    summaries = []
    for name, (model, weight_name) in MODELS.items():
        weight = ROOT/"models"/"mmocr"/weight_name
        detector = TextDetInferencer(model=model, weights=str(weight), device="cpu")
        output = args.output_root/name
        output.mkdir(parents=True, exist_ok=True)
        results = []
        for index, item in enumerate(records, 1):
            try:
                result = process(detector, item, output, args.threshold, args.padding)
            except Exception as error:
                result = {"document": item["document"], "crop": item["crop"], "status": "ERROR", "error": str(error)}
            results.append(result)
            print(f"{name} [{index}/{len(records)}] {item['document']} / {item['crop']} status={result['status']} boxes={result.get('detected_text_regions',0)}", flush=True)
        for document in sorted({r["document"] for r in results}):
            folder = output/document
            folder.mkdir(parents=True, exist_ok=True)
            (folder/"document_results.json").write_text(json.dumps([r for r in results if r["document"] == document], ensure_ascii=False, indent=2), encoding="utf-8")
        good = [r for r in results if r["status"] == "OK"]
        summary = {
            "model": model, "backend": "MMOCR PyTorch CPU", "weight_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            "threshold": args.threshold, "padding_px": args.padding,
            "documents": len({r["document"] for r in results}), "crops_attempted": len(results),
            "crops_processed": len(good), "errors": len(results)-len(good),
            "crops_with_text": sum(r["detected_text_regions"] > 0 for r in good),
            "detected_text_regions": sum(r["detected_text_regions"] for r in good),
            "pixel_integrity_pass": all(r["pixel_integrity_pass"] for r in good), "results": results,
        }
        (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with (output/"summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["document", "crop", "status", "detected_text_regions", "masked_area_percent", "elapsed_seconds", "output_folder", "error"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        summaries.append({k: v for k, v in summary.items() if k != "results"})
        del detector
    (args.output_root/"comparison_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    if any(s["errors"] or not s["pixel_integrity_pass"] for s in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()