"""Whole-image YOLOv8 handwritten-word masking experiment; no tiling/enhancement."""
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
from ultralytics import YOLO

from craft_material_text_experiment import material_inputs, read_image, write_image

ROOT = Path(__file__).resolve().parent
PROVENANCE = {
    "repository": "armvectores/yolov8n_handwritten_text_detection",
    "revision": "e144d41b93d51e4b4eac005c5e4a5a38bd92b44d",
    "file": "best.pt", "training_domain": "handwritten text (model card lists Armenian)",
    "license": "Weight license not specified in model card; review before production use. Ultralytics code is AGPL-3.0.",
}


def process(model, item: dict, root: Path, confidence: float) -> dict:
    start = time.perf_counter()
    image = read_image(item["path"])
    height, width = image.shape[:2]
    prediction = model.predict(image, device="cpu", imgsz=640, conf=confidence, verbose=False)[0]
    mask = np.zeros((height, width), np.uint8)
    preview = image.copy()
    regions = []
    for box, score, category in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy(), prediction.boxes.cls.cpu().numpy()):
        x0, y0, x1, y1 = box.astype(float)
        a, b = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        c, d = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
        if c <= a or d <= b:
            continue
        mask[b:d, a:c] = 255
        cv2.rectangle(preview, (a,b), (c-1,d-1), (0,0,255), 2)
        page_box = item.get("page_bbox")
        mapped = None
        if page_box and len(page_box) == 4:
            sx, sy = (page_box[2]-page_box[0])/width, (page_box[3]-page_box[1])/height
            mapped = [a*sx+page_box[0], b*sy+page_box[1], c*sx+page_box[0], d*sy+page_box[1]]
        regions.append({"label": model.names[int(category)], "score": float(score), "bbox_crop_px": [a,b,c,d], "bbox_page_image_px": mapped, "recognized_text": None})
    redacted = image.copy()
    redacted[mask > 0] = 0
    folder = root/item["document"]/Path(item["crop"]).stem
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in (("original.png", image), ("text_regions.png", preview), ("mask.png", mask), ("redacted.png", redacted)):
        write_image(folder/name, value)
    result = {
        "status": "OK", "document": item["document"], "crop": item["crop"],
        "page_label": item["page_label"], "material_bbox_page_image_px": item["page_bbox"],
        "detected_text_regions": len(regions), "masked_area_percent": float(np.count_nonzero(mask)/mask.size*100),
        "elapsed_seconds": round(time.perf_counter()-start,3), "output_folder": str(folder),
        "pixel_integrity_pass": bool(np.array_equal(redacted[mask == 0],image[mask == 0]) and np.all(redacted[mask > 0] == 0) and np.array_equal(read_image(folder/"redacted.png"),redacted)),
        "detections": regions,
    }
    (folder/"detections.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=ROOT/"output"/"paddleocr_vl_image_only_haiku")
    parser.add_argument("--output-root",type=Path,default=ROOT/"output"/"yolov8_material_text_experiment")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--confidence",type=float,default=0.25)
    args = parser.parse_args()
    records = material_inputs(args.input_root,None)
    if args.limit:
        records=records[:args.limit]
    if not records or not 0 <= args.confidence <= 1:
        parser.error("Missing material crops or invalid confidence")
    weights=ROOT/"models"/"yolov8_text"/"handwritten_text.pt"
    torch.set_num_threads(4)
    model=YOLO(str(weights))
    if set(model.names.values()) != {"word"}:
        raise RuntimeError("Unexpected detector classes")
    args.output_root.mkdir(parents=True,exist_ok=True)
    results=[]
    for index,item in enumerate(records,1):
        try:
            result=process(model,item,args.output_root,args.confidence)
        except Exception as error:
            result={"status":"ERROR","document":item["document"],"crop":item["crop"],"error":str(error)}
        results.append(result)
        print(f"[{index}/{len(records)}] {item['crop']} status={result['status']} boxes={result.get('detected_text_regions',0)}",flush=True)
    for doc in sorted({r["document"] for r in results}):
        folder=args.output_root/doc
        folder.mkdir(parents=True,exist_ok=True)
        (folder/"document_results.json").write_text(json.dumps([r for r in results if r["document"]==doc],ensure_ascii=False,indent=2),encoding="utf-8")
    good=[r for r in results if r["status"]=="OK"]
    summary={"model":"YOLOv8n handwritten text","provenance":PROVENANCE,"weight_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),
             "mode":"whole image, standard YOLO resize/letterbox only; no tiles/zoom/enhancement", "confidence":args.confidence,
             "limitations":"Handwriting-trained community checkpoint; printed/curved/faint text may be missed. All detected words masked, not sensitive-only. No accuracy/security guarantee.",
             "documents":len({r["document"] for r in results}),"crops_processed":len(good),"errors":len(results)-len(good),
             "crops_with_text":sum(r["detected_text_regions"]>0 for r in good),"detected_text_regions":sum(r["detected_text_regions"] for r in good),
             "pixel_integrity_pass":all(r["pixel_integrity_pass"] for r in good),"results":results}
    (args.output_root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    with (args.output_root/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=["document","crop","status","detected_text_regions","masked_area_percent","elapsed_seconds","output_folder","error"],extrasaction="ignore")
        writer.writeheader(); writer.writerows(results)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"results","provenance"}},indent=2))
    if summary["errors"] or not summary["pixel_integrity_pass"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()