"""Isolated overlapping-tile/zoom/CLAHE experiment for DBNet++ and PAN."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mmocr.apis import TextDetInferencer

from mmocr_material_text_experiment import ROOT, MODELS, inputs, process, save_png


def tile_starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    return sorted(set([*range(0, length - size + 1, size - overlap), length - size]))


def map_points(polygon, offset: tuple[int, int], scale_x: float, scale_y: float) -> np.ndarray:
    points = np.asarray(polygon, dtype=float).reshape(-1, 2).copy()
    points[:, 0] = points[:, 0] / scale_x + offset[0]
    points[:, 1] = points[:, 1] / scale_y + offset[1]
    return points


def near_duplicate(first: np.ndarray, second: np.ndarray) -> bool:
    """Suppress only nearly equal polygons; partial word/line boxes are retained."""
    a, b = first.min(axis=0), first.max(axis=0)
    c, d = second.min(axis=0), second.max(axis=0)
    intersection = float(np.maximum(0, np.minimum(b, d) - np.maximum(a, c)).prod())
    area_a, area_b = float((b-a).prod()), float((d-c).prod())
    union = area_a + area_b - intersection
    return union > 0 and intersection / union >= 0.8


class TiledDetector:
    def __init__(self, detector, folder: Path, args):
        self.detector, self.folder, self.args = detector, folder, args
        self.stats = {}

    def __call__(self, image, **kwargs):
        height, width = image.shape[:2]
        candidates = []
        passes = []
        tile_dir = self.folder / "zoomed_tiles"
        tile_dir.mkdir(parents=True, exist_ok=True)

        def detect(value, offset, scale_x, scale_y, name):
            result = self.detector(value, return_vis=False, progress_bar=False)["predictions"][0]
            count = 0
            for poly, score in zip(result.get("polygons", []), result.get("scores", [])):
                if float(score) < self.args.threshold:
                    continue
                points = map_points(poly, offset, scale_x, scale_y)
                if not np.isfinite(points).all():
                    continue
                candidates.append((points, float(score), name))
                count += 1
            passes.append({"pass": name, "offset_original_px": offset, "scale_x": scale_x,
                           "scale_y": scale_y, "candidate_count": count})

        detect(image, (0, 0), 1, 1, "whole_image")
        tile_count = 0
        for y in tile_starts(height, self.args.tile_size, self.args.overlap):
            for x in tile_starts(width, self.args.tile_size, self.args.overlap):
                tile_count += 1
                tile = image[y:min(height, y+self.args.tile_size), x:min(width, x+self.args.tile_size)]
                th, tw = tile.shape[:2]
                enlarged = cv2.resize(tile, (round(tw*self.args.zoom), round(th*self.args.zoom)), interpolation=cv2.INTER_CUBIC)
                sx, sy = enlarged.shape[1]/tw, enlarged.shape[0]/th
                name = f"tile_{tile_count:03d}_x{x}_y{y}"
                save_png(tile_dir / f"{name}_zoom.png", enlarged)
                detect(enlarged, (x, y), sx, sy, name+"_zoom")
                lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                save_png(tile_dir / f"{name}_clahe.png", enhanced)
                detect(enhanced, (x, y), sx, sy, name+"_clahe")
        kept = []
        for poly, score, name in sorted(candidates, key=lambda p: p[1], reverse=True):
            if not any(near_duplicate(poly, entry[0]) for entry in kept):
                kept.append((poly, score, name))
        self.stats = {"tile_count": tile_count, "inference_passes": len(passes),
                      "raw_candidates": len(candidates), "consolidated_candidates": len(kept),
                      "passes": passes, "retained_detection_sources": [entry[2] for entry in kept]}
        return {"predictions": [{"polygons": [p.reshape(-1).tolist() for p, _, _ in kept],
                                 "scores": [s for _, s, _ in kept]}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT/"output"/"paddleocr_vl_image_only_haiku")
    parser.add_argument("--output-root", type=Path, default=ROOT/"output"/"mmocr_tiled_material_experiment")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--zoom", type=float, default=2)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--padding", type=int, default=2)
    args = parser.parse_args()
    if args.tile_size < 32 or not 0 <= args.overlap < args.tile_size or args.zoom <= 0 or args.padding < 0:
        parser.error("Invalid tile, overlap, zoom or padding")
    # Direct verification of original-coordinate mapping, including noninteger scales.
    assert np.allclose(map_points([0, 0, 20, 40], (10, 30), 2, 2), [[10, 30], [20, 50]])
    records = inputs(args.input_root)
    if args.limit:
        records = records[:args.limit]
    if not records:
        parser.error("No material crops found")
    torch.set_num_threads(4)
    summaries = []
    for name, (config, weights) in MODELS.items():
        detector = TextDetInferencer(model=config, weights=str(ROOT/"models"/"mmocr"/weights), device="cpu")
        out = args.output_root/name
        out.mkdir(parents=True, exist_ok=True)
        results = []
        for i, item in enumerate(records, 1):
            folder = out/item["document"]/Path(item["crop"]).stem
            try:
                wrapper = TiledDetector(detector, folder, args)
                result = process(wrapper, item, out, args.threshold, args.padding)
                result["tiling"] = wrapper.stats
                (folder/"detections.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as error:
                result = {"status": "ERROR", "document": item["document"], "crop": item["crop"], "error": str(error)}
            results.append(result)
            print(f"{name} [{i}/{len(records)}] {item['document']} / {item['crop']} {result['status']} boxes={result.get('detected_text_regions',0)}", flush=True)
            (out/"progress.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        for doc in sorted({r["document"] for r in results}):
            folder = out/doc
            folder.mkdir(parents=True, exist_ok=True)
            (folder/"document_results.json").write_text(json.dumps([r for r in results if r["document"]==doc], ensure_ascii=False, indent=2), encoding="utf-8")
        good = [r for r in results if r["status"]=="OK"]
        summary = {"model": config, "method": "whole image + overlapping 512px tiles + zoom + CLAHE",
                   "tile_size": args.tile_size, "overlap": args.overlap, "zoom": args.zoom,
                   "documents": len({r["document"] for r in results}), "processed": len(good),
                   "errors": len(results)-len(good), "crops_with_text": sum(r["detected_text_regions"]>0 for r in good),
                   "text_regions": sum(r["detected_text_regions"] for r in good),
                   "pixel_integrity_pass": all(r["pixel_integrity_pass"] for r in good),
                   "limitations": "All-text masking only; more boxes can include duplicates and false positives. Upscaling does not recover missing detail.",
                   "results": results}
        (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append({k:v for k,v in summary.items() if k!="results"})
        del detector
    (args.output_root/"comparison_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    if any(r["errors"] or not r["pixel_integrity_pass"] for r in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()