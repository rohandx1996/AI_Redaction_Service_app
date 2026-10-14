"""Vision-LLM text localisation and masking on material-photo crops (isolated experiment)."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from config import settings

ROOT = Path(__file__).resolve().parent
MODELS = {
    "haiku_fastest": "claude-haiku-4-5@20251001",
    "opus_best": "claude-opus-5-5",
}
PROMPT = """This is a photograph of a material sample, label or specimen. Its size is {w}x{h} pixels.
Find EVERY piece of visible text: printed labels, handwriting, stamped or engraved marks, numbers, ruler or scale markings.
Return one tight bounding box per text line or word group, in pixel coordinates of this image
(origin top-left, x to the right, y downward, x1>x0, y1>y0, within 0..{w} and 0..{h}).
Mark sensitive=true for text that identifies a company, person, address, sample, product, lot or reference;
false for generic scale numbers or technical markings.
Return JSON only:
{{"text_regions":[{{"text":"","x0":0,"y0":0,"x1":0,"y1":0,"sensitive":false}}]}}
Return {{"text_regions":[]}} if there is no text."""

SELECTIVE_PROMPT = """This is a material-sample photograph, {w}x{h} pixels.
Locate EVERY visible text line or word group, but decide separately whether to redact it.
Policy material-identifiers-v1:
- REDACT material/sample names or identifiers, product/part codes, batch/lot/serial numbers,
    specimen/test/report/reference numbers associated with the sample, and company/brand,
    person/address/contact information identifying the sample or its source.
- PRESERVE ruler/scale graduations and numbers, units, dimensions, generic measurement
    labels, generic headings, and non-identifying technical information.
- A number is NOT sensitive merely because it is a number. Use its visual context and
    association with the sample. A technical-looking code identifying a material IS sensitive.
- For mixed text, use separate boxes for identifying values and generic labels where possible.
- Unreadable or ambiguous text whose role cannot be established must be REVIEW, not guessed
    as safe or invented. A recognisable identifier may be REDACT even if not fully legible.
Return tight boxes in original-image pixels (origin top-left), within image bounds.
Return JSON only:
{{"text_regions":[{{"text":"","x0":0,"y0":0,"x1":0,"y1":0,
"decision":"redact|preserve|review","category":"","reason":""}}]}}
Each decision must be exactly redact, preserve or review. Explain the contextual reason.
Return {{"text_regions":[]}} if no text is visible. Do not remove the sample itself."""


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
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    path.write_bytes(encoded.tobytes())


def call_model(model: str, png: bytes, width: int, height: int, policy: str = "all-text") -> tuple[dict, dict, float]:
    body = {
        "model": model, "max_tokens": 8192,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png).decode("ascii")}},
            {"type": "text", "text": (SELECTIVE_PROMPT if policy == "selective" else PROMPT).format(w=width, h=height)},
        ]}],
    }
    if "haiku" in model:
        body["temperature"] = 0
    headers = {"x-api-key": settings.bosch_gateway_token, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    url = settings.bosch_gateway_url.rstrip("/") + "/v1/messages"
    last_error = None
    for attempt in range(5):
        started = time.perf_counter()
        try:
            response = requests.post(url, headers=headers, json=body, timeout=240)
            if response.status_code in (429, 500, 502, 503, 529):
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            payload = response.json()
            text = "".join(b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text")
            return parse(text), payload.get("usage", {}), time.perf_counter() - started
        except (requests.RequestException, RuntimeError, ValueError) as error:
            last_error = error
            if isinstance(error, requests.HTTPError) and error.response is not None and error.response.status_code < 500:
                break
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"Model call failed: {last_error}")


def parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def process(model: str, item: dict, root: Path, padding: int, policy: str = "all-text") -> dict:
    started = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(item["path"].read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable crop")
    height, width = image.shape[:2]
    reply, usage, latency = call_model(model, item["path"].read_bytes(), width, height, policy)
    mask = np.zeros((height, width), np.uint8)
    preview = image.copy()
    regions, rejected = [], 0
    box = item["page_bbox"]
    for raw in reply.get("text_regions", []):
        try:
            x0, y0, x1, y1 = (float(raw[k]) for k in ("x0", "y0", "x1", "y1"))
        except (KeyError, TypeError, ValueError):
            rejected += 1
            continue
        x0, x1 = max(0.0, min(x0, x1)), min(float(width), max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(float(height), max(y0, y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            rejected += 1
            continue
        decision = raw.get("decision", "review") if policy == "selective" else "redact"
        if decision not in ("redact", "preserve", "review"):
            decision = "review"
        if decision == "redact":
            cv2.rectangle(mask, (int(x0), int(y0)), (int(np.ceil(x1)), int(np.ceil(y1))), 255, -1)
        colour = {"redact": (0, 0, 255), "preserve": (0, 180, 0), "review": (0, 180, 255)}[decision]
        cv2.rectangle(preview, (int(x0), int(y0)), (int(x1), int(y1)), colour, 2)
        page = None
        if box and len(box) == 4:
            sx, sy = (box[2] - box[0]) / width, (box[3] - box[1]) / height
            page = [x0 * sx + box[0], y0 * sy + box[1], x1 * sx + box[0], y1 * sy + box[1]]
        regions.append({"text": str(raw.get("text", "")),
                "sensitive": decision == "redact" if policy == "selective" else bool(raw.get("sensitive")),
                "decision": decision, "category": str(raw.get("category", "")),
                "reason": str(raw.get("reason", "")), "masked": decision == "redact",
                        "bbox_crop_px": [x0, y0, x1, y1], "bbox_page_image_px": page})
    if padding:
        mask = cv2.dilate(mask, np.ones((2 * padding + 1, 2 * padding + 1), np.uint8))
    redacted = image.copy()
    redacted[mask > 0] = 0
    folder = root / item["document"] / Path(item["crop"]).stem
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in (("original.png", image), ("text_regions.png", preview), ("mask.png", mask), ("redacted.png", redacted)):
        save_png(folder / name, value)
    saved = cv2.imdecode(np.frombuffer((folder / "redacted.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    result = {
        "status": "OK", "model": model, "document": item["document"], "crop": item["crop"],
        "page_label": item["page_label"], "material_bbox_page_image_px": box,
        "width": width, "height": height, "detected_text_regions": len(regions),
        "sensitive_regions": sum(r["sensitive"] for r in regions), "rejected_boxes": rejected,
        "policy": "material-identifiers-v1" if policy == "selective" else "all-text",
        "redacted_regions": sum(r["masked"] for r in regions),
        "preserved_regions": sum(r["decision"] == "preserve" for r in regions),
        "review_regions": sum(r["decision"] == "review" for r in regions),
        "requires_review": bool(rejected or any(r["decision"] == "review" for r in regions)),
        "masked_area_percent": float(np.count_nonzero(mask) / mask.size * 100),
        "pixel_integrity_pass": bool(np.array_equal(image[mask == 0], redacted[mask == 0])
                                     and np.all(redacted[mask > 0] == 0) and np.array_equal(saved, redacted)),
        "api_latency_seconds": round(latency, 3), "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": usage, "output_folder": str(folder), "regions": regions,
        "limitations": "LLM boxes and policy decisions are unscored; no ground truth; review regions remain visible; padding may overlap nearby preserved text; not release-approved",
    }
    (folder / "detections.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "output" / "paddleocr_vl_image_only_haiku")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output" / "llm_material_text_experiment")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--padding", type=int, default=3)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--policy", choices=["all-text", "selective"], default="all-text")
    args = parser.parse_args()
    if args.policy == "selective" and args.output_root == ROOT / "output" / "llm_material_text_experiment":
        args.output_root = ROOT / "output" / "llm_material_selective_experiment"
    records = inputs(args.input_root)[:args.limit] if args.limit else inputs(args.input_root)
    if not records or args.padding < 0:
        parser.error("No inputs or invalid padding")
    summaries = []
    for key in args.models:
        model = MODELS[key]
        output = args.output_root / key
        output.mkdir(parents=True, exist_ok=True)
        results = []
        for index, item in enumerate(records, 1):
            try:
                result = process(model, item, output, args.padding, args.policy)
            except Exception as error:
                result = {"status": "ERROR", "model": model, "document": item["document"], "crop": item["crop"], "error": str(error)}
            results.append(result)
            print(f"{key} [{index}/{len(records)}] {item['crop']} status={result['status']} "
                  f"boxes={result.get('detected_text_regions', 0)} {result.get('api_latency_seconds', '')}s {result.get('error', '')[:120]}", flush=True)
        for document in sorted({r["document"] for r in results}):
            folder = output / document
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "document_results.json").write_text(json.dumps([r for r in results if r["document"] == document], ensure_ascii=False, indent=2), encoding="utf-8")
        good = [r for r in results if r["status"] == "OK"]
        summary = {
            "key": key, "model": model, "documents": len({r["document"] for r in results}),
            "crops_attempted": len(results), "crops_processed": len(good), "errors": len(results) - len(good),
            "crops_with_text": sum(r["detected_text_regions"] > 0 for r in good),
            "detected_text_regions": sum(r["detected_text_regions"] for r in good),
            "sensitive_regions": sum(r["sensitive_regions"] for r in good),
            "policy": "material-identifiers-v1" if args.policy == "selective" else "all-text",
            "redacted_regions": sum(r["redacted_regions"] for r in good),
            "preserved_regions": sum(r["preserved_regions"] for r in good),
            "review_regions": sum(r["review_regions"] for r in good),
            "crops_requiring_review": sum(r["requires_review"] for r in good),
            "rejected_boxes": sum(r["rejected_boxes"] for r in good),
            "total_api_seconds": round(sum(r["api_latency_seconds"] for r in good), 1),
            "mean_api_seconds": round(sum(r["api_latency_seconds"] for r in good) / max(1, len(good)), 2),
            "input_tokens": sum(r["usage"].get("input_tokens", 0) for r in good),
            "output_tokens": sum(r["usage"].get("output_tokens", 0) for r in good),
            "pixel_integrity_pass": all(r["pixel_integrity_pass"] for r in good), "results": results,
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["document", "crop", "status", "policy", "detected_text_regions", "sensitive_regions", "redacted_regions", "preserved_regions", "review_regions", "requires_review", "api_latency_seconds", "masked_area_percent", "output_folder", "error"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        summaries.append({k: v for k, v in summary.items() if k != "results"})
    (args.output_root / "comparison_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    if any(s["errors"] or not s["pixel_integrity_pass"] for s in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
