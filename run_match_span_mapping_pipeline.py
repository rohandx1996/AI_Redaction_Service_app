"""Run original-vs-published extraction, OCR-span mapping, and PDF redaction."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

from config import settings
from identify_redactions import call_llm, patch_tiktoken
from redact import detect_machine_readable_regions, redact_pdf

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MATCH_ROOT = REPO_ROOT / "output" / "matched_original_counterparts" / "matches_20260916_214743"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "timestamped_batches"
POLICY_ROOT = REPO_ROOT / "output" / "reports"
MAPPING_SYSTEM_PROMPT = """You map known redaction values to OCR spans. Return JSON only with this shape:
{"mappings":[{"value":"...","category":"...","span_indices":[0],"confidence":0.0,"reason":"..."}],"unmatched":[{"value":"...","category":"...","reason":"..."}]}
Use only span indices supplied in the OCR JSON. OCR may contain multilingual recognition errors, broken
characters, or split spans. Use page context, labels, neighboring spans, and common sense to map the
known redaction value to the closest visible OCR occurrence. A value may require adjacent spans.
Do not invent indices. Map every distinct physical occurrence separately. If there is no defensible
mapping, put it in unmatched. Do not map technical text merely because it is nearby."""

ORIGINAL_ONLY_SYSTEM_PROMPT = """You identify sensitive information to redact from an original supplier compliance report page.
You receive only the original page image. Return JSON only:
{"redacted_information":[{"category":"...","field_label":"...","original_value":"...","confidence":0.0}]}
Protect supplier/customer/applicant identity, protected addresses, people/signatures, submitted sample names,
product/style/model/part identifiers, report/access identifiers, contact details, and proprietary references.
Do not redact laboratory branding, technical methods, chemicals, standards, units, results, procedures, or
generic technical content. A generic test-report number is not sensitive by itself; redact an identifier
only when page context marks it as a private client reference, order/reference number, PIN, QR/access code,
or other confidential submission identifier. Do not redact routine testing-laboratory mastheads, logos,
footers, public lab addresses, or standard contact blocks. Preserve values exactly as visibly written.
Use the OCR text only as supporting evidence; the image and page context control. Read Chinese, Japanese,
German, and mixed-language identity/address blocks. Include complete compound address/contact blocks when
they are tied to a protected party. Do not invent hidden values."""


def load_policy_context() -> str:
    retained_path = POLICY_ROOT / "published_knowledge_inventory" / "retained_knowledge_summary.json"
    patterns_path = POLICY_ROOT / "published_redaction_analysis" / "redaction_patterns_summary.json"
    parts = []
    if retained_path.exists():
        data = json.loads(retained_path.read_text(encoding="utf-8"))
        knowledge = data.get("unique_retained_knowledge", {})
        for key in ("technical_terms", "chemical_names", "standards_and_regulations", "test_methods_and_instruments"):
            values = knowledge.get(key, [])
            if values:
                parts.append(f"RETAINED {key}: {json.dumps(values[:80], ensure_ascii=False)}")
    if patterns_path.exists():
        data = json.loads(patterns_path.read_text(encoding="utf-8"))
        values = data.get("recurring_redacted_values", [])
        if values:
            parts.append(f"RECURRING REDACTION PATTERNS: {json.dumps(values[:80], ensure_ascii=False)}")
    return "\n".join(parts)


POLICY_CONTEXT = load_policy_context()


def clean_json(raw: str) -> dict:
    value = (raw or "").strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
        if value.lower().startswith("json"):
            value = value[4:].strip()
    return json.loads(value or "{}")


def document_name(match_dir: Path) -> str:
    relative = match_dir.name
    safe = "".join(char if char.isalnum() else "_" for char in relative)
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:180]}_{digest}"


def find_published(match_dir: Path) -> Path | None:
    candidates = sorted(match_dir.glob("published_*.pdf"))
    return candidates[0] if candidates else None


def ocr_page(ocr, page: fitz.Page, page_number: int) -> tuple[list[dict], int, int]:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    result = ocr.ocr(np.asarray(image), cls=False)
    spans = []
    for index, detection in enumerate(result[0] if result and result[0] else []):
        polygon, recognition = detection
        text, confidence = recognition
        spans.append({
            "index": index,
            "text": text,
            "confidence": float(confidence),
            "polygon": [[float(point[0]), float(point[1])] for point in polygon],
        })
    return spans, image.width, image.height


def page_jpeg(document: fitz.Document, page_number: int, dpi: int = 120) -> bytes:
    pixmap = document[page_number].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=78, optimize=True)
    return buffer.getvalue()


def call_original_vlm(image_bytes: bytes, page_number: int, ocr_text: str = "") -> dict:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    prompt = f"Original page number: {page_number + 1}. Identify redaction values from this original page only.\nPUBLISHED-POLICY REPRESENTATION (guidance only; do not infer hidden values):\n{POLICY_CONTEXT}\nOCR supporting text from this same page:\n{ocr_text[:16000]}"
    if settings.llm_model.startswith("claude-"):
        import requests
        response = requests.post(
            settings.bosch_gateway_url.rstrip("/") + "/v1/messages",
            headers={"x-api-key": settings.bosch_gateway_token, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": settings.llm_model, "system": ORIGINAL_ONLY_SYSTEM_PROMPT, "max_tokens": 4096, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}}]}]},
            timeout=180,
        )
        response.raise_for_status()
        raw = "".join(block.get("text", "") for block in response.json().get("content", []) if block.get("type") == "text")
        return clean_json(raw)
    import litellm
    response = litellm.completion(
        model=f"openai/{settings.llm_model}", api_base=settings.bosch_gateway_url.rstrip("/") + "/v1", api_key=settings.bosch_gateway_token,
        messages=[{"role": "system", "content": ORIGINAL_ONLY_SYSTEM_PROMPT}, {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}]}],
        max_tokens=4096, temperature=0,
    )
    return clean_json(response["choices"][0]["message"]["content"] or "{}")


def extraction_for_page(original: fitz.Document, page_number: int, ocr_text: str, output_path: Path) -> dict:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    result = call_original_vlm(page_jpeg(original, page_number), page_number, ocr_text)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def map_page(spans: list[dict], redaction_info: dict, page_number: int, output_path: Path) -> dict:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    candidates = []
    for item in redaction_info.get("redacted_information", []) or []:
        value = str(item.get("original_value", "")).strip()
        if value:
            candidates.append({
                "value": value,
                "category": item.get("category", "OTHER_IDENTITY"),
                "field_label": item.get("field_label", ""),
                "confidence": item.get("confidence", 0.0),
            })
    numbered = "\n".join(f"{s['index']}. {s['text']} | polygon={s['polygon']}" for s in spans)
    prompt = (
        f"Page number: {page_number + 1}\nKNOWN REDACTION INFORMATION JSON:\n"
        f"{json.dumps({'redactions': candidates}, ensure_ascii=False)}\n\nOCR SPANS:\n{numbered}\n\n"
        "Map the known values to OCR spans. Return only the requested JSON."
    )
    parsed = clean_json(call_llm(prompt, system_prompt=MAPPING_SYSTEM_PROMPT))
    span_by_index = {int(span["index"]): span for span in spans}
    mappings = []
    for item in parsed.get("mappings", []) or []:
        try:
            indices = [int(index) for index in item.get("span_indices", [])]
        except (TypeError, ValueError):
            continue
        if indices and all(index in span_by_index for index in indices):
            mappings.append({
                "page_number": page_number,
                "value": str(item.get("value", "")),
                "category": str(item.get("category", "OTHER_IDENTITY")),
                "span_indices": indices,
                "confidence": float(item.get("confidence", 0.0)),
                "reason": str(item.get("reason", "")),
            })
    result = {"page_number": page_number, "mappings": mappings, "unmatched": parsed.get("unmatched", [])}
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_document(match_dir: Path, output_root: Path, ocr, workers: int = 1) -> dict:
    started = time.perf_counter()
    original_path = match_dir / "original_01.pdf"
    published_path = find_published(match_dir)
    if not original_path.exists():
        return {"document": match_dir.name, "status": "SKIPPED", "reason": "missing original PDF"}
    name = document_name(match_dir)
    document_dir = output_root / name
    extraction_dir = document_dir / "redaction_information"
    mapping_dir = document_dir / "span_mappings"
    ocr_dir = document_dir / "ocr_pages"
    for directory in (extraction_dir, mapping_dir, ocr_dir):
        directory.mkdir(parents=True, exist_ok=True)
    regions = []
    sensitive_values = []
    extraction_started = time.perf_counter()
    with fitz.open(str(original_path)) as original:
        page_count = len(original)
        if page_count == 0:
            return {"document": match_dir.name, "status": "ERROR", "reason": "empty PDF"}
        ocr_records = []
        for page_number in range(page_count):
            ocr_started = time.perf_counter()
            spans, image_width, image_height = ocr_page(ocr, original[page_number], page_number)
            ocr_record = {"page_number": page_number, "image_width": image_width, "image_height": image_height, "spans": spans, "ocr_seconds": round(time.perf_counter() - ocr_started, 4)}
            (ocr_dir / f"page_{page_number + 1:04d}.json").write_text(json.dumps(ocr_record, ensure_ascii=False, indent=2), encoding="utf-8")
            ocr_records.append(ocr_record)
        def process_page(page_number: int) -> dict:
            ocr_text = "\n".join(str(span.get("text", "")) for span in ocr_records[page_number]["spans"])
            extraction_started_page = time.perf_counter()
            info = extraction_for_page(original, page_number, ocr_text, extraction_dir / f"page_{page_number + 1:04d}.json")
            extraction_seconds = time.perf_counter() - extraction_started_page
            mapping_started_page = time.perf_counter()
            mapping = map_page(ocr_records[page_number]["spans"], info, page_number, mapping_dir / f"page_{page_number + 1:04d}.json")
            mapping_seconds = time.perf_counter() - mapping_started_page
            return {"page_number": page_number, "redaction_information": info, "ocr": ocr_records[page_number], "mapping": mapping, "timings_sec": {"vlm_extraction": round(extraction_seconds, 4), "span_mapping": round(mapping_seconds, 4)}}
        if workers > 1 and page_count > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                page_records = list(executor.map(process_page, range(page_count)))
        else:
            page_records = [process_page(page_number) for page_number in range(page_count)]
        for page_record in page_records:
            page_number = page_record["page_number"]
            spans = page_record["ocr"]["spans"]
            mapping = page_record["mapping"]
            scale_x = original[page_number].rect.width / page_record["ocr"]["image_width"]
            scale_y = original[page_number].rect.height / page_record["ocr"]["image_height"]
            for mapped in mapping["mappings"]:
                points = [point for index in mapped["span_indices"] for point in next(span["polygon"] for span in spans if span["index"] == index)]
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                regions.append({"page_number": page_number, "x0": min(xs) * scale_x, "y0": min(ys) * scale_y, "x1": max(xs) * scale_x, "y1": max(ys) * scale_y})
                sensitive_values.append(mapped["value"])
    extraction_latency = time.perf_counter() - extraction_started
    ocr_latency = sum(page["ocr"].get("ocr_seconds", 0) for page in page_records)
    vlm_latency = sum(page["timings_sec"].get("vlm_extraction", 0) for page in page_records)
    mapping_latency = sum(page["timings_sec"].get("span_mapping", 0) for page in page_records)
    output_pdf = document_dir / "redacted.pdf"
    redaction_started = time.perf_counter()
    machine_readable_started = time.perf_counter()
    machine_readable_regions, machine_readable_detections = detect_machine_readable_regions(original_path)
    machine_readable_latency = time.perf_counter() - machine_readable_started
    audit = redact_pdf(
        original_path,
        output_pdf,
        regions,
        apply_bbox_filter=False,
        sensitive_values=sensitive_values,
        machine_readable_regions=machine_readable_regions,
        machine_readable_detections=machine_readable_detections,
    )
    redaction_latency = time.perf_counter() - redaction_started
    result = {
        "document": match_dir.name,
        "source_pdf": str(original_path),
        "published_pdf": str(published_path),
        "output_pdf": str(output_pdf),
        "page_count": page_count,
        "mapped_redactions": len(regions),
        "machine_readable_regions": len(machine_readable_regions),
        "machine_readable_detections": len(machine_readable_detections),
        "qr_detected_count": sum(item.get("type") == "QR" for item in machine_readable_detections),
        "qr_decoded_count": sum(item.get("type") == "QR" and item.get("decoded") for item in machine_readable_detections),
        "barcode_detected_count": sum(item.get("type") != "QR" for item in machine_readable_detections),
        "qr_remaining_decoded_count": sum(item.get("type") == "QR" for item in audit.get("findings", {}).get("qr_barcodes_remaining", [])),
        "pages": page_records,
        "audit": audit,
        "timings_sec": {"ocr": round(ocr_latency, 2), "vlm_extraction": round(vlm_latency, 2), "span_mapping": round(mapping_latency, 2), "extraction_ocr_mapping": round(extraction_latency, 2), "machine_readable_detection": round(machine_readable_latency, 2), "redaction_audit": round(redaction_latency, 2), "document_total": round(time.perf_counter() - started, 2)},
        "status": "OK",
    }
    (document_dir / "pipeline_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: value for key, value in result.items() if key != "pages"}


def create_ocr():
    import paddle
    from paddleocr import PaddleOCR
    try:
        paddle.set_flags({"FLAGS_use_mkldnn": False, "FLAGS_use_onednn": False})
    except ValueError:
        paddle.set_flags({"FLAGS_use_mkldnn": False})
    return PaddleOCR(det_algorithm="DB", rec_algorithm="SVTR_LCNet", use_gpu=False, lang="en", use_angle_cls=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-root", type=Path, default=DEFAULT_MATCH_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--document")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    patch_tiktoken()
    match_root = args.match_root.resolve()
    output_root = args.output_root or DEFAULT_OUTPUT_ROOT / f"match_span_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    matches = sorted(path for path in match_root.iterdir() if path.is_dir() and (path / "original_01.pdf").exists())
    if args.document:
        matches = [path for path in matches if path.name == args.document]
    elif args.smoke:
        matches = matches[:1]
    if not matches:
        raise SystemExit("No matched original_01.pdf documents found")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    ocr = create_ocr()
    results = []
    for path in matches:
        try:
            results.append(run_document(path, output_root, ocr, workers=args.workers))
        except Exception as error:
            results.append({"document": path.name, "status": "ERROR", "reason": str(error)})
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "match_span_pipeline", "created_at": datetime.now().isoformat(timespec="seconds"), "document_count": len(results), "results": results}
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "documents": len(results), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
