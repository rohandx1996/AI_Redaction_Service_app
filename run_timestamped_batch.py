"""Run multilingual redaction extraction and build timestamped review outputs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import fitz
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from aggregate_ocr_json import OCR_ROOT
from identify_redactions import build_user_prompt, call_llm, parse_and_validate, page_dirs_sorted
from identify_redactions import patch_tiktoken, settings
from release_audit import write_release_audit
from redact import detect_machine_readable_regions, redact_pdf

REPO_ROOT = Path(__file__).resolve().parent
BATCH_ROOT = REPO_ROOT / "output" / "timestamped_batches"
console = Console()

REVIEW_PROMPT = """You are an OCR Quality Review Engine.
You will receive OCR text spans in the format <SPAN_ID>. <OCR_TEXT>.
Identify only spans that may contain OCR recognition errors. Do not translate, correct OCR,
perform redaction, classify entities, or explain text meaning. A span is suspicious when it has
corrupted characters, strange symbols replacing language text, broken words, damaged multilingual
text, malformed URLs or phones, damaged names/addresses, truncation, appended garbage, or a likely
language-model mismatch. Return only valid JSON with suspicious_ocr_spans containing span_index,
ocr_text, issue_code, and confidence. Allowed issue_code values are:
CORRUPTED_MULTILINGUAL_TEXT, BROKEN_WORD, MERGED_WORDS, CHARACTER_SUBSTITUTION,
UNEXPECTED_SYMBOLS, TRUNCATED_TEXT, APPENDED_GARBAGE, DAMAGED_URL, DAMAGED_PHONE,
DAMAGED_COMPANY_NAME, DAMAGED_PERSON_NAME, DAMAGED_ADDRESS, SMALL_TEXT_OR_FOOTER_FAILURE,
POSSIBLE_LANGUAGE_MODEL_MISMATCH, OTHER_OCR_FAILURE.
"""

REPAIR_PROMPT = """You are performing OCR.
Return the exact visible text contained in the supplied image.
Preserve original language, characters, punctuation, visible spacing, and capitalization.
Do not translate, summarize, explain, identify entities, redact, normalize, correct grammar,
or infer missing text. Return only the raw recognized text, line by line, with no markdown,
JSON, or explanation.
"""

VISUAL_REDACTION_PROMPT = """You are a visual redaction detector for supplier compliance reports.
Inspect the supplied full page image, including text printed inside photographs, scans, labels,
packaging, diagrams, and embedded material images. Return only sensitive identity-bearing values
that link the report to a supplier, customer, applicant, submitter, manufacturer, commercial
partner, facility, contact, submitted sample, proprietary product/material, or specific commercial
submission. Return every separate visible occurrence with its own tight bbox, even when the same
value appears elsewhere on the page.

Interpret different laboratory wording contextually. Client Name, Customer, Applicant, Submitted
By, Sample Submitted By, 送樣廠商, and equivalents identify the aligned visible organization value.
Sample Name, Sample Description, Style/Item No., Model No., Part Number, Material, Lot No., Order
No., Client Ref., and equivalents may identify a protected submitted sample/product/reference.
Protect identity-bearing text inside sample photos, bottle/package labels, product markings,
QR labels, and barcodes.

Do not automatically redact the testing laboratory's routine logo, masthead, accreditation name,
standard footer, terms-of-service block, or generic laboratory address. Redact a laboratory name
only when local context makes it a protected customer, supplier, applicant, subcontractor,
commercial partner, or confidential submission identity. Do not redact generic field labels alone.

Do not redact technical knowledge that Bosch retains: chemical names, CAS numbers, standards,
test methods, instruments, procedures, flowcharts, measurements, units, conclusions, or generic
material imagery without an identity-bearing value. Do not cover a technical paragraph/table/
diagram merely because a small identity value occurs nearby; return a tight box for the identity.

Visual examples:
- A bottle label reading "LINTEC LC2850" beside "Sample Name" -> return a tight SAMPLE_NAME box.
- The same protected company printed twice on one page -> return two separate boxes.
- A routine SGS logo/header/footer with no protected-party context -> return no box.
- A diagram reading "Sample pretreatment -> GC/MS -> result" -> return no box.

Return JSON only. Coordinates must be pixel coordinates in the supplied image:
{"redactions":[{"text":"exact visible text","category":"SAMPLE_NAME","x0":0,"y0":0,"x1":0,"y1":0,"confidence":0.95}]}
Use an empty list when no sensitive visual text is present. Do not invent or infer hidden text.
"""


def long_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    resolved = path.resolve(strict=False)
    text = str(resolved)
    if text.startswith("\\\\?\\") or len(text) < 260:
        return resolved
    return Path("\\\\?\\" + text)


def parse_json_response(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)


def filter_large_spans(spans: list[dict], image_width: int, image_height: int) -> tuple[list[dict], list[int]]:
    """Drop OCR span height outliers derived from the full OCR corpus."""
    kept = []
    removed = []
    for span in spans:
        polygon = span.get("polygon", [])
        if len(polygon) < 4:
            kept.append(span)
            continue
        height = max(point[1] for point in polygon) - min(point[1] for point in polygon)
        if height >= image_height * 0.10:
            removed.append(span.get("index"))
        else:
            kept.append(span)
    return kept, removed


def call_vision(prompt: str, image: Image.Image) -> tuple[str, float]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    started = time.perf_counter()
    if settings.llm_model.startswith("claude-"):
        import requests

        last_error = None
        for attempt in range(6):
            try:
                response = requests.post(
                    settings.bosch_gateway_url.rstrip("/") + "/v1/messages",
                    headers={
                        "x-api-key": settings.bosch_gateway_token,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": settings.llm_model,
                        "max_tokens": 2048,
                        "messages": [{"role": "user", "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": "image/png", "data": encoded,
                            }},
                            {"type": "text", "text": prompt},
                        ]}],
                    },
                    timeout=180,
                )
                response.raise_for_status()
                content = response.json().get("content", [])
                return "".join(
                    block.get("text", "") for block in content if block.get("type") == "text"
                ) or "{}", time.perf_counter() - started
            except Exception as error:
                last_error = error
                if attempt == 5:
                    break
                response = getattr(error, "response", None)
                retry_after = response.headers.get("Retry-After") if response is not None else None
                time.sleep(float(retry_after) if retry_after else min(60.0, 5.0 * (2 ** attempt)))
        raise RuntimeError(f"Vision LLM call failed after 6 attempts: {last_error}")

    import litellm

    response = litellm.completion(
        model=f"openai/{settings.llm_model}",
        api_base=settings.bosch_gateway_url.rstrip("/") + "/v1",
        api_key=settings.bosch_gateway_token,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ]}
        ],
        max_tokens=2048,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"] or "{}", time.perf_counter() - started


def visual_redactions(page_image: Image.Image) -> tuple[list[dict], float]:
    raw, latency = call_vision(VISUAL_REDACTION_PROMPT, page_image)
    try:
        data = parse_json_response(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return [], latency
    redactions = []
    width, height = page_image.size
    for item in data.get("redactions", []) or []:
        try:
            x0 = max(0.0, min(float(item["x0"]), width))
            y0 = max(0.0, min(float(item["y0"]), height))
            x1 = max(0.0, min(float(item["x1"]), width))
            y1 = max(0.0, min(float(item["y1"]), height))
            confidence = float(item.get("confidence", 0.0))
            text = str(item.get("text", "")).strip()
            category = str(item.get("category", "OTHER_IDENTITY"))
        except (KeyError, TypeError, ValueError):
            continue
        if not text or confidence < 0.80 or x1 <= x0 or y1 <= y0:
            continue
        redactions.append({
            "text": text, "category": category, "confidence": confidence,
            "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "source": "vlm_visual_detection",
        })
    return redactions, latency


def review_spans(spans: list[dict]) -> tuple[list[int], float]:
    numbered = "\n".join(
        f"{span.get('index', index)}. {span.get('text', '')} (ocr_confidence={span.get('confidence', 0.0)})"
        for index, span in enumerate(spans)
    )
    started = time.perf_counter()
    raw = call_llm(f"{REVIEW_PROMPT}\nOCR spans:\n{numbered}")
    data = parse_json_response(raw)
    ids = []
    for value in data.get("suspicious_spans", []):
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(ids)), time.perf_counter() - started


def repair_suspicious_spans(page_dir: Path, spans: list[dict], suspicious_ids: list[int]) -> tuple[list[dict], float]:
    if not suspicious_ids:
        return spans, 0.0
    image_path = long_path(page_dir / f"{page_dir.name}.png")
    with Image.open(image_path) as page_image:
        repaired = []
        for span in spans:
            if span.get("index") not in suspicious_ids or not span.get("polygon"):
                continue
            points = span["polygon"]
            left = max(0, int(min(point[0] for point in points)) - 8)
            top = max(0, int(min(point[1] for point in points)) - 8)
            right = min(page_image.width, int(max(point[0] for point in points)) + 8)
            bottom = min(page_image.height, int(max(point[1] for point in points)) + 8)
            crop = page_image.crop((left, top, right, bottom))
            raw, latency = call_vision(REPAIR_PROMPT, crop)
            corrected = raw.strip().strip("`")
            if corrected:
                original = span.get("text", "")
                span["text_original"] = original
                span["text"] = corrected
                span["text_corrected"] = corrected
                span["ocr_source"] = "vlm_repair"
                span["repair_latency_sec"] = round(latency, 2)
                repaired.append(span.get("index"))
    return spans, sum(float(span.get("repair_latency_sec", 0.0)) for span in spans if span.get("index") in repaired)


def read_page(page_dir: Path, apply_bbox_filter: bool = True) -> dict:
    started = time.perf_counter()
    path = long_path(page_dir / f"{page_dir.name}.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    ocr_latency = time.perf_counter() - started
    spans = data.get("spans", [])
    if not spans:
        return {
            "page_label": page_dir.name,
            "page_number": data.get("page_number", 0),
            "source_pdf": data.get("source_pdf"),
            "text": "",
            "spans": [],
            "entities": {},
            "status": "NO_SPANS",
            "ocr_latency_sec": round(ocr_latency, 4),
            "llm_latency_sec": 0.0,
        }
    with Image.open(long_path(page_dir / f"{page_dir.name}.png")) as image:
        page_image = image.copy()
        if apply_bbox_filter:
            spans, oversized_ids = filter_large_spans(spans, image.width, image.height)
        else:
            oversized_ids = []
    suspicious_ids, review_latency = review_spans(spans)
    spans, repair_latency = repair_suspicious_spans(page_dir, spans, suspicious_ids)
    spans_by_index = {s.get("index", i): s for i, s in enumerate(spans)}
    redaction_started = time.perf_counter()
    raw = call_llm(build_user_prompt(spans))
    redaction_latency = time.perf_counter() - redaction_started
    visual_started = time.perf_counter()
    visual_entities, visual_latency = visual_redactions(page_image)
    visual_latency = time.perf_counter() - visual_started
    entities = parse_and_validate(raw, spans_by_index)
    page_text = " ".join(str(span.get("text", "")) for span in spans).casefold()
    if "sgs" in page_text and ("laboratory" in page_text or "sgs group" in page_text):
        for category in ("PHONE", "FAX", "EMAIL"):
            entities[category] = []
    return {
        "page_label": page_dir.name,
        "page_number": data.get("page_number", 0),
        "source_pdf": data.get("source_pdf"),
        "text": "\n".join(str(span.get("text", "")) for span in spans),
        "spans": spans,
        "entities": entities,
        "visual_redactions": visual_entities,
        "status": "OK",
        "ocr_latency_sec": round(ocr_latency, 4),
        "review_latency_sec": round(review_latency, 2),
        "repair_latency_sec": round(repair_latency, 2),
        "llm_latency_sec": round(redaction_latency, 2),
        "visual_llm_latency_sec": round(visual_latency, 2),
        "suspicious_span_ids": suspicious_ids,
        "suspicious_count": len(suspicious_ids),
        "oversized_span_ids_removed": oversized_ids,
    }


def bbox_review_pdf(document_name: str, pages: list[dict], output_path: Path) -> int:
    pdf = fitz.open()
    regions_by_page = []
    for page_record in sorted(pages, key=lambda item: item["page_label"]):
        page_dir = long_path(OCR_ROOT / document_name / page_record["page_label"])
        image_path = long_path(page_dir / f"{page_record['page_label']}.png")
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        page = pdf.new_page(width=width, height=height)
        page.insert_image(page.rect, filename=str(image_path))
        for entries in page_record.get("entities", {}).values():
            for entity in entries or []:
                bbox = entity.get("bbox")
                if not bbox:
                    continue
                try:
                    rect = fitz.Rect(
                        float(bbox["x0"]), float(bbox["y0"]),
                        float(bbox["x1"]), float(bbox["y1"]),
                    ) & page.rect
                except (KeyError, TypeError, ValueError):
                    continue
                regions_by_page.append({
                    "page_number": len(pdf) - 1,
                    "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1,
                })
        for entity in page_record.get("visual_redactions", []):
            bbox = entity.get("bbox")
            if not bbox:
                continue
            try:
                rect = fitz.Rect(
                    float(bbox["x0"]), float(bbox["y0"]),
                    float(bbox["x1"]), float(bbox["y1"]),
                ) & page.rect
            except (KeyError, TypeError, ValueError):
                continue
            regions_by_page.append({
                "page_number": len(pdf) - 1,
                "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1,
            })
    for region in regions_by_page:
        pdf[region["page_number"]].draw_rect(
            fitz.Rect(region["x0"], region["y0"], region["x1"], region["y1"]),
            color=(1, 0, 0), width=3,
        )
    boxes = len(regions_by_page)
    if not len(pdf):
        pdf.close()
        raise RuntimeError(f"No page images available for {document_name}")
    output_path = long_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(str(output_path), garbage=4, deflate=True)
    pdf.close()
    return boxes


def actual_redacted_pdf(
    document_name: str,
    pages: list[dict],
    source_pdf: str,
    output_path: Path,
    apply_bbox_filter: bool = True,
) -> dict:
    source_dir = OCR_ROOT / document_name
    regions = []
    sensitive_values = []
    qr_started = time.perf_counter()
    machine_readable_regions, machine_readable_detections = detect_machine_readable_regions(
        long_path(Path(source_pdf))
    )
    qr_latency = time.perf_counter() - qr_started
    selected_values = set()
    for page_record in pages:
        for entries in page_record.get("entities", {}).values():
            for entity in entries or []:
                value = str(entity.get("value", "")).strip()
                if value:
                    selected_values.add(re.sub(r"[^a-z0-9]+", "", value.casefold()))

    def native_text_regions(page: fitz.Page) -> list[fitz.Rect]:
        spans = [
            span
            for block in page.get_text("dict").get("blocks", [])
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
        normalized_spans = [re.sub(r"[^a-z0-9]+", "", str(span.get("text", "")).casefold()) for span in spans]
        joined = "".join(normalized_spans)
        offsets = []
        position = 0
        for span, normalized in zip(spans, normalized_spans):
            offsets.append((position, position + len(normalized), span))
            position += len(normalized)
        regions = []
        for value in selected_values:
            start = joined.find(value)
            best_match = None
            if start < 0 and len(value) >= 8:
                for first in range(len(normalized_spans)):
                    candidate = ""
                    for last in range(first, min(first + 4, len(normalized_spans))):
                        candidate += normalized_spans[last]
                        if not candidate:
                            continue
                        similarity = SequenceMatcher(None, value, candidate).ratio()
                        if best_match is None or similarity > best_match[0]:
                            best_match = (similarity, first, last + 1)
                if best_match and best_match[0] >= 0.96:
                    matching_spans = spans[best_match[1]:best_match[2]]
                    if matching_spans:
                        regions.append(fitz.Rect(
                            min(span["bbox"][0] for span in matching_spans), min(span["bbox"][1] for span in matching_spans),
                            max(span["bbox"][2] for span in matching_spans), max(span["bbox"][3] for span in matching_spans),
                        ))
                    continue
            while start >= 0:
                end = start + len(value)
                matching_spans = [span for span_start, span_end, span in offsets if span_end > start and span_start < end]
                if matching_spans:
                    regions.append(fitz.Rect(
                        min(span["bbox"][0] for span in matching_spans), min(span["bbox"][1] for span in matching_spans),
                        max(span["bbox"][2] for span in matching_spans), max(span["bbox"][3] for span in matching_spans),
                    ))
                start = joined.find(value, start + 1)
        return regions

    with fitz.open(str(long_path(Path(source_pdf)))) as document:
        for page_record in pages:
            page_number = int(page_record.get("page_number", 0))
            if page_number < 0 or page_number >= len(document):
                continue
            page_dir = long_path(source_dir / page_record["page_label"])
            image_path = long_path(page_dir / f"{page_record['page_label']}.png")
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            page_rect = document[page_number].rect
            scale_x = page_rect.width / image_width
            scale_y = page_rect.height / image_height
            for rect in native_text_regions(document[page_number]):
                regions.append({
                    "page_number": page_number,
                    "x0": rect.x0, "y0": rect.y0,
                    "x1": rect.x1, "y1": rect.y1,
                })
            spans_by_index = {
                int(span.get("index", position)): span
                for position, span in enumerate(page_record.get("spans", []))
            }
            selected_span_indices = {
                span_index
                for span_index, span in spans_by_index.items()
                if re.sub(r"[^a-z0-9]+", "", str(span.get("text", "")).casefold()) in selected_values
            }
            for entries in page_record.get("entities", {}).values():
                for entity in entries or []:
                    if entity.get("value"):
                        sensitive_values.append(str(entity["value"]))
                    bbox = entity.get("bbox")
                    if not bbox:
                        continue
                    try:
                        rect = fitz.Rect(
                            float(bbox["x0"]) * scale_x,
                            float(bbox["y0"]) * scale_y,
                            float(bbox["x1"]) * scale_x,
                            float(bbox["y1"]) * scale_y,
                        ) & page_rect
                    except (KeyError, TypeError, ValueError):
                        continue
                    regions.append({
                        "page_number": page_number,
                        "x0": rect.x0, "y0": rect.y0,
                        "x1": rect.x1, "y1": rect.y1,
                    })
                    selected_span_indices.discard(int(entity.get("span_indices", [-1])[0]))
            for span_index in selected_span_indices:
                span = spans_by_index[span_index]
                polygon = span.get("polygon", [])
                if not polygon:
                    continue
                xs = [float(point[0]) for point in polygon]
                ys = [float(point[1]) for point in polygon]
                rect = fitz.Rect(
                    min(xs) * scale_x, min(ys) * scale_y,
                    max(xs) * scale_x, max(ys) * scale_y,
                ) & page_rect
                if not rect.is_empty:
                    regions.append({
                        "page_number": page_number,
                        "x0": rect.x0, "y0": rect.y0,
                        "x1": rect.x1, "y1": rect.y1,
                    })
                    sensitive_values.append(str(span.get("text", "")))
            for entity in page_record.get("visual_redactions", []):
                bbox = entity.get("bbox")
                if not bbox:
                    continue
                try:
                    rect = fitz.Rect(
                        float(bbox["x0"]) * scale_x,
                        float(bbox["y0"]) * scale_y,
                        float(bbox["x1"]) * scale_x,
                        float(bbox["y1"]) * scale_y,
                    ) & page_rect
                except (KeyError, TypeError, ValueError):
                    continue
                regions.append({
                    "page_number": page_number,
                    "x0": rect.x0, "y0": rect.y0,
                    "x1": rect.x1, "y1": rect.y1,
                })
    pdf_started = time.perf_counter()
    audit = redact_pdf(
        long_path(Path(source_pdf)),
        long_path(output_path),
        regions,
        apply_bbox_filter=apply_bbox_filter,
        sensitive_values=sensitive_values,
        machine_readable_regions=machine_readable_regions,
        machine_readable_detections=machine_readable_detections,
    )
    pdf_latency = time.perf_counter() - pdf_started
    return {
        "redacted_boxes": len(regions), "release_audit": audit,
        "timings_sec": {"qr_barcode_detection": round(qr_latency, 2), "pdf_redaction_and_audit": round(pdf_latency, 2)},
    }


def process_document(document_name: str, batch_dir: Path, workers: int, logger: logging.Logger, progress: Progress, page_task: int, stats: dict, stats_lock: threading.Lock, page_limit: int | None = None, apply_bbox_filter: bool = True) -> dict:
    source_dir = long_path(OCR_ROOT / document_name)
    page_dirs = page_dirs_sorted(source_dir)
    if page_limit:
        page_dirs = page_dirs[:page_limit]
    pages: list[dict] = []
    errors: list[dict] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(read_page, page_dir, apply_bbox_filter): page_dir.name
            for page_dir in page_dirs
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                page = future.result()
                pages.append(page)
                with stats_lock:
                    stats["pages_done"] += 1
                    stats["ocr_total"] += page.get("ocr_latency_sec", 0.0)
                    stats["llm_total"] += page.get("llm_latency_sec", 0.0)
                    stats["llm_calls"] += page.get("status") == "OK"
                    stats["review_total"] += page.get("review_latency_sec", 0.0)
                    stats["repair_total"] += page.get("repair_latency_sec", 0.0)
                    stats["suspicious_total"] += page.get("suspicious_count", 0)
                    avg_ocr = stats["ocr_total"] / stats["pages_done"]
                    avg_llm = stats["llm_total"] / stats["llm_calls"] if stats["llm_calls"] else 0.0
                    avg_review = stats["review_total"] / stats["pages_done"]
                    progress.update(page_task, advance=1, status=f"pages={stats['pages_done']}/{stats['total_pages']} suspicious={stats['suspicious_total']} review={avg_review:.2f}s repair={stats['repair_total']:.2f}s redaction={avg_llm:.2f}s")
            except Exception as error:
                logger.error("PAGE_FAIL | %s | %s | %s", document_name, label, error)
                errors.append({"page_label": label, "error": str(error)})
                with stats_lock:
                    stats["pages_done"] += 1
                    progress.update(page_task, advance=1, status=f"pages={stats['pages_done']}/{stats['total_pages']} page_error={label}")
    pages.sort(key=lambda item: item["page_label"])
    source_pdf = next((page.get("source_pdf") for page in pages if page.get("source_pdf")), None)
    document_export = {
        "format": "timestamped_ocr_recognition_export",
        "document": document_name,
        "source_pdf": source_pdf,
        "page_count": len(pages),
        "pages": pages,
    }
    redaction_export = {
        "format": "timestamped_redaction_information_export",
        "document": document_name,
        "source_pdf": source_pdf,
        "page_count": len(pages),
        "pages": [
            {
                "page_label": page["page_label"],
                "page_number": page["page_number"],
                "status": page["status"],
                "entities": page.get("entities", {}),
                "visual_redactions": page.get("visual_redactions", []),
            }
            for page in pages
        ],
        "errors": errors,
    }
    package_dir = long_path(batch_dir / document_name)
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "recognition.json").write_text(json.dumps(document_export, ensure_ascii=False, indent=2), encoding="utf-8")
    (package_dir / "redaction_information.json").write_text(json.dumps(redaction_export, ensure_ascii=False, indent=2), encoding="utf-8")
    boxes = bbox_review_pdf(document_name, pages, package_dir / "bbox_review.pdf")
    release_name = f"redacted_document_{hashlib.sha1(document_name.encode('utf-8')).hexdigest()[:12]}.pdf"
    quarantine_pdf = long_path(package_dir / "quarantine" / release_name)
    quarantine_pdf.parent.mkdir(parents=True, exist_ok=True)
    redaction_result = actual_redacted_pdf(
        document_name,
        pages,
        source_pdf,
        quarantine_pdf,
        apply_bbox_filter=apply_bbox_filter,
    ) if source_pdf else 0
    redacted_boxes = redaction_result["redacted_boxes"] if source_pdf else 0
    release_audit = redaction_result["release_audit"] if source_pdf else {"status": "FAIL", "findings": {"source_pdf": "missing"}}
    timings_sec = redaction_result.get("timings_sec", {}) if source_pdf else {}
    status = "OK" if release_audit["status"] == "PASS" else "RELEASE_FAILED"
    release_file = ""
    if status == "OK":
        release_pdf = long_path(package_dir / "release" / release_name)
        release_pdf.parent.mkdir(parents=True, exist_ok=True)
        quarantine_pdf.replace(release_pdf)
        release_file = str(release_pdf)
        release_audit["pdf"] = release_file
    timings_sec["document_total"] = round(time.perf_counter() - started, 2)
    logger.info("DOCUMENT_%s | %s | pages=%d | boxes=%d | redacted_boxes=%d | errors=%d | timings=%s", status, document_name, len(pages), boxes, redacted_boxes, len(errors), timings_sec)
    return {"document": document_name, "pages": len(pages), "boxes": boxes, "redacted_boxes": redacted_boxes, "errors": len(errors), "status": status, "release_file": release_file, "quarantine_file": str(quarantine_pdf) if status != "OK" else "", "release_audit": release_audit, "timings_sec": timings_sec}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent LLM page workers (default: 8).")
    parser.add_argument("--smoke", action="store_true", help="Process one page from one document.")
    parser.add_argument("--document", help="Process one complete OCR document folder.")
    parser.add_argument("--document-limit", type=int, help="Process only the first N OCR document folders.")
    parser.add_argument("--no-bbox-filter", action="store_true", help="Keep every valid OCR and redaction bbox.")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    documents = sorted(path.name for path in OCR_ROOT.iterdir() if path.is_dir())
    if args.document:
        if args.document not in documents:
            raise SystemExit(f"Document folder not found: {args.document}")
        documents = [args.document]
    if args.document_limit:
        documents = documents[:args.document_limit]
    if args.smoke and not args.document_limit:
        documents = documents[:1]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = BATCH_ROOT / f"batch_{run_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_path = batch_dir / "batch.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(log_path, encoding="utf-8")])
    logger = logging.getLogger("timestamped_batch")
    logger.info("START | documents=%d | workers=%d", len(documents), args.workers)
    results = []
    total_pages = sum((1 if args.smoke else len(page_dirs_sorted(long_path(OCR_ROOT / name)))) for name in documents)
    stats = {"pages_done": 0, "total_pages": total_pages, "ocr_total": 0.0, "llm_total": 0.0, "llm_calls": 0, "review_total": 0.0, "repair_total": 0.0, "suspicious_total": 0}
    stats_lock = threading.Lock()
    with Progress(TextColumn("Redaction batch"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("documents", total=len(documents), status="starting")
        page_task = progress.add_task("pages", total=total_pages, status=f"pages=0/{total_pages} avg_ocr=0.000s avg_llm=0.00s")
        for document_name in documents:
            try:
                result = process_document(
                    document_name,
                    batch_dir,
                    args.workers,
                    logger,
                    progress,
                    page_task,
                    stats,
                    stats_lock,
                    1 if args.smoke else None,
                    apply_bbox_filter=not args.no_bbox_filter,
                )
            except Exception as error:
                logger.exception("DOCUMENT_FAIL | %s | %s", document_name, error)
                result = {"document": document_name, "pages": 0, "boxes": 0, "redacted_boxes": 0, "errors": 1, "status": "ERROR", "error": str(error)}
            results.append(result)
            progress.advance(task)
            progress.update(task, description=f"documents | ok={sum(r['status'] == 'OK' for r in results)} errors={sum(r.get('errors', 0) for r in results)}")
    recognition_documents = []
    redaction_documents = []
    for result in results:
        package_dir = batch_dir / result["document"]
        recognition_path = package_dir / "recognition.json"
        redaction_path = package_dir / "redaction_information.json"
        if recognition_path.exists():
            recognition_documents.append(json.loads(recognition_path.read_text(encoding="utf-8")))
        if redaction_path.exists():
            redaction_documents.append(json.loads(redaction_path.read_text(encoding="utf-8")))
    consolidated_recognition = {
        "format": "consolidated_timestamped_ocr_recognition_export",
        "document_count": len(recognition_documents),
        "total_page_count": sum(document.get("page_count", 0) for document in recognition_documents),
        "documents": recognition_documents,
    }
    consolidated_redaction = {
        "format": "consolidated_timestamped_redaction_information_export",
        "bbox_filter_used": not args.no_bbox_filter,
        "document_count": len(redaction_documents),
        "total_page_count": sum(document.get("page_count", 0) for document in redaction_documents),
        "documents": redaction_documents,
    }
    (batch_dir / "consolidated_recognition.json").write_text(
        json.dumps(consolidated_recognition, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (batch_dir / "consolidated_redaction.json").write_text(
        json.dumps(consolidated_redaction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit_path = write_release_audit(batch_dir / "release_audit.xlsx", results)
    manifest = {
        "batch_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workers": args.workers,
        "bbox_filter_used": not args.no_bbox_filter,
        "document_count": len(documents),
        "documents": results,
        "consolidated_recognition": str(batch_dir / "consolidated_recognition.json"),
        "consolidated_redaction": str(batch_dir / "consolidated_redaction.json"),
        "release_audit": str(audit_path),
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("DONE | batch=%s | documents=%d", batch_dir, len(results))
    console.print(f"Batch complete: {batch_dir}")
    console.print(f"Manifest: {batch_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
