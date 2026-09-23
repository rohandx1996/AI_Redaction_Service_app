"""Benchmark gateway VLM/text extraction models on one original document."""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

import fitz
import requests
from dotenv import load_dotenv

load_dotenv()
BASE = os.getenv("BOSCH_GATEWAY_URL", "").rstrip("/")
TOKEN = os.getenv("BOSCH_GATEWAY_TOKEN", "")
MATCH_ROOT = Path("output/matched_original_counterparts/matches_20260916_214743")
DOCUMENT = MATCH_ROOT / "0001_Bosch_BC01_Backside_Coating_2025-02-07_RoHS_Halogens_EKR25101752"
ORIGINAL = DOCUMENT / "original_01.pdf"
GROUND_TRUTH = MATCH_ROOT / "litellm_all_original_vs_published_redactions.json"
OUTPUT = Path("output/model_benchmarks/vlm_original_document_01_20260921")

POLICY = """Protect supplier/customer/applicant identity, protected addresses, people/signatures, submitted sample names, product/style/model/part identifiers, private references, PIN/access codes, and contact details. Retain laboratory branding, technical methods, chemicals, standards, units, results, procedures, and generic report identifiers unless clearly private. Preserve visible values exactly. Read multilingual text. Return JSON only: {\"redacted_information\":[{\"category\":\"...\",\"original_value\":\"...\",\"confidence\":0.0}]}"""
MODELS = [
    ("claude-sonnet-5", "anthropic"),
    ("claude-sonnet-4-5@20250929", "anthropic"),
    ("claude-opus-4-5@20251101", "anthropic"),
    ("gpt-4o", "openai"),
    ("gpt-4o-mini", "openai"),
    ("Qwen/Qwen3.6-35B-A3B", "openai"),
    ("zai-glm-52", "openai"),
]


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def clean_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text or "{}")


def page_jpeg(page: fitz.Page) -> str:
    pix = page.get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72), alpha=False)
    return base64.b64encode(pix.tobytes("jpeg")).decode("ascii")


def call_model(model: str, api: str, image: str, page_number: int) -> dict:
    prompt = f"Original page {page_number}. Extract every sensitive value that should be redacted.\nPolicy:\n{POLICY}"
    if api == "anthropic":
        response = requests.post(
            BASE + "/v1/messages",
            headers={"x-api-key": TOKEN, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": 4096, "system": POLICY, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image}}]}]},
            timeout=180,
        )
        response.raise_for_status()
        text = "".join(block.get("text", "") for block in response.json().get("content", []) if block.get("type") == "text")
        return clean_json(text)
    response = requests.post(
        BASE + "/v1/chat/completions",
        headers={"Authorization": "Bearer " + TOKEN, "content-type": "application/json"},
        json={"model": model, "messages": [{"role": "system", "content": POLICY}, {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}}]}], "max_tokens": 4096, "temperature": 0},
        timeout=180,
    )
    response.raise_for_status()
    body = response.json()
    return clean_json(body.get("choices", [{}])[0].get("message", {}).get("content", ""))


def truth_values() -> set[str]:
    data = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    document = next(item for item in data["documents"] if "EKR25101752" in item.get("original_pdf", ""))
    return {normalize(item.get("original_value")) for item in document.get("redacted_information", []) if item.get("original_value")}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    truth = truth_values()
    with fitz.open(str(ORIGINAL)) as document:
        pages = [page_jpeg(page) for page in document]
    summary = []
    for model, api in MODELS:
        started = time.perf_counter()
        values = []
        pages_out = []
        error = ""
        try:
            for page_number, image in enumerate(pages, start=1):
                page_started = time.perf_counter()
                result = call_model(model, api, image, page_number)
                page_values = result.get("redacted_information", []) or []
                values.extend(item for item in page_values if isinstance(item, dict))
                pages_out.append({"page_number": page_number, "result": result, "seconds": round(time.perf_counter() - page_started, 2)})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        predicted = {normalize(item.get("original_value")) for item in values if item.get("original_value")}
        tp = len(predicted & truth)
        summary.append({"model": model, "api": api, "status": "OK" if not error else "ERROR", "seconds": round(time.perf_counter() - started, 2), "seconds_per_page": round((time.perf_counter() - started) / len(pages), 2), "predicted_values": len(predicted), "truth_values": len(truth), "true_positive_values": tp, "recall": tp / len(truth) if truth else 0, "precision": tp / len(predicted) if predicted else 0, "error": error})
        (OUTPUT / (model.replace("/", "_").replace("@", "_") + ".json")).write_text(json.dumps({"model": model, "api": api, "pages": pages_out, "error": error}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(summary[-1])
    (OUTPUT / "summary.json").write_text(json.dumps({"document": str(ORIGINAL), "ground_truth": str(GROUND_TRUTH), "results": summary}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
