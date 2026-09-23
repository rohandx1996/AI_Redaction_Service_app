"""Infer redaction patterns by comparing published PDFs with their source reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from config import settings

REPO_ROOT = Path(__file__).resolve().parent
PUBLISHED_ROOT = Path(r"C:\Users\DHR1RT\Downloads\OneDrive_1_9-7-2026\00_Published")
REPORT_ROOT = REPO_ROOT / "output" / "reports" / "published_redaction_analysis"
console = Console()

SYSTEM_PROMPT = """You are auditing Bosch published supplier-compliance reports to identify every redaction key/value visible or inferable from the published report.
You receive only the exact published PDF text from the 00_Published folder.
Report document-level findings individually. Do not average, summarize away, or return only recurring values.

Capture:
- every visible Redacted_by_Bosch marker and its nearby field/label;
- every visible Bosch redaction marker and nearby field label;
- the redaction key/category, such as SAMPLE_NAME, SUPPLIER_NAME, COMPANY, LABORATORY_NAME,
  PERSON_NAME, ADDRESS, PHONE, FAX, URL, REPORT_NUMBER, PIN_CODE, SUPPLIER_IDENTIFIER,
  PRODUCT_NAME, PART_NUMBER, or OTHER_IDENTITY.

Do not claim the hidden value from a marker alone. For marker-only findings, set value to the exact marker text and explain that the original value is unavailable.

Return JSON only:
{
    "redacted_items": [
        {"value":"exact visible marker or exact original candidate", "field_label":"nearby label or empty", "category":"SAMPLE_NAME|SUPPLIER_NAME|MANUFACTURER_NAME|LABORATORY_NAME|PERSON_NAME|ADDRESS|CONTACT|URL|REPORT_ID|PIN_CODE|SUPPLIER_IDENTIFIER|PRODUCT_NAME|PART_NUMBER|OTHER_IDENTITY", "source":"published_marker|original_absent_candidate", "confidence":0.0, "reason":"brief"}
  ],
  "kept_examples": ["technical values clearly retained"],
  "policy_observations": ["short observations about what appears consistently redacted"]
}
Do not invent values. Treat candidate differences as evidence, not proof: layout extraction can create false differences.
Do not classify chemical names, standards, methods, results, units, or scientific content as redactions."""


def all_pdfs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.pdf")) if root.exists() else []


def extract_pages(path: Path) -> list[str]:
    with fitz.open(str(path)) as doc:
        return [" ".join(page.get_text("text").split()) for page in doc]


def call_llm(payload: dict, retries: int = 2) -> dict:
    import litellm
    prompt = json.dumps(payload, ensure_ascii=False)
    for attempt in range(retries + 1):
        try:
            response = litellm.completion(
                model=f"openai/{settings.llm_model}",
                api_base=settings.bosch_gateway_url.rstrip("/") + "/v1",
                api_key=settings.bosch_gateway_token,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=3000,
                temperature=0,
            )
            raw = response["choices"][0]["message"]["content"] or "{}"
            raw = raw.strip().strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            return json.loads(raw)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def analyze_one(published: Path) -> dict:
    published_pages = extract_pages(published)
    published_text = "\n".join(published_pages)
    marker_candidates = [
        line.strip() for line in published_text.splitlines()
        if "redacted_by_bosch" in line.lower() or "redacted by bosch" in line.lower()
    ]
    payload = {
        "published_file": published.name,
        "published_text": published_text[:30000],
        "visible_bosch_redaction_markers": marker_candidates,
    }
    result = call_llm(payload)
    return {
        "published": str(published), "status": "ANALYZED",
        "marker_count": len(marker_candidates), "redacted_items": result.get("redacted_items", []),
        "kept_examples": result.get("kept_examples", []),
        "policy_observations": result.get("policy_observations", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not settings.bosch_gateway_token or not settings.bosch_gateway_url:
        raise SystemExit("Bosch Gateway credentials not configured (.env).")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    published = all_pdfs(PUBLISHED_ROOT)
    if args.limit:
        published = published[:args.limit]
    results = []
    with Progress(TextColumn("Published redaction analysis"), BarColumn(), "{task.completed}/{task.total}", console=console) as progress:
        task = progress.add_task("documents", total=len(published))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(analyze_one, path): path for path in published}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append({"published": str(path), "status": "ERROR", "error": str(error)})
                progress.advance(task)
    results.sort(key=lambda row: row["published"])
    (REPORT_ROOT / "document_analysis.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {}
    observations = []
    for row in results:
        for item in row.get("redacted_items", []):
            key = (item.get("category", "UNKNOWN"), item.get("value", ""))
            counts[key] = counts.get(key, 0) + 1
        observations.extend(row.get("policy_observations", []))
    summary = {
        "documents_seen": len(published),
        "documents_analyzed": sum(row.get("status") == "ANALYZED" for row in results),
        "documents_with_errors": sum(row.get("status") == "ERROR" for row in results),
        "recurring_redacted_values": [
            {"category": category, "value": value, "document_count": count}
            for (category, value), count in sorted(counts.items(), key=lambda item: -item[1])
        ],
        "policy_observations": sorted(set(observations)),
    }
    (REPORT_ROOT / "redaction_patterns_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Wrote {REPORT_ROOT / 'document_analysis.json'}")
    console.print(f"Wrote {REPORT_ROOT / 'redaction_patterns_summary.json'}")


if __name__ == "__main__":
    main()
