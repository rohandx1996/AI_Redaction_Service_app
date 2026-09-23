"""Inventory knowledge retained in Bosch published reports using LiteLLM."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn

from config import settings

PUBLISHED_ROOT = Path(r"C:\Users\DHR1RT\Downloads\OneDrive_1_9-7-2026\00_Published")
REPORT_ROOT = Path(__file__).resolve().parent / "output" / "reports" / "published_knowledge_inventory"
console = Console()

SYSTEM_PROMPT = """You are auditing Bosch published supplier-compliance reports.
Extract the knowledge that remains visible and was NOT redacted by Bosch engineers.
Use only text present in the supplied published PDF text. Do not invent, translate, or infer hidden text.
Do not list supplier/private identities as retained knowledge unless they are visibly present and clearly part of the published content.
Return JSON only with these categories:
{
  "technical_terms": [], "chemical_names": [], "cas_numbers": [],
  "materials_and_products": [], "standards_and_regulations": [],
  "test_methods_and_instruments": [], "analytical_procedures": [],
  "flowcharts_and_workflows": [], "measurement_results_and_units": [],
  "table_headings_and_fields": [], "sample_or_product_descriptions": [],
  "conclusions_and_compliance_statements": [], "dates_and_page_structure": [],
  "other_retained_knowledge": []
}
Each item must be an exact visible phrase from the published text. Deduplicate within each category.
Keep scientific, regulatory, analytical, technical, table, workflow, diagram-description, and result content.
Do not return explanations outside JSON."""


def extract_text(path: Path) -> tuple[str, int]:
    with fitz.open(str(path)) as doc:
        return "\n".join(page.get_text("text") for page in doc), len(doc)


def call_llm(filename: str, text: str, retries: int = 2) -> dict:
    import litellm
    payload = f"Published report filename: {filename}\n\nPublished report text:\n{text[:50000]}"
    for attempt in range(retries + 1):
        try:
            response = litellm.completion(
                model=f"openai/{settings.llm_model}",
                api_base=settings.bosch_gateway_url.rstrip("/") + "/v1",
                api_key=settings.bosch_gateway_token,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": payload}],
                max_tokens=4000, temperature=0,
            )
            raw = (response["choices"][0]["message"]["content"] or "{}").strip().strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            return json.loads(raw)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def analyze(path: Path) -> dict:
    text, page_count = extract_text(path)
    result = call_llm(path.name, text)
    return {"published_file": str(path), "page_count": page_count, "knowledge": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not settings.bosch_gateway_token or not settings.bosch_gateway_url:
        raise SystemExit("Bosch Gateway credentials not configured (.env).")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    files = sorted(PUBLISHED_ROOT.rglob("*.pdf"))
    if args.limit:
        files = files[:args.limit]
    results = []
    with Progress(TextColumn("Published knowledge inventory"), BarColumn(), "{task.completed}/{task.total}", console=console) as progress:
        task = progress.add_task("documents", total=len(files))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(analyze, path): path for path in files}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append({"published_file": str(path), "status": "ERROR", "error": str(error)})
                progress.advance(task)
    results.sort(key=lambda item: item["published_file"])
    (REPORT_ROOT / "document_knowledge_inventory.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    categories = {}
    for row in results:
        for category, values in row.get("knowledge", {}).items():
            categories.setdefault(category, set()).update(str(value) for value in values if value)
    summary = {
        "documents_seen": len(files),
        "documents_analyzed": sum(row.get("status", "OK") == "OK" for row in results),
        "documents_with_errors": sum(row.get("status") == "ERROR" for row in results),
        "unique_retained_knowledge": {category: sorted(values) for category, values in sorted(categories.items())},
        "unique_counts": {category: len(values) for category, values in sorted(categories.items())},
    }
    (REPORT_ROOT / "retained_knowledge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Wrote {REPORT_ROOT / 'document_knowledge_inventory.json'}")
    console.print(f"Wrote {REPORT_ROOT / 'retained_knowledge_summary.json'}")


if __name__ == "__main__":
    main()
