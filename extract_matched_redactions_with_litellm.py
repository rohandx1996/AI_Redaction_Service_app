"""Compare matched original/published PDFs with the configured LLM gateway."""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time

import fitz
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from config import settings


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MATCH_ROOT = REPO_ROOT / "output" / "matched_original_counterparts" / "matches_20260916_214743"
PAGE_RESULT_DIR = "litellm_pair_pages"
DOCUMENT_RESULT = "litellm_original_vs_published_redactions.json"
AGGREGATE_RESULT = "litellm_all_original_vs_published_redactions.json"
REDACTED_RESULT = "litellm_redacted_information.json"
RETAINED_RESULT = "litellm_retained_information.json"
SUMMARY_RESULT = "LITELLM_ORIGINAL_VS_PUBLISHED_SUMMARY.md"
console = Console()

SYSTEM_PROMPT = """You compare an ORIGINAL supplier compliance report page with its Bosch PUBLISHED page.
Identify exact information values visible in ORIGINAL that Bosch removed, covered, replaced with a
Redacted_by_Bosch marker, or otherwise made unavailable in PUBLISHED.

Use visual comparison first; extracted text is supporting evidence because either PDF may be flattened.
Different laboratories use different labels for the same semantic role. Infer the field role from nearby
labels and layout, but copy original values exactly. Examples of equivalent context include Client Name,
Customer, Applicant, Sample Submitted By, Provided By Applicant, Sample Name, Sample Description,
Style/Item No., Client Ref., Order No., Lot No., and their multilingual equivalents.

Do not report technical information that remains visible, including chemicals, CAS numbers, standards,
methods, instruments, procedures, flowcharts, measurements, limits, results, and conclusions. Do not
assume all missing extracted text is redacted when page extraction differs. Report only visually supported
differences. Generic labels are context, not values, unless Bosch removed the label itself.

Return JSON only:
{
  "redacted_information": [
    {
      "category": "SUPPLIER_NAME|CUSTOMER_NAME|APPLICANT_NAME|LABORATORY_NAME|PERSON_NAME|ADDRESS|CONTACT|SAMPLE_NAME|PRODUCT_OR_PART_ID|CLIENT_OR_SAMPLE_REFERENCE|REPORT_OR_ACCESS_ID|OTHER_IDENTITY",
      "field_label": "nearby original label or empty",
      "original_value": "exact original visible value",
      "published_state": "marker|covered|removed|partially_redacted",
      "published_visible_value": "marker or remaining fragment, otherwise empty",
      "confidence": 0.0,
      "evidence": "brief visual reason"
    }
  ],
  "retained_technical_examples": ["short exact examples that confirm technical knowledge remains"],
  "comparison_notes": ["brief uncertainty or layout notes"]
}
Do not invent hidden values or report a redaction without visual evidence."""


def clean_json(raw: str) -> dict:
    value = (raw or "").strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
        if value.lower().startswith("json"):
            value = value[4:].strip()
    return json.loads(value)


def page_jpeg(document: fitz.Document, page_number: int, dpi: int = 120) -> bytes:
    pixmap = document[page_number].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=78, optimize=True)
    return buffer.getvalue()


def page_text(document: fitz.Document, page_number: int) -> str:
    return " ".join(document[page_number].get_text("text").split())[:16000]


def image_data(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def call_gateway(original_image: bytes, published_image: bytes, prompt: str, retries: int = 2) -> dict:
    last_error = None
    for attempt in range(retries + 1):
        try:
            if settings.llm_model.startswith("claude-"):
                import requests

                response = requests.post(
                    settings.bosch_gateway_url.rstrip("/") + "/v1/messages",
                    headers={
                        "x-api-key": settings.bosch_gateway_token,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": settings.llm_model,
                        "system": SYSTEM_PROMPT,
                        "max_tokens": 4096,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "ORIGINAL PAGE IMAGE:"},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data(original_image)}},
                                {"type": "text", "text": "PUBLISHED PAGE IMAGE:"},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data(published_image)}},
                                {"type": "text", "text": prompt},
                            ],
                        }],
                    },
                    timeout=180,
                )
                response.raise_for_status()
                raw = "".join(
                    block.get("text", "")
                    for block in response.json().get("content", [])
                    if block.get("type") == "text"
                )
                return clean_json(raw)

            import litellm

            response = litellm.completion(
                model=f"openai/{settings.llm_model}",
                api_base=settings.bosch_gateway_url.rstrip("/") + "/v1",
                api_key=settings.bosch_gateway_token,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data(original_image)}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data(published_image)}"}},
                    ]},
                ],
                max_tokens=4096,
                temperature=0,
            )
            return clean_json(response["choices"][0]["message"]["content"] or "{}")
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Gateway comparison failed: {last_error}")


def best_published(group: dict) -> Path | None:
    candidates = [Path(item["path"]) for item in group["counterparts"] if item["role"] == "published"]
    if not candidates:
        return None
    scored = []
    for path in candidates:
        try:
            with fitz.open(str(path)) as document:
                characters = sum(len(page.get_text("text")) for page in document)
            scored.append((characters, path))
        except Exception:
            continue
    return max(scored, default=(0, None), key=lambda item: item[0])[1]


def compare_page(original_path: Path, published_path: Path, page_number: int, output_path: Path) -> dict:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    with fitz.open(str(original_path)) as original, fitz.open(str(published_path)) as published:
        prompt = (
            f"Page number: {page_number + 1}\n"
            f"Original extracted text:\n{page_text(original, page_number)}\n\n"
            f"Published extracted text:\n{page_text(published, page_number)}"
        )
        result = call_gateway(
            page_jpeg(original, page_number),
            page_jpeg(published, page_number),
            prompt,
        )
    record = {
        "page_number": page_number + 1,
        "model": settings.llm_model,
        "redacted_information": result.get("redacted_information", []),
        "retained_technical_examples": result.get("retained_technical_examples", []),
        "comparison_notes": result.get("comparison_notes", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def finalize_document(match_folder: Path, original_path: Path, published_path: Path, pages: list[dict], errors: list[dict]) -> dict:
    pages.sort(key=lambda item: item["page_number"])
    redactions = []
    for page in pages:
        for item in page["redacted_information"]:
            redactions.append({"page_number": page["page_number"], **item})
    result = {
        "format": "litellm_original_vs_published_redaction_extraction",
        "model": settings.llm_model,
        "original_pdf": str(original_path),
        "published_pdf": str(published_path),
        "page_count": len(pages),
        "redacted_item_count": len(redactions),
        "redacted_information": redactions,
        "pages": pages,
        "errors": errors,
    }
    (match_folder / DOCUMENT_RESULT).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_separated_outputs(match_root: Path, aggregate: dict) -> None:
    redacted_items = []
    retained_items = []
    category_counts = defaultdict(int)
    label_counts = defaultdict(int)
    for document in aggregate["documents"]:
        for item in document["redacted_information"]:
            record = {
                "original_pdf": document["original_pdf"],
                "published_pdf": document["published_pdf"],
                **item,
            }
            redacted_items.append(record)
            category_counts[item.get("category", "UNKNOWN")] += 1
            label = str(item.get("field_label", "")).strip()
            if label:
                label_counts[label] += 1
        for page in document["pages"]:
            for value in page.get("retained_technical_examples", []):
                retained_items.append({
                    "original_pdf": document["original_pdf"],
                    "published_pdf": document["published_pdf"],
                    "page_number": page["page_number"],
                    "value": value,
                    "evidence": "visible in matched original and published page",
                })

    redacted_output = {
        "format": "litellm_redacted_information_evidence",
        "model": aggregate["model"],
        "document_count": aggregate["document_count"],
        "page_count": aggregate["page_count"],
        "item_count": len(redacted_items),
        "category_counts": dict(sorted(category_counts.items())),
        "items": redacted_items,
    }
    retained_output = {
        "format": "litellm_retained_information_evidence",
        "model": aggregate["model"],
        "document_count": aggregate["document_count"],
        "page_count": aggregate["page_count"],
        "item_count": len(retained_items),
        "items": retained_items,
    }
    (match_root / REDACTED_RESULT).write_text(
        json.dumps(redacted_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (match_root / RETAINED_RESULT).write_text(
        json.dumps(retained_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# LiteLLM Original vs Published Redaction Findings",
        "",
        "## Coverage",
        "",
        f"- Model: `{aggregate['model']}`",
        f"- Matched original/published documents: {aggregate['document_count']}",
        f"- Pages compared visually: {aggregate['page_count']}",
        f"- Redacted key/value findings: {len(redacted_items)}",
        f"- Retained technical examples: {len(retained_items)}",
        f"- Documents with errors: {aggregate['documents_with_errors']}",
        f"- Match groups skipped without a published PDF: {len(aggregate['skipped'])}",
        "",
        "## Redacted Categories",
        "",
        "| Category | Findings |",
        "|---|---:|",
    ]
    for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {category} | {count} |")
    lines.extend([
        "",
        "## Contextual Field Wording",
        "",
        "Different laboratories use different labels for equivalent information. Frequent labels observed in the extracted evidence include:",
        "",
    ])
    for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[:40]:
        lines.append(f"- `{label}`: {count}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Bosch redaction decisions are contextual. Identity-bearing values tied to the customer, supplier, applicant, submitted sample, proprietary product/part, contact, address, or access/reference identifier are removed. Generic field labels and technical knowledge are generally retained.",
        "",
        "Each finding retains its original PDF, published PDF, page number, exact original value, field label, published state, confidence, and visual evidence in `litellm_redacted_information.json`.",
        "",
        "## Files",
        "",
        f"- `{REDACTED_RESULT}`: flattened redacted key/value evidence.",
        f"- `{RETAINED_RESULT}`: retained technical examples.",
        f"- `{AGGREGATE_RESULT}`: complete document/page responses.",
        f"- Each matched folder contains `{DOCUMENT_RESULT}` and resumable page JSON files.",
    ])
    (match_root / SUMMARY_RESULT).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-root", type=Path, default=DEFAULT_MATCH_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--document-limit", type=int)
    parser.add_argument("--document", help="Process one match-folder name.")
    parser.add_argument("--output-root", type=Path, help="Write fresh per-document and aggregate outputs here.")
    args = parser.parse_args()
    if not settings.bosch_gateway_url or not settings.bosch_gateway_token:
        raise SystemExit("Bosch gateway credentials are not configured")
    manifest = json.loads((args.match_root / "match_manifest.json").read_text(encoding="utf-8"))
    output_root = args.output_root.resolve() if args.output_root else args.match_root
    output_root.mkdir(parents=True, exist_ok=True)
    groups = manifest["matches"]
    if args.document:
        groups = [group for group in groups if Path(group["match_folder"]).name == args.document]
    if args.document_limit:
        groups = groups[:args.document_limit]

    jobs = []
    skipped = []
    for group in groups:
        match_folder = output_root / Path(group["match_folder"]).name
        original_path = Path(group["original"])
        published_path = best_published(group)
        if published_path is None:
            skipped.append({"original_pdf": str(original_path), "reason": "no published counterpart"})
            continue
        with fitz.open(str(original_path)) as original, fitz.open(str(published_path)) as published:
            page_count = min(len(original), len(published))
        for page_number in range(page_count):
            output_path = match_folder / PAGE_RESULT_DIR / f"page_{page_number + 1:04d}.json"
            jobs.append((match_folder, original_path, published_path, page_number, output_path))

    page_results: dict[Path, list[dict]] = defaultdict(list)
    errors: dict[Path, list[dict]] = defaultdict(list)
    with Progress(TextColumn("Matched pair extraction"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("pages", total=len(jobs))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(compare_page, original, published, page_number, output):
                (folder, original, published, page_number)
                for folder, original, published, page_number, output in jobs
            }
            for future in as_completed(futures):
                folder, original, published, page_number = futures[future]
                try:
                    page_results[folder].append(future.result())
                except Exception as error:
                    errors[folder].append({"page_number": page_number + 1, "error": str(error)})
                progress.advance(task)

    documents = []
    seen = set()
    for folder, original, published, _, _ in jobs:
        if folder in seen:
            continue
        seen.add(folder)
        documents.append(finalize_document(folder, original, published, page_results[folder], errors[folder]))
    aggregate = {
        "format": "litellm_all_original_vs_published_redaction_extractions",
        "model": settings.llm_model,
        "match_root": str(args.match_root),
        "document_count": len(documents),
        "page_count": sum(document["page_count"] for document in documents),
        "redacted_item_count": sum(document["redacted_item_count"] for document in documents),
        "documents_with_errors": sum(bool(document["errors"]) for document in documents),
        "skipped": skipped,
        "documents": documents,
    }
    aggregate_path = output_root / AGGREGATE_RESULT
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    write_separated_outputs(output_root, aggregate)
    print(json.dumps({
        "aggregate": str(aggregate_path),
        "redacted_information": str(output_root / REDACTED_RESULT),
        "retained_information": str(output_root / RETAINED_RESULT),
        "summary": str(output_root / SUMMARY_RESULT),
        "document_count": aggregate["document_count"],
        "page_count": aggregate["page_count"],
        "redacted_item_count": aggregate["redacted_item_count"],
        "documents_with_errors": aggregate["documents_with_errors"],
        "skipped": len(skipped),
    }, indent=2))


if __name__ == "__main__":
    main()