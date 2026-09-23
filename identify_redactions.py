"""Stage 2 of the redaction pipeline: identify sensitive-information entities per page.

Input: the OCR span JSONs already produced by PaddleOCR (output/batch/paddle_dbnet_svtr_pages).
Each span is grounded with an index, so the LLM only has to point back at existing
spans (span_indices) instead of inventing coordinates - the stored polygon per span
is then reused directly for redaction (see redact.py), no re-matching text needed.

Output: one `<page_label>_redactions.json` per page, next to the existing
`<page_label>.json`/`.txt`/`.png`, plus a per-run summary CSV in output/reports/.

Usage:
    python identify_redactions.py                    # smoke test: first MAX_PAGES_TEST_RUN pages only
    python identify_redactions.py --doc <folder_name>  # one document, all its pages
    python identify_redactions.py --all                # every scored document/page (full batch)
    python identify_redactions.py --all --limit 50      # cap total pages processed
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

from config import settings

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "output" / "batch" / "paddle_dbnet_svtr_pages"
REPORTS_DIR = REPO_ROOT / "output" / "reports"
LOGS_DIR = REPO_ROOT / "logs"
MAX_PAGES_TEST_RUN = 5  # safety cap for a first smoke test when run with no flags


def to_windows_long_path(path: Path) -> Path:
    """Use the Windows extended-length path form when needed to avoid MAX_PATH failures."""
    if os.name != "nt":
        return path
    abs_path = path.resolve(strict=False)
    if abs_path.as_posix().startswith("//?/") or abs_path.as_posix().startswith("\\\\?\\"):
        return abs_path
    path_str = str(abs_path)
    if len(path_str) < 260:
        return abs_path
    return Path("\\\\?\\" + path_str)


console = Console()

SCHEMA_KEYS = [
    "SUPPLIER_NAME", "PARENT_COMPANY_NAME", "MANUFACTURER_NAME", "SUBCONTRACTOR_NAME",
    "COMMERCIAL_PARTNER_NAME", "CUSTOMER_NAME", "LABORATORY_NAME", "COMPANY", "PERSON_NAME",
    "CONTACT_ROLE", "EMAIL", "PHONE", "FAX", "ADDRESS", "LOCATION", "PLANT_NAME", "SITE_NAME",
    "SAMPLE_NAME",
    "SIGNATURE_TEXT", "SUPPLIER_CODE", "SUPPLIER_IDENTIFIER", "CUSTOMER_IDENTIFIER", "REPORT_NUMBER",
    "CERTIFICATE_NUMBER", "DOCUMENT_ID", "PIN_CODE", "INTERNAL_REFERENCE", "PROJECT_NAME",
    "NDA_REFERENCE", "CONTRACT_REFERENCE", "PRODUCT_NAME", "PRODUCT_CODE", "PART_NUMBER",
    "MATERIAL_ID", "BOM_ID", "FORMULATION_NAME", "TRADE_NAME", "PROPRIETARY_MATERIAL_DATA",
    "URL", "QR_CODE_TEXT", "BARCODE_TEXT",
]

SYSTEM_PROMPT = r"""You are a supplier information extraction engine.

You will receive OCR text spans from one document page. Each span has:
<SPAN_ID>. <TEXT>

Your task is occurrence-level extraction of identity-bearing information Bosch protects before
publishing supplier compliance reports. Identify values that directly or indirectly link the report
to a supplier, customer, applicant, submitter, manufacturer, commercial partner, facility, contact,
submitted sample, proprietary product/material, or specific commercial submission.

Use only text present in the OCR spans. Do not translate, normalize, repair OCR, summarize,
paraphrase, infer missing text, or invent values. Return values exactly as they appear.

Extract only these categories:
SUPPLIER_NAME, PARENT_COMPANY_NAME, MANUFACTURER_NAME, SUBCONTRACTOR_NAME,
COMMERCIAL_PARTNER_NAME, CUSTOMER_NAME, LABORATORY_NAME, COMPANY, PERSON_NAME, CONTACT_ROLE,
EMAIL, PHONE, FAX, ADDRESS, LOCATION, PLANT_NAME, SITE_NAME, SAMPLE_NAME, SIGNATURE_TEXT,
SUPPLIER_CODE, SUPPLIER_IDENTIFIER, CUSTOMER_IDENTIFIER, REPORT_NUMBER, CERTIFICATE_NUMBER,
DOCUMENT_ID, PIN_CODE, INTERNAL_REFERENCE, PROJECT_NAME, NDA_REFERENCE, CONTRACT_REFERENCE,
PRODUCT_NAME, PRODUCT_CODE, PART_NUMBER, MATERIAL_ID, BOM_ID, FORMULATION_NAME, TRADE_NAME,
PROPRIETARY_MATERIAL_DATA, URL, QR_CODE_TEXT, BARCODE_TEXT.

OCCURRENCE COMPLETENESS IS MANDATORY:
- Return every sensitive occurrence on this page, even when the same value appears more than once.
- Each separate OCR span occurrence requires a separate entity with its own span_indices.
- Once a value is protected anywhere on this page, return every other exact occurrence of that
    value on the page, even when another occurrence has no nearby field label.
- Never assume that returning one occurrence covers repeated headers, footers, tables, labels,
    photographs, or a second occurrence beside a field such as Sample Submitted By.
- Do not merge distant occurrences of the same value into one entity or one large bbox.
- When one sensitive value is split across adjacent spans, combine only those adjacent spans that
    form that single visible occurrence.

INTERPRET FIELD ROLES FROM CONTEXT, NOT ONE FIXED VOCABULARY:
- Client Name, Customer, Applicant, Submitted By, Sample Submitted By, Submitted and Identified By,
    Provided By Applicant, 送樣廠商, and equivalent multilingual labels identify a supplier/customer/
    applicant/submitter value. Extract the visible value following or aligned with the label.
- Sample Name, Sample Description, Sample Identification, Style/Item No., Model No., Material,
    Composition, Product Name, Part Number, and equivalent labels may identify the submitted test
    object. Extract their values when proprietary or submission-specific.
- Client Ref., Customer Ref., Order No., Lot No., Sample No., Sample ID, internal reference, and
    equivalent fields are protected when they link the report to a specific commercial submission.
- Identity-bearing text printed inside sample photographs, bottle/package labels, QR labels, and
    product markings is protected when represented in the supplied OCR spans.
- LOCATION, PLANT_NAME, and SITE_NAME are extractable only when they identify a protected party's
    business/facility identity, not when they merely name a public test location or generic geography.

ORGANIZATION CLASSIFICATION:
- Use the explicit nearby field label and page context to determine the role. An organization beside
    Sample Submitted By or Applicant is the submitter/customer even if it is also a manufacturer.
- LABORATORY_NAME is not automatically sensitive. Do not extract a testing laboratory's routine
    masthead, logo, accreditation identity, standard report footer, terms-of-service block, or generic
    lab address merely because it appears on the report.
- Extract LABORATORY_NAME only when the laboratory is itself a protected customer, supplier,
    applicant, subcontractor, commercial partner, or when the local context explicitly marks that
    laboratory identity as confidential/proprietary submission information.
- Assign each physical span occurrence to only one organization category. This category rule must
    never remove repeated physical occurrences: identical protected text at two span indices must be
    returned twice, using the same contextually supported category. Explicit contextual role wins;
    use COMPANY only when no more specific role is supported anywhere on the page.

VALUE VERSUS LABEL:
- Extract the actual visible sensitive value, not a generic label by itself.
- Labels such as Sample Name, Client Name, Client Address, Sample Receiving Date, Test Requested,
    and Sample Submitted By are context only unless the label itself contains identity-bearing data.
- Dates alone are not protected unless they are part of a protected identifier/reference value.
- QR_CODE_TEXT and BARCODE_TEXT are extractable only when their decoded/visible text exists in OCR.

The published Bosch corpus shows that technical knowledge is intentionally retained. Never extract
chemical names, CAS numbers, chemical/material compositions, RoHS, REACH, PFAS, PFOS, PFOA,
IEC/ISO/ASTM/EN/EPA standards, test methods, instruments, analytical flow charts, laboratory
procedures, sample preparation/pretreatment, extraction/digestion/filtration procedures, GC/MS,
GC/FPD, LC/MS, ICP, ICP-OES, XRF, measurements, units, detection limits, method limits, table
headings, workflow/diagram content, test results, pass/fail status, scientific conclusions, or
compliance statements. Keep these even when commercially valuable. Extract only a separate
identity-bearing value embedded in the same span; never redact the whole technical sentence for it.

Matched original-versus-published Bosch evidence shows recurring protection of customer/supplier/
applicant identity, addresses, submitted sample names, proprietary style/item/model/part values,
lot/order/client references, PIN/report-access identifiers, and identity-bearing sample-photo text.
This prompt operates on original OCR: extract the exact original visible value, never a guessed
replacement and never a Redacted_by_Bosch marker.

Keep examples such as Sample pretreatment, Analysis by GC/MS, RoHS 2011/65/EU, IEC62321,
Cadmium, Lead, Mercury, PFOS, PFOA, and RoHS10物質の分析は最新版のIEC62321を参照しました。

FEW-SHOT EXAMPLES:
The expected snippets below show only non-empty categories for brevity. Your actual response must
still contain every schema key.

Example 1 - repeated customer identity and proprietary sample:
Input spans:
0. SGS
1. MITSUBISHI GAS CHEMICAL CO., INC. ELECTRONICS MATERIAL DIVISION
2. Sample Submitted By
3. MITSUBISHI GAS CHEMICAL CO., INC. ELECTRONICS MATERIAL DIVISION
4. Sample Name
5. CCL-HL830NS(TYPE-HX)
6. Cadmium, Lead, Mercury
7. Analysis according to IEC 62321 by ICP-OES
Expected non-empty output:
{"CUSTOMER_NAME":[
    {"value":"MITSUBISHI GAS CHEMICAL CO., INC. ELECTRONICS MATERIAL DIVISION","span_indices":[1],"confidence":0.95},
    {"value":"MITSUBISHI GAS CHEMICAL CO., INC. ELECTRONICS MATERIAL DIVISION","span_indices":[3],"confidence":0.95}
],"SAMPLE_NAME":[
    {"value":"CCL-HL830NS(TYPE-HX)","span_indices":[5],"confidence":0.95}
]}
Reason: both customer occurrences need separate bboxes. Routine SGS branding and technical spans
6-7 remain visible.

Example 2 - different laboratory vocabulary for the same protected roles:
Input spans:
0. Client Name
1. KYOCERA CORPORATION
2. Client Address
3. 1810 TAKI-CHO SATSUMASENDAI-SHI KAGOSHIMA JAPAN
4. SAMPLE NAME
5. CERAMIC A440
6. CLIENT REF.NO
7. 1-01-PH7-25
Expected non-empty output:
{"CUSTOMER_NAME":[{"value":"KYOCERA CORPORATION","span_indices":[1],"confidence":0.95}],
 "ADDRESS":[{"value":"1810 TAKI-CHO SATSUMASENDAI-SHI KAGOSHIMA JAPAN","span_indices":[3],"confidence":0.95}],
 "SAMPLE_NAME":[{"value":"CERAMIC A440","span_indices":[5],"confidence":0.95}],
 "CUSTOMER_IDENTIFIER":[{"value":"1-01-PH7-25","span_indices":[7],"confidence":0.95}]}

Example 3 - routine testing laboratory identity is retained:
Input spans:
0. SGS Taiwan Ltd.
1. Chemical Laboratory-Kaohsiung
2. Member of the SGS Group
3. Client Name
4. LINTEC Corporation
Expected non-empty output:
{"CUSTOMER_NAME":[{"value":"LINTEC Corporation","span_indices":[4],"confidence":0.95}]}
Reason: spans 0-2 are routine testing-laboratory masthead/footer identity, not the protected client.

Example 4 - technical knowledge is retained:
Input spans:
0. Sample pretreatment
1. Analysis by GC/MS
2. RoHS 2011/65/EU
3. Cadmium result: Not Detected, MDL 2 mg/kg
Expected non-empty output:
{}
Reason: these are procedure, method, regulation, chemical, result, limit, and unit information.

Confidence rules: 0.95-1.00 explicit, 0.85-0.94 strong, 0.80-0.84 likely. Below 0.80 do not extract.

EXPLANATION IS REQUIRED FOR EVERY ENTITY:
- redaction_reason: give one concise reason grounded only in the supplied OCR spans and nearby
    labels, such as "value follows Client Name" or "submission-specific Part Number".
- reason_type: use EXPLICIT_CONTEXT when a label or direct page evidence supports the decision.
    Use MODEL_INFERENCE when protection is inferred without an explicit label; this marks the
    decision as a model assumption.
- Never claim evidence absent from the OCR spans.

FINAL OCCURRENCE CHECK BEFORE RESPONDING:
For every value you selected, scan all input spans once more. If the exact protected value occurs at
another span index, add a separate entity for that index in the same category. Then verify that no
technical-only or routine testing-laboratory span was selected.

Return JSON only, with every category key present. Entity format:
{"value":"", "span_indices":[0], "confidence":0.95, "redaction_reason":"value follows Client Name", "reason_type":"EXPLICIT_CONTEXT"}

Schema:
{
    "SUPPLIER_NAME": [], "PARENT_COMPANY_NAME": [], "MANUFACTURER_NAME": [], "SUBCONTRACTOR_NAME": [],
    "COMMERCIAL_PARTNER_NAME": [], "LABORATORY_NAME": [], "CUSTOMER_NAME": [], "COMPANY": [],
    "PERSON_NAME": [], "CONTACT_ROLE": [], "EMAIL": [], "PHONE": [], "FAX": [],
    "ADDRESS": [], "LOCATION": [], "PLANT_NAME": [], "SITE_NAME": [],
    "SAMPLE_NAME": [], "SIGNATURE_TEXT": [], "SUPPLIER_CODE": [], "SUPPLIER_IDENTIFIER": [], "REPORT_NUMBER": [],
    "CERTIFICATE_NUMBER": [], "DOCUMENT_ID": [], "PIN_CODE": [], "CUSTOMER_IDENTIFIER": [],
    "INTERNAL_REFERENCE": [], "PROJECT_NAME": [], "NDA_REFERENCE": [], "CONTRACT_REFERENCE": [],
    "PRODUCT_NAME": [], "PRODUCT_CODE": [], "PART_NUMBER": [], "MATERIAL_ID": [], "BOM_ID": [],
    "FORMULATION_NAME": [], "TRADE_NAME": [], "PROPRIETARY_MATERIAL_DATA": [],
    "URL": [], "QR_CODE_TEXT": [], "BARCODE_TEXT": []
}"""


def patch_tiktoken() -> None:
    """Avoid litellm's import-time tiktoken download, which the corporate firewall blocks."""
    import tiktoken

    class _ByteEncoding:
        def encode(self, text, **kwargs):
            return list(text.encode("utf-8"))

        def decode(self, tokens):
            return bytes(tokens).decode("utf-8", errors="ignore")

    tiktoken.get_encoding = lambda name: _ByteEncoding()
    tiktoken.encoding_for_model = lambda name: _ByteEncoding()


def page_dirs_sorted(doc_dir: Path) -> list[Path]:
    return sorted((p for p in doc_dir.iterdir() if p.is_dir() and p.name.startswith("page_")), key=lambda p: p.name)


def build_user_prompt(spans: list[dict]) -> str:
    numbered = "\n".join(f"{s.get('index', i)}. {s.get('text', '')}" for i, s in enumerate(spans))
    return (
        "OCR spans for this page. The page may contain English, Chinese, German, or mixed "
        "languages; preserve every protected value exactly as recognized.\n"
        f"{numbered}\n\nReturn the JSON per schema."
    )


def span_bbox(spans_by_index: dict[int, dict], index: int) -> dict | None:
    """Return the tight PaddleOCR rectangle for exactly one OCR span."""
    xs: list[float] = []
    ys: list[float] = []
    span = spans_by_index.get(index)
    if not span:
        return None
    for point in span.get("polygon", []):
        xs.append(point[0])
        ys.append(point[1])
    if not xs or not ys:
        return None
    return {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}


def infer_span_indices(value: str, spans_by_index: dict[int, dict]) -> list[int]:
    """Recover the narrowest OCR span when a model returns a bare string."""
    if not value:
        return []
    exact = [index for index, span in spans_by_index.items() if str(span.get("text", "")).strip() == value.strip()]
    if exact:
        return exact[:1]
    candidates = []
    for index, span in spans_by_index.items():
        text = str(span.get("text", ""))
        if value in text:
            polygon = span.get("polygon", [])
            if polygon:
                width = max(point[0] for point in polygon) - min(point[0] for point in polygon)
                height = max(point[1] for point in polygon) - min(point[1] for point in polygon)
                candidates.append((width * height, index))
    return [min(candidates)[1]] if candidates else []


def parse_and_validate(raw: str, spans_by_index: dict[int, dict]) -> dict:
    """Parse MUST_REDACT entities and retain REVIEW_REQUIRED candidates without bboxes."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw)

    result: dict[str, list[dict]] = {key: [] for key in SCHEMA_KEYS}
    result["REVIEW_REQUIRED"] = []
    must_redact = parsed.get("MUST_REDACT", parsed)
    for key in SCHEMA_KEYS:
        seen_occurrences = set()
        for item in must_redact.get(key, []) or []:
            try:
                if isinstance(item, str):
                    value = item
                    span_indices = infer_span_indices(value, spans_by_index)
                    confidence = 0.80
                else:
                    value = str(item["value"])
                    span_indices = [int(i) for i in item.get("span_indices", [])]
                    confidence = float(item.get("confidence", 0.0))
                    redaction_reason = str(item.get("redaction_reason", "Protected information inferred from page context"))
                    reason_type = str(item.get("reason_type", "MODEL_INFERENCE")).upper()
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(item, str):
                redaction_reason = "Protected information inferred from page context"
                reason_type = "MODEL_INFERENCE"
            if reason_type not in {"EXPLICIT_CONTEXT", "MODEL_INFERENCE"}:
                reason_type = "MODEL_INFERENCE"
            if not value or not span_indices:
                continue
            for span_index in dict.fromkeys(span_indices):
                occurrence = (value, (span_index,))
                if occurrence in seen_occurrences or span_bbox(spans_by_index, span_index) is None:
                    continue
                seen_occurrences.add(occurrence)
                result[key].append({
                    "value": value,
                    "span_indices": [span_index],
                    "confidence": confidence,
                    "redaction_reason": redaction_reason,
                    "reason_type": reason_type,
                    "bbox": span_bbox(spans_by_index, span_index),
                })

        for entity in list(result[key]):
            if len(entity["span_indices"]) != 1:
                continue
            selected_span = spans_by_index.get(entity["span_indices"][0])
            if not selected_span or str(selected_span.get("text", "")).strip() != entity["value"].strip():
                continue
            for index, span in spans_by_index.items():
                if str(span.get("text", "")).strip() != entity["value"].strip():
                    continue
                occurrence = (entity["value"], (index,))
                if occurrence in seen_occurrences:
                    continue
                seen_occurrences.add(occurrence)
                result[key].append({
                    "value": entity["value"],
                    "span_indices": [index],
                    "confidence": entity["confidence"],
                    "redaction_reason": entity["redaction_reason"],
                    "reason_type": entity["reason_type"],
                    "bbox": span_bbox(spans_by_index, index),
                    "propagated_exact_occurrence": True,
                })
    for item in parsed.get("REVIEW_REQUIRED", []) or []:
        try:
            if isinstance(item, str):
                value = item
                span_indices = infer_span_indices(value, spans_by_index)
                category = "OTHER_POLICY_REVIEW"
                reason = "OCR_AMBIGUITY"
                confidence = 0.0
            else:
                value = str(item["value"])
                span_indices = [int(i) for i in item.get("span_indices", [])]
                category = str(item.get("candidate_category", "OTHER_POLICY_REVIEW"))
                reason = str(item.get("reason", "OTHER_POLICY_REVIEW"))
                confidence = float(item.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if value:
            result["REVIEW_REQUIRED"].append({
                "value": value, "span_indices": span_indices,
                "candidate_category": category, "reason": reason,
                "confidence": confidence,
            })
    return result


def call_llm(user_prompt: str, retries: int = 5, system_prompt: str | None = None) -> str:
    """Call the configured model with the default or caller-supplied extraction policy."""
    model_name = settings.llm_model
    active_system_prompt = system_prompt or SYSTEM_PROMPT
    last_error = None
    for attempt in range(retries + 1):
        try:
            if model_name.startswith("claude-"):
                import requests

                response = requests.post(
                    settings.bosch_gateway_url.rstrip("/") + "/v1/messages",
                    headers={
                        "x-api-key": settings.bosch_gateway_token,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "system": active_system_prompt,
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                    timeout=120,
                )
                response.raise_for_status()
                return "".join(
                    block.get("text", "")
                    for block in response.json().get("content", [])
                    if block.get("type") == "text"
                ) or "{}"

            import litellm

            api_base = settings.bosch_gateway_url.rstrip("/") + "/v1"
            response = litellm.completion(
                model=f"openai/{model_name}", api_base=api_base, api_key=settings.bosch_gateway_token,
                messages=[
                    {"role": "system", "content": active_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048, temperature=0,
            )
            return response["choices"][0]["message"]["content"] or "{}"
        except Exception as error:
            last_error = error
            response = getattr(error, "response", None)
            if response is not None and getattr(response, "status_code", None) == 429:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(60.0, 5.0 * (2 ** attempt))
            else:
                delay = 1.5 * (attempt + 1)
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_error}")


def process_page(doc_name: str, page_dir: Path) -> dict | None:
    page_dir = to_windows_long_path(page_dir.resolve())
    label = page_dir.name
    json_path = to_windows_long_path(page_dir / f"{label}.json")
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    spans = data.get("spans", [])
    if not spans:
        return None
    spans_by_index = {s.get("index", i): s for i, s in enumerate(spans)}
    raw = call_llm(build_user_prompt(spans))
    entities = parse_and_validate(raw, spans_by_index)
    record = {
        "document": doc_name,
        "page_label": label,
        "page_number": data.get("page_number", 0),
        "model": settings.llm_model,
        "entities": entities,
    }
    out_path = to_windows_long_path(page_dir / f"{label}_redactions.json")
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def iter_target_pages(doc_filter: str | None) -> list[tuple[str, Path]]:
    if not OUTPUT_ROOT.exists():
        raise SystemExit(f"OCR output root not found: {OUTPUT_ROOT}")
    all_dirs = sorted(p for p in OUTPUT_ROOT.iterdir() if p.is_dir())
    if doc_filter:
        all_dirs = [d for d in all_dirs if d.name == doc_filter]
        if not all_dirs:
            raise SystemExit(f"No document folder matching: {doc_filter}")
    return [(doc_dir.name, page_dir.resolve()) for doc_dir in all_dirs for page_dir in page_dirs_sorted(doc_dir)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Process every document/page.")
    parser.add_argument("--doc", type=str, default=None, help="Process only this document folder name.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of pages processed.")
    parser.add_argument("--force", action="store_true", help="Reprocess pages that already have a *_redactions.json.")
    args = parser.parse_args()

    if not settings.bosch_gateway_token or not settings.bosch_gateway_url:
        raise SystemExit("Bosch Gateway credentials not configured (.env).")
    patch_tiktoken()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"identify_redactions_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            RichHandler(console=console, show_path=False, show_time=False),
        ],
    )
    log = logging.getLogger("identify_redactions")

    all_targets = iter_target_pages(args.doc)
    if not args.all and not args.doc:
        all_targets = all_targets[:MAX_PAGES_TEST_RUN]
        log.info(f"Smoke-test mode: first {len(all_targets)} pages only. Use --all or --doc to scale up.")
    if args.limit:
        all_targets = all_targets[:args.limit]

    if args.force:
        targets = all_targets
        skipped = 0
    else:
        targets = [(d, p) for d, p in all_targets if not (p / f"{p.name}_redactions.json").exists()]
        skipped = len(all_targets) - len(targets)

    log.info(
        f"Run {run_id}: {len(targets)} pages to process"
        + (f", {skipped} already done (skipped, use --force to redo)" if skipped else "")
    )
    log.info(f"Log file: {log_path}")

    summary_rows = []
    ok = 0
    failed = 0
    latencies: list[float] = []
    run_start = time.perf_counter()

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("elapsed •"),
        TimeRemainingColumn(),
        TextColumn("remaining •"),
        TextColumn("{task.fields[status]}"),
    ]

    with Progress(*progress_columns, console=console) as progress:
        task = progress.add_task("Identifying redactions", total=len(targets), status="starting...")
        for doc_name, page_dir in targets:
            page_start = time.perf_counter()
            try:
                record = process_page(doc_name, page_dir)
                latency = time.perf_counter() - page_start
                latencies.append(latency)
                if record is None:
                    log.info(f"SKIP (no spans) | {doc_name} / {page_dir.name}")
                    progress.advance(task)
                    continue
                counts = {k: len(v) for k, v in record["entities"].items() if v}
                total = sum(counts.values())
                summary_rows.append({
                    "document": doc_name, "page": page_dir.name, "latency_sec": round(latency, 2),
                    "total_entities": total, "by_category": json.dumps(counts),
                })
                ok += 1
                log.info(f"OK   {latency:5.2f}s | {total:3} entities | {doc_name} / {page_dir.name}")
            except Exception as error:
                latency = time.perf_counter() - page_start
                latencies.append(latency)
                failed += 1
                summary_rows.append({
                    "document": doc_name, "page": page_dir.name, "latency_sec": round(latency, 2),
                    "total_entities": "ERROR", "by_category": str(error),
                })
                log.error(f"FAIL {latency:5.2f}s | {doc_name} / {page_dir.name} | {error}")

            avg_latency = sum(latencies) / len(latencies)
            progress.update(task, advance=1, status=f"ok={ok} failed={failed} avg={avg_latency:.1f}s/page")

    run_elapsed = time.perf_counter() - run_start
    summary_path = REPORTS_DIR / "redaction_extraction_summary.csv"
    write_header = not summary_path.exists() or args.force or not args.doc and args.all
    mode = "w" if (write_header and not (args.doc and not args.force)) else "a"
    file_exists = summary_path.exists()
    with summary_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["document", "page", "latency_sec", "total_entities", "by_category"])
        if mode == "w" or not file_exists:
            writer.writeheader()
        writer.writerows(summary_rows)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    log.info(
        f"Done: {ok} ok, {failed} failed, {skipped} skipped (already done). "
        f"Total time: {run_elapsed / 60:.1f} min, avg latency: {avg_latency:.2f}s/page."
    )
    log.info(f"Per-page results: *_redactions.json next to each page. Summary CSV: {summary_path}")
    log.info(f"Full run log: {log_path}")


if __name__ == "__main__":
    main()
