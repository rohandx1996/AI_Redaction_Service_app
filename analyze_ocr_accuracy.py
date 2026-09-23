"""Evaluate PaddleOCR (DBNet + SVTR_LCNet) accuracy against each PDF's embedded
text layer, validate the worst pages with an LLM (via litellm + Bosch Gateway),
and emit a manager-facing Markdown + CSV report.

Ground truth: PyMuPDF's embedded text layer per page (no manual transcription
exists in this repo). Pages without a usable text layer are flagged
"no_baseline" and excluded from the numeric metrics but still sampled for LLM
vision validation. Chinese-language documents (detected via CJK characters in
the output folder name) are excluded from scoring entirely; only counted and
logged.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import base64
import csv
import json
import os
import re
import statistics

import fitz
import jiwer
from tqdm import tqdm

from config import settings

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

OUTPUT_ROOT = Path("output/batch/paddle_dbnet_svtr_pages")
REPORTS_DIR = Path("output/reports")
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
TOKEN_RE = re.compile(r"[a-z0-9]+")

# LLM validation budget: worst page per document + every no-baseline page, capped.
MAX_LLM_CALLS = 60
# Set to True to run a pure PyMuPDF-based evaluation without any gateway/LLM calls.
SKIP_LLM = True


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


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


@dataclass
class PageResult:
    document: str
    page_label: str
    page_number: int
    has_baseline: bool
    cer: float | None
    wer: float | None
    token_f1: float | None
    token_precision: float | None
    token_recall: float | None
    avg_confidence: float
    empty_page: bool
    missed_words: Counter = field(default_factory=Counter)
    extra_words: Counter = field(default_factory=Counter)
    ground_truth: str = ""
    ocr_text: str = ""


@dataclass
class DocResult:
    document: str
    source_pdf: str
    pages: list[PageResult]
    complete: bool
    expected_pages: int
    actual_pages: int


def load_page_json(page_dir: Path, label: str) -> dict | None:
    path = page_dir / f"{label}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def page_dirs_sorted(doc_dir: Path) -> list[Path]:
    return sorted((p for p in doc_dir.iterdir() if p.is_dir() and p.name.startswith("page_")), key=lambda p: p.name)


def find_source_pdf(doc_dir: Path) -> str | None:
    for page_dir in page_dirs_sorted(doc_dir):
        data = load_page_json(page_dir, page_dir.name)
        if data and data.get("source_pdf"):
            return data["source_pdf"]
    return None


def is_page_complete(page_dir: Path, label: str) -> bool:
    return (
        (page_dir / f"{label}.json").exists()
        and (page_dir / f"{label}.png").exists()
        and (page_dir / f"{label}.txt").exists()
    )


def compute_token_diff(gt_tokens: list[str], ocr_tokens: list[str]) -> tuple[float, float, float, Counter, Counter]:
    """Order-independent bag-of-words diff. Recall is the key redaction-risk signal:
    it approximates whether a ground-truth word was recovered at all (detection+recognition
    combined), regardless of position - unlike CER/WER which penalize reordering."""
    gt_counter = Counter(gt_tokens)
    ocr_counter = Counter(ocr_tokens)
    overlap = sum((gt_counter & ocr_counter).values())
    precision = overlap / sum(ocr_counter.values()) if ocr_counter else 0.0
    recall = overlap / sum(gt_counter.values()) if gt_counter else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    missed = gt_counter - ocr_counter
    extra = ocr_counter - gt_counter
    return precision, recall, f1, missed, extra


def evaluate_document(doc_dir: Path) -> DocResult | None:
    source_pdf = find_source_pdf(doc_dir)
    if not source_pdf or not Path(source_pdf).exists():
        return None

    try:
        pdf = fitz.open(source_pdf)
    except Exception:
        return None

    dirs = page_dirs_sorted(doc_dir)
    page_results: list[PageResult] = []
    try:
        for page_dir in dirs:
            label = page_dir.name
            if not is_page_complete(page_dir, label):
                continue
            data = load_page_json(page_dir, label)
            if not data:
                continue
            page_number = data.get("page_number", 0)
            if page_number >= len(pdf):
                continue

            ocr_text = (page_dir / f"{label}.txt").read_text(encoding="utf-8", errors="ignore")
            spans = data.get("spans", [])
            avg_confidence = statistics.fmean(s["confidence"] for s in spans) if spans else 0.0
            empty_page = len(spans) == 0

            gt_text = pdf[page_number].get_text("text")
            gt_tokens = tokenize(gt_text)
            has_baseline = len(gt_tokens) >= 5

            cer = wer = token_f1 = token_precision = token_recall = None
            missed: Counter = Counter()
            extra: Counter = Counter()
            if has_baseline:
                gt_norm = " ".join(gt_tokens)
                ocr_norm = " ".join(tokenize(ocr_text))
                if ocr_norm.strip():
                    cer = jiwer.cer(gt_norm, ocr_norm)
                    wer = jiwer.wer(gt_norm, ocr_norm)
                else:
                    cer, wer = 1.0, 1.0
                token_precision, token_recall, token_f1, missed, extra = compute_token_diff(gt_tokens, tokenize(ocr_text))

            page_results.append(PageResult(
                document=doc_dir.name,
                page_label=label,
                page_number=page_number,
                has_baseline=has_baseline,
                cer=cer,
                wer=wer,
                token_f1=token_f1,
                token_precision=token_precision,
                token_recall=token_recall,
                avg_confidence=avg_confidence,
                empty_page=empty_page,
                missed_words=missed,
                extra_words=extra,
                ground_truth=gt_text,
                ocr_text=ocr_text,
            ))
    finally:
        pdf.close()

    expected_pages = len(fitz.open(source_pdf))
    return DocResult(
        document=doc_dir.name,
        source_pdf=source_pdf,
        pages=page_results,
        complete=len(page_results) >= expected_pages,
        expected_pages=expected_pages,
        actual_pages=len(page_results),
    )


def gateway_reachable() -> bool:
    """Probe the gateway via an authenticated API call. A plain GET on the root URL
    triggers a browser OAuth redirect (fails through corp proxies without interactive
    login), so this hits the actual chat completions endpoint with the Bearer token."""
    import requests

    if not settings.bosch_gateway_token or not settings.bosch_gateway_url:
        return False
    try:
        url = settings.bosch_gateway_url.rstrip("/") + "/v1/chat/completions"
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {settings.bosch_gateway_token}", "Content-Type": "application/json"},
            json={"model": settings.llm_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
            timeout=20,
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def print_status(message: str) -> None:
    print(f"[status] {message}", flush=True)


def check_llm_connection() -> bool:
    """Print a live OK/FAIL summary of gateway reachability + a real chat completion."""
    print_status(f"LiteLLM model configured: {settings.llm_model} via {settings.bosch_gateway_url}")
    if not settings.bosch_gateway_token or not settings.bosch_gateway_url:
        print_status("LiteLLM connection: SKIPPED (no Bosch Gateway credentials configured)")
        return False
    if not gateway_reachable():
        print_status("LiteLLM connection: FAILED (gateway host unreachable - check VPN/PX proxy)")
        return False
    try:
        patch_tiktoken()
        import litellm
        response = litellm.completion(
            model=f"openai/{settings.llm_model}",
            api_base=settings.bosch_gateway_url.rstrip("/") + "/v1",
            api_key=settings.bosch_gateway_token,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=10,
        )
        content = response["choices"][0]["message"]["content"] or ""
        if "OK" in content:
            print_status(f"LiteLLM connection: OK (model={settings.llm_model}, response={content.strip()!r})")
            return True
        print_status(f"LiteLLM connection: UNEXPECTED response: {content!r}")
        return False
    except Exception as error:
        print_status(f"LiteLLM connection: FAILED ({error})")
        return False


def run_llm_validation(doc_results: list[DocResult]) -> tuple[list[dict], str]:
    """Spot-check the worst page per doc + no-baseline pages via Bosch Gateway LLM."""
    if not settings.bosch_gateway_token or not settings.bosch_gateway_url:
        return [], "LLM validation skipped: Bosch Gateway credentials not configured."
    if not gateway_reachable():
        return [], (
            "LLM validation skipped: Bosch Gateway host is not resolvable from this network "
            "(likely requires Bosch VPN/corporate network access)."
        )

    try:
        patch_tiktoken()
        import litellm
    except ImportError:
        return [], "LLM validation skipped: litellm is not installed."

    print_status("LLM validation starting: OCR text/vision spot-check against ground truth...")
    ok_count = 0
    fail_count = 0

    model = f"openai/{settings.llm_model}"
    api_base = settings.bosch_gateway_url.rstrip("/") + "/v1"

    candidates: list[tuple[DocResult, PageResult]] = []
    for doc in doc_results:
        scored = [p for p in doc.pages if p.has_baseline and p.cer is not None]
        if scored:
            worst = max(scored, key=lambda p: p.cer)
            candidates.append((doc, worst))
        for page in doc.pages:
            if not page.has_baseline:
                candidates.append((doc, page))

    candidates = candidates[:MAX_LLM_CALLS]
    print_status(f"LLM validation sample size: {len(candidates)} pages (worst page/doc + no-baseline pages)")
    validations = []
    progress = tqdm(candidates, desc="LLM validation")
    for doc, page in progress:
        try:
            if page.has_baseline:
                prompt = (
                    "You are auditing OCR quality for a compliance pipeline.\n"
                    f"Ground truth (from PDF text layer):\n{page.ground_truth[:3000]}\n\n"
                    f"OCR output (PaddleOCR DBNet+SVTR):\n{page.ocr_text[:3000]}\n\n"
                    "In under 80 words: rate OCR accuracy 0-100, list up to 3 specific "
                    "misrecognized words/phrases (ground truth -> OCR), and the likely cause."
                )
                messages = [{"role": "user", "content": prompt}]
            else:
                img_path = OUTPUT_ROOT / doc.document / page.page_label / f"{page.page_label}.png"
                b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
                prompt = (
                    "This page has no embedded PDF text layer (likely a scanned image), so "
                    "there is no automatic ground truth. Compare the image to this OCR output:\n"
                    f"{page.ocr_text[:3000]}\n\n"
                    "In under 80 words: rate OCR accuracy 0-100, list up to 3 specific errors, "
                    "and the likely cause (blur, small font, table, skew, etc.)."
                )
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]}]

            response = litellm.completion(
                model=model, api_base=api_base, api_key=settings.bosch_gateway_token,
                messages=messages, max_tokens=200,
            )
            note = response["choices"][0]["message"]["content"]
            ok_count += 1
        except Exception as error:
            note = f"LLM call failed: {error}"
            fail_count += 1
        progress.set_postfix_str(f"ok={ok_count} failed={fail_count}")

        validations.append({
            "document": doc.document,
            "page": page.page_label,
            "has_baseline": page.has_baseline,
            "note": note,
        })

    print_status(f"LLM validation done: {ok_count} succeeded, {fail_count} failed out of {len(candidates)}")
    return validations, ""


def run_llm_synthesis(summary: dict, top_missed: list[tuple[str, int]]) -> str:
    if not settings.bosch_gateway_token or not settings.bosch_gateway_url:
        return "_LLM narrative unavailable: Bosch Gateway credentials not configured._"
    if not gateway_reachable():
        return (
            "_LLM narrative unavailable: Bosch Gateway host is not resolvable from this "
            "network (likely requires Bosch VPN/corporate network access)._"
        )
    try:
        patch_tiktoken()
        import litellm
    except ImportError:
        return "_LLM narrative unavailable: litellm is not installed._"

    model = f"openai/{settings.llm_model}"
    api_base = settings.bosch_gateway_url.rstrip("/") + "/v1"
    prompt = (
        "You are writing an executive summary for a manager about PaddleOCR "
        "(DBNet detector + SVTR_LCNet recognizer) text recognition quality on a batch "
        "of compliance PDFs. Here are the aggregated automatic metrics:\n"
        f"{json.dumps(summary, indent=2)}\n\n"
        f"Most frequently missed/misrecognized words (ground truth word, count): {top_missed[:20]}\n\n"
        "Write: 1) overall verdict, 2) where it performs best, 3) where it lacks "
        "(with concrete word/pattern examples), 4) 2-3 actionable recommendations. "
        "Keep it under 300 words, plain business language."
    )
    try:
        response = litellm.completion(
            model=model, api_base=api_base, api_key=settings.bosch_gateway_token,
            messages=[{"role": "user", "content": prompt}], max_tokens=500,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as error:
        return f"_LLM narrative generation failed: {error}_"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print_status(f"OCR ground-truth engine: PyMuPDF (fitz) {fitz.__doc__.splitlines()[0] if fitz.__doc__ else ''}".strip())
    if not SKIP_LLM:
        check_llm_connection()

    all_dirs = sorted(p for p in OUTPUT_ROOT.iterdir() if p.is_dir())
    chinese_docs = [d for d in all_dirs if CJK_RE.search(d.name)]
    candidate_dirs = [d for d in all_dirs if d not in chinese_docs]
    print_status(f"Found {len(all_dirs)} document folders ({len(chinese_docs)} Chinese excluded, {len(candidate_dirs)} to score)")

    doc_results: list[DocResult] = []
    partial_docs: list[DocResult] = []
    running_f1: list[float] = []
    progress = tqdm(candidate_dirs, desc="Scoring documents")
    for i, doc_dir in enumerate(progress, start=1):
        result = evaluate_document(doc_dir)
        if result is None:
            continue
        if result.complete:
            doc_results.append(result)
            running_f1.extend(p.token_f1 for p in result.pages if p.token_f1 is not None)
        else:
            partial_docs.append(result)
        if running_f1:
            progress.set_postfix_str(f"avg F1={statistics.fmean(running_f1):.1%}")
        if i % 20 == 0 or i == len(candidate_dirs):
            avg = f"{statistics.fmean(running_f1):.1%}" if running_f1 else "n/a"
            print_status(f"Scored {i}/{len(candidate_dirs)} documents so far - running avg token F1: {avg}")

    with (REPORTS_DIR / "chinese_documents_excluded.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder_name", "source_pdf"])
        for d in chinese_docs:
            writer.writerow([d.name, find_source_pdf(d) or ""])

    with (REPORTS_DIR / "partial_documents.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder_name", "actual_pages", "expected_pages", "source_pdf"])
        for d in partial_docs:
            writer.writerow([d.document, d.actual_pages, d.expected_pages, d.source_pdf])

    all_pages = [p for d in doc_results for p in d.pages]
    scored_pages = [p for p in all_pages if p.has_baseline]
    no_baseline_pages = [p for p in all_pages if not p.has_baseline]

    with (REPORTS_DIR / "ocr_accuracy_per_page.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "document", "page", "has_baseline", "empty_page", "token_precision", "token_recall",
            "token_f1", "cer", "wer", "avg_confidence",
        ])
        for p in all_pages:
            writer.writerow([
                p.document, p.page_label, p.has_baseline, p.empty_page, p.token_precision, p.token_recall,
                p.token_f1, p.cer, p.wer, p.avg_confidence,
            ])

    missed_total: Counter = Counter()
    for p in scored_pages:
        missed_total.update(p.missed_words)
    top_missed = missed_total.most_common(50)

    overall_cer = statistics.fmean(p.cer for p in scored_pages) if scored_pages else None
    overall_wer = statistics.fmean(p.wer for p in scored_pages) if scored_pages else None
    overall_f1 = statistics.fmean(p.token_f1 for p in scored_pages) if scored_pages else None
    overall_precision = statistics.fmean(p.token_precision for p in scored_pages) if scored_pages else None
    overall_recall = statistics.fmean(p.token_recall for p in scored_pages) if scored_pages else None
    total_empty_pages = sum(1 for p in all_pages if p.empty_page)

    per_doc_rows = []
    for d in doc_results:
        scored = [p for p in d.pages if p.has_baseline]
        empty_pages = sum(1 for p in d.pages if p.empty_page)
        if not scored:
            per_doc_rows.append({
                "document": d.document,
                "pages": len(d.pages),
                "scored_pages": 0,
                "avg_precision": None,
                "avg_recall": None,
                "avg_token_f1": None,
                "avg_cer": None,
                "avg_wer": None,
                "empty_pages": empty_pages,
                "worst_page": "n/a",
            })
            continue
        per_doc_rows.append({
            "document": d.document,
            "pages": len(d.pages),
            "scored_pages": len(scored),
            "avg_precision": statistics.fmean(p.token_precision for p in scored),
            "avg_recall": statistics.fmean(p.token_recall for p in scored),
            "avg_token_f1": statistics.fmean(p.token_f1 for p in scored),
            "avg_cer": statistics.fmean(p.cer for p in scored),
            "avg_wer": statistics.fmean(p.wer for p in scored),
            "empty_pages": empty_pages,
            "worst_page": min(scored, key=lambda p: p.token_recall).page_label,
        })
    # Recall (did we recover the words at all) is the primary redaction-risk ranking signal.
    per_doc_rows.sort(key=lambda r: (r["avg_recall"] is None, r["avg_recall"]))

    print_status(
        f"Scoring complete: {len(doc_results)} documents, {len(scored_pages)} pages scored, "
        f"overall token recall={overall_recall:.1%} precision={overall_precision:.1%} F1={overall_f1:.1%} "
        f"(CER={overall_cer:.1%} WER={overall_wer:.1%}, secondary), empty-detection pages={total_empty_pages}"
        if scored_pages else "Scoring complete: no scored pages found"
    )

    summary = {
        "documents_analyzed": len(doc_results),
        "documents_partial_excluded": len(partial_docs),
        "documents_chinese_excluded": len(chinese_docs),
        "pages_scored": len(scored_pages),
        "pages_no_baseline": len(no_baseline_pages),
        "pages_empty_detection": total_empty_pages,
        "overall_token_recall": overall_recall,
        "overall_token_precision": overall_precision,
        "overall_token_f1": overall_f1,
        "overall_cer": overall_cer,
        "overall_wer": overall_wer,
    }

    if SKIP_LLM:
        print_status("SKIP_LLM=True: running pure PyMuPDF-based evaluation, no gateway/LLM calls.")
        validations, llm_skip_reason = [], "LLM validation skipped: SKIP_LLM=True (pure PyMuPDF-based evaluation)."
        narrative = llm_skip_reason
    else:
        print_status("Starting LLM validation pass (spot-checking worst pages)...")
        validations, llm_skip_reason = run_llm_validation(doc_results)
        if llm_skip_reason:
            print_status(llm_skip_reason)
        print_status("Starting LLM narrative synthesis for the executive summary...")
        narrative = run_llm_synthesis(summary, top_missed) if not llm_skip_reason else llm_skip_reason
        print_status("LLM narrative synthesis done.")

    with (REPORTS_DIR / "llm_validations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "page", "has_baseline", "note"])
        for v in validations:
            writer.writerow([v["document"], v["page"], v["has_baseline"], v["note"]])

    lines = []
    lines.append("# PaddleOCR (DBNet + SVTR_LCNet) Text Recognition Accuracy Report\n")
    lines.append("## Executive Summary\n")
    lines.append(f"- Documents analyzed (non-Chinese, fully processed): **{summary['documents_analyzed']}**")
    lines.append(f"- Documents excluded (Chinese-language): **{summary['documents_chinese_excluded']}** (see `chinese_documents_excluded.csv`)")
    lines.append(f"- Documents excluded (partial/in-progress): **{summary['documents_partial_excluded']}** (see `partial_documents.csv`)")
    lines.append(f"- Pages with a usable ground truth (PDF text layer): **{summary['pages_scored']}**")
    lines.append(f"- Pages with no text layer (image-only, no baseline): **{summary['pages_no_baseline']}**")
    lines.append(f"- Pages with zero OCR detections (empty page, worst-case for redaction): **{summary['pages_empty_detection']}**")
    if overall_recall is not None:
        lines.append(f"- Overall Token Recall (ground-truth words recovered - primary redaction-risk metric): **{overall_recall:.2%}**")
        lines.append(f"- Overall Token Precision: **{overall_precision:.2%}**")
        lines.append(f"- Overall token-level F1 (order-independent): **{overall_f1:.2%}**")
        lines.append(f"- Overall Character Error Rate (CER, secondary diagnostic): **{overall_cer:.2%}**")
        lines.append(f"- Overall Word Error Rate (WER, secondary diagnostic): **{overall_wer:.2%}**")
    lines.append("")
    lines.append(
        "> **Note on metrics:** For a redaction pipeline, **recall** (did OCR recover a given "
        "ground-truth word at all, anywhere on the page) is the metric that matters most - a missed "
        "word can never be redacted downstream, regardless of the LLM. Token-level precision/F1 are "
        "secondary quality signals. CER/WER are sequence-aligned and penalize reading-order differences "
        "between PyMuPDF's line extraction and PaddleOCR's detection order (common on tables/multi-column "
        "pages), so they overstate error and are kept only as a secondary diagnostic. Documents below are "
        "ranked worst-to-best by recall.\n"
    )
    lines.append("## Qualitative Insights\n")
    lines.append(narrative + "\n")
    lines.append("## Per-Document Accuracy (worst to best by Token Recall)\n")
    lines.append("| Document | Pages | Scored Pages | Empty Pages | Avg Recall | Avg Precision | Avg Token F1 | Avg CER | Avg WER | Worst Page |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in per_doc_rows:
        if r["scored_pages"] == 0:
            lines.append(
                f"| {r['document'][:60]} | {r['pages']} | 0 | {r['empty_pages']} | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            f"| {r['document'][:60]} | {r['pages']} | {r['scored_pages']} | {r['empty_pages']} | "
            f"{r['avg_recall']:.2%} | {r['avg_precision']:.2%} | {r['avg_token_f1']:.2%} | "
            f"{r['avg_cer']:.2%} | {r['avg_wer']:.2%} | {r['worst_page']} |"
        )
    lines.append("")
    lines.append("## Top Missed/Misrecognized Words (ground truth word, times missed)\n")
    lines.append("| Word | Times Missed |")
    lines.append("|---|---|")
    for word, count in top_missed[:30]:
        lines.append(f"| {word} | {count} |")
    lines.append("")
    lines.append("## LLM Spot-Check Validation Notes\n")
    if validations:
        for v in validations:
            lines.append(f"- **{v['document']} / {v['page']}** (baseline={v['has_baseline']}): {v['note']}")
    else:
        lines.append(f"_{llm_skip_reason}_")
    lines.append("")
    lines.append(f"_Full per-page data: `ocr_accuracy_per_page.csv`. LLM notes: `llm_validations.csv`._")

    (REPORTS_DIR / "ocr_accuracy_report.md").write_text("\n".join(lines), encoding="utf-8")
    print_status(f"Report written to {REPORTS_DIR / 'ocr_accuracy_report.md'}")
    print_status("Done. Run build_excel_report.py to refresh the Excel workbook.")


if __name__ == "__main__":
    main()
