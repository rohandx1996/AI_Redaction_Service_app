from pathlib import Path

from rich.console import Console
from langdetect import detect, LangDetectException

from config import load_settings
from llm import (
    format_targets_summary,
    identify_sensitive_spans,
    test_gateway_connection,
)
from ocr import build_page_texts, ocr_engine
from redact import redact_pdf
from utils import ensure_dir, log_event


console = Console()


def detect_document_languages(page_texts: list[dict]) -> list[str]:
    """Detect languages represented in the OCR text, returning ISO codes."""
    languages = set()
    for page in page_texts:
        text = " ".join(span["text"] for span in page["spans"])
        if not text.strip():
            continue
        try:
            languages.add(detect(text))
        except LangDetectException:
            continue
    return sorted(languages) or ["en"]


def targets_to_regions(ocr_results: list, targets: list) -> list[dict]:
    """Map LLM span targets back to their OCR line-level coordinates."""
    regions = []
    pages = {result.page_number: result for result in ocr_results}
    for target in targets:
        page = pages.get(target.page_number)
        if page is None:
            continue
        span = next(
            (candidate for candidate in page.spans if candidate.index == target.span_index),
            None,
        )
        if span is None:
            continue
        regions.append(
            {
                "page_number": span.page_number,
                "x0": span.x0,
                "y0": span.y0,
                "x1": span.x1,
                "y1": span.y1,
            }
        )
    return regions


def process_pdf(pdf_path: Path, output_dir: Path) -> Path:
    """Run OCR, sensitive-span identification, and PDF redaction for one file."""
    console.print(f"Processing: {pdf_path.name}", style="cyan")
    ocr_results = ocr_engine.process_document(str(pdf_path))
    page_texts = build_page_texts(ocr_results)
    languages = detect_document_languages(page_texts)
    targets = identify_sensitive_spans(page_texts, languages)
    regions = targets_to_regions(ocr_results, targets)
    output_path = output_dir / f"{pdf_path.stem}_redacted.pdf"
    redact_pdf(pdf_path, output_path, regions)
    console.print(f"Redacted PDF saved to: {output_path}", style="green")
    summary = format_targets_summary(targets)
    console.print(summary or "No sensitive spans identified.")
    return output_path


def main() -> None:
    """Process every PDF in the input directory."""
    settings = load_settings()
    ensure_dir(settings.INPUT_DIR)
    ensure_dir(settings.OUTPUT_DIR)
    ensure_dir(settings.LOGS_DIR)
    log_event("Pipeline started", settings.LOGS_DIR)

    pdf_files = sorted(Path(settings.INPUT_DIR).glob("*.pdf"))
    if not pdf_files:
        console.print(
            f"No PDF files found in {settings.INPUT_DIR}. Add a PDF and run again.",
            style="yellow",
        )
        return

    console.print(f"Found {len(pdf_files)} PDF file(s).")
    if not test_gateway_connection():
        raise RuntimeError("Bosch Gateway connection failed")

    for pdf_path in pdf_files:
        log_event(f"Processing PDF: {pdf_path.name}", settings.LOGS_DIR)
        output_path = process_pdf(pdf_path, Path(settings.OUTPUT_DIR))
        log_event(f"Created redacted PDF: {output_path.name}", settings.LOGS_DIR)

    console.print("Pipeline complete.", style="green")
    log_event("Pipeline completed", settings.LOGS_DIR)


if __name__ == "__main__":
    main()
