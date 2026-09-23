"""Combine selected OCR page JSON files into one document-grouped JSON export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
OCR_ROOT = REPO_ROOT / "output" / "batch" / "paddle_dbnet_svtr_pages"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "reports" / "selected_documents_ocr_text.json"


def page_json(page_dir: Path) -> dict:
    path = page_dir / f"{page_dir.name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def page_text(data: dict) -> str:
    return "\n".join(str(span.get("text", "")) for span in data.get("spans", []))


def build_export(document_names: list[str]) -> dict:
    documents = []
    for document_name in document_names:
        document_dir = OCR_ROOT / document_name
        if not document_dir.is_dir():
            raise FileNotFoundError(f"Document folder not found: {document_dir}")

        pages = []
        for page_dir in sorted(document_dir.glob("page_*")):
            if not page_dir.is_dir():
                continue
            if not (page_dir / f"{page_dir.name}.json").exists():
                continue
            data = page_json(page_dir)
            pages.append({
                "page_break_before": bool(pages),
                "page_label": page_dir.name,
                "page_number": data.get("page_number", len(pages)),
                "text": page_text(data),
                "spans": data.get("spans", []),
                "width": data.get("width"),
                "height": data.get("height"),
                "source_pdf": data.get("source_pdf"),
            })

        documents.append({
            "document_break_before": bool(documents),
            "document": document_name,
            "source_pdf": pages[0].get("source_pdf") if pages else None,
            "page_count": len(pages),
            "pages": pages,
        })

    return {
        "format": "ocr_text_export",
        "description": "PaddleOCR recognition grouped by document, with page breaks.",
        "document_count": len(documents),
        "total_page_count": sum(document["page_count"] for document in documents),
        "documents": documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", help="OCR document folder names.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    export = build_export(args.documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {export['total_page_count']} pages from {export['document_count']} documents to {args.output}")


if __name__ == "__main__":
    main()
