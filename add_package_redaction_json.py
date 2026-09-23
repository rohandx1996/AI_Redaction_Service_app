"""Add consolidated redaction-information JSON to every document package."""

from __future__ import annotations

import json
import os
from pathlib import Path

from aggregate_ocr_json import OCR_ROOT

REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT / "output" / "document_packages"


def long_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    resolved = path.resolve(strict=False)
    text = str(resolved)
    if text.startswith("\\\\?\\") or len(text) < 260:
        return resolved
    return Path("\\\\?\\" + text)


def build_redaction_export(document_name: str) -> dict:
    document_dir = long_path(OCR_ROOT / document_name)
    pages = []
    for page_dir in sorted(
        (path for path in document_dir.iterdir() if path.is_dir() and path.name.startswith("page_")),
        key=lambda path: path.name,
    ):
        path = long_path(page_dir / f"{page_dir.name}_redactions.json")
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        pages.append({
            "page_break_before": bool(pages),
            "page_label": record.get("page_label", page_dir.name),
            "page_number": record.get("page_number", len(pages)),
            "model": record.get("model"),
            "entities": record.get("entities", {}),
        })

    return {
        "format": "redaction_information_export",
        "description": "LLM-identified redaction entities and bbox coordinates grouped by page.",
        "document": document_name,
        "page_count": len(pages),
        "pages": pages,
    }


def main() -> None:
    packages = sorted(path for path in PACKAGE_ROOT.iterdir() if path.is_dir())
    added = 0
    for package_dir in packages:
        document_name = package_dir.name
        export = build_redaction_export(document_name)
        output_path = long_path(package_dir / f"{document_name}_redactions.json")
        output_path.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
        added += 1
        print(f"OK | {document_name} | pages={export['page_count']} | {output_path}")
    print(f"Added redaction JSON: {added}/{len(packages)} packages")


if __name__ == "__main__":
    main()
