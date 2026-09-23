"""Rebuild source/redacted PDF pairs from a completed timestamped batch."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from run_timestamped_batch import actual_redacted_pdf, long_path


def merged_pages(package_dir: Path) -> tuple[str | None, list[dict]]:
    recognition = json.loads((package_dir / "recognition.json").read_text(encoding="utf-8"))
    redaction = json.loads((package_dir / "redaction_information.json").read_text(encoding="utf-8"))
    decisions = {page["page_label"]: page for page in redaction.get("pages", [])}
    pages = []
    for page in recognition.get("pages", []):
        merged = dict(page)
        decision = decisions.get(page.get("page_label"), {})
        merged["entities"] = decision.get("entities", {})
        merged["visual_redactions"] = decision.get("visual_redactions", [])
        pages.append(merged)
    return recognition.get("source_pdf"), pages


def rebuild(batch_dir: Path, document: str | None = None) -> dict[str, int]:
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    pair_root = batch_dir / "document_pairs"
    counts = {"release": 0, "quarantine": 0, "errors": 0, "missing": 0}
    for record in manifest.get("documents", []):
        document_name = record.get("document", "")
        if document and document_name != document:
            continue
        if record.get("status") == "ERROR":
            counts["errors"] += 1
            continue
        package_dir = long_path(batch_dir / document_name)
        recognition_path = package_dir / "recognition.json"
        redaction_path = package_dir / "redaction_information.json"
        if not recognition_path.exists() or not redaction_path.exists():
            counts["missing"] += 1
            continue
        source_pdf, pages = merged_pages(package_dir)
        if not source_pdf or not long_path(Path(source_pdf)).exists():
            counts["missing"] += 1
            continue
        destination_name = "release" if record.get("status") == "OK" else "quarantine"
        destination = long_path(pair_root / destination_name / document_name)
        destination.mkdir(parents=True, exist_ok=True)
        source_destination = destination / "source.pdf"
        redacted_destination = destination / "redacted.pdf"
        shutil.copy2(long_path(Path(source_pdf)), source_destination)
        actual_redacted_pdf(
            document_name,
            pages,
            source_pdf,
            redacted_destination,
            apply_bbox_filter=bool(manifest.get("bbox_filter_used", True)),
        )
        counts[destination_name] += 1
        print(f"{destination_name.upper()} | {document_name}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--document")
    args = parser.parse_args()
    print(json.dumps(rebuild(args.batch_dir, args.document), indent=2))


if __name__ == "__main__":
    main()