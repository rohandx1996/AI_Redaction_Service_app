"""Build one bbox-PDF plus one recognition-JSON package per OCR document."""

from __future__ import annotations

import json
import os
from pathlib import Path

import fitz
from PIL import Image
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from aggregate_ocr_json import OCR_ROOT, build_export

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "output" / "document_packages"


def long_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    resolved = path.resolve(strict=False)
    text = str(resolved)
    if text.startswith("\\\\?\\") or len(text) < 260:
        return resolved
    return Path("\\\\?\\" + text)


def build_bbox_pdf(document_name: str, output_path: Path) -> tuple[int, int]:
    source_dir = long_path(OCR_ROOT / document_name)
    output = fitz.open()
    page_count = 0
    bbox_count = 0

    page_dirs = sorted(
        (p for p in source_dir.iterdir() if p.is_dir() and p.name.startswith("page_")),
        key=lambda p: p.name,
    )
    for page_dir in page_dirs:
        image_path = long_path(page_dir / f"{page_dir.name}.png")
        redaction_path = long_path(page_dir / f"{page_dir.name}_redactions.json")
        if not image_path.exists() or not redaction_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        page = output.new_page(width=width, height=height)
        page.insert_image(page.rect, filename=str(image_path))
        record = json.loads(redaction_path.read_text(encoding="utf-8"))
        for entries in record.get("entities", {}).values():
            for entity in entries or []:
                bbox = entity.get("bbox")
                if not bbox:
                    continue
                try:
                    rect = fitz.Rect(
                        float(bbox["x0"]), float(bbox["y0"]),
                        float(bbox["x1"]), float(bbox["y1"]),
                    ) & page.rect
                except (KeyError, TypeError, ValueError):
                    continue
                if rect.is_empty or rect.width <= 0 or rect.height <= 0:
                    continue
                page.draw_rect(rect, color=(1, 0, 0), width=3)
                bbox_count += 1
        page_count += 1

    if not page_count:
        output.close()
        raise RuntimeError(f"No complete page image/redaction pairs: {document_name}")
    output.save(str(long_path(output_path)), garbage=4, deflate=True)
    output.close()
    return page_count, bbox_count


def main() -> None:
    documents = sorted(path.name for path in OCR_ROOT.iterdir() if path.is_dir())
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    failures = []

    with Progress(TextColumn("Building document packages"), BarColumn(), MofNCompleteColumn()) as progress:
        task = progress.add_task("documents", total=len(documents))
        for document_name in documents:
            package_dir = OUTPUT_ROOT / document_name
            long_path(package_dir).mkdir(parents=True, exist_ok=True)
            try:
                recognition = build_export([document_name])
                recognition_path = package_dir / f"{document_name}_recognition.json"
                long_path(recognition_path).write_text(
                    json.dumps(recognition, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                bbox_path = package_dir / f"{document_name}_bbox.pdf"
                pages, boxes = build_bbox_pdf(document_name, bbox_path)
                print(f"OK | {document_name} | pages={pages} | boxes={boxes}")
            except Exception as error:
                failures.append((document_name, str(error)))
                print(f"FAIL | {document_name} | {error}")
            progress.advance(task)

    print(f"Completed document packages: {len(documents) - len(failures)}/{len(documents)}")
    print(f"Output root: {OUTPUT_ROOT}")
    if failures:
        print("Failures:")
        for name, error in failures:
            print(f"- {name}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
