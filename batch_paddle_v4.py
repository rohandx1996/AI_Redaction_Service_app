"""Batch PP-OCRv4 OCR with per-document JSON, PNG, and text artifacts."""

from pathlib import Path
import json
import os
import sys
import time

import fitz
from PIL import Image, ImageDraw
from tqdm import tqdm


SOURCE_ROOT = Path(r"C:\Users\DHR1RT\Downloads\OneDrive_1_9-7-2026\02_Original")
OUTPUT_ROOT = Path("output/batch/paddle_ocr_v4")
OCR_LANGUAGE = "en"


def safe_name(pdf_path: Path) -> str:
    """Create a stable output name from a PDF relative path."""
    relative = pdf_path.relative_to(SOURCE_ROOT).with_suffix("")
    return "__".join(relative.parts).replace(" ", "_")


def process_document(ocr, pdf_path: Path) -> None:
    """OCR every page of one PDF and save all artifacts."""
    name = safe_name(pdf_path)
    json_path = OUTPUT_ROOT / "json" / f"{name}.json"
    text_path = OUTPUT_ROOT / "text" / f"{name}.txt"
    png_dir = OUTPUT_ROOT / "png" / name
    if json_path.exists() and text_path.exists():
        print(f"[SKIP] {pdf_path.name} -> already complete", flush=True)
        return

    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    document = fitz.open(str(pdf_path))
    pages = []
    text_lines = []
    try:
        for page_number in range(len(document)):
            page = document[page_number]
            pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            result = next(iter(ocr.predict(image)), None)
            payload = result.json if result is not None else {}
            if callable(payload):
                payload = payload()
            payload = json.loads(payload) if isinstance(payload, str) else payload
            pages.append({"page_number": page_number, "result": payload})
            page_text = []
            draw = ImageDraw.Draw(image)
            boxes = payload.get("res", {}).get("rec_boxes", [])
            texts = payload.get("res", {}).get("rec_texts", [])
            scores = payload.get("res", {}).get("rec_scores", [])
            for index, box in enumerate(boxes):
                coordinates = [float(value) for value in box]
                draw.polygon(
                    [(coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]),
                     (coordinates[4], coordinates[5]), (coordinates[6], coordinates[7])],
                    outline="red",
                )
                text = texts[index] if index < len(texts) else ""
                score = scores[index] if index < len(scores) else None
                page_text.append(text)
                draw.text((coordinates[0], coordinates[1]), str(index), fill="blue")
                payload.setdefault("bbox_spans", []).append({
                    "index": index, "text": text, "score": score,
                    "polygon": coordinates,
                })
            text_lines.append(f"--- Page {page_number + 1} ---\n" + "\n".join(page_text))
            image.save(png_dir / f"page_{page_number + 1:04d}.png")
            print(f"    page {page_number + 1}/{len(document)} saved", flush=True)
    finally:
        document.close()
    json_path.write_text(json.dumps({"source_pdf": str(pdf_path), "pages": pages}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    text_path.write_text("\n\n".join(text_lines), encoding="utf-8")
    print(f"[DONE] {pdf_path.name} | {len(pages)} pages | {time.time() - started:.1f}s | JSON={json_path} | PNG={png_dir}", flush=True)


def main() -> None:
    """Process all PDFs recursively with document-level progress."""
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(SOURCE_ROOT)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    import paddle

    paddle.set_flags({"FLAGS_use_mkldnn": False, "FLAGS_use_onednn": False})
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        ocr_version="PP-OCRv4",
        lang=OCR_LANGUAGE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
    )
    pdfs = sorted(SOURCE_ROOT.rglob("*.pdf"))
    print(f"PP-OCRv4 batch starting: {len(pdfs)} PDFs | language={OCR_LANGUAGE}", flush=True)
    for pdf_path in tqdm(pdfs, desc="PP-OCRv4 documents", unit="document"):
        try:
            process_document(ocr, pdf_path)
        except Exception as error:
            print(f"[ERROR] {pdf_path}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()