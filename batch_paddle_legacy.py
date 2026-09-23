"""Run legacy PaddleOCR DBNet + SVTR over every source PDF page.

Layout: OUTPUT_ROOT/<document>/page_NNNN/page_NNNN.{json,png,txt}
Each page is written to disk immediately after OCR, before the next page
starts, so a crash never loses already-processed pages. All exceptions are
caught per page, logged to ERROR_LOG, and do not stop the batch.
"""

from datetime import datetime
from pathlib import Path
import hashlib
import json
import traceback

import fitz
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


SOURCE_ROOT = Path(r"C:\Users\DHR1RT\Downloads\OneDrive_1_9-7-2026\02_Original")
OUTPUT_ROOT = Path("output/batch/paddle_dbnet_svtr_pages")
ERROR_LOG = OUTPUT_ROOT / "error_log.txt"
OCR_LANGUAGE = "en"


def output_name(pdf_path: Path) -> str:
    """Return a collision-resistant output name for a source PDF."""
    relative = str(pdf_path.relative_to(SOURCE_ROOT)).replace("\\", "__")
    readable = "".join(char if char.isalnum() else "_" for char in relative)
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:160]}_{digest}"


def log_error(context: str) -> None:
    """Append the current exception's traceback to the shared error log."""
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat()}] {context}\n")
        handle.write(traceback.format_exc())
        handle.write("\n" + "-" * 80 + "\n")


def page_is_complete(page_dir: Path, label: str) -> bool:
    """Return True if a page's JSON, PNG, and text outputs all exist."""
    return (
        (page_dir / f"{label}.json").exists()
        and (page_dir / f"{label}.png").exists()
        and (page_dir / f"{label}.txt").exists()
    )


def process_page(ocr, pdf_path: Path, page_number: int, page, doc_dir: Path) -> bool:
    """OCR one page and write its JSON, PNG, and text before returning."""
    label = f"page_{page_number + 1:04d}"
    page_dir = doc_dir / label
    if page_is_complete(page_dir, label):
        return True
    page_dir.mkdir(parents=True, exist_ok=True)

    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        result = ocr.ocr(np.asarray(image), cls=False)
        detections = result[0] if result and result[0] else []
        draw = ImageDraw.Draw(image)
        spans = []
        page_text = []
        for index, detection in enumerate(detections):
            polygon, recognition = detection
            text, confidence = recognition
            polygon = [[float(point[0]), float(point[1])] for point in polygon]
            draw.polygon([tuple(point) for point in polygon], outline="red", width=2)
            draw.text(tuple(polygon[0]), str(index), fill="blue")
            spans.append({
                "index": index,
                "text": text,
                "confidence": float(confidence),
                "polygon": polygon,
            })
            page_text.append(text)

        (page_dir / f"{label}.json").write_text(
            json.dumps({
                "model": "PaddleOCR DBNet + SVTR_LCNet",
                "source_pdf": str(pdf_path),
                "page_number": page_number,
                "spans": spans,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (page_dir / f"{label}.txt").write_text("\n".join(page_text), encoding="utf-8")
        image.save(page_dir / f"{label}.png")
        return True
    except Exception:
        log_error(f"{pdf_path} | {label}")
        (page_dir / f"{label}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return False


def main() -> None:
    """Process all PDFs recursively, one page at a time, with nested progress bars."""
    import paddle
    from paddleocr import PaddleOCR

    paddle.set_flags({"FLAGS_use_mkldnn": False})
    ocr = PaddleOCR(
        det_algorithm="DB",
        rec_algorithm="SVTR_LCNet",
        use_gpu=False,
        lang=OCR_LANGUAGE,
        use_angle_cls=False,
    )
    pdfs = sorted(SOURCE_ROOT.rglob("*.pdf"))
    print(f"PaddleOCR per-page batch starting: {len(pdfs)} PDFs | language={OCR_LANGUAGE}", flush=True)

    doc_bar = tqdm(pdfs, desc="Documents", unit="doc", position=0)
    for pdf_path in doc_bar:
        doc_bar.set_postfix_str(pdf_path.name[:40])
        try:
            document = fitz.open(str(pdf_path))
        except Exception:
            log_error(f"OPEN FAILED: {pdf_path}")
            continue

        try:
            doc_dir = OUTPUT_ROOT / output_name(pdf_path)
            doc_dir.mkdir(parents=True, exist_ok=True)
            page_bar = tqdm(
                range(len(document)),
                desc=f"  Pages [{pdf_path.name[:30]}]",
                unit="page",
                position=1,
                leave=False,
            )
            for page_number in page_bar:
                ok = process_page(ocr, pdf_path, page_number, document[page_number], doc_dir)
                page_bar.set_postfix_str("OK" if ok else "ERROR")
            page_bar.close()
        finally:
            document.close()
    doc_bar.close()
    print("PaddleOCR per-page batch complete.", flush=True)


if __name__ == "__main__":
    main()