from pathlib import Path
from collections import defaultdict
import re

import cv2
import fitz
import numpy as np


MAX_REDACTION_PAGE_HEIGHT_RATIO = 0.20


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _remove_pdf_objects(document: fitz.Document) -> dict:
    """Remove non-page PDF objects that can expose information outside redactions."""
    removed = {"annotations": 0, "form_fields": 0, "links": 0, "attachments": 0, "bookmarks": 0}
    for page in document:
        annotation = page.first_annot
        while annotation:
            next_annotation = annotation.next
            page.delete_annot(annotation)
            removed["annotations"] += 1
            annotation = next_annotation
        widget = page.first_widget
        while widget:
            next_widget = widget.next
            page.delete_widget(widget)
            removed["form_fields"] += 1
            widget = next_widget
        for link in page.get_links():
            page.delete_link(link)
            removed["links"] += 1
    for name in document.embfile_names():
        document.embfile_del(name)
        removed["attachments"] += 1
    removed["bookmarks"] = len(document.get_toc(simple=False))
    if removed["bookmarks"]:
        document.set_toc([])
    document.set_metadata({})
    document.del_xml_metadata()
    return removed


# PDF comment-style markup (sticky notes) vs. other markup/drawing annotations
COMMENT_ANNOT_SUBTYPES = {"Text", "FreeText", "Popup"}


def audit_pdf(
    pdf_path: str | Path,
    sensitive_values: list[str] | None = None,
    redaction_regions: list[dict] | None = None,
    post_redaction_detections: list[dict] | None = None,
) -> dict:
    """Inspect a saved PDF for hidden objects and searchable selected values."""
    pdf_file = Path(pdf_path)
    expected_values = sorted({
        normalized for value in (sensitive_values or [])
        if (normalized := _normalized_text(str(value)))
    })
    with fitz.open(str(pdf_file)) as document:
        extracted_pages = [page.get_text("text") for page in document]
        normalized_pages = [_normalized_text(text) for text in extracted_pages]
        leaked_values = [
            {"page_number": page_number, "value_index": value_index}
            for page_number, text in enumerate(normalized_pages)
            for value_index, value in enumerate(expected_values)
            if value in text
        ]
        raw_metadata = {
            key: value
            for key, value in document.metadata.items()
            if value and key not in {"format", "encryption"}
        }
        author = raw_metadata.get("author", "")
        creator = raw_metadata.get("creator", "")
        producer = raw_metadata.get("producer", "")
        metadata = {
            key: value
            for key, value in raw_metadata.items()
            if key not in {"author", "creator", "producer"}
        }
        comments = 0
        annotations = 0
        for page in document:
            for annotation in page.annots() or []:
                if annotation.type[1] in COMMENT_ANNOT_SUBTYPES:
                    comments += 1
                else:
                    annotations += 1
        form_fields = sum(1 for page in document for _ in page.widgets() or [])
        links = sum(len(page.get_links()) for page in document)
        attachments = list(document.embfile_names())
        bookmarks = document.get_toc(simple=False)
        hidden_ocr_text = []
        for region in redaction_regions or []:
            page_number = int(region["page_number"])
            if page_number < 0 or page_number >= len(document):
                continue
            rect = fitz.Rect(region["x0"], region["y0"], region["x1"], region["y1"])
            leftover = document[page_number].get_textbox(rect).strip()
            if leftover:
                hidden_ocr_text.append({"page_number": page_number, "text": leftover})
    qr_barcodes_remaining = [
        detection for detection in (post_redaction_detections or []) if detection.get("decoded")
    ]
    findings = {
        "author": author,
        "creator": creator,
        "producer": producer,
        "metadata": metadata,
        "attachments": attachments,
        "bookmarks": len(bookmarks),
        "comments": comments,
        "annotations": annotations,
        "form_fields": form_fields,
        "links": links,
        "hidden_ocr_text": hidden_ocr_text,
        "qr_barcodes_remaining": qr_barcodes_remaining,
        "leaked_values": leaked_values,
    }
    passed = not any(findings.values())
    return {
        "pdf": str(pdf_file),
        "status": "PASS" if passed else "FAIL",
        "page_count": len(extracted_pages),
        "sensitive_value_count": len(expected_values),
        "findings": findings,
    }


def detect_machine_readable_regions(pdf_path: str | Path) -> tuple[list[dict], list[dict]]:
    """Detect QR codes and barcodes on original PDF pages for automatic redaction."""
    regions = []
    detections = []
    qr_detector = cv2.QRCodeDetector()
    barcode_detector = cv2.barcode_BarcodeDetector()
    with fitz.open(str(pdf_path)) as document:
        for page_number, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
            scale_x = page.rect.width / pixmap.width
            scale_y = page.rect.height / pixmap.height
            decoded, points, _ = qr_detector.detectAndDecode(image)
            if points is not None:
                points = points.reshape(-1, 2)
                rect = fitz.Rect(
                    points[:, 0].min() * scale_x, points[:, 1].min() * scale_y,
                    points[:, 0].max() * scale_x, points[:, 1].max() * scale_y,
                ) & page.rect
                if not rect.is_empty:
                    regions.append({"page_number": page_number, "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1})
                    detections.append({"page_number": page_number, "type": "QR", "decoded": bool(decoded)})
            barcode_result = barcode_detector.detectAndDecode(image)
            if len(barcode_result) == 3:
                decoded_value, barcode_type, points = barcode_result
                decoded_info = [decoded_value] if points is not None else []
                barcode_types = [barcode_type] if points is not None else []
                corners = [points] if points is not None else []
            else:
                _, decoded_info, barcode_types, corners = barcode_result
            decoded_info = decoded_info if decoded_info is not None else []
            barcode_types = barcode_types if barcode_types is not None else []
            corners = corners if corners is not None else []
            for decoded_value, barcode_type, points in zip(decoded_info, barcode_types, corners):
                points = np.asarray(points).reshape(-1, 2)
                rect = fitz.Rect(
                    points[:, 0].min() * scale_x, points[:, 1].min() * scale_y,
                    points[:, 0].max() * scale_x, points[:, 1].max() * scale_y,
                ) & page.rect
                if not rect.is_empty:
                    regions.append({"page_number": page_number, "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1})
                    detections.append({"page_number": page_number, "type": str(barcode_type), "decoded": bool(decoded_value)})
    return regions, detections


def filter_redaction_regions(regions: list[dict], page_rects: dict[int, fitz.Rect]) -> list[dict]:
    """Reject only extreme page-height outliers from proposed redaction regions."""
    kept = []
    for region in regions:
        try:
            page_number = int(region["page_number"])
            page_rect = page_rects[page_number]
            rect = fitz.Rect(
                float(region["x0"]), float(region["y0"]),
                float(region["x1"]), float(region["y1"]),
            ) & page_rect
        except (KeyError, TypeError, ValueError):
            continue
        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            continue
        if rect.height >= page_rect.height * MAX_REDACTION_PAGE_HEIGHT_RATIO:
            continue
        kept.append({
            "page_number": page_number,
            "x0": rect.x0, "y0": rect.y0,
            "x1": rect.x1, "y1": rect.y1,
        })
    return kept


def redact_pdf(
    pdf_path: str | Path,
    output_path: str | Path,
    regions: list[dict],
    apply_bbox_filter: bool = True,
    sensitive_values: list[str] | None = None,
    machine_readable_regions: list[dict] | None = None,
    machine_readable_detections: list[dict] | None = None,
    text_redaction_mode: int = 0,
) -> dict:
    """Apply irreversible redactions, remove hidden PDF objects, and audit the result."""
    input_file = Path(pdf_path)
    output_file = Path(output_path)
    if text_redaction_mode not in (0, 1, 2):
        raise ValueError("text_redaction_mode must be 0, 1, or 2")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    document = fitz.open(str(input_file))
    try:
        page_rects = {page_number: document[page_number].rect for page_number in range(len(document))}
        if apply_bbox_filter:
            regions = filter_redaction_regions(regions, page_rects)
        regions = [*regions, *(machine_readable_regions or [])]
        regions_by_page = defaultdict(list)
        for region in regions:
            page_number = int(region["page_number"])
            if page_number < 0 or page_number >= len(document):
                raise IndexError(f"Page number out of range: {page_number}")
            regions_by_page[page_number].append(region)

        for page_number, page_regions in regions_by_page.items():
            page = document[page_number]
            for region in page_regions:
                rect = fitz.Rect(
                    region["x0"],
                    region["y0"],
                    region["x1"],
                    region["y1"],
                )
                page.add_redact_annot(rect, fill=(0, 0, 0))
            # PyMuPDF enum: 0 removes text, 1 leaves it intact, 2 removes only invisible text.
            page.apply_redactions(images=2, graphics=2, text=text_redaction_mode)

        removed_objects = _remove_pdf_objects(document)
        document.save(str(output_file), garbage=4, deflate=True)
    finally:
        document.close()
    _, post_redaction_detections = detect_machine_readable_regions(output_file)
    audit = audit_pdf(output_file, sensitive_values, regions, post_redaction_detections)
    audit["removed_objects"] = removed_objects
    audit["machine_readable_detections"] = machine_readable_detections or []
    return audit
