from pathlib import Path
from collections import defaultdict

import fitz


def redact_pdf(pdf_path: str | Path, output_path: str | Path, regions: list[dict]):
    """Apply black redactions to zero-based page regions in a PDF."""
    input_file = Path(pdf_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    document = fitz.open(str(input_file))
    try:
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
            page.apply_redactions()

        document.save(str(output_file), garbage=4, deflate=True)
    finally:
        document.close()
