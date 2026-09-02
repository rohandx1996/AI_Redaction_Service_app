from pathlib import Path

import fitz


def redact_pdf(pdf_path: str | Path, output_path: str | Path, regions: list[dict]):
    input_file = Path(pdf_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    document = fitz.open(str(input_file))
    for page_number, region in enumerate(regions, start=1):
        page = document[page_number - 1]
        rect = fitz.Rect(region["x0"], region["y0"], region["x1"], region["y1"])
        page.add_redact_annot(rect, fill=(0, 0, 0))
        page.apply_redactions()

    document.save(str(output_file))
    document.close()
    print(f"Redacted PDF saved to: {output_file}")
