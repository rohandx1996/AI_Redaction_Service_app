from pathlib import Path


def extract_text_from_pdf(pdf_path: str | Path):
    pdf_path = Path(pdf_path)
    print(f"OCR placeholder: would process {pdf_path.name} with GOT-OCR2 or PaddleOCR.")
    return {
        "text": "Placeholder OCR output",
        "pages": 1,
        "confidence": 0.85,
    }
