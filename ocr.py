# GOT-OCR2 runs on CPU only — no GPU required.
# Expected speed: 30 to 60 seconds per page on CPU.
# Model must be downloaded first using download_model.bat
# Bounding boxes are line-level estimates, not word-level.
# This is sufficient for POC validation.
# For production: use Azure Document Intelligence for
# pixel-perfect word-level bounding boxes.

from pathlib import Path
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import fitz
from PIL import Image
import tempfile
import os
from rich.console import Console
from rich.progress import track

from config import settings


console = Console()


@dataclass
class TextSpan:
    """A line of OCR text and its estimated page coordinates."""

    index: int
    text: str
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0

    def to_pymupdf_rect(self) -> tuple:
        """Return the span coordinates in PyMuPDF rectangle format."""
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class PageOCRResult:
    """OCR output for one PDF page."""

    page_number: int
    width: float
    height: float
    spans: list[TextSpan] = field(default_factory=list)
    raw_text: str = ""
    has_tables: bool = False


class GotOCREngine:
    """Run GOT-OCR2 locally on rendered PDF pages using the CPU."""

    def __init__(self):
        """Initialize the engine without loading the model."""
        self.model = None
        self.tokenizer = None
        self.model_path = Path(settings.got_ocr_model_path)
        self.loaded = False
        self.console = Console()

    def load_model(self):
        """Load the local GOT-OCR2 tokenizer and CPU model."""
        if not self.model_path.exists():
            self.console.print(f"Model not found at {self.model_path}", style="red")
            self.console.print(
                "Run download_model.bat to download GOT-OCR2 first",
                style="red",
            )
            raise FileNotFoundError(self.model_path)

        self.console.print(
            "Loading GOT-OCR2 model, please wait...",
            style="yellow",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        self.model.eval()
        self.loaded = True
        self.console.print("GOT-OCR2 loaded successfully", style="green")

    def process_page(
        self,
        page_image: Image.Image,
        page_number: int,
        page_width: float,
        page_height: float,
    ) -> PageOCRResult:
        """Run GOT-OCR2 on one page image and create line-level spans."""
        if not self.loaded:
            self.load_model()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        page_image.save(tmp_path)

        try:
            with torch.no_grad():
                result_text = self.model.chat(
                    self.tokenizer,
                    tmp_path,
                    ocr_type="ocr",
                )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        lines = [line.strip() for line in result_text.splitlines() if line.strip()]
        total_lines = max(len(lines), 1)
        spans = []
        # NOTE: Bounding boxes are estimated per line (equal height split).
        # GOT-OCR2 ocr_type="ocr" does not return word-level coordinates.
        # For pixel-perfect bboxes in production use Azure Document
        # Intelligence or GOT-OCR2 with ocr_type="format" + coord parsing.
        for index, line in enumerate(lines):
            line_height = page_height / total_lines
            y0 = index * line_height
            y1 = y0 + line_height
            spans.append(
                TextSpan(
                    index=index,
                    text=line,
                    page_number=page_number,
                    x0=0.0,
                    y0=y0,
                    x1=page_width,
                    y1=y1,
                    confidence=1.0,
                )
            )

        has_tables = (
            "<table" in result_text.lower()
            or any(line.count("|") >= 2 for line in lines)
        )
        return PageOCRResult(
            page_number=page_number,
            width=page_width,
            height=page_height,
            spans=spans,
            raw_text="\n".join(lines),
            has_tables=has_tables,
        )

    def process_document(self, pdf_path: str) -> list[PageOCRResult]:
        """Render every PDF page and return its GOT-OCR2 results."""
        doc = fitz.open(pdf_path)
        try:
            page_count = len(doc)
            self.console.print(f"Starting OCR on {page_count} page(s)...")
            results = []
            for page_num in track(range(page_count), description="OCR pages"):
                self.console.print(
                    f"Processing page {page_num + 1} of {page_count}"
                )
                page = doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                result = self.process_page(
                    image,
                    page_num,
                    page_width,
                    page_height,
                )
                results.append(result)
                self.console.print(
                    f"Page {page_num + 1} done: {len(result.spans)} spans found"
                )
            return results
        finally:
            doc.close()

    def is_available(self) -> bool:
        """Return whether the configured GOT-OCR2 model directory exists."""
        return self.model_path.exists() and self.model_path.is_dir()


def build_page_texts(ocr_results: list[PageOCRResult]) -> list[dict]:
    """Convert OCR results to the page-and-span format expected by llm.py."""
    return [
        {
            "page_number": result.page_number,
            "spans": [
                {"index": span.index, "text": span.text}
                for span in result.spans
            ],
        }
        for result in ocr_results
    ]


ocr_engine = GotOCREngine()
