# GOT-OCR2 runs on CPU only — no GPU required.
# Expected speed: 30 to 60 seconds per page on CPU.
# Model must be downloaded first using download_model.bat
# Text-layer PDFs use precise PyMuPDF word coordinates.
# Scanned PDFs use line-level estimates because GOT-OCR2 OCR output
# does not include word coordinates in ocr_type="ocr" mode.
# This is sufficient for POC validation.

from pathlib import Path
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModel
import torch
import fitz
from PIL import Image
import tempfile
import os
from contextlib import nullcontext
from unittest.mock import patch
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
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32,
            use_safetensors=True,
            pad_token_id=self.tokenizer.eos_token_id,
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
        source_words: list[tuple] | None = None,
    ) -> PageOCRResult:
        """Run GOT-OCR2 and attach precise text-layer or fallback line boxes."""
        if not self.loaded:
            self.load_model()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        page_image.save(tmp_path)

        try:
            with torch.no_grad():
                result_text = self._chat_on_cpu(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        lines = [line.strip() for line in result_text.splitlines() if line.strip()]
        spans = self._build_spans(
            lines,
            page_number,
            page_width,
            page_height,
            source_words or [],
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

    def _build_spans(
        self,
        lines: list[str],
        page_number: int,
        page_width: float,
        page_height: float,
        source_words: list[tuple],
    ) -> list[TextSpan]:
        """Build spans from PDF word boxes, falling back to estimated line boxes."""
        if source_words:
            spans = []
            word_index = 0
            for index, line in enumerate(lines):
                line_tokens = line.split()
                matched = source_words[word_index:word_index + len(line_tokens)]
                if not matched:
                    continue
                word_index += len(matched)
                spans.append(
                    TextSpan(
                        index=index,
                        text=line,
                        page_number=page_number,
                        x0=min(word[0] for word in matched),
                        y0=min(word[1] for word in matched),
                        x1=max(word[2] for word in matched),
                        y1=max(word[3] for word in matched),
                        confidence=1.0,
                    )
                )
            if spans:
                return spans

        # GOT-OCR2 ocr_type="ocr" does not return word-level coordinates.
        # For scanned pages without a PDF text layer, use line-level estimates.
        total_lines = max(len(lines), 1)
        line_height = page_height / total_lines
        return [
            TextSpan(
                index=index,
                text=line,
                page_number=page_number,
                x0=0.0,
                y0=index * line_height,
                x1=page_width,
                y1=(index + 1) * line_height,
                confidence=1.0,
            )
            for index, line in enumerate(lines)
        ]

    def _chat_on_cpu(self, image_path: str) -> str:
        """Run GOT-OCR2 chat while adapting its CUDA-only helper to CPU."""
        def keep_on_cpu(tensor, *args, **kwargs):
            return tensor

        def keep_float32(tensor, *args, **kwargs):
            return tensor.float()

        with patch.object(torch.Tensor, "cuda", keep_on_cpu), \
                patch.object(torch.Tensor, "half", keep_float32), \
                patch.object(torch, "autocast", lambda *args, **kwargs: nullcontext()):
            return self.model.chat(
                self.tokenizer,
                image_path,
                ocr_type="ocr",
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
                source_words = page.get_text("words")
                result = self.process_page(
                    image,
                    page_num,
                    page_width,
                    page_height,
                    source_words,
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
