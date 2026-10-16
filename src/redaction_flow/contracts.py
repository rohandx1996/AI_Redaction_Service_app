"""Shared, serializable contracts for the integrated page-redaction flow."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Decision = Literal["redact", "preserve", "review"]


@dataclass
class TextSpan:
    index: int
    text: str
    confidence: float
    polygon: list[list[float]]


@dataclass
class Region:
    kind: str
    bbox: list[float]
    score: float = 0.0
    crop_path: str | None = None
    source: str = ""


@dataclass
class PageRecord:
    page_number: int
    page_label: str
    image_path: str
    image_width: int
    image_height: int
    pdf_width: float
    pdf_height: float
    pdf_cropbox: list[float] = field(default_factory=list)
    derotation_matrix: list[float] = field(default_factory=list)
    spans: list[TextSpan] = field(default_factory=list)
    image_regions: list[Region] = field(default_factory=list)
    code_regions: list[Region] = field(default_factory=list)