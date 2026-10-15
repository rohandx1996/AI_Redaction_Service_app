"""Named coordinate transforms: crop pixels -> page pixels -> PDF page points."""
from __future__ import annotations

import math
from typing import Sequence

import fitz


def crop_bbox_to_page_pixels(
    crop_bbox: Sequence[float], text_bbox: Sequence[float], crop_width: int, crop_height: int,
) -> list[float]:
    if len(crop_bbox) != 4 or len(text_bbox) != 4 or crop_width <= 0 or crop_height <= 0:
        raise ValueError("Invalid crop/page bbox or crop dimensions")
    x0, y0, x1, y1 = map(float, crop_bbox)
    a, b, c, d = map(float, text_bbox)
    sx, sy = (x1 - x0) / crop_width, (y1 - y0) / crop_height
    mapped = [x0 + a * sx, y0 + b * sy, x0 + c * sx, y0 + d * sy]
    return [max(x0, mapped[0]), max(y0, mapped[1]), min(x1, mapped[2]), min(y1, mapped[3])]


def page_pixels_to_pdf_rect(
    bbox: Sequence[float], image_width: int, image_height: int, page_rect: fitz.Rect,
    derotation_matrix: Sequence[float] | None = None,
    cropbox: Sequence[float] | None = None,
) -> fitz.Rect:
    if len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid page-pixel bbox or page-image dimensions")
    x0, y0, x1, y1 = map(float, bbox)
    rect = fitz.Rect(
        x0 * page_rect.width / image_width,
        y0 * page_rect.height / image_height,
        x1 * page_rect.width / image_width,
        y1 * page_rect.height / image_height,
    ) & page_rect
    if derotation_matrix:
        if len(derotation_matrix) != 6:
            raise ValueError("derotation_matrix must have six affine coefficients")
        rect = rect * fitz.Matrix(*map(float, derotation_matrix))
        clip = fitz.Rect(*map(float, cropbox)) if cropbox and len(cropbox) == 4 else None
        if clip is not None:
            rect &= clip
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        raise ValueError("Mapped PDF rectangle is empty")
    return rect


def crop_bbox_to_pdf_rect(
    crop_bbox: Sequence[float], text_bbox: Sequence[float], crop_width: int, crop_height: int,
    image_width: int, image_height: int, page_rect: fitz.Rect,
    derotation_matrix: Sequence[float] | None = None,
    cropbox: Sequence[float] | None = None,
) -> fitz.Rect:
    page_bbox = crop_bbox_to_page_pixels(crop_bbox, text_bbox, crop_width, crop_height)
    return page_pixels_to_pdf_rect(page_bbox, image_width, image_height, page_rect,
                                   derotation_matrix, cropbox)


def pdf_rect_to_region(page_number: int, rect: fitz.Rect) -> dict[str, float | int]:
    return {"page_number": page_number, "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1}


def clamp_pixel_bbox(bbox: Sequence[float], width: int, height: int) -> list[float]:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        raise ValueError("Invalid bbox or dimensions")
    x0, y0, x1, y1 = map(float, bbox)
    values = [max(0.0, min(x0, width)), max(0.0, min(y0, height)),
              max(0.0, min(x1, width)), max(0.0, min(y1, height))]
    if not all(math.isfinite(value) for value in values) or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("Invalid or out-of-bounds bbox")
    return values
