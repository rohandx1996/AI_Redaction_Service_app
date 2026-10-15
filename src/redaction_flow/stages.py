"""Model-backed page extraction, OCR-span mapping, image triage and material decisions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .coordinates import clamp_pixel_bbox, crop_bbox_to_pdf_rect, pdf_rect_to_region
from .gateway import VisionGateway
from .prompts import CROP_CLASSIFICATION_SYSTEM, MATERIAL_TEXT_SYSTEM, MAPPING_SYSTEM, ORIGINAL_EXTRACTION_SYSTEM


class RedactionStages:
    def __init__(self, gateway: VisionGateway, extraction_model: str, mapping_model: str,
                 classifier_model: str, material_model: str, classification_threshold: float = 0.75,
                 material_padding_px: int = 3):
        self.gateway = gateway
        self.extraction_model = extraction_model
        self.mapping_model = mapping_model
        self.classifier_model = classifier_model
        self.material_model = material_model
        self.classification_threshold = classification_threshold
        if material_padding_px < 0:
            raise ValueError("material_padding_px must be nonnegative")
        self.material_padding_px = material_padding_px

    def extract_page(self, image_path: Path, page_number: int) -> dict[str, Any]:
        with Image.open(image_path) as image:
            png = image.convert("RGB")
            import io
            buffer = io.BytesIO()
            png.save(buffer, format="PNG")
        result, usage = self.gateway.complete(
            self.extraction_model, ORIGINAL_EXTRACTION_SYSTEM,
            f"Original PDF page {page_number + 1}. Extract sensitive visible values from this page image. Return the requested JSON only.",
            buffer.getvalue(), max_tokens=4096,
        )
        if "redacted_information" not in result or not isinstance(result["redacted_information"], list):
            raise ValueError("Extraction JSON redacted_information must be a list")
        result["_usage"] = usage
        return result

    def map_redactions(self, spans: list[dict[str, Any]], redaction_info: dict[str, Any], page_number: int) -> dict[str, Any]:
        candidates = []
        for entry in redaction_info.get("redacted_information", []) or []:
            value = str(entry.get("original_value", "")).strip()
            if value:
                candidates.append({"value": value, "category": entry.get("category", "OTHER_IDENTITY"),
                                  "field_label": entry.get("field_label", ""),
                                  "confidence": entry.get("confidence", 0.0)})
        numbered = "\n".join(
            f"{int(span['index'])}. {span.get('text', '')} | polygon={json.dumps(span.get('polygon', []))}"
            for span in spans
        )
        prompt = (
            f"Page number: {page_number + 1}\nREDACTION JSON VALUES:\n"
            f"{json.dumps(candidates, ensure_ascii=False)}\nOCR RECOGNITION SPANS:\n{numbered}\n"
            "Map all physical occurrences of the known redaction values to OCR spans."
        )
        parsed, usage = self.gateway.complete(self.mapping_model, MAPPING_SYSTEM, prompt, max_tokens=8192)
        valid = {int(span["index"]): span for span in spans}
        mappings = []
        for mapping in parsed.get("mappings", []) or []:
            try:
                indices = [int(index) for index in mapping.get("span_indices", [])]
                if not indices or any(index not in valid for index in indices):
                    continue
                polygons = [valid[index].get("polygon", []) for index in indices]
                points = [point for polygon in polygons for point in polygon]
                if not points:
                    continue
                bbox = [min(float(p[0]) for p in points), min(float(p[1]) for p in points),
                        max(float(p[0]) for p in points), max(float(p[1]) for p in points)]
                mappings.append({"value": str(mapping.get("value", "")),
                                 "category": str(mapping.get("category", "OTHER_IDENTITY")),
                                 "span_indices": indices, "confidence": float(mapping.get("confidence", 0.0)),
                                 "reason": str(mapping.get("reason", "")), "bbox_page_image_px": bbox})
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        unmatched = parsed.get("unmatched", []) or []
        mapped_values = {self._norm(item["value"]) for item in mappings if item["value"]}
        explicitly_unmatched = {self._norm(item.get("value", "")) for item in unmatched if isinstance(item, dict)}
        for candidate in candidates:
            if self._norm(candidate["value"]) not in mapped_values | explicitly_unmatched:
                unmatched.append({"value": candidate["value"], "category": candidate["category"],
                                  "reason": "Mapper did not account for this extracted value"})
        return {"page_number": page_number, "mappings": mappings, "unmatched": unmatched, "usage": usage}

    def classify_crop(self, crop_path: Path) -> dict[str, Any]:
        with Image.open(crop_path) as image:
            import io
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
        result, usage = self.gateway.complete(
            self.classifier_model, CROP_CLASSIFICATION_SYSTEM,
            "Classify this image-tag crop. Return the requested JSON only.", buffer.getvalue(), max_tokens=512,
        )
        category = str(result.get("category", "")).upper()
        if category not in {"QR_BARCODE", "MATERIAL_SAMPLE", "NOISE"}:
            raise ValueError(f"Unexpected crop classification {category!r}")
        confidence = float(result.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0 or confidence < self.classification_threshold:
            raise ValueError(f"Image-tag classification confidence {confidence:.3f} is below threshold "
                             f"{self.classification_threshold:.3f}")
        return {"category": category, "confidence": confidence,
                "reason": str(result.get("reason", "")), "usage": usage}

    def inspect_material_crop(self, crop_path: Path, crop_bbox: list[float], page_record: dict[str, Any], page_rect) -> dict[str, Any]:
        with Image.open(crop_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            import io
            buffer = io.BytesIO()
            rgb.save(buffer, format="PNG")
        parsed, usage = self.gateway.complete(
            self.material_model, MATERIAL_TEXT_SYSTEM,
            f"Crop dimensions: {width}x{height} pixels. Return tight crop-pixel boxes.",
            buffer.getvalue(), max_tokens=8192,
        )
        text_regions = []
        pdf_regions = []
        reviews = []
        sensitive_values = []
        for raw in parsed.get("text_regions", []) or []:
            try:
                bbox = clamp_pixel_bbox([raw[key] for key in ("x0", "y0", "x1", "y1")], width, height)
            except (KeyError, TypeError, ValueError):
                reviews.append({"page_number": page_record["page_number"], "crop": str(crop_path),
                                "text": str(raw.get("text", "")), "reason": "Invalid/missing model bbox"})
                continue
            decision = str(raw.get("decision", "review")).casefold()
            if decision not in {"redact", "preserve", "review"}:
                decision = "review"
            text_item = {"text": str(raw.get("text", "")), "decision": decision,
                         "category": str(raw.get("category", "")), "reason": str(raw.get("reason", "")),
                         "bbox_crop_px": bbox}
            text_regions.append(text_item)
            if decision == "redact":
                pad = self.material_padding_px
                padded_bbox = clamp_pixel_bbox(
                    [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], width, height,
                )
                rect = crop_bbox_to_pdf_rect(
                    crop_bbox, padded_bbox, width, height,
                    int(page_record["image_width"]), int(page_record["image_height"]), page_rect,
                    page_record.get("derotation_matrix"), page_record.get("pdf_cropbox_pt"),
                )
                pdf_regions.append(pdf_rect_to_region(int(page_record["page_number"]), rect))
                text_item["bbox_crop_px_padded"] = padded_bbox
                if text_item["text"]:
                    sensitive_values.append(text_item["text"])
            elif decision == "review":
                reviews.append({"page_number": page_record["page_number"], "crop": str(crop_path), **text_item})
        return {"text_regions": text_regions, "pdf_regions": pdf_regions,
                "review_items": reviews, "sensitive_values": sensitive_values, "usage": usage,
                "crop_size_px": [width, height]}

    @staticmethod
    def _norm(value: str) -> str:
        return "".join(char.casefold() for char in value if char.isalnum())
