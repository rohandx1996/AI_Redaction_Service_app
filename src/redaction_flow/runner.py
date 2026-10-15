"""Resumable orchestration for original-PDF extraction, OCR mapping and image redaction."""
from __future__ import annotations

import json
import hashlib
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz

from .analyzers import PaddlePageAnalyzer, regions_to_json, spans_to_json
from .coordinates import page_pixels_to_pdf_rect, pdf_rect_to_region
from .pdf import redact_pdf
from .policy import POLICY_VERSION, can_release, release_blockers
from .stages import RedactionStages


class IntegratedRedactionRunner:
    def __init__(self, analyzer: PaddlePageAnalyzer, stages: RedactionStages,
                 text_redaction_mode: int = 0):
        self.analyzer = analyzer
        self.stages = stages
        self.text_redaction_mode = text_redaction_mode

    def run_document(self, source_pdf: Path, batch_root: Path, page_limit: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        identity = str(source_pdf.resolve())
        document_key = f"{source_pdf.stem}_{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:8]}"
        print(f"START document={document_key} source={source_pdf.name}", flush=True)
        document_dir = batch_root / document_key
        paths = {
            "pages": document_dir / "pages",
            "redaction": document_dir / "redaction_information",
            "recognition": document_dir / "recognition",
            "layout": document_dir / "layout_regions",
            "mapping": document_dir / "span_mappings",
            "classification": document_dir / "image_classification",
            "material": document_dir / "material_decisions",
            "quarantine": document_dir / "quarantine",
            "release": document_dir / "release",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        page_records = []
        redaction_regions: list[dict[str, float | int]] = []
        sensitive_values: list[str] = []
        review_items: list[dict[str, Any]] = []
        unmatched_values: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        extraction_seconds = ocr_seconds = mapping_seconds = classification_seconds = material_seconds = 0.0

        try:
            original = fitz.open(str(source_pdf))
        except Exception as exc:
            return {"document": document_key, "source_pdf": str(source_pdf), "status": "ERROR",
                    "errors": [{"stage": "open_pdf", "error": str(exc)}]}
        with original:
            count = min(len(original), page_limit) if page_limit else len(original)
            if count < len(original):
                review_items.append({
                    "kind": "partial_document_run",
                    "reason": f"Processed {count} of {len(original)} pages; full-document release is blocked.",
                })
            # Stage 1: original page image -> best model redaction JSON. This is deliberately before OCR.
            page_images = []
            redaction_records = []
            for page_number in range(count):
                page_dir = paths["pages"] / f"page_{page_number + 1:04d}"
                try:
                    rendered = self.analyzer.render_page(original[page_number], page_number, page_dir)
                    page_images.append(rendered)
                    extraction_path = paths["redaction"] / f"page_{page_number + 1:04d}.json"
                    extracted = self._cached_json(extraction_path, self.stages.extraction_model)
                    if extracted is None:
                        t0 = time.perf_counter()
                        extracted = self.stages.extract_page(rendered[0], page_number)
                        extraction_seconds += time.perf_counter() - t0
                        self._write_json(extraction_path, {"page_number": page_number,
                                                           "model": self.stages.extraction_model,
                                                           "redacted_information": extracted.get("redacted_information", []),
                                                           "usage": extracted.get("_usage", {})})
                    redaction_records.append(extracted)
                    print(f"EXTRACT document={document_key} page={page_number + 1}/{count} "
                          f"values={len(extracted.get('redacted_information', []))}", flush=True)
                except Exception as exc:
                    errors.append({"page_number": page_number, "stage": "vision_extraction", "error": str(exc)})
                    redaction_records.append({"redacted_information": [], "error": str(exc)})
                    if len(page_images) <= page_number:
                        page_images.append(None)

            # Stage 2: OCR recognition + layout/image-tag regions on the original PDF.
            for page_number in range(count):
                if page_images[page_number] is None:
                    page_records.append(None)
                    continue
                page_dir = paths["pages"] / f"page_{page_number + 1:04d}"
                try:
                    t0 = time.perf_counter()
                    record, _ = self.analyzer.analyze(
                        original[page_number], page_number, page_dir, rendered=page_images[page_number],
                    )
                    ocr_seconds += time.perf_counter() - t0
                    page_dict = {
                        "page_number": record.page_number, "page_label": record.page_label,
                        "image_path": record.image_path, "image_width": record.image_width,
                        "image_height": record.image_height, "pdf_width": record.pdf_width,
                        "pdf_height": record.pdf_height, "pdf_cropbox_pt": record.pdf_cropbox,
                        "derotation_matrix": record.derotation_matrix,
                        "spans": [{"index": s.index, "text": s.text, "confidence": s.confidence,
                                   "polygon": s.polygon} for s in record.spans],
                        "image_regions": [{"bbox": r.bbox, "score": r.score, "crop_path": r.crop_path,
                                           "source": r.source} for r in record.image_regions],
                        "code_regions": [{"kind": r.kind, "bbox": r.bbox, "score": r.score,
                                          "decoded": r.decoded, "crop_path": r.crop_path,
                                          "source": r.source} for r in record.code_regions],
                    }
                    self._write_json(paths["recognition"] / f"page_{page_number + 1:04d}.json", {
                        **spans_to_json(record), "spans": page_dict["spans"]})
                    self._write_json(paths["layout"] / f"page_{page_number + 1:04d}.json", regions_to_json(record))
                    page_records.append(page_dict)
                    print(f"ANALYZE document={document_key} page={page_number + 1}/{count} "
                          f"ocr_spans={len(page_dict['spans'])} image_tags={len(page_dict['image_regions'])} "
                          f"codes={len(page_dict['code_regions'])}", flush=True)
                except Exception as exc:
                    errors.append({"page_number": page_number, "stage": "ocr_layout", "error": str(exc)})
                    page_records.append(None)

            # Stage 3/4: best-model value->span mapping; fast image-tag triage; Opus material decisions.
            for page_number, page_dict in enumerate(page_records):
                if page_dict is None:
                    continue
                page_redactions = redaction_records[page_number]
                if page_redactions.get("error"):
                    continue
                page_rect = original[page_number].rect
                spans = page_dict["spans"]
                mapping_path = paths["mapping"] / f"page_{page_number + 1:04d}.json"
                try:
                    cached_mapping = self._cached_json(mapping_path, self.stages.mapping_model)
                    if cached_mapping is None:
                        t0 = time.perf_counter()
                        mapping = self.stages.map_redactions(spans, page_redactions, page_number)
                        mapping_seconds += time.perf_counter() - t0
                        self._write_json(mapping_path, {"model": self.stages.mapping_model, **mapping})
                    else:
                        mapping = cached_mapping
                    page_dict["mapping"] = mapping
                    unmatched_values.extend({"page_number": page_number, **item}
                                            for item in mapping.get("unmatched", []) if isinstance(item, dict))
                    for item in mapping.get("mappings", []):
                        rect = page_pixels_to_pdf_rect(item["bbox_page_image_px"], page_dict["image_width"],
                                                       page_dict["image_height"], page_rect,
                                                       page_dict["derotation_matrix"], page_dict["pdf_cropbox_pt"])
                        redaction_regions.append(pdf_rect_to_region(page_number, rect))
                        if item.get("value"):
                            sensitive_values.append(str(item["value"]))
                except Exception as exc:
                    errors.append({"page_number": page_number, "stage": "span_mapping", "error": str(exc)})
                    page_dict["mapping"] = {"mappings": [], "unmatched": []}

                image_classifications = []
                material_index = 0
                for region_index, region in enumerate(page_dict["image_regions"], 1):
                    crop_path = Path(region["crop_path"])
                    try:
                        classification_path = paths["classification"] / f"page_{page_number + 1:04d}_image_{region_index:03d}.json"
                        classification = self._cached_json(classification_path, self.stages.classifier_model)
                        code_overlaps = self._overlapping_codes(region["bbox"], page_dict["code_regions"])
                        if classification is None and code_overlaps:
                            classification = {
                                "category": "QR_BARCODE", "confidence": 1.0,
                                "reason": "Overlaps a QR/barcode region detected on the page",
                                "model": self.stages.classifier_model,
                                "source": "page_detector_overlap",
                                "overlapping_code_regions": code_overlaps,
                            }
                        elif classification is None:
                            t0 = time.perf_counter()
                            classification = self.stages.classify_crop(crop_path)
                            classification_seconds += time.perf_counter() - t0
                            classification["model"] = self.stages.classifier_model
                        if not code_overlaps and not classification.get("source"):
                            classification["source"] = "vision_api"
                        classification_record = {
                            "page_number": page_number, "page_label": page_dict["page_label"],
                            "crop_path": str(crop_path), "bbox_page_image_px": region["bbox"],
                            "layout_score": region["score"], **classification,
                        }
                        image_classifications.append(classification_record)
                        self._write_json(classification_path, classification_record)
                        category = classification["category"]
                        if category == "QR_BARCODE":
                            rect = page_pixels_to_pdf_rect(region["bbox"], page_dict["image_width"],
                                                           page_dict["image_height"], page_rect,
                                                           page_dict["derotation_matrix"], page_dict["pdf_cropbox_pt"])
                            redaction_regions.append(pdf_rect_to_region(page_number, rect))
                            continue
                        if category != "MATERIAL_SAMPLE":
                            continue
                        material_index += 1
                        material_path = paths["material"] / f"page_{page_number + 1:04d}_material_{material_index:03d}.json"
                        decision = self._cached_json(material_path, self.stages.material_model)
                        if decision is None:
                            t0 = time.perf_counter()
                            decision = self.stages.inspect_material_crop(
                                crop_path, region["bbox"], page_dict, page_rect,
                            )
                            material_seconds += time.perf_counter() - t0
                        decision_record = {
                            "page_number": page_number, "crop_path": str(crop_path),
                            "bbox_page_image_px": region["bbox"], "model": self.stages.material_model,
                            "policy": POLICY_VERSION, **decision,
                        }
                        self._write_json(material_path, decision_record)
                        redaction_regions.extend(decision["pdf_regions"])
                        sensitive_values.extend(decision["sensitive_values"])
                        review_items.extend(decision["review_items"])
                    except Exception as exc:
                        errors.append({"page_number": page_number, "stage": "image_classification_or_material", "crop": str(crop_path), "error": str(exc)})
                        review_items.append({"page_number": page_number, "crop": str(crop_path),
                                             "reason": "Image-tag classification/material review stage failed"})
                page_dict["image_classifications"] = image_classifications
                print(f"MAP_AND_CLASSIFY document={document_key} page={page_number + 1}/{count} "
                      f"mappings={len(page_dict.get('mapping', {}).get('mappings', []))} "
                      f"image_crops={len(image_classifications)}", flush=True)

                # Code boxes outside/inside image-tag boxes are independently redacted as well.
                for code in page_dict["code_regions"]:
                    rect = page_pixels_to_pdf_rect(code["bbox"], page_dict["image_width"],
                                                   page_dict["image_height"], page_rect,
                                                   page_dict["derotation_matrix"], page_dict["pdf_cropbox_pt"])
                    redaction_regions.append(pdf_rect_to_region(page_number, rect))

        # Missing extraction records are release blockers; retain exact unmatched extraction values.
        for page_number, redaction in enumerate(redaction_records):
            if redaction.get("error"):
                continue
            page_mapping = next((record.get("mapping", {}) for record in page_records
                                 if record is not None and record.get("page_number") == page_number), {})
            mapped_norm = {self.stages._norm(str(item.get("value", "")))
                           for item in page_mapping.get("mappings", []) if item.get("value")}
            for item in redaction.get("redacted_information", []) or []:
                value = str(item.get("original_value", "")).strip()
                if value and self.stages._norm(value) not in mapped_norm and not any(
                    int(entry.get("page_number", -1)) == page_number and
                    self.stages._norm(str(entry.get("value", ""))) == self.stages._norm(value)
                    for entry in unmatched_values
                ):
                    unmatched_values.append({"page_number": page_number, "value": value,
                                             "category": item.get("category", ""),
                                             "reason": "Extracted value has no confirmed OCR mapping"})

        self._write_json(document_dir / "recognition.json", {
            "source_pdf": str(source_pdf), "page_count": count,
            "ocr_engine": "PaddleOCR DB + SVTR_LCNet",
            "pages": [record for record in page_records if record is not None],
        })
        self._write_json(document_dir / "redaction_information.json", {
            "source_pdf": str(source_pdf), "model": self.stages.extraction_model,
            "pages": [{"page_number": i, **record} for i, record in enumerate(redaction_records)],
        })

        blockers = release_blockers(review_items, unmatched_values, errors)
        quarantine_path = paths["quarantine"] / f"{source_pdf.stem}_redacted.pdf"
        release_path = paths["release"] / f"{source_pdf.stem}_redacted.pdf"
        audit = None
        if not any(error.get("stage") == "open_pdf" for error in errors):
            try:
                audit = redact_pdf(
                    source_pdf, quarantine_path, redaction_regions,
                    apply_bbox_filter=False, sensitive_values=sensitive_values,
                    text_redaction_mode=self.text_redaction_mode,
                )
            except Exception as exc:
                errors.append({"stage": "pdf_redaction_audit", "error": str(exc)})
                blockers = release_blockers(review_items, unmatched_values, errors)
        if audit is None:
            status = "REVIEW_REQUIRED" if blockers else "ERROR"
        elif can_release(audit, blockers):
            shutil.move(str(quarantine_path), str(release_path))
            audit["pdf"] = str(release_path)
            status = "RELEASED"
        else:
            status = "REVIEW_REQUIRED" if blockers else "RELEASE_FAILED"

        result = {
            "format": "integrated_original_pdf_redaction_run",
            "document": document_key, "source_pdf": str(source_pdf),
            "output_pdf": str(release_path if status == "RELEASED" else quarantine_path if quarantine_path.exists() else ""),
            "page_count": count, "status": status,
            "policy_version": POLICY_VERSION,
            "models": {"extraction": self.stages.extraction_model,
                       "mapping": self.stages.mapping_model,
                       "image_classifier": self.stages.classifier_model,
                       "material_text": self.stages.material_model},
            "region_counts": {"total": len(redaction_regions),
                              "text_mappings": sum(len(json.loads(path.read_text(encoding="utf-8")).get("mappings", []))
                                                   for path in paths["mapping"].glob("page_*.json")),
                              "review_regions": len(review_items), "unmatched_values": len(unmatched_values)},
            "release_blockers": blockers, "errors": errors, "audit": audit,
            "timings_sec": {"vision_extraction": round(extraction_seconds, 2),
                            "ocr_layout": round(ocr_seconds, 2), "span_mapping": round(mapping_seconds, 2),
                            "image_classification": round(classification_seconds, 2),
                            "material_decisions": round(material_seconds, 2),
                            "document_total": round(time.perf_counter() - started, 2)},
        }
        self._write_json(document_dir / "pipeline_result.json", result)
        print(
            f"DONE document={document_key} status={status} pages={count} "
            f"regions={len(redaction_regions)} review={len(blockers)} "
            f"total_sec={result['timings_sec']['document_total']:.2f}",
            flush=True,
        )
        return result

    def run(self, pdfs: list[Path], batch_root: Path, page_limit: int | None = None) -> dict[str, Any]:
        results = []
        batch_root.mkdir(parents=True, exist_ok=True)
        try:
            for pdf_path in pdfs:
                try:
                    results.append(self.run_document(pdf_path, batch_root, page_limit))
                except Exception as exc:
                    results.append({"document": pdf_path.stem, "source_pdf": str(pdf_path),
                                    "status": "ERROR", "errors": [{"error": str(exc)}]})
        finally:
            close = getattr(self.analyzer, "close", None)
            if callable(close):
                close()
        manifest = {
            "format": "integrated_original_pdf_redaction_manifest",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "document_count": len(results), "page_limit": page_limit,
            "statuses": {status: sum(item.get("status") == status for item in results)
                         for status in ("RELEASED", "REVIEW_REQUIRED", "RELEASE_FAILED", "ERROR")},
            "results": results,
        }
        self._write_json(batch_root / "manifest.json", manifest)
        return manifest

    @staticmethod
    def _overlapping_codes(image_bbox: list[float], codes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        x0, y0, x1, y1 = image_bbox
        matches = []
        for code in codes:
            a, b, c, d = code["bbox"]
            intersection = max(0.0, min(x1, c) - max(x0, a)) * max(0.0, min(y1, d) - max(y0, b))
            code_area = max(1.0, (c - a) * (d - b))
            if intersection / code_area >= 0.5:
                matches.append(code)
        return matches

    @staticmethod
    def _cached_json(path: Path, model: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if data.get("model") == model else None
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
