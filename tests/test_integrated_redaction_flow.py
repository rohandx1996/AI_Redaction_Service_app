"""Offline contract and PDF-safety tests for the integrated redaction flow."""
from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import fitz
from types import SimpleNamespace

from redaction_flow.coordinates import crop_bbox_to_page_pixels, page_pixels_to_pdf_rect
from redaction_flow.analyzers import PaddlePageAnalyzer
from redaction_flow.gateway import VisionGateway
from redaction_flow.policy import can_release, release_blockers
from redaction_flow.contracts import PageRecord, Region
from redaction_flow.runner import IntegratedRedactionRunner
from redaction_flow.stages import RedactionStages
from redaction_flow.pdf import redact_pdf


class CoordinateTests(unittest.TestCase):
    def test_crop_to_page_pixels_uses_crop_origin_and_scale(self):
        mapped = crop_bbox_to_page_pixels([100, 200, 300, 400], [10, 20, 50, 60], 100, 100)
        self.assertEqual(mapped, [120.0, 240.0, 200.0, 320.0])

    def test_page_pixel_bbox_maps_to_pdf_points(self):
        rect = page_pixels_to_pdf_rect([100, 200, 300, 400], 600, 800, fitz.Rect(0, 0, 300, 400))
        self.assertEqual(rect, fitz.Rect(50, 100, 150, 200))

    def test_rotated_page_box_maps_back_to_unrotated_pdf_coordinates(self):
        document = fitz.open()
        page = document.new_page(width=300, height=200)
        page.set_rotation(90)
        expected = fitz.Rect(50, 60, 70, 80)
        display = expected * page.rotation_matrix
        image_width, image_height = int(page.rect.width * 2), int(page.rect.height * 2)
        pixel_bbox = [display.x0 * 2, display.y0 * 2, display.x1 * 2, display.y1 * 2]
        actual = page_pixels_to_pdf_rect(
            pixel_bbox, image_width, image_height, page.rect,
            [page.derotation_matrix.a, page.derotation_matrix.b, page.derotation_matrix.c,
             page.derotation_matrix.d, page.derotation_matrix.e, page.derotation_matrix.f],
            [page.cropbox.x0, page.cropbox.y0, page.cropbox.x1, page.cropbox.y1],
        )
        self.assertAlmostEqual(actual.x0, expected.x0, places=4)
        self.assertAlmostEqual(actual.y0, expected.y0, places=4)
        self.assertAlmostEqual(actual.x1, expected.x1, places=4)
        self.assertAlmostEqual(actual.y1, expected.y1, places=4)
        document.close()


class GatewayTests(unittest.TestCase):
    @patch("redaction_flow.gateway.requests.post")
    def test_model_specific_temperature_and_json_parsing(self, post):
        post.return_value.status_code = 200
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {
            "content": [{"type": "text", "text": '```json\n{"ok":true}\n```'}],
            "usage": {"input_tokens": 7},
        }
        gateway = VisionGateway("https://example.invalid", "token")
        parsed, usage = gateway.complete("claude-opus-5-5", "sys", "prompt")
        self.assertEqual(parsed, {"ok": True})
        self.assertEqual(usage["input_tokens"], 7)
        self.assertNotIn("temperature", post.call_args.kwargs["json"])
        gateway.complete("claude-haiku-4-5@20251001", "sys", "prompt")
        self.assertEqual(post.call_args.kwargs["json"]["temperature"], 0)


class PageAnalyzerContractTests(unittest.TestCase):
    def test_ocr_worker_record_maps_to_text_spans(self):
        analyzer = PaddlePageAnalyzer.__new__(PaddlePageAnalyzer)
        analyzer._ocr_worker = SimpleNamespace(
            stdin=StringIO(),
            stdout=StringIO('OCR_WORKER_JSON:{"status":"ok","spans":[{"index":0,"text":"Sample ID","confidence":0.97,"polygon":[[10,20],[90,20],[90,40],[10,40]]}]}\n'),
            poll=lambda: None,
        )
        spans = analyzer._ocr_spans(Path("page.png"))
        self.assertEqual(spans[0].text, "Sample ID")
        self.assertEqual(spans[0].polygon[0], [10.0, 20.0])

    def test_pp_doclayout_output_provides_image_tag_boxes(self):
        analyzer = PaddlePageAnalyzer.__new__(PaddlePageAnalyzer)
        analyzer.layout_threshold = 0.5
        result = {"res": {"boxes": [
            {"label": "image", "score": 0.91, "coordinate": [100, 120, 300, 320]},
            {"label": "text", "score": 0.99, "coordinate": [5, 5, 90, 50]},
        ]}}

        class FakePipeline:
            def predict(self, *_args, **_kwargs):
                return [SimpleNamespace(json=result)]

        analyzer.layout = FakePipeline()
        images = analyzer._layout_images(Path("page.png"), 400, 500)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].bbox, [100.0, 120.0, 300.0, 320.0])


class MappingTests(unittest.TestCase):
    def test_invalid_span_indices_are_not_accepted_and_values_block_release(self):
        stages = RedactionStages(VisionGateway("https://example.invalid", "token"), "opus", "opus", "haiku", "opus")
        fake = {"mappings": [{"value": "ACME", "span_indices": [99], "confidence": 0.99}], "unmatched": []}
        with patch.object(stages.gateway, "complete", return_value=(fake, {})):
            result = stages.map_redactions(
                [{"index": 0, "text": "ACME", "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]]}],
                {"redacted_information": [{"original_value": "ACME", "category": "CUSTOMER_NAME"}]}, 0,
            )
        self.assertEqual(result["mappings"], [])
        self.assertTrue(result["unmatched"])
        self.assertTrue(release_blockers([], result["unmatched"], []))

    def test_low_crop_classification_confidence_is_rejected(self):
        stages = RedactionStages(VisionGateway("https://example.invalid", "token"), "opus", "opus", "haiku", "opus")
        with tempfile.TemporaryDirectory() as directory:
            crop = Path(directory) / "crop.png"
            from PIL import Image
            Image.new("RGB", (2, 2), "white").save(crop)
            with patch.object(stages.gateway, "complete", return_value=({"category": "NOISE", "confidence": 0.2}, {})):
                with self.assertRaises(ValueError):
                    stages.classify_crop(crop)


class ReleaseGateTests(unittest.TestCase):
    def test_review_unmatched_and_errors_prevent_release(self):
        audit = {"status": "PASS"}
        self.assertTrue(can_release(audit, []))
        blockers = release_blockers([{"text": "ambiguous"}], [], [])
        self.assertFalse(can_release(audit, blockers))


class OrchestrationTests(unittest.TestCase):
    def test_original_image_extraction_precedes_ocr_and_clean_run_releases(self):
        events = []

        class FakeAnalyzer:
            def render_page(self, page, page_number, output_dir):
                import numpy as np
                from PIL import Image
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / "page_0001.png"
                Image.fromarray(np.full((40, 60, 3), 255, dtype=np.uint8)).save(path)
                return path, np.full((40, 60, 3), 255, dtype=np.uint8)

            def analyze(self, page, page_number, output_dir, rendered=None):
                events.append("ocr_layout")
                return PageRecord(page_number, "page_0001", str(rendered[0]), 60, 40,
                                  float(page.rect.width), float(page.rect.height)), rendered[1]

        class FakeStages:
            extraction_model = "opus"
            mapping_model = "opus"
            classifier_model = "haiku"
            material_model = "opus"

            def extract_page(self, image_path, page_number):
                events.append("vision_extract")
                return {"redacted_information": []}

            def map_redactions(self, spans, info, page_number):
                events.append("span_map")
                return {"mappings": [], "unmatched": []}

            @staticmethod
            def _norm(value):
                return "".join(char.casefold() for char in value if char.isalnum())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.pdf"
            doc = fitz.open()
            doc.new_page(width=300, height=200)
            doc.save(source)
            doc.close()

            def fake_redact(_source, output, *_args, **_kwargs):
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_bytes(b"fixture redacted output")
                return {"status": "PASS", "findings": {}}

            runner = IntegratedRedactionRunner(FakeAnalyzer(), FakeStages())
            with patch("redaction_flow.runner.redact_pdf", side_effect=fake_redact):
                result = runner.run_document(source, root / "batch", page_limit=1)
            self.assertEqual(events, ["vision_extract", "ocr_layout", "span_map"])
            self.assertEqual(result["status"], "RELEASED")
            self.assertTrue(Path(result["output_pdf"]).is_file())

    def test_qr_classification_redacts_entire_image_tag_bbox(self):
        class FakeAnalyzer:
            def render_page(self, page, page_number, output_dir):
                import numpy as np
                from PIL import Image
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / "page_0001.png"
                Image.fromarray(np.full((40, 60, 3), 255, dtype=np.uint8)).save(path)
                return path, np.full((40, 60, 3), 255, dtype=np.uint8)

            def analyze(self, page, page_number, output_dir, rendered=None):
                crop = output_dir / "image.png"
                from PIL import Image
                Image.new("RGB", (20, 20), "white").save(crop)
                return PageRecord(0, "page_0001", str(rendered[0]), 60, 40,
                                  float(page.rect.width), float(page.rect.height),
                                  image_regions=[Region("image", [10, 5, 30, 25], 0.9, crop_path=str(crop))]), rendered[1]

        class FakeStages:
            extraction_model = "opus"
            mapping_model = "opus"
            classifier_model = "haiku"
            material_model = "opus"

            def extract_page(self, image_path, page_number):
                return {"redacted_information": []}

            def map_redactions(self, spans, info, page_number):
                return {"mappings": [], "unmatched": []}

            def classify_crop(self, crop_path):
                return {"category": "QR_BARCODE", "confidence": 0.99, "reason": "QR code"}

            @staticmethod
            def _norm(value):
                return "".join(char.casefold() for char in value if char.isalnum())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.pdf"
            doc = fitz.open()
            doc.new_page(width=300, height=200)
            doc.new_page(width=300, height=200)
            doc.save(source)
            doc.close()

            def fake_redact(_source, output, regions, **_kwargs):
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_bytes(b"fixture")
                self.assertEqual(len(regions), 1)
                self.assertEqual([regions[0][key] for key in ("x0", "y0", "x1", "y1")],
                                 [50.0, 25.0, 150.0, 125.0])
                return {"status": "PASS", "findings": {}}

            runner = IntegratedRedactionRunner(FakeAnalyzer(), FakeStages())
            with patch("redaction_flow.runner.redact_pdf", side_effect=fake_redact):
                result = runner.run_document(source, root / "batch", page_limit=1)
            self.assertEqual(result["status"], "REVIEW_REQUIRED")


class PdfRedactionTests(unittest.TestCase):
    def test_text_mode_zero_removes_searchable_redacted_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.pdf", root / "redacted.pdf"
            doc = fitz.open()
            page = doc.new_page(width=300, height=120)
            page.insert_text((40, 60), "SECRETXYZ", fontsize=18)
            text_rect = page.search_for("SECRETXYZ")[0]
            doc.save(source)
            doc.close()
            result = redact_pdf(
                source, output,
                [{"page_number": 0, "x0": text_rect.x0 - 1, "y0": text_rect.y0 - 1,
                  "x1": text_rect.x1 + 1, "y1": text_rect.y1 + 1}],
                apply_bbox_filter=False, sensitive_values=["SECRETXYZ"], text_redaction_mode=0,
            )
            with fitz.open(output) as redacted:
                extracted = redacted[0].get_text("text")
            self.assertNotIn("SECRETXYZ", extracted)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["findings"]["hidden_ocr_text"], [])
            self.assertEqual(result["findings"]["leaked_values"], [])

    def test_invalid_text_mode_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "in.pdf"
            doc = fitz.open()
            doc.new_page()
            doc.save(source)
            doc.close()
            with self.assertRaises(ValueError):
                redact_pdf(source, Path(directory) / "out.pdf", [], text_redaction_mode=3)


if __name__ == "__main__":
    unittest.main()
