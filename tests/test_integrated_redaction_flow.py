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
from redaction_flow.contracts import PageRecord, Region, TextSpan
from redaction_flow.runner import IntegratedRedactionRunner
from redaction_flow.stages import RedactionStages
from redaction_flow.pdf import audit_pdf, redact_pdf
from redaction_flow.prompts import ORIGINAL_EXTRACTION_SYSTEM


def _copy_pdf_fixture(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(source) as document:
        document.save(output)


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

    def test_code_candidate_over_dense_text_is_filtered_but_image_candidate_kept(self):
        spans = [
            TextSpan(
                i, f"text-{i}", 0.99,
                [[10, y], [90, y], [90, y + 8], [10, y + 8]],
            )
            for i, y in enumerate((10, 30, 50))
        ]
        false_positive = Region("qr_code", [10, 5, 30, 65], 0.8)
        image_candidate = Region("qr_code", [40, 5, 60, 65], 0.8)
        containing_image = Region("image", [35, 0, 70, 70], 0.9)

        filtered = PaddlePageAnalyzer._filter_text_overlapped_code_candidates(
            [false_positive, image_candidate], spans, [containing_image],
        )

        self.assertEqual(filtered, [image_candidate])

    def test_code_detection_uses_geometry_without_decoding(self):
        import numpy as np

        qr_detector = SimpleNamespace(
            detect=lambda _image: (True, np.asarray([[[1, 2], [8, 2], [8, 9], [1, 9]]], dtype=float)),
        )
        barcode_detector = SimpleNamespace(detect=lambda _image: (False, None))
        with patch("redaction_flow.analyzers.cv2.QRCodeDetector", return_value=qr_detector), \
                patch("redaction_flow.analyzers.cv2.barcode_BarcodeDetector", return_value=barcode_detector):
            regions = PaddlePageAnalyzer._codes(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].kind, "qr_code")
        self.assertEqual(regions[0].bbox, [1.0, 2.0, 8.0, 9.0])


class MappingTests(unittest.TestCase):
    def test_extraction_policy_preserves_lab_signoff(self):
        self.assertIn("Preserve the testing laboratory's", ORIGINAL_EXTRACTION_SYSTEM)
        self.assertIn("signatures and names belonging to those", ORIGINAL_EXTRACTION_SYSTEM)

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

    def test_targeted_review_returns_only_supplied_flag_ids(self):
        from PIL import Image

        stages = RedactionStages(
            VisionGateway("https://example.invalid", "token"), "opus", "opus", "haiku", "opus",
        )
        import io
        buffer = io.BytesIO()
        Image.new("RGB", (40, 30), "white").save(buffer, format="PNG")
        response = {
            "overall_decision": "APPROVE", "confidence": 0.99,
            "checks": [{"id": "B4", "decision": "APPROVE", "confidence": 0.98,
                        "remark": "Flagged mark is clearly laboratory print"}],
            "remarks": ["Only supplied flag checked"],
        }
        with patch.object(stages.reviewer_gateway, "complete", return_value=(response, {})) as complete:
            result = stages.review_flagged_page(
                buffer.getvalue(), buffer.getvalue(), 3,
                [{"id": "B4", "reason": "Faint mark of unclear role"}],
            )
        self.assertEqual(result["checks"][0]["id"], "B4")
        self.assertEqual(result["checks"][0]["remark"], "Flagged mark is clearly laboratory print")
        self.assertIsNotNone(complete.call_args.kwargs["image"])

    def test_targeted_review_rejects_missing_flag_id(self):
        from PIL import Image
        import io

        stages = RedactionStages(VisionGateway("https://example.invalid", "token"), "opus", "opus", "haiku", "opus")
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
        with patch.object(stages.reviewer_gateway, "complete", return_value=(
            {"overall_decision": "APPROVE", "confidence": 1.0, "checks": [], "remarks": []}, {}
        )):
            with self.assertRaises(ValueError):
                stages.review_flagged_page(
                    buffer.getvalue(), buffer.getvalue(), 0,
                    [{"id": "B0", "reason": "Needs review"}],
                )


class ReleaseGateTests(unittest.TestCase):
    def test_review_unmatched_and_errors_prevent_release(self):
        audit = {"status": "PASS"}
        self.assertTrue(can_release(audit, []))
        blockers = release_blockers([{"text": "ambiguous"}], [], [])
        self.assertFalse(can_release(audit, blockers))

    def test_qr_description_is_machine_code_specific(self):
        self.assertTrue(IntegratedRedactionRunner._is_machine_code_description(
            {"value": "CHECK REPORT QR code", "reason": "No OCR span"},
        ))
        self.assertFalse(IntegratedRedactionRunner._is_machine_code_description(
            {"value": "Troy Chang signature", "reason": "No OCR span"},
        ))


class OrchestrationTests(unittest.TestCase):
    def test_second_review_is_called_only_for_flagged_pages(self):
        from PIL import Image
        import io

        class FakeStages:
            reviewer_model = "fast-reviewer"
            review_threshold = 0.95

            def __init__(self):
                self.pages = []

            def review_flagged_page(self, _original, _redacted, page_number, flags):
                self.pages.append(page_number)
                return {
                    "page_number": page_number, "model": self.reviewer_model,
                    "overall_decision": "APPROVE", "confidence": 0.99,
                    "checks": [{"id": flag["id"], "decision": "APPROVE",
                                "confidence": 0.99, "remark": "Visibly benign"} for flag in flags],
                    "remarks": [],
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "pair.pdf"
            doc = fitz.open()
            doc.new_page(width=200, height=100)
            doc.new_page(width=200, height=100)
            doc.save(pdf_path)
            doc.close()
            image_path = root / "page.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            page_image = (image_path, None)
            stages = FakeStages()
            runner = IntegratedRedactionRunner(SimpleNamespace(dpi=72), stages)
            reports, _elapsed, approved, errors = runner._review_flagged_pages(
                pdf_path, [page_image, page_image],
                [{"page_number": 1, "reason": "A faint mark needs a second look"}],
                root / "review",
            )

        self.assertEqual(stages.pages, [1])
        self.assertEqual(len(reports), 1)
        self.assertEqual(approved, {0})
        self.assertEqual(errors, [])

    def test_original_image_extraction_precedes_ocr_and_clean_run_releases(self):
        events = []

        class FakeAnalyzer:
            dpi = 150

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
            reviewer_model = "haiku"
            review_threshold = 0.95

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
                _copy_pdf_fixture(_source, Path(output))
                return {"status": "PASS", "findings": {}}

            runner = IntegratedRedactionRunner(FakeAnalyzer(), FakeStages())
            with patch("redaction_flow.runner.redact_pdf", side_effect=fake_redact):
                result = runner.run_document(source, root / "batch", page_limit=1)
            self.assertEqual(events, ["vision_extract", "ocr_layout", "span_map"])
            self.assertEqual(result["status"], "RELEASED", result.get("release_blockers"))
            self.assertTrue(Path(result["output_pdf"]).is_file())

    def test_qr_classification_redacts_entire_image_tag_bbox(self):
        class FakeAnalyzer:
            dpi = 150

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
            reviewer_model = "haiku"
            review_threshold = 0.95

            def extract_page(self, image_path, page_number):
                return {"redacted_information": []}

            def map_redactions(self, spans, info, page_number):
                return {"mappings": [], "unmatched": []}

            def classify_crop(self, crop_path):
                return {"category": "QR_BARCODE", "confidence": 0.99, "reason": "QR code"}

            def review_flagged_page(self, _original, _redacted, page_number, flags):
                return {"overall_decision": "REVIEW", "confidence": 0.5,
                        "checks": [{"id": flag["id"], "decision": "REVIEW", "confidence": 0.5,
                                    "remark": "Keep for human review"} for flag in flags], "remarks": []}

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
                _copy_pdf_fixture(_source, Path(output))
                self.assertEqual(len(regions), 1)
                self.assertEqual([regions[0][key] for key in ("x0", "y0", "x1", "y1")],
                                 [50.0, 25.0, 150.0, 125.0])
                return {"status": "PASS", "findings": {}}

            runner = IntegratedRedactionRunner(FakeAnalyzer(), FakeStages())
            with patch("redaction_flow.runner.redact_pdf", side_effect=fake_redact):
                result = runner.run_document(source, root / "batch", page_limit=1)
            self.assertEqual(result["status"], "REVIEW_REQUIRED")

    def test_code_candidate_is_classified_and_qr_bbox_redacted(self):
        class FakeAnalyzer:
            dpi = 150

            def render_page(self, page, page_number, output_dir):
                import numpy as np
                from PIL import Image
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / "page_0001.png"
                image = np.full((40, 60, 3), 255, dtype=np.uint8)
                Image.fromarray(image).save(path)
                return path, image

            def analyze(self, page, page_number, output_dir, rendered=None):
                from PIL import Image
                crop = output_dir / "qr_candidate.png"
                Image.new("RGB", (20, 20), "white").save(crop)
                return PageRecord(
                    0, "page_0001", str(rendered[0]), 60, 40,
                    float(page.rect.width), float(page.rect.height),
                    code_regions=[Region("qr_code", [10, 10, 30, 30], 0.8,
                                         crop_path=str(crop))],
                ), rendered[1]

        class FakeStages:
            extraction_model = "opus"
            mapping_model = "opus"
            classifier_model = "haiku"
            material_model = "opus"
            reviewer_model = "haiku"
            review_threshold = 0.95
            classified_crops = []

            def extract_page(self, image_path, page_number):
                return {"redacted_information": [{
                    "category": "access_code", "field_label": "QR code",
                    "original_value": "CHECK REPORT QR code",
                }]}

            def map_redactions(self, spans, info, page_number):
                return {"mappings": [], "unmatched": [{
                    "value": "CHECK REPORT QR code", "category": "access_code",
                    "reason": "Graphic code has no OCR text span",
                }]}

            def classify_crop(self, crop_path):
                self.classified_crops.append(str(crop_path))
                return {"category": "QR_BARCODE", "confidence": 0.99,
                        "reason": "Visually identified as a QR code"}

            def review_flagged_page(self, _original, _redacted, page_number, flags):
                return {"overall_decision": "REVIEW", "confidence": 0.5,
                        "checks": [{"id": flag["id"], "decision": "REVIEW", "confidence": 0.5,
                                    "remark": "Keep for human review"} for flag in flags], "remarks": []}

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

            def fake_redact(_source, output, regions, **_kwargs):
                _copy_pdf_fixture(_source, Path(output))
                self.assertEqual(len(regions), 1)
                self.assertEqual([regions[0][key] for key in ("x0", "y0", "x1", "y1")],
                                 [50.0, 50.0, 150.0, 150.0])
                return {"status": "PASS", "findings": {}}

            runner = IntegratedRedactionRunner(FakeAnalyzer(), FakeStages())
            with patch("redaction_flow.runner.redact_pdf", side_effect=fake_redact):
                result = runner.run_document(source, root / "batch")

            self.assertEqual(result["status"], "RELEASED", result.get("release_blockers"))
            self.assertEqual(len(FakeStages.classified_crops), 1)

    def test_detected_code_candidate_can_be_classified_as_noise(self):
        class FakeAnalyzer:
            dpi = 150

            def render_page(self, page, page_number, output_dir):
                import numpy as np
                from PIL import Image
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / "page_0001.png"
                image = np.full((40, 60, 3), 255, dtype=np.uint8)
                Image.fromarray(image).save(path)
                return path, image

            def analyze(self, page, page_number, output_dir, rendered=None):
                from PIL import Image
                crop = output_dir / "candidate.png"
                Image.new("RGB", (20, 20), "white").save(crop)
                return PageRecord(
                    0, "page_0001", str(rendered[0]), 60, 40,
                    float(page.rect.width), float(page.rect.height),
                    code_regions=[Region("qr_code", [10, 10, 30, 30], 0.8,
                                         crop_path=str(crop))],
                ), rendered[1]

        class FakeStages:
            extraction_model = "opus"
            mapping_model = "opus"
            classifier_model = "haiku"
            material_model = "opus"
            reviewer_model = "haiku"
            review_threshold = 0.95

            def extract_page(self, image_path, page_number):
                return {"redacted_information": []}

            def map_redactions(self, spans, info, page_number):
                return {"mappings": [], "unmatched": []}

            def classify_crop(self, crop_path):
                return {"category": "NOISE", "confidence": 0.99,
                        "reason": "False-positive code candidate"}

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

            def fake_redact(_source, output, regions, **_kwargs):
                _copy_pdf_fixture(_source, Path(output))
                self.assertEqual(regions, [])
                return {"status": "PASS", "findings": {}}

            runner = IntegratedRedactionRunner(FakeAnalyzer(), FakeStages())
            with patch("redaction_flow.runner.redact_pdf", side_effect=fake_redact):
                result = runner.run_document(source, root / "batch")

            self.assertEqual(result["status"], "RELEASED", result.get("release_blockers"))

    def test_code_candidate_classified_as_material_uses_opus_material_policy(self):
        class FakeAnalyzer:
            dpi = 150

            def render_page(self, page, page_number, output_dir):
                import numpy as np
                from PIL import Image
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / "page_0001.png"
                image = np.full((40, 60, 3), 255, dtype=np.uint8)
                Image.fromarray(image).save(path)
                return path, image

            def analyze(self, page, page_number, output_dir, rendered=None):
                from PIL import Image
                crop = output_dir / "candidate.png"
                Image.new("RGB", (20, 20), "white").save(crop)
                return PageRecord(
                    0, "page_0001", str(rendered[0]), 60, 40,
                    float(page.rect.width), float(page.rect.height),
                    code_regions=[Region("qr_code", [10, 10, 30, 30], 0.8,
                                         crop_path=str(crop))],
                ), rendered[1]

        class FakeStages:
            extraction_model = "opus"
            mapping_model = "opus"
            classifier_model = "haiku"
            material_model = "opus"
            reviewer_model = "haiku"
            review_threshold = 0.95
            inspected_crops = []

            def extract_page(self, image_path, page_number):
                return {"redacted_information": []}

            def map_redactions(self, spans, info, page_number):
                return {"mappings": [], "unmatched": []}

            def classify_crop(self, crop_path):
                return {"category": "MATERIAL_SAMPLE", "confidence": 0.99,
                        "reason": "Material sample label"}

            def inspect_material_crop(self, crop_path, crop_bbox, page_record, page_rect):
                self.inspected_crops.append((str(crop_path), crop_bbox))
                return {"pdf_regions": [{"page_number": 0, "x0": 55.0, "y0": 55.0,
                                         "x1": 95.0, "y1": 95.0}],
                        "review_items": [], "text_regions": [], "sensitive_values": []}

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

            def fake_redact(_source, output, regions, **_kwargs):
                _copy_pdf_fixture(_source, Path(output))
                self.assertEqual(regions, [{"page_number": 0, "x0": 55.0, "y0": 55.0,
                                            "x1": 95.0, "y1": 95.0}])
                return {"status": "PASS", "findings": {}}

            stages = FakeStages()
            runner = IntegratedRedactionRunner(FakeAnalyzer(), stages)
            with patch("redaction_flow.runner.redact_pdf", side_effect=fake_redact):
                result = runner.run_document(source, root / "batch")

            self.assertEqual(result["status"], "RELEASED", result.get("release_blockers"))
            self.assertEqual(len(stages.inspected_crops), 1)


class PdfRedactionTests(unittest.TestCase):
    def test_page_scoped_sensitive_values_do_not_flag_other_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two-pages.pdf"
            doc = fitz.open()
            for _ in range(2):
                page = doc.new_page(width=300, height=120)
                page.insert_text((30, 50), "REPORTID")
            doc.save(path)
            doc.close()

            audit = audit_pdf(path, sensitive_values_by_page={0: ["REPORTID"]})

            self.assertEqual(audit["findings"]["leaked_values"], [{"page_number": 0, "value_index": 0}])

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
