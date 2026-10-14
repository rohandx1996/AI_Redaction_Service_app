"""Offline regression checks for selective image masking."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import llm_material_text_experiment as experiment


class SelectiveMaskTests(unittest.TestCase):
    def test_only_redact_decisions_change_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.png"
            image = np.full((50, 100, 3), 255, dtype=np.uint8)
            experiment.save_png(source, image)
            regions = []
            for index, decision in enumerate(("redact", "preserve", "review", "invalid")):
                regions.append({"text": str(index), "x0": index * 20 + 2, "y0": 5,
                                "x1": index * 20 + 12, "y1": 15, "decision": decision})
            regions.append({"text": "missing decision", "x0": 82, "y0": 5, "x1": 92, "y1": 15})
            item = {"path": source, "document": "doc", "crop": "sample.png",
                    "page_label": "1", "page_bbox": [0, 0, 100, 50]}
            with patch.object(experiment, "call_model", return_value=({"text_regions": regions}, {}, 0.1)):
                result = experiment.process("mock", item, root / "out", 3, "selective")
            saved = cv2.imdecode(np.frombuffer((Path(result["output_folder"]) / "redacted.png").read_bytes(),
                                               np.uint8), cv2.IMREAD_COLOR)
            self.assertTrue(np.all(saved[10, 7] == 0))
            for x in (27, 47, 67, 87):
                self.assertTrue(np.array_equal(saved[10, x], image[10, x]))
            self.assertEqual(result["redacted_regions"], 1)
            self.assertEqual(result["preserved_regions"], 1)
            self.assertEqual(result["review_regions"], 3)
            self.assertTrue(result["requires_review"])
            self.assertTrue(result["pixel_integrity_pass"])

    def test_selective_request_contains_contextual_policy(self):
        with patch.object(experiment.requests, "post") as post:
            post.return_value.json.return_value = {"content": [{"type": "text", "text": '{"text_regions":[]}'}]}
            experiment.call_model("claude-opus-5-5", b"png", 100, 50, "selective")
            body = post.call_args.kwargs["json"]
            self.assertNotIn("temperature", body)
            self.assertIn("material-identifiers-v1", body["messages"][0]["content"][1]["text"])


if __name__ == "__main__":
    unittest.main()