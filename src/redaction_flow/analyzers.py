"""PaddleOCR DB/SVTR text recognition, PP-DocLayoutV3 tags and OpenCV codes."""
from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
from typing import Any

import cv2
import fitz
import numpy as np
from PIL import Image

from .contracts import PageRecord, Region, TextSpan


class PaddlePageAnalyzer:
    def __init__(self, dpi: int = 150, layout_threshold: float = 0.5,
                 language: str = "en", device: str = "cpu",
                 ocr_python: Path | None = None,
                 layout_packages: Path | None = None):
        self.dpi = dpi
        self.layout_threshold = layout_threshold
        self.language = language
        self.device = device
        configured_packages = os.environ.get("PADDLE_LAYOUT_PACKAGES")
        packages = layout_packages or (Path(configured_packages) if configured_packages else None)
        if packages is None:
            raise ValueError("Pass layout_packages or set PADDLE_LAYOUT_PACKAGES for PP-DocLayoutV3")
        if not packages.is_dir():
            raise FileNotFoundError(f"PaddleX/PaddlePaddle 3 package directory not found: {packages}")
        package_path = str(packages.resolve())
        if package_path in sys.path:
            sys.path.remove(package_path)
        sys.path.insert(0, package_path)
        legacy_python = ocr_python or Path(os.environ.get("OCR_LEGACY_PYTHON", sys.executable))
        if not legacy_python.is_file():
            raise FileNotFoundError(f"PaddleOCR 2.x worker Python not found: {legacy_python}")
        worker_env = os.environ.copy()
        worker_env.pop("PYTHONPATH", None)
        self._ocr_worker = subprocess.Popen(
            [str(legacy_python), "-u", str(Path(__file__).with_name("ocr_worker.py")), language],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", bufsize=1, env=worker_env,
        )
        ready_record = self._read_worker_record()
        if ready_record.get("status") != "ready":
            self._ocr_worker.terminate()
            raise RuntimeError(f"Legacy PaddleOCR worker failed to initialize: {ready_record}")
        from paddlex import create_model
        self.layout = create_model("PP-DocLayoutV3", device=device)

    def render_page(self, page: fitz.Page, page_number: int, output_dir: Path) -> tuple[Path, np.ndarray]:
        output_dir.mkdir(parents=True, exist_ok=True)
        page_label = f"page_{page_number + 1:04d}"
        image_path = output_dir / f"{page_label}.png"
        pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi / 72, self.dpi / 72), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
        Image.fromarray(image).save(image_path)
        return image_path, image

    def analyze(self, page: fitz.Page, page_number: int, output_dir: Path,
                rendered: tuple[Path, np.ndarray] | None = None) -> tuple[PageRecord, np.ndarray]:
        output_dir.mkdir(parents=True, exist_ok=True)
        page_label = f"page_{page_number + 1:04d}"
        if rendered is None:
            image_path, image = self.render_page(page, page_number, output_dir)
        else:
            image_path, image = rendered
        height, width = image.shape[:2]
        spans = self._ocr_spans(image_path)
        image_regions = self._layout_images(image_path, width, height)
        code_regions = self._codes(image)
        all_regions = [*image_regions, *code_regions]
        for index, region in enumerate(all_regions, 1):
            x0, y0, x1, y1 = [int(round(value)) for value in region.bbox]
            crop_path = output_dir / f"{page_label}_{region.kind}_{index:03d}.png"
            Image.fromarray(image[y0:y1, x0:x1]).save(crop_path)
            region.crop_path = str(crop_path)
        record = PageRecord(
            page_number=page_number, page_label=page_label, image_path=str(image_path),
            image_width=width, image_height=height,
            pdf_width=float(page.rect.width), pdf_height=float(page.rect.height),
            pdf_cropbox=[float(page.cropbox.x0), float(page.cropbox.y0),
                         float(page.cropbox.x1), float(page.cropbox.y1)],
            derotation_matrix=[float(page.derotation_matrix.a), float(page.derotation_matrix.b),
                               float(page.derotation_matrix.c), float(page.derotation_matrix.d),
                               float(page.derotation_matrix.e), float(page.derotation_matrix.f)],
            spans=spans, image_regions=image_regions, code_regions=code_regions,
        )
        return record, image

    def _ocr_spans(self, image_path: Path) -> list[TextSpan]:
        if not self._ocr_worker.stdin or not self._ocr_worker.stdout or self._ocr_worker.poll() is not None:
            raise RuntimeError("Legacy PaddleOCR worker is not running")
        self._ocr_worker.stdin.write(json.dumps({"image_path": str(image_path)}) + "\n")
        self._ocr_worker.stdin.flush()
        result = self._read_worker_record()
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Legacy PaddleOCR worker failed"))
        return [TextSpan(
            index=int(item["index"]), text=str(item["text"]), confidence=float(item["confidence"]),
            polygon=[[float(point[0]), float(point[1])] for point in item["polygon"]],
        ) for item in result.get("spans", [])]

    def _read_worker_record(self) -> dict[str, Any]:
        if not self._ocr_worker.stdout:
            raise RuntimeError("Legacy PaddleOCR worker has no output stream")
        while True:
            line = self._ocr_worker.stdout.readline()
            if not line:
                raise RuntimeError("Legacy PaddleOCR worker exited without returning results")
            if line.startswith("OCR_WORKER_JSON:"):
                return json.loads(line[len("OCR_WORKER_JSON:"):])

    def close(self) -> None:
        worker = getattr(self, "_ocr_worker", None)
        if worker is None or worker.poll() is not None:
            return
        try:
            if worker.stdin:
                worker.stdin.write("{\"command\": \"quit\"}\n")
                worker.stdin.flush()
            worker.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            worker.terminate()
            worker.wait(timeout=5)

    def _layout_images(self, image_path: Path, width: int, height: int) -> list[Region]:
        predictions = list(self.layout.predict(str(image_path), threshold=self.layout_threshold))
        if not predictions:
            return []
        payload = predictions[0].json
        result = payload.get("res", payload)
        regions = []
        for box in result.get("boxes", []):
            if str(box.get("label", "")).casefold() != "image":
                continue
            coords = box.get("coordinate", [])
            if len(coords) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in coords]
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(width), x1), min(float(height), y1)
            if x1 > x0 and y1 > y0:
                regions.append(Region("image", [x0, y0, x1, y1], float(box.get("score", 0.0)), source="PP-DocLayoutV3"))
        return regions

    @staticmethod
    def _codes(image: np.ndarray) -> list[Region]:
        height, width = image.shape[:2]
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        regions = []
        try:
            decoded, points, _ = cv2.QRCodeDetector().detectAndDecode(image_bgr)
            if points is not None:
                points = np.asarray(points).reshape(-1, 2)
                bbox = [max(0.0, float(points[:, 0].min())), max(0.0, float(points[:, 1].min())),
                        min(float(width), float(points[:, 0].max())), min(float(height), float(points[:, 1].max()))]
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    regions.append(Region("qr_code", bbox, 1.0 if decoded else 0.8, bool(decoded), source="OpenCV"))
        except cv2.error as exc:
            raise RuntimeError(f"QR detection failed: {exc}") from exc
        try:
            result = cv2.barcode_BarcodeDetector().detectAndDecode(image_bgr)
            if len(result) == 3:
                value, barcode_type, points = result
                entries = [(value, barcode_type, points)] if points is not None else []
            else:
                _, values, types, corners = result
                values = values if values is not None else []
                types = types if types is not None else []
                corners = corners if corners is not None else []
                entries = list(zip(values, types, corners))
            for value, barcode_type, points in entries:
                points = np.asarray(points).reshape(-1, 2)
                bbox = [max(0.0, float(points[:, 0].min())), max(0.0, float(points[:, 1].min())),
                        min(float(width), float(points[:, 0].max())), min(float(height), float(points[:, 1].max()))]
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    regions.append(Region("barcode", bbox, 1.0 if value else 0.8, bool(value), source=f"OpenCV:{barcode_type}"))
        except (cv2.error, ValueError, TypeError, AttributeError) as exc:
            raise RuntimeError(f"Barcode detection failed: {exc}") from exc
        return regions

def regions_to_json(record: PageRecord) -> dict[str, Any]:
    def serialize(regions: list[Region]) -> list[dict[str, Any]]:
        return [{"type": r.kind, "bbox_page_image_px": r.bbox, "score": r.score,
                 "decoded": r.decoded, "crop_path": r.crop_path, "source": r.source} for r in regions]
    return {
        "page_number": record.page_number, "page_label": record.page_label,
        "image_path": record.image_path, "image_size_px": [record.image_width, record.image_height],
        "pdf_size_pt": [record.pdf_width, record.pdf_height],
        "pdf_cropbox_pt": record.pdf_cropbox,
        "derotation_matrix": record.derotation_matrix,
        "layout_engine": "PP-DocLayoutV3", "code_engine": "OpenCV",
        "image_regions": serialize(record.image_regions), "code_regions": serialize(record.code_regions),
    }


def spans_to_json(record: PageRecord) -> dict[str, Any]:
    return {
        "page_number": record.page_number, "page_label": record.page_label,
        "image_size_px": [record.image_width, record.image_height],
        "pdf_size_pt": [record.pdf_width, record.pdf_height],
        "ocr_engine": "PaddleOCR DB + SVTR_LCNet",
        "spans": [{"index": s.index, "text": s.text, "confidence": s.confidence, "polygon": s.polygon}
                  for s in record.spans],
    }
