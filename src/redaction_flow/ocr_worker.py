"""Persistent worker for PaddleOCR 2.x to keep its Paddle runtime isolated from PaddleX 3.x."""
from __future__ import annotations

import json
import sys
import traceback

import cv2
import paddle
from paddleocr import PaddleOCR


def emit(payload: dict) -> None:
    print("OCR_WORKER_JSON:" + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    language = sys.argv[1] if len(sys.argv) > 1 else "en"
    try:
        try:
            paddle.set_flags({"FLAGS_use_mkldnn": False, "FLAGS_use_onednn": False})
        except ValueError:
            paddle.set_flags({"FLAGS_use_mkldnn": False})
        engine = PaddleOCR(det_algorithm="DB", rec_algorithm="SVTR_LCNet", use_gpu=False,
                           lang=language, use_angle_cls=False)
        emit({"status": "ready", "paddle": paddle.__version__})
    except Exception as exc:
        emit({"status": "error", "error": str(exc), "traceback": traceback.format_exc()})
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "quit":
                emit({"status": "bye"})
                return 0
            image = cv2.imread(request["image_path"], cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image {request['image_path']}")
            detections = engine.ocr(image, cls=False)
            page_detections = detections[0] if detections and detections[0] else []
            spans = []
            for index, (polygon, recognition) in enumerate(page_detections):
                text, confidence = recognition
                spans.append({"index": index, "text": str(text), "confidence": float(confidence),
                              "polygon": [[float(point[0]), float(point[1])] for point in polygon]})
            emit({"status": "ok", "spans": spans})
        except Exception as exc:
            emit({"status": "error", "error": str(exc), "traceback": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
