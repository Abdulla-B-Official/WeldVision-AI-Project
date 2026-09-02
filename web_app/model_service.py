"""
model_service.py — Pure ONNX Runtime engine.
Eliminates PyTorch dependencies to maintain low memory usage (<80MB RAM).
"""

import logging
import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

# Ensure local folder is in sys.path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import (
        CLASS_DISPLAY_NAMES,
        COLOR_DEFECTIVE,
        COLOR_GOOD,
        DEFAULT_CONF_THRESHOLD,
        DEFECTIVE_IDS,
        GOOD_IDS,
        MODEL_PATH,
    )
except ModuleNotFoundError:
    from web_app.config import (
        CLASS_DISPLAY_NAMES,
        COLOR_DEFECTIVE,
        COLOR_GOOD,
        DEFAULT_CONF_THRESHOLD,
        DEFECTIVE_IDS,
        GOOD_IDS,
        MODEL_PATH,
    )

logger = logging.getLogger(__name__)

_session = None
_model_loaded = False
_model_error = None
_input_name = None
_output_names = None

try:
    if MODEL_PATH.exists():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        _session = ort.InferenceSession(
            str(MODEL_PATH), opts, providers=["CPUExecutionProvider"]
        )
        _input_name = _session.get_inputs()[0].name
        _output_names = [o.name for o in _session.get_outputs()]

        _model_loaded = True
        logger.info(f"Native ONNX Engine loaded from {MODEL_PATH}")
    else:
        _model_error = f"Model file not found: {MODEL_PATH}"
except Exception as exc:
    _model_error = str(exc)
    logger.error(f"Failed to load ONNX model: {exc}")


def get_status() -> dict:
    return {
        "status": "online",
        "model_loaded": _model_loaded,
        "model_name": MODEL_PATH.name if MODEL_PATH else "N/A",
        "device": "cpu (pure-onnx)",
        "error": _model_error,
    }


def preprocess(image: np.ndarray, target_size: int = 640):
    """Resize, letterbox, normalize image to (1, 3, 640, 640) float32 array."""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    top = (target_size - nh) // 2
    left = (target_size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized

    blob = canvas[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(blob, axis=0), scale, top, left


def run_inference(
    image: np.ndarray, conf_threshold: float = DEFAULT_CONF_THRESHOLD
) -> dict:
    if not _model_loaded or _session is None:
        return {
            "success": False,
            "annotated_image": image,
            "detections": [],
            "count": 0,
            "inference_time_ms": 0.0,
            "error": _model_error,
        }

    try:
        t_start = time.perf_counter()

        blob, scale, pad_top, pad_left = preprocess(image)
        outputs = _session.run(_output_names, {_input_name: blob})

        preds = np.squeeze(outputs[0])
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # Transpose to shape (8400, 84)

        boxes, confidences, class_ids = [], [], []

        for row in preds:
            scores = row[4:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])

            if confidence >= conf_threshold:
                cx, cy, w, h = row[0:4]
                x1 = int((cx - w / 2 - pad_left) / scale)
                y1 = int((cy - h / 2 - pad_top) / scale)
                bw = int(w / scale)
                bh = int(h / scale)

                boxes.append([x1, y1, bw, bh])
                confidences.append(confidence)
                class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.45)

        t_end = time.perf_counter()
        inference_ms = round((t_end - t_start) * 1000, 2)

        annotated = image.copy()
        detections = []

        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, bw, bh = boxes[i]
                x2, y2 = x1 + bw, y1 + bh
                confidence = confidences[i]
                class_id = class_ids[i]

                class_name = CLASS_DISPLAY_NAMES.get(
                    class_id, f"Class {class_id}"
                )
                color = (
                    COLOR_GOOD if class_id in GOOD_IDS else COLOR_DEFECTIVE
                )

                cv2.rectangle(
                    annotated,
                    (max(0, x1), max(0, y1)),
                    (min(image.shape[1], x2), min(image.shape[0], y2)),
                    color,
                    3,
                )

                label = f"{class_name} {confidence:.0%}"
                (lw, lh), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
                )
                label_y = max(y1 - 10, lh + baseline)
                cv2.rectangle(
                    annotated,
                    (x1, label_y - lh - baseline),
                    (x1 + lw + 4, label_y + baseline),
                    color,
                    cv2.FILLED,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "is_defective": class_id in DEFECTIVE_IDS,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

        return {
            "success": True,
            "annotated_image": annotated,
            "detections": detections,
            "count": len(detections),
            "inference_time_ms": inference_ms,
            "error": None,
        }

    except Exception as exc:
        logger.exception("Inference failed")
        return {
            "success": False,
            "annotated_image": image,
            "detections": [],
            "count": 0,
            "inference_time_ms": 0.0,
            "error": str(exc),
        }