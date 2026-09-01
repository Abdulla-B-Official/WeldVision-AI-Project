"""
model_service.py — Loads YOLOv8 once at startup and provides inference.

The model is loaded at module import time so every Flask request reuses
the same in-memory model (no per-request disk I/O).
"""

import logging
import time
import cv2
import numpy as np
import torch

from config import (
    CLASS_DISPLAY_NAMES,
    COLOR_DEFECTIVE,
    COLOR_GOOD,
    DEFAULT_CONF_THRESHOLD,
    DEFECTIVE_IDS,
    DEVICE,
    GOOD_IDS,
    MAX_IMAGE_SIZE,
    MODEL_PATH,
)

logger = logging.getLogger(__name__)

# ── Disable gradient computation globally to conserve memory ──
torch.set_grad_enabled(False)

# ── Load model once ────────────────────────────────────────────────────────────
_model = None
_model_loaded = False
_model_error = None

try:
    if MODEL_PATH.exists():
        from ultralytics import YOLO

        _model = YOLO(str(MODEL_PATH))
        _model.to(DEVICE)
        _model_loaded = True
        logger.info(f"Model loaded from {MODEL_PATH} on {DEVICE}")
    else:
        _model_error = f"Model file not found: {MODEL_PATH}"
        logger.warning(_model_error)
except Exception as exc:
    _model_error = str(exc)
    logger.error(f"Failed to load model: {exc}")


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_status() -> dict:
    """Return current model status information."""
    return {
        "status": "online",
        "model_loaded": _model_loaded,
        "model_name": MODEL_PATH.name,
        "model_path": str(MODEL_PATH),
        "device": DEVICE,
        "error": _model_error,
    }


def run_inference(
    image: np.ndarray,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> dict:
    """
    Run YOLO inference on a BGR OpenCV image.

    Returns:
        {
            "success": bool,
            "annotated_image": np.ndarray (BGR),
            "detections": list[dict],
            "count": int,
            "inference_time_ms": float,
            "error": str | None,
        }
    """
    if not _model_loaded or _model is None:
        return {
            "success": False,
            "annotated_image": image,
            "detections": [],
            "count": 0,
            "inference_time_ms": 0.0,
            "error": _model_error or "Model not loaded.",
        }

    try:
        t_start = time.perf_counter()

        # Enforce inference mode and constrain image size (imgsz=640) to prevent OOM errors
        with torch.inference_mode():
            results = _model.predict(
                source=image,
                conf=conf_threshold,
                imgsz=640,
                device=DEVICE,
                verbose=False,
            )

        t_end = time.perf_counter()
        inference_ms = round((t_end - t_start) * 1000, 2)

        annotated = image.copy()
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                # Use corrected display name (CLASS_DISPLAY_NAMES overrides
                # the model's embedded names which are swapped vs reality)
                class_name = CLASS_DISPLAY_NAMES.get(
                    class_id, _model.names[class_id]
                )

                color = COLOR_GOOD if class_id in GOOD_IDS else COLOR_DEFECTIVE

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                # Draw label background
                label = f"{class_name}  {confidence:.0%}"
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

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "is_defective": class_id in DEFECTIVE_IDS,
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    }
                )

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