"""
model_service.py
WeldVision AI - ONNX Runtime inference engine
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from config import (
    MODEL_PATH,
    INPUT_SIZE,
    DEFAULT_CONF_THRESHOLD,
    NMS_IOU_THRESHOLD,
    GOOD_IDS,
    DEFECTIVE_IDS,
    CLASS_DISPLAY_NAMES,
    COLOR_GOOD,
    COLOR_DEFECTIVE,
)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# =========================================================
# GLOBAL MODEL STATE
# =========================================================

_session = None
_input_name = None
_output_names = None

_model_loaded = False
_model_error = None


# =========================================================
# LOAD ONNX MODEL
# =========================================================

def load_model():

    global _session
    global _input_name
    global _output_names
    global _model_loaded
    global _model_error

    try:

        if not MODEL_PATH.exists():

            _model_error = (
                f"Model not found: {MODEL_PATH}"
            )

            logger.error(_model_error)

            return

        logger.info(
            f"Loading ONNX model: {MODEL_PATH}"
        )

        options = ort.SessionOptions()

        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        _session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=[
                "CPUExecutionProvider"
            ],
        )

        _input_name = (
            _session.get_inputs()[0].name
        )

        _output_names = [
            output.name
            for output in _session.get_outputs()
        ]

        _model_loaded = True
        _model_error = None

        logger.info(
            "ONNX model loaded successfully."
        )

        logger.info(
            f"Input: {_input_name}"
        )

        logger.info(
            f"Outputs: {_output_names}"
        )

    except Exception as exc:

        _model_loaded = False

        _model_error = str(exc)

        logger.exception(
            "Failed to load ONNX model."
        )


# Load model when Flask starts
load_model()


# =========================================================
# STATUS
# =========================================================

def get_status():

    return {
        "status": "online",
        "api": "online",

        "model_loaded": _model_loaded,

        "model_name": (
            MODEL_PATH.name
            if MODEL_PATH
            else "Unknown"
        ),

        "device": "CPU / ONNX Runtime",

        "model_path": str(MODEL_PATH),

        "error": _model_error,

        "class_mapping": {
            "0": "Bad Weld",
            "1": "Good Weld",
            "2": "Defect",
        },

        "good_classes": [1],

        "defective_classes": [0, 2],
    }


# =========================================================
# LETTERBOX PREPROCESSING
# =========================================================

def preprocess(image):

    if image is None:

        raise ValueError(
            "Input image is None."
        )

    if image.size == 0:

        raise ValueError(
            "Input image is empty."
        )

    original_height, original_width = image.shape[:2]

    scale = min(
        INPUT_SIZE / original_width,
        INPUT_SIZE / original_height,
    )

    new_width = int(
        round(original_width * scale)
    )

    new_height = int(
        round(original_height * scale)
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (
            INPUT_SIZE,
            INPUT_SIZE,
            3,
        ),
        114,
        dtype=np.uint8,
    )

    pad_left = (
        INPUT_SIZE - new_width
    ) // 2

    pad_top = (
        INPUT_SIZE - new_height
    ) // 2

    canvas[
        pad_top:
        pad_top + new_height,
        pad_left:
        pad_left + new_width
    ] = resized

    # BGR -> RGB
    rgb = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB,
    )

    # HWC -> CHW
    blob = rgb.transpose(
        2,
        0,
        1,
    )

    # Normalize
    blob = blob.astype(
        np.float32
    ) / 255.0

    # Add batch dimension
    blob = np.expand_dims(
        blob,
        axis=0,
    )

    return (
        blob,
        scale,
        pad_left,
        pad_top,
    )


# =========================================================
# OUTPUT PARSER
# =========================================================

def parse_predictions(
    output,
    scale,
    pad_left,
    pad_top,
    image_width,
    image_height,
    confidence_threshold,
):

    predictions = np.asarray(output)

    # Remove batch dimension
    predictions = np.squeeze(
        predictions
    )

    if predictions.ndim != 2:

        raise ValueError(
            f"Unexpected ONNX output shape: "
            f"{predictions.shape}"
        )

    # YOLOv8 commonly returns:
    #
    # (84, 8400)
    #
    # Convert to:
    #
    # (8400, 84)

    if predictions.shape[0] < predictions.shape[1]:

        predictions = predictions.T

    boxes = []
    confidences = []
    class_ids = []

    number_of_classes = len(
        CLASS_DISPLAY_NAMES
    )

    for row in predictions:

        if len(row) < 5:

            continue

        # -------------------------------------------------
        # Standard YOLOv8:
        #
        # cx, cy, w, h, class0, class1, class2
        # -------------------------------------------------

        if len(row) == 4 + number_of_classes:

            class_scores = row[4:]

            class_id = int(
                np.argmax(class_scores)
            )

            confidence = float(
                class_scores[class_id]
            )

        # -------------------------------------------------
        # Some YOLO exports include objectness:
        #
        # cx, cy, w, h, objectness, class0...
        # -------------------------------------------------

        elif len(row) == 5 + number_of_classes:

            objectness = float(
                row[4]
            )

            class_scores = row[5:]

            class_id = int(
                np.argmax(class_scores)
            )

            confidence = (
                objectness
                * float(class_scores[class_id])
            )

        else:

            # Fallback
            class_scores = row[4:]

            class_id = int(
                np.argmax(class_scores)
            )

            confidence = float(
                class_scores[class_id]
            )

        if confidence < confidence_threshold:

            continue

        cx, cy, width, height = (
            map(float, row[:4])
        )

        # Convert from model coordinates
        # back to original image coordinates.

        x1 = (
            cx - width / 2 - pad_left
        ) / scale

        y1 = (
            cy - height / 2 - pad_top
        ) / scale

        x2 = (
            cx + width / 2 - pad_left
        ) / scale

        y2 = (
            cy + height / 2 - pad_top
        ) / scale

        # Clip bounding box
        x1 = max(
            0,
            min(
                image_width - 1,
                x1,
            ),
        )

        y1 = max(
            0,
            min(
                image_height - 1,
                y1,
            ),
        )

        x2 = max(
            0,
            min(
                image_width - 1,
                x2,
            ),
        )

        y2 = max(
            0,
            min(
                image_height - 1,
                y2,
            ),
        )

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 2 or box_height <= 2:

            continue

        boxes.append(
            [
                int(x1),
                int(y1),
                int(box_width),
                int(box_height),
            ]
        )

        confidences.append(
            confidence
        )

        class_ids.append(
            class_id
        )

    return (
        boxes,
        confidences,
        class_ids,
    )


# =========================================================
# INFERENCE
# =========================================================

def run_inference(
    image,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
):

    if not _model_loaded:

        return {
            "success": False,
            "annotated_image": image,
            "detections": [],
            "count": 0,
            "inference_time_ms": 0,
            "error": _model_error
            or "Model is not loaded.",
        }

    try:

        start_time = time.perf_counter()

        image_height, image_width = (
            image.shape[:2]
        )

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        blob, scale, pad_left, pad_top = (
            preprocess(image)
        )

        # -------------------------------------------------
        # ONNX INFERENCE
        # -------------------------------------------------

        outputs = _session.run(
            _output_names,
            {
                _input_name: blob
            },
        )

        # -------------------------------------------------
        # PARSE OUTPUT
        # -------------------------------------------------

        (
            boxes,
            confidences,
            class_ids,
        ) = parse_predictions(
            outputs[0],
            scale,
            pad_left,
            pad_top,
            image_width,
            image_height,
            conf_threshold,
        )

        # -------------------------------------------------
        # NMS
        # -------------------------------------------------

        indices = []

        if boxes:

            indices = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                conf_threshold,
                NMS_IOU_THRESHOLD,
            )

        # -------------------------------------------------
        # DRAW RESULTS
        # -------------------------------------------------

        annotated = image.copy()

        detections = []

        if len(indices) > 0:

            for index in np.array(
                indices
            ).flatten():

                x, y, width, height = (
                    boxes[index]
                )

                x2 = x + width
                y2 = y + height

                confidence = confidences[
                    index
                ]

                class_id = class_ids[
                    index
                ]

                # -------------------------------------------------
                # CORRECT CLASS MAPPING
                # -------------------------------------------------

                class_name = (
                    CLASS_DISPLAY_NAMES.get(
                        class_id,
                        f"Class {class_id}",
                    )
                )

                is_good = (
                    class_id in GOOD_IDS
                )

                is_defective = (
                    class_id in DEFECTIVE_IDS
                )

                color = (
                    COLOR_GOOD
                    if is_good
                    else COLOR_DEFECTIVE
                )

                # -------------------------------------------------
                # DRAW BOX
                # -------------------------------------------------

                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x2, y2),
                    color,
                    3,
                )

                # -------------------------------------------------
                # LABEL
                # -------------------------------------------------

                label = (
                    f"{class_name} "
                    f"{confidence:.0%}"
                )

                (
                    text_width,
                    text_height,
                ), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    2,
                )

                label_top = max(
                    0,
                    y - text_height - baseline - 6,
                )

                label_bottom = (
                    label_top
                    + text_height
                    + baseline
                    + 6
                )

                label_right = (
                    x
                    + text_width
                    + 8
                )

                cv2.rectangle(
                    annotated,
                    (x, label_top),
                    (
                        label_right,
                        label_bottom,
                    ),
                    color,
                    -1,
                )

                cv2.putText(
                    annotated,
                    label,
                    (
                        x + 4,
                        label_bottom - baseline - 3,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # -------------------------------------------------
                # SAVE DETECTION
                # -------------------------------------------------

                detections.append(
                    {
                        "class_id": class_id,

                        "class_name": class_name,

                        # Compatibility with older frontend
                        "class": class_name,

                        "confidence": round(
                            confidence,
                            4,
                        ),

                        "is_good": is_good,

                        "is_defective": is_defective,

                        "bbox": {
                            "x1": x,
                            "y1": y,
                            "x2": x2,
                            "y2": y2,
                        },
                    }
                )

        # -------------------------------------------------
        # OVERALL VERDICT
        # -------------------------------------------------

        has_defect = any(
            detection["class_id"]
            in DEFECTIVE_IDS
            for detection in detections
        )

        has_good = any(
            detection["class_id"]
            in GOOD_IDS
            for detection in detections
        )

        if has_defect:

            verdict = "DEFECT"

        elif has_good:

            verdict = "PASS"

        else:

            verdict = "NO_WELD"

        # -------------------------------------------------
        # INFERENCE TIME
        # -------------------------------------------------

        inference_time_ms = round(
            (
                time.perf_counter()
                - start_time
            )
            * 1000,
            2,
        )

        return {

            "success": True,

            "annotated_image": annotated,

            "detections": detections,

            "count": len(detections),

            "inference_time_ms":
                inference_time_ms,

            "verdict": verdict,

            "has_defect": has_defect,

            "has_good": has_good,

            "error": None,
        }

    except Exception as exc:

        logger.exception(
            "Inference failed."
        )

        return {

            "success": False,

            "annotated_image": image,

            "detections": [],

            "count": 0,

            "inference_time_ms": 0,

            "error": str(exc),
        }