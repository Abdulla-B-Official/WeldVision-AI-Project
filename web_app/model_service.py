"""
model_service.py
WeldVision AI - YOLO ONNX inference service

Class mapping:
    0 = Bad Weld  -> DEFECT
    1 = Good Weld -> PASS
    2 = Defect    -> DEFECT
"""

import time
import threading

import cv2
import numpy as np
import onnxruntime as ort

from web_app.config import (
    CLASS_DISPLAY_NAMES,
    COLOR_DEFECTIVE,
    COLOR_GOOD,
    DEFAULT_CONF_THRESHOLD,
    DEFECTIVE_IDS,
    GOOD_IDS,
    INPUT_SIZE,
    MODEL_PATH,
    NMS_IOU_THRESHOLD,
)

# ============================================================
# GLOBAL MODEL STATE
# ============================================================

_session = None
_model_error = None
_model_lock = threading.Lock()


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """
    Load the ONNX model using ONNX Runtime.

    The model is loaded only once and reused for all predictions.
    """

    global _session
    global _model_error

    if _session is not None:
        return _session

    with _model_lock:

        if _session is not None:
            return _session

        try:
            model_path = str(MODEL_PATH)

            print("=" * 60)
            print("WELDVISION AI - LOADING ONNX MODEL")
            print("=" * 60)
            print(f"Model path: {model_path}")

            providers = ["CPUExecutionProvider"]

            _session = ort.InferenceSession(
                model_path,
                providers=providers
            )

            _model_error = None

            print("Model loaded successfully.")
            print(f"Providers: {_session.get_providers()}")

            # Print input information
            for inp in _session.get_inputs():
                print(
                    f"Input: {inp.name} | "
                    f"Shape: {inp.shape} | "
                    f"Type: {inp.type}"
                )

            # Print output information
            for out in _session.get_outputs():
                print(
                    f"Output: {out.name} | "
                    f"Shape: {out.shape} | "
                    f"Type: {out.type}"
                )

            print("=" * 60)

            return _session

        except Exception as exc:
            _session = None
            _model_error = str(exc)

            print("=" * 60)
            print("ERROR LOADING ONNX MODEL")
            print("=" * 60)
            print(_model_error)
            print("=" * 60)

            return None


# ============================================================
# MODEL STATUS
# ============================================================

def get_status():
    """
    Return the current API/model status.
    """

    session = load_model()

    return {
        "status": "online" if session is not None else "error",
        "model_loaded": session is not None,
        "model_name": MODEL_PATH.name,
        "model_path": str(MODEL_PATH),
        "device": "CPU / ONNX Runtime",
        "error": _model_error,
    }


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(image):
    """
    Convert an OpenCV BGR image into YOLO ONNX input format.

    Output:
        numpy array with shape:
        (1, 3, INPUT_SIZE, INPUT_SIZE)
    """

    if image is None:
        raise ValueError("Input image is None.")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    # Resize to model input size
    resized = cv2.resize(
        image,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # BGR -> RGB
    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    # Convert uint8 -> float32
    tensor = rgb.astype(np.float32) / 255.0

    # HWC -> CHW
    tensor = np.transpose(
        tensor,
        (2, 0, 1)
    )

    # Add batch dimension
    tensor = np.expand_dims(
        tensor,
        axis=0
    )

    return tensor


# ============================================================
# PREDICTION PARSER
# ============================================================

def parse_predictions(
    output,
    original_width,
    original_height,
    confidence_threshold=DEFAULT_CONF_THRESHOLD,
):
    """
    Parse YOLO ONNX predictions.

    Supports common YOLO output formats:

        YOLOv8:
        [x, y, w, h, class_scores...]

    and formats containing objectness:

        [x, y, w, h, objectness, class_scores...]

    Returns a list of detection dictionaries.
    """

    predictions = np.asarray(output)

    # --------------------------------------------------------
    # Remove unnecessary dimensions
    # --------------------------------------------------------

    predictions = np.squeeze(predictions)

    if predictions.ndim != 2:
        raise ValueError(
            f"Unexpected YOLO output shape: {predictions.shape}"
        )

    # --------------------------------------------------------
    # YOLO commonly returns:
    #
    #   (84, 8400)
    #
    # or:
    #
    #   (8400, 84)
    #
    # For 3 classes:
    #
    #   (7, 8400)
    # --------------------------------------------------------

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    num_columns = predictions.shape[1]

    # We have 3 classes.
    #
    # YOLOv8 format:
    #   4 + 3 = 7
    #
    # Objectness format:
    #   5 + 3 = 8
    # --------------------------------------------------------

    if num_columns not in (7, 8):
        raise ValueError(
            f"Unexpected YOLO prediction shape: "
            f"{predictions.shape}. "
            f"Expected 7 or 8 values per detection."
        )

    # --------------------------------------------------------
    # Determine scale from model input to original image
    # --------------------------------------------------------

    x_scale = original_width / float(INPUT_SIZE)
    y_scale = original_height / float(INPUT_SIZE)

    boxes = []
    scores = []
    class_ids = []

    # ========================================================
    # LOOP THROUGH PREDICTIONS
    # ========================================================

    for prediction in predictions:

        # ----------------------------------------------------
        # YOLOv8 standard format
        #
        # x, y, w, h, class0, class1, class2
        # ----------------------------------------------------

        if num_columns == 7:

            x_center = float(prediction[0])
            y_center = float(prediction[1])
            width = float(prediction[2])
            height = float(prediction[3])

            class_scores = prediction[4:]

            class_id = int(
                np.argmax(class_scores)
            )

            confidence = float(
                class_scores[class_id]
            )

        # ----------------------------------------------------
        # Objectness format
        #
        # x, y, w, h, objectness,
        # class0, class1, class2
        # ----------------------------------------------------

        else:

            x_center = float(prediction[0])
            y_center = float(prediction[1])
            width = float(prediction[2])
            height = float(prediction[3])

            objectness = float(
                prediction[4]
            )

            class_scores = prediction[5:]

            class_id = int(
                np.argmax(class_scores)
            )

            class_confidence = float(
                class_scores[class_id]
            )

            confidence = (
                objectness * class_confidence
            )

        # ----------------------------------------------------
        # Confidence filtering
        # ----------------------------------------------------

        if confidence < confidence_threshold:
            continue

        # ----------------------------------------------------
        # Convert center coordinates to corners
        # ----------------------------------------------------

        x1 = (x_center - width / 2.0) * x_scale
        y1 = (y_center - height / 2.0) * y_scale
        x2 = (x_center + width / 2.0) * x_scale
        y2 = (y_center + height / 2.0) * y_scale

        # ----------------------------------------------------
        # Clamp coordinates to image boundaries
        # ----------------------------------------------------

        x1 = max(
            0,
            min(int(round(x1)), original_width - 1)
        )

        y1 = max(
            0,
            min(int(round(y1)), original_height - 1)
        )

        x2 = max(
            0,
            min(int(round(x2)), original_width - 1)
        )

        y2 = max(
            0,
            min(int(round(y2)), original_height - 1)
        )

        box_width = max(
            0,
            x2 - x1
        )

        box_height = max(
            0,
            y2 - y1
        )

        if box_width <= 0 or box_height <= 0:
            continue

        boxes.append(
            [
                x1,
                y1,
                box_width,
                box_height
            ]
        )

        scores.append(
            confidence
        )

        class_ids.append(
            class_id
        )

    # ========================================================
    # NON-MAXIMUM SUPPRESSION
    # ========================================================

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        confidence_threshold,
        NMS_IOU_THRESHOLD
    )

    if indices is None or len(indices) == 0:
        return []

    # OpenCV can return:
    #
    # [[0], [1], [2]]
    #
    # or:
    #
    # [0, 1, 2]
    #
    indices = np.array(indices).reshape(-1)

    detections = []

    # ========================================================
    # BUILD DETECTION RESULTS
    # ========================================================

    for index in indices:

        index = int(index)

        x, y, width, height = boxes[index]

        class_id = int(
            class_ids[index]
        )

        confidence = float(
            scores[index]
        )

        # ----------------------------------------------------
        # Get class name
        # ----------------------------------------------------

        class_name = CLASS_DISPLAY_NAMES.get(
            class_id,
            f"Class {class_id}"
        )

        # ----------------------------------------------------
        # Determine Good / Defective
        # ----------------------------------------------------

        is_good = (
            class_id in GOOD_IDS
        )

        is_defective = (
            class_id in DEFECTIVE_IDS
        )

        # ----------------------------------------------------
        # Final coordinates
        # ----------------------------------------------------

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + width)
        y2 = int(y + height)

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,

                # Compatibility field
                "class": class_name,

                "confidence": round(
                    confidence,
                    4
                ),

                "is_good": is_good,
                "is_defective": is_defective,

                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
            }
        )

    return detections


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image,
    detections,
):
    """
    Draw bounding boxes and labels on the image.
    """

    annotated = image.copy()

    for detection in detections:

        class_id = detection["class_id"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        bbox = detection["bbox"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        # ----------------------------------------------------
        # Color based on class
        # ----------------------------------------------------

        if class_id in GOOD_IDS:
            color = COLOR_GOOD

        else:
            color = COLOR_DEFECTIVE

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"{class_name} "
            f"{confidence * 100:.1f}%"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )
        )

        # Keep label inside image
        label_x = max(
            0,
            x1
        )

        label_y = max(
            text_height + baseline,
            y1
        )

        # Background rectangle
        cv2.rectangle(
            annotated,
            (
                label_x,
                label_y - text_height - baseline
            ),
            (
                label_x + text_width,
                label_y
            ),
            color,
            -1
        )

        # Text
        cv2.putText(
            annotated,
            label,
            (
                label_x,
                label_y - baseline
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return annotated


# ============================================================
# RUN INFERENCE
# ============================================================

def run_inference(
    image,
    confidence_threshold=DEFAULT_CONF_THRESHOLD,
):
    """
    Run YOLO ONNX inference on an OpenCV BGR image.

    Returns:
        {
            "detections": [...],
            "verdict": "PASS" / "DEFECT" / "NO_WELD",
            "has_defect": bool,
            "has_good": bool,
            "inference_time_ms": float,
            "annotated_image": numpy.ndarray
        }
    """

    # ========================================================
    # VALIDATE IMAGE
    # ========================================================

    if image is None:
        raise ValueError(
            "Input image is None."
        )

    if image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    session = load_model()

    if session is None:
        raise RuntimeError(
            f"ONNX model could not be loaded: "
            f"{_model_error}"
        )

    # ========================================================
    # ORIGINAL IMAGE SIZE
    # ========================================================

    original_height, original_width = image.shape[:2]

    # ========================================================
    # PREPROCESS
    # ========================================================

    input_tensor = preprocess(
        image
    )

    # ========================================================
    # GET INPUT NAME
    # ========================================================

    input_name = session.get_inputs()[0].name

    # ========================================================
    # RUN ONNX INFERENCE
    # ========================================================

    start_time = time.perf_counter()

    outputs = session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    # ========================================================
    # VALIDATE OUTPUT
    # ========================================================

    if not outputs:
        raise RuntimeError(
            "ONNX model returned no output."
        )

    # First output is normally YOLO predictions
    output = outputs[0]

    # ========================================================
    # PARSE PREDICTIONS
    # ========================================================

    detections = parse_predictions(
        output=output,
        original_width=original_width,
        original_height=original_height,
        confidence_threshold=confidence_threshold,
    )

    # ========================================================
    # DETERMINE OVERALL VERDICT
    #
    # 0 = Bad Weld  -> DEFECT
    # 1 = Good Weld -> PASS
    # 2 = Defect    -> DEFECT
    #
    # Priority:
    # DEFECT > PASS > NO_WELD
    # ========================================================

    has_defect = any(
        detection["class_id"] in DEFECTIVE_IDS
        for detection in detections
    )

    has_good = any(
        detection["class_id"] in GOOD_IDS
        for detection in detections
    )

    if has_defect:
        verdict = "DEFECT"

    elif has_good:
        verdict = "PASS"

    else:
        verdict = "NO_WELD"

    # ========================================================
    # DRAW RESULTS
    # ========================================================

    annotated_image = draw_detections(
        image,
        detections
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "detections": detections,

        "verdict": verdict,

        "has_defect": has_defect,

        "has_good": has_good,

        "inference_time_ms": round(
            inference_time_ms,
            2
        ),

        "annotated_image": annotated_image,
    }