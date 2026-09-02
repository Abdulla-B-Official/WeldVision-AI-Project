"""
app.py — WeldVision AI Flask Server

YOLO Native Class Mapping
-------------------------
0 = Good Weld -> PASS
1 = Bad Weld  -> DEFECT
2 = Defect    -> DEFECT

Overall Verdict Priority
------------------------
DEFECT > PASS > NO_WELD
"""

import base64
import logging
import os

import cv2
import numpy as np

from flask import Flask, jsonify, render_template, request

from web_app.config import (
    DEFAULT_CONF_THRESHOLD,
    DEFECTIVE_IDS,
    GOOD_IDS,
)

from web_app.model_service import (
    get_status,
    run_inference,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# Maximum upload size = 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAPPING = {
    0: "Good Weld",
    1: "Bad Weld",
    2: "Defect",
}


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def index():
    """Render the WeldVision AI web application."""
    return render_template("index.html")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    """
    Health endpoint used by Render and the frontend.

    Returns HTTP 200 even if the model has an error so that
    the frontend can distinguish API status from model status.
    """
    try:
        status = get_status()
        return jsonify(status), 200

    except Exception as exc:
        logger.exception("Health check failed.")
        return jsonify(
            {
                "status": "error",
                "api": "online",
                "model_loaded": False,
                "device": "CPU / ONNX Runtime",
                "error": str(exc),
            }
        ), 200


# ============================================================
# API STATUS
# ============================================================

@app.route("/api/status", methods=["GET"])
def api_status():
    """Compatibility endpoint for the frontend."""
    try:
        status = get_status()
        return jsonify(status), 200

    except Exception as exc:
        logger.exception("API status check failed.")
        return jsonify(
            {
                "status": "error",
                "api": "online",
                "model_loaded": False,
                "device": "CPU / ONNX Runtime",
                "error": str(exc),
            }
        ), 200


# ============================================================
# IMAGE DECODER
# ============================================================

def decode_uploaded_image(file):
    """
    Convert uploaded file into an OpenCV BGR image.
    """
    if file is None:
        return None

    data = file.read()
    if not data:
        return None

    image_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# ============================================================
# IMAGE ENCODER
# ============================================================

def encode_image(image):
    """
    Convert OpenCV image to base64 JPEG data URL.
    """
    if image is None:
        raise ValueError("Cannot encode an empty image.")

    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        raise ValueError("Could not encode result image.")

    encoded = base64.b64encode(buffer).decode("utf-8")
    return "data:image/jpeg;base64," + encoded


# ============================================================
# CONFIDENCE
# ============================================================

def get_confidence():
    """
    Read confidence threshold from the request.
    Keeps the value between 0.05 and 0.99.
    """
    try:
        confidence = float(
            request.form.get("confidence", DEFAULT_CONF_THRESHOLD)
        )
    except (TypeError, ValueError):
        confidence = DEFAULT_CONF_THRESHOLD

    return max(0.05, min(0.99, confidence))


# ============================================================
# VERDICT CALCULATION
# ============================================================

def calculate_verdict(detections):
    """
    Calculate overall weld condition directly from native model output.

    Mapping:
        0 -> Good Weld -> PASS
        1 -> Bad Weld  -> DEFECT
        2 -> Defect    -> DEFECT

    Priority:
        DEFECT > PASS > NO_WELD
    """
    if not detections:
        return ("NO_WELD", False, False)

    has_defect = any(
        int(detection.get("class_id", -1)) in DEFECTIVE_IDS
        for detection in detections
    )

    has_good = any(
        int(detection.get("class_id", -1)) in GOOD_IDS
        for detection in detections
    )

    if has_defect:
        verdict = "DEFECT"
    elif has_good:
        verdict = "PASS"
    else:
        verdict = "NO_WELD"

    return (verdict, has_defect, has_good)


# ============================================================
# FORMAT PREDICTION RESPONSE
# ============================================================

def build_prediction_response(result, include_image=True):
    """
    Convert model_service inference result into
    the JSON structure expected by the frontend.
    """
    detections = result.get("detections", [])

    (
        verdict,
        has_defect,
        has_good,
    ) = calculate_verdict(detections)

    response = {
        "success": True,
        "detections": detections,
        "count": len(detections),
        "inference_time_ms": result.get("inference_time_ms", 0),
        "verdict": verdict,
        "has_defect": has_defect,
        "has_good": has_good,
        "class_mapping": {
            "0": "Good Weld",
            "1": "Bad Weld",
            "2": "Defect",
        },
    }

    if include_image:
        response["image"] = encode_image(result["annotated_image"])

    return response


# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    """Process one uploaded image."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No image file uploaded."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "No image selected."}), 400

    try:
        image = decode_uploaded_image(file)
        if image is None:
            return jsonify({"success": False, "error": "Invalid image file."}), 400

        confidence = get_confidence()

        logger.info(
            "Prediction request: filename=%s size=%sx%s confidence=%.2f",
            file.filename,
            image.shape[1],
            image.shape[0],
            confidence,
        )

        result = run_inference(image, confidence_threshold=confidence)

        response = build_prediction_response(result, include_image=True)

        logger.info(
            "Prediction successful: verdict=%s detections=%d time=%.2fms",
            response["verdict"],
            response["count"],
            response["inference_time_ms"],
        )

        return jsonify(response), 200

    except Exception as exc:
        logger.exception("Inference failed for /predict.")
        return jsonify(
            {
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        ), 503


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Process multiple uploaded images."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"success": False, "error": "No files uploaded."}), 400

    try:
        confidence = get_confidence()
        results = []
        failed_files = []

        for file in files:
            if not file.filename:
                continue

            try:
                image = decode_uploaded_image(file)
                if image is None:
                    failed_files.append(
                        {
                            "filename": file.filename,
                            "error": "Invalid image file.",
                        }
                    )
                    continue

                result = run_inference(image, confidence_threshold=confidence)

                response = build_prediction_response(result, include_image=True)
                response["filename"] = file.filename

                results.append(response)

            except Exception as exc:
                logger.exception("Batch inference failed: %s", file.filename)
                failed_files.append(
                    {
                        "filename": file.filename,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )

        return jsonify(
            {
                "success": True,
                "processed_count": len(results),
                "failed_count": len(failed_files),
                "results": results,
                "failed_files": failed_files,
            }
        ), 200

    except Exception as exc:
        logger.exception("Batch prediction failed.")
        return jsonify(
            {
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        ), 500


# ============================================================
# FILE TOO LARGE HANDLER
# ============================================================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify(
        {
            "success": False,
            "error": "Image is too large. Maximum size is 10 MB.",
        }
    ), 413


# ============================================================
# GENERAL ERROR HANDLER
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):
    logger.exception("Internal server error.")
    return jsonify(
        {
            "success": False,
            "error": "Internal server error.",
        }
    ), 500


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )