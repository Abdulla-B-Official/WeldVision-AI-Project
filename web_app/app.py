"""
app.py — Flask REST API Server for WeldVision AI.
Handles UI rendering, single image inference, batch processing, and verdict calculation.
"""

import base64
import os
import sys
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

# Ensure local folder is in sys.path for Gunicorn / Cloud execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import DEFAULT_CONF_THRESHOLD
    from model_service import get_status, run_inference
except ModuleNotFoundError:
    from web_app.config import DEFAULT_CONF_THRESHOLD
    from web_app.model_service import get_status, run_inference

app = Flask(__name__)

# Classes explicitly treated as non-defective
GOOD_WELD_CLASSES = {"good_weld", "good weld", "good", "pass", "acceptable"}


def process_verdict(detections):
    """
    Evaluates detections to determine overall verdict (PASS / DEFECT)
    and extracts maximum confidence.
    """
    if not detections:
        return {"verdict": "PASS", "has_defect": False, "top_confidence": 0.0}

    has_defect = False
    max_conf = 0.0

    for det in detections:
        # Normalize class label matching
        label = str(det.get("class", "")).strip().lower()
        conf = float(det.get("confidence", 0.0))
        
        if conf > max_conf:
            max_conf = conf

        # If class is NOT in good weld set, flag as defect
        if label not in GOOD_WELD_CLASSES:
            has_defect = True

    verdict = "DEFECT" if has_defect else "PASS"
    return {
        "verdict": verdict,
        "has_defect": has_defect,
        "top_confidence": round(max_conf, 1),
    }


@app.route("/")
def index():
    """Render main dashboard interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """API health check and model status route."""
    return jsonify(get_status())


@app.route("/predict", methods=["POST"])
def predict():
    """Single image inference route."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    try:
        conf_threshold = float(
            request.form.get("confidence", DEFAULT_CONF_THRESHOLD)
        )

        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return (
                jsonify({"success": False, "error": "Invalid image format"}),
                400,
            )

        # Run ONNX / YOLO inference
        result = run_inference(image, conf_threshold=conf_threshold)

        if not result.get("success", False):
            return jsonify(result), 500

        # Encode annotated output image to Base64
        _, buffer = cv2.imencode(".jpg", result["annotated_image"])
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        # Determine PASS / DEFECT status
        verdict_data = process_verdict(result["detections"])

        return jsonify({
            "success": True,
            "verdict": verdict_data["verdict"],
            "has_defect": verdict_data["has_defect"],
            "top_confidence": verdict_data["top_confidence"],
            "image": f"data:image/jpeg;base64,{encoded_image}",
            "detections": result["detections"],
            "count": result["count"],
            "inference_time_ms": result["inference_time_ms"],
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Multiple images batch processing route."""
    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"success": False, "error": "No files uploaded"}), 400

    conf_threshold = float(
        request.form.get("confidence", DEFAULT_CONF_THRESHOLD)
    )
    batch_results = []

    for file in files:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:
            res = run_inference(image, conf_threshold=conf_threshold)
            if res.get("success", False):
                _, buffer = cv2.imencode(".jpg", res["annotated_image"])
                encoded = base64.b64encode(buffer).decode("utf-8")
                verdict_data = process_verdict(res["detections"])

                batch_results.append({
                    "filename": file.filename,
                    "verdict": verdict_data["verdict"],
                    "has_defect": verdict_data["has_defect"],
                    "top_confidence": verdict_data["top_confidence"],
                    "image": f"data:image/jpeg;base64,{encoded}",
                    "detections": res["detections"],
                    "count": res["count"],
                    "inference_time_ms": res["inference_time_ms"],
                })

    return jsonify({
        "success": True,
        "processed_count": len(batch_results),
        "results": batch_results,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)