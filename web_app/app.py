"""
app.py — Flask REST API Server for WeldVision AI.
"""

import base64
import os
import sys
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

# Ensure local web_app folder is in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from config import DEFAULT_CONF_THRESHOLD
    from model_service import get_status, run_inference
except ModuleNotFoundError:
    from web_app.config import DEFAULT_CONF_THRESHOLD
    from web_app.model_service import get_status, run_inference

app = Flask(__name__)


@app.route("/")
def index():
    """Render main dashboard interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """API health check route expected by JavaScript."""
    try:
        return jsonify(get_status())
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


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
            return jsonify({"success": False, "error": "Invalid image format"}), 400

        result = run_inference(image, conf_threshold=conf_threshold)

        if not result.get("success", False):
            return jsonify(result), 500

        _, buffer = cv2.imencode(".jpg", result["annotated_image"])
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        # Minimal Defect vs Good Weld classification logic
        detections = result.get("detections", [])
        has_defect = any(
            str(d.get("class", "")).strip().lower() not in ["good_weld", "good weld", "good", "pass"]
            for d in detections
        )
        verdict = "DEFECT" if has_defect else "PASS"

        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{encoded_image}",
            "detections": detections,
            "count": result["count"],
            "inference_time_ms": result["inference_time_ms"],
            "verdict": verdict,
            "has_defect": has_defect
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Batch processing route."""
    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"success": False, "error": "No files uploaded"}), 400

    conf_threshold = float(request.form.get("confidence", DEFAULT_CONF_THRESHOLD))
    batch_results = []

    for file in files:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:
            res = run_inference(image, conf_threshold=conf_threshold)
            if res.get("success", False):
                _, buffer = cv2.imencode(".jpg", res["annotated_image"])
                encoded = base64.b64encode(buffer).decode("utf-8")

                has_defect = any(
                    str(d.get("class", "")).strip().lower() not in ["good_weld", "good weld", "good", "pass"]
                    for d in res.get("detections", [])
                )

                batch_results.append({
                    "filename": file.filename,
                    "image": f"data:image/jpeg;base64,{encoded}",
                    "detections": res["detections"],
                    "count": res["count"],
                    "inference_time_ms": res["inference_time_ms"],
                    "verdict": "DEFECT" if has_defect else "PASS",
                    "has_defect": has_defect
                })

    return jsonify({
        "success": True,
        "processed_count": len(batch_results),
        "results": batch_results,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)