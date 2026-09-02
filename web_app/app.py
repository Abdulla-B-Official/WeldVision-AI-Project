"""
app.py — Flask REST API Server for WeldVision AI.
Handles UI rendering, API health checks, single image inspection, and batch processing.
"""

import base64
import os
import sys
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

# Force absolute path resolution so Render/Gunicorn locate modules correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from config import DEFAULT_CONF_THRESHOLD
    from model_service import get_status, run_inference
except ModuleNotFoundError:
    from web_app.config import DEFAULT_CONF_THRESHOLD
    from web_app.model_service import get_status, run_inference

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    """Render main dashboard interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Fail-safe health endpoint structure.
    Guarantees a 200 OK JSON response to unfreeze the frontend JS UI.
    """
    try:
        status = get_status()
        if isinstance(status, dict):
            status.setdefault("status", "online")
            status.setdefault("api", "online")
            status.setdefault("model_loaded", True)
            status.setdefault("device", status.get("device", "CPU"))
            status.setdefault("model_name", status.get("model_name", "best.pt"))
            return jsonify(status), 200
    except Exception as e:
        # Fallback payload to clear 'API Checking...' if model service has startup delay
        return jsonify({
            "status": "online",
            "api": "online",
            "model_loaded": False,
            "device": "CPU",
            "model_name": "best.pt",
            "warning": str(e)
        }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Single image inference route with simplified Defect / Pass logic."""
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

        detections = result.get("detections", [])

        # Simple logic: If any detection label is NOT a good weld variant -> DEFECT
        has_defect = any(
            str(d.get("class", "")).strip().lower() not in ["good_weld", "good weld", "good", "pass"]
            for d in detections
        )
        verdict = "DEFECT" if has_defect else "PASS"

        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{encoded_image}",
            "detections": detections,
            "count": result.get("count", len(detections)),
            "inference_time_ms": result.get("inference_time_ms", 0),
            "verdict": verdict,
            "has_defect": has_defect
        }), 200

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

                dets = res.get("detections", [])
                has_defect = any(
                    str(d.get("class", "")).strip().lower() not in ["good_weld", "good weld", "good", "pass"]
                    for d in dets
                )

                batch_results.append({
                    "filename": file.filename,
                    "image": f"data:image/jpeg;base64,{encoded}",
                    "detections": dets,
                    "count": res.get("count", len(dets)),
                    "inference_time_ms": res.get("inference_time_ms", 0),
                    "verdict": "DEFECT" if has_defect else "PASS",
                    "has_defect": has_defect
                })

    return jsonify({
        "success": True,
        "processed_count": len(batch_results),
        "results": batch_results,
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)