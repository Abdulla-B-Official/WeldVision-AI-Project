"""
app.py — Flask REST API Server for WeldVision AI.
Handles UI rendering, image uploads, batch inference, and API endpoints.
"""

import base64
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from config import DEFAULT_CONF_THRESHOLD
from model_service import get_status, run_inference

app = Flask(__name__)


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
        # Fetch confidence threshold from request if provided
        conf_threshold = float(request.form.get("confidence", DEFAULT_CONF_THRESHOLD))
        
        # Read image file into OpenCV array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"success": False, "error": "Invalid image format"}), 400

        # Run ONNX inference
        result = run_inference(image, conf_threshold=conf_threshold)

        if not result["success"]:
            return jsonify(result), 500

        # Encode annotated output image to Base64 format for HTML rendering
        _, buffer = cv2.imencode(".jpg", result["annotated_image"])
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
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

    conf_threshold = float(request.form.get("confidence", DEFAULT_CONF_THRESHOLD))
    batch_results = []

    for file in files:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:
            res = run_inference(image, conf_threshold=conf_threshold)
            if res["success"]:
                _, buffer = cv2.imencode(".jpg", res["annotated_image"])
                encoded = base64.b64encode(buffer).decode("utf-8")
                
                batch_results.append({
                    "filename": file.filename,
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