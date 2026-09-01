"""
app.py — WeldVision AI  |  Production-Ready Entry Point.
"""

import io
import logging
import os
from pathlib import Path
import sys
import threading
import webbrowser

from flask import Flask, jsonify, render_template, request
from PIL import Image

# ── Ensure sibling modules resolve regardless of entry point ──
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_service
import utils
from config import DEFAULT_CONF_THRESHOLD, MAX_IMAGE_SIZE

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── UI route ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the WeldVision AI web interface."""
    return render_template("index.html")


# ── API routes ─────────────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def status():
    """Return model and device status."""
    return jsonify(model_service.get_status())


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accept multipart/form-data image upload.
    Optional field: conf_threshold (float, default 0.50)
    Returns annotated image (base64) + detections JSON.
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify(
            {"success": False, "error": "Invalid file. Supported: JPG, JPEG, PNG, WEBP."}
        ), 400

    try:
        conf = float(request.form.get("conf_threshold", DEFAULT_CONF_THRESHOLD))
        conf = max(0.01, min(conf, 1.0))
    except (ValueError, TypeError):
        conf = DEFAULT_CONF_THRESHOLD

    try:
        pil_image = Image.open(io.BytesIO(file.read()))
        cv2_image = utils.pil_to_cv2(pil_image)
        cv2_image = utils.resize_if_large(cv2_image, MAX_IMAGE_SIZE)
    except Exception as exc:
        logger.warning(f"Image decode error: {exc}")
        return jsonify({"success": False, "error": "Could not read image file."}), 400

    result = model_service.run_inference(cv2_image, conf_threshold=conf)

    if not result["success"]:
        return jsonify({"success": False, "error": result["error"]}), 500

    try:
        b64 = utils.cv2_to_base64(result["annotated_image"])
    except Exception as exc:
        return jsonify({"success": False, "error": f"Image encoding failed: {exc}"}), 500

    return jsonify({
        "success":           True,
        "annotated_image":   b64,
        "detections":        result["detections"],
        "count":             result["count"],
        "inference_time_ms": result["inference_time_ms"],
    })


@app.route("/api/webcam", methods=["POST"])
def webcam():
    """
    Accept JSON: { "frame": "<base64 string>", "conf_threshold": 0.5 }
    Returns annotated frame + detections.
    """
    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"success": False, "error": "No frame data provided."}), 400

    try:
        conf = float(data.get("conf_threshold", DEFAULT_CONF_THRESHOLD))
        conf = max(0.01, min(conf, 1.0))
    except (ValueError, TypeError):
        conf = DEFAULT_CONF_THRESHOLD

    try:
        cv2_image = utils.base64_to_cv2(data["frame"])
        cv2_image = utils.resize_if_large(cv2_image, MAX_IMAGE_SIZE)
    except Exception as exc:
        logger.warning(f"Webcam frame decode error: {exc}")
        return jsonify({"success": False, "error": "Could not decode webcam frame."}), 400

    result = model_service.run_inference(cv2_image, conf_threshold=conf)

    if not result["success"]:
        return jsonify({"success": False, "error": result["error"]}), 500

    try:
        b64 = utils.cv2_to_base64(result["annotated_image"], quality=80)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Image encoding failed: {exc}"}), 500

    return jsonify({
        "success":           True,
        "annotated_image":   b64,
        "detections":        result["detections"],
        "count":             result["count"],
        "inference_time_ms": result["inference_time_ms"],
    })


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error."}), 500


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 5000))

    info = model_service.get_status()
    logger.info(f"Model loaded={info['model_loaded']}  device={info['device']}")
    if not info["model_loaded"]:
        logger.warning(f"Model NOT loaded: {info.get('error')}")

    if os.environ.get("RENDER") is None:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()

    logger.info(f"Starting WeldVision AI at http://{HOST}:{PORT}")
    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )