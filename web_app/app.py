"""
app.py — WeldVision AI Flask Server

Class Mapping:
    0 = Bad Weld  -> DEFECT
    1 = Good Weld -> PASS
    2 = Defect    -> DEFECT
"""

import base64
import os

import cv2
import numpy as np

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)

from web_app.config import DEFAULT_CONF_THRESHOLD

from web_app.model_service import (
    get_status,
    run_inference,
)

# =========================================================
# CLASS MAPPING
# =========================================================

# IMPORTANT:
# The YOLO model uses these class IDs.
#
# 0 -> Bad Weld -> DEFECTIVE
# 1 -> Good Weld -> GOOD
# 2 -> Defect -> DEFECTIVE

GOOD_IDS = {1}
DEFECTIVE_IDS = {0, 2}


# =========================================================
# FLASK APP
# =========================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    status = get_status()

    # API is online even if model loading failed.
    # This allows the frontend to distinguish:
    #
    # API ONLINE
    # MODEL OFFLINE
    #
    # instead of remaining stuck on "API Checking..."

    return jsonify(status), 200


# =========================================================
# COMPATIBILITY STATUS ENDPOINT
# =========================================================

@app.route("/api/status", methods=["GET"])
def api_status():

    return jsonify(
        get_status()
    ), 200


# =========================================================
# IMAGE DECODER
# =========================================================

def decode_uploaded_image(file):

    if file is None:
        return None

    data = file.read()

    if not data:
        return None

    array = np.frombuffer(
        data,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    return image


# =========================================================
# IMAGE ENCODER
# =========================================================

def encode_image(image):

    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            90,
        ],
    )

    if not success:
        raise ValueError(
            "Could not encode result image."
        )

    encoded = base64.b64encode(
        buffer
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# =========================================================
# CONFIDENCE
# =========================================================

def get_confidence():

    try:

        confidence = float(
            request.form.get(
                "confidence",
                DEFAULT_CONF_THRESHOLD,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        confidence = DEFAULT_CONF_THRESHOLD

    # Keep confidence between 0.05 and 0.99
    return max(
        0.05,
        min(0.99, confidence),
    )


# =========================================================
# VERDICT LOGIC
# =========================================================

def calculate_verdict(detections):
    """
    Determine the overall weld condition using CLASS IDs.

    Correct model mapping:

        0 -> Bad Weld -> DEFECT
        1 -> Good Weld -> PASS
        2 -> Defect -> DEFECT

    Priority:

        DEFECT > PASS > NO_WELD

    This means if even one defective detection exists,
    the entire weld is considered defective.
    """

    has_defect = any(
        int(d.get("class_id", -1)) in DEFECTIVE_IDS
        for d in detections
    )

    has_good = any(
        int(d.get("class_id", -1)) in GOOD_IDS
        for d in detections
    )

    if has_defect:

        verdict = "DEFECT"

    elif has_good:

        verdict = "PASS"

    else:

        verdict = "NO_WELD"

    return (
        verdict,
        has_defect,
        has_good,
    )


# =========================================================
# SINGLE IMAGE PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"],
)
def predict():

    # -----------------------------------------------------
    # Check uploaded file
    # -----------------------------------------------------

    if "file" not in request.files:

        return jsonify(
            {
                "success": False,
                "error": "No image file uploaded.",
            }
        ), 400

    file = request.files["file"]

    if not file.filename:

        return jsonify(
            {
                "success": False,
                "error": "No image selected.",
            }
        ), 400

    try:

        # -------------------------------------------------
        # Decode image
        # -------------------------------------------------

        image = decode_uploaded_image(file)

        if image is None:

            return jsonify(
                {
                    "success": False,
                    "error": "Invalid image file.",
                }
            ), 400

        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        confidence = get_confidence()

        # -------------------------------------------------
        # YOLO inference
        # -------------------------------------------------

        result = run_inference(
            image,
            confidence,
        )

        # -------------------------------------------------
        # Model error
        # -------------------------------------------------

        if not result.get("success", False):

            return jsonify(
                {
                    "success": False,
                    "error": result.get(
                        "error",
                        "Model inference failed.",
                    ),
                }
            ), 503

        # -------------------------------------------------
        # Detections
        # -------------------------------------------------

        detections = result.get(
            "detections",
            [],
        )

        # -------------------------------------------------
        # IMPORTANT:
        # Calculate verdict using CLASS IDs.
        #
        # 0 = defective
        # 1 = good
        # 2 = defective
        # -------------------------------------------------

        (
            verdict,
            has_defect,
            has_good,
        ) = calculate_verdict(
            detections
        )

        # -------------------------------------------------
        # Encode annotated image
        # -------------------------------------------------

        encoded_image = encode_image(
            result["annotated_image"]
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return jsonify(
            {
                "success": True,

                "image": encoded_image,

                "detections": detections,

                "count": len(detections),

                "inference_time_ms":
                    result.get(
                        "inference_time_ms",
                        0,
                    ),

                "verdict": verdict,

                "has_defect": has_defect,

                "has_good": has_good,

                # Useful for frontend/debugging
                "class_mapping": {
                    "0": "Bad Weld",
                    "1": "Good Weld",
                    "2": "Defect",
                },
            }
        ), 200

    except Exception as exc:

        return jsonify(
            {
                "success": False,
                "error": str(exc),
            }
        ), 500


# =========================================================
# BATCH PREDICTION
# =========================================================

@app.route(
    "/predict/batch",
    methods=["POST"],
)
def predict_batch():

    files = request.files.getlist(
        "files"
    )

    if not files:

        return jsonify(
            {
                "success": False,
                "error": "No files uploaded.",
            }
        ), 400

    try:

        confidence = get_confidence()

        results = []

        # -------------------------------------------------
        # Process every image
        # -------------------------------------------------

        for file in files:

            if not file.filename:
                continue

            image = decode_uploaded_image(
                file
            )

            if image is None:
                continue

            # ---------------------------------------------
            # YOLO inference
            # ---------------------------------------------

            result = run_inference(
                image,
                confidence,
            )

            if not result.get(
                "success",
                False,
            ):
                continue

            # ---------------------------------------------
            # Detections
            # ---------------------------------------------

            detections = result.get(
                "detections",
                [],
            )

            # ---------------------------------------------
            # Calculate verdict using CLASS IDs
            # ---------------------------------------------

            (
                verdict,
                has_defect,
                has_good,
            ) = calculate_verdict(
                detections
            )

            # ---------------------------------------------
            # Encode result image
            # ---------------------------------------------

            encoded_image = encode_image(
                result["annotated_image"]
            )

            # ---------------------------------------------
            # Store result
            # ---------------------------------------------

            results.append(
                {
                    "filename":
                        file.filename,

                    "image":
                        encoded_image,

                    "detections":
                        detections,

                    "count":
                        len(detections),

                    "inference_time_ms":
                        result.get(
                            "inference_time_ms",
                            0,
                        ),

                    "verdict":
                        verdict,

                    "has_defect":
                        has_defect,

                    "has_good":
                        has_good,
                }
            )

        # -------------------------------------------------
        # Return batch results
        # -------------------------------------------------

        return jsonify(
            {
                "success": True,

                "processed_count":
                    len(results),

                "results":
                    results,
            }
        ), 200

    except Exception as exc:

        return jsonify(
            {
                "success": False,
                "error": str(exc),
            }
        ), 500


# =========================================================
# FILE TOO LARGE
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify(
        {
            "success": False,
            "error":
                "Image is too large. "
                "Maximum size is 10 MB.",
        }
    ), 413


# =========================================================
# SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )