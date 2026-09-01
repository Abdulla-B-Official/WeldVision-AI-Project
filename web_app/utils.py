"""
utils.py — Image encoding / decoding helpers for WeldVision AI.
"""

import base64
import io
import numpy as np
import cv2
from PIL import Image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to an OpenCV array (BGR)."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_base64(cv2_image: np.ndarray, quality: int = 90) -> str:
    """Encode an OpenCV (BGR) image to a JPEG base64 string."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", cv2_image, encode_param)
    if not success:
        raise ValueError("Failed to encode image to JPEG.")
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_cv2(b64_string: str) -> np.ndarray:
    """Decode a base64 string (JPEG/PNG) into an OpenCV (BGR) array."""
    # Strip data-URI prefix if present  (e.g. "data:image/jpeg;base64,...")
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    data = base64.b64decode(b64_string)
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode base64 image data.")
    return img


def resize_if_large(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale the image so that neither dimension exceeds max_side."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
