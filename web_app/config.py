"""
config.py — Central configuration for WeldVision AI.

Path resolution: this file lives at web_app/config.py
  parents[0] = web_app/
  parents[1] = project root  ← Weld-Detection-Project/
"""

from pathlib import Path
import torch

# ── Project paths ──────────────────────────────────────────────────────────────
WEB_APP_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_APP_DIR.parent

# Primary path: web_app/models/best.onnx (Lightweight ONNX model for deployment)
# Fallback paths for PyTorch (.pt) models if ONNX is missing locally
WEB_ONNX_MODEL_PATH = WEB_APP_DIR / "models" / "best.onnx"
WEB_PT_MODEL_PATH   = WEB_APP_DIR / "models" / "best.pt"
LOCAL_MODEL_PATH    = PROJECT_ROOT / "runs" / "detect" / "weld_yolov8s" / "weights" / "best.pt"

# Automatically choose ONNX first, then fall back to PyTorch weights
if WEB_ONNX_MODEL_PATH.exists():
    MODEL_PATH = WEB_ONNX_MODEL_PATH
elif WEB_PT_MODEL_PATH.exists():
    MODEL_PATH = WEB_PT_MODEL_PATH
else:
    MODEL_PATH = LOCAL_MODEL_PATH

DATA_PATH = PROJECT_ROOT / "data.yaml"

# ── Inference settings ─────────────────────────────────────────────────────────
DEFAULT_CONF_THRESHOLD = 0.50
MAX_IMAGE_SIZE = 640          # Cap to 640px max edge to optimize RAM usage on Render

# ── Class semantics ──────────────────────────────────────────────────────────
# Aligned with web_cam.py dataset indices:
#   0 -> Bad Weld
#   1 -> Good Weld
#   2 -> Defect
GOOD_IDS      = {1}       # class 1 → Good Weld
DEFECTIVE_IDS = {0, 2}    # class 0 → Bad Weld, class 2 → Defect

# Display names shown in bounding-box labels and detection cards.
CLASS_DISPLAY_NAMES = {
    0: 'Bad Weld',    # class 0
    1: 'Good Weld',   # class 1
    2: 'Defect',      # class 2
}

# BGR colours for OpenCV bounding-box drawing
COLOR_GOOD      = (0, 220, 100)   # vibrant green
COLOR_DEFECTIVE = (0, 60, 230)    # vibrant red (BGR)

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"