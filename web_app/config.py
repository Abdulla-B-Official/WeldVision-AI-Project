"""
config.py — Central configuration for WeldVision AI.

Path resolution: this file lives at web_app/config.py
  parents[0] = web_app/
  parents[1] = project root  ← Weld-Detection-Project/
"""

from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────────
WEB_APP_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_APP_DIR.parent

# Primary path: web_app/models/best.onnx (Lightweight ONNX model for deployment)
# Fallback paths for model configuration
WEB_ONNX_MODEL_PATH = WEB_APP_DIR / "models" / "best.onnx"
WEB_PT_MODEL_PATH   = WEB_APP_DIR / "models" / "best.pt"
LOCAL_MODEL_PATH    = PROJECT_ROOT / "runs" / "detect" / "weld_yolov8s" / "weights" / "best.pt"

# Automatically choose ONNX first
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
# Swapped index 1 and 2 to correct model dataset mapping:
#   0 -> Bad Weld
#   1 -> Defect
#   2 -> Good Weld
GOOD_IDS      = {2}       # class 2 → Good Weld
DEFECTIVE_IDS = {0, 1}    # class 0 → Bad Weld, class 1 → Defect

# Display names shown in bounding-box labels and detection cards.
CLASS_DISPLAY_NAMES = {
    0: 'Bad Weld',    # class 0
    1: 'Defect',      # class 1
    2: 'Good Weld',   # class 2
}

# BGR colours for OpenCV bounding-box drawing
COLOR_GOOD      = (0, 220, 100)   # vibrant green
COLOR_DEFECTIVE = (0, 60, 230)    # vibrant red (BGR)

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cpu"  # Pure ONNX CPU execution target