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

# Primary path: web_app/models/best.pt (for Render and GitHub tracking)
# Fallback path: runs/detect/weld_yolov8s/weights/best.pt (for local computer training runs)
LOCAL_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "weld_yolov8s" / "weights" / "best.pt"
WEB_MODEL_PATH = WEB_APP_DIR / "models" / "best.pt"

# Automatically choose whichever model path exists
if WEB_MODEL_PATH.exists():
    MODEL_PATH = WEB_MODEL_PATH
else:
    MODEL_PATH = LOCAL_MODEL_PATH

DATA_PATH = PROJECT_ROOT / "data.yaml"

# ── Inference settings ─────────────────────────────────────────────────────────
DEFAULT_CONF_THRESHOLD = 0.50
MAX_IMAGE_SIZE = 640          # Cap to 640px max edge to optimize RAM usage on Render

# ── Class semantics ──────────────────────────────────────────────────────────
GOOD_IDS      = {0}       # class 0 → Good Weld  (corrected)
DEFECTIVE_IDS = {1, 2}   # class 1 → Bad Weld, class 2 → Defect (corrected)

# Corrected display names shown in bounding-box labels and detection cards.
CLASS_DISPLAY_NAMES = {
    0: 'Good Weld',   # corrected: model says "Bad Weld"  but weld is good
    1: 'Bad Weld',    # corrected: model says "Good Weld" but weld is bad
    2: 'Defect',      # unchanged
}

# BGR colours for OpenCV bounding-box drawing
COLOR_GOOD      = (0, 220, 100)   # vibrant green
COLOR_DEFECTIVE = (0, 60, 230)    # vibrant red (BGR)

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"