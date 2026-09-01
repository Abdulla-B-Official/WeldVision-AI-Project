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

MODEL_PATH = PROJECT_ROOT / "runs" / "weld_yolov8m" / "weights" / "best.pt"
DATA_PATH  = PROJECT_ROOT / "data.yaml"

# ── Inference settings ─────────────────────────────────────────────────────────
DEFAULT_CONF_THRESHOLD = 0.50
MAX_IMAGE_SIZE = 1280          # pixels — longer edge resized if larger

# ── Class semantics ──────────────────────────────────────────────────────────
#
# data.yaml / model.names:
#   0 → "Bad Weld"   label
#   1 → "Good Weld"  label
#   2 → "Defect"     label
#
# EMPIRICAL FINDING: the training dataset was annotated with swapped labels.
# The model predicts class 0 for physically-good welds and class 1 for
# physically-bad welds.  We correct this here WITHOUT retraining the model.
#
#   class 0 (model label "Bad Weld")  → is actually a GOOD weld
#   class 1 (model label "Good Weld") → is actually a BAD  weld
#   class 2 (model label "Defect")    → is defective (unchanged)
#
GOOD_IDS      = {0}       # class 0 → Good Weld  (corrected)
DEFECTIVE_IDS = {1, 2}   # class 1 → Bad Weld, class 2 → Defect (corrected)

# Corrected display names shown in bounding-box labels and detection cards.
# These OVERRIDE the names embedded in the model so the UI always shows
# the semantically-correct text.
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
