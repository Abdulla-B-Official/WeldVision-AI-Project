"""
config.py — Central configuration for WeldVision AI
"""

from pathlib import Path

# =========================================================
# PROJECT PATHS
# =========================================================

WEB_APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_APP_DIR.parent

MODEL_PATH = WEB_APP_DIR / "models" / "best.onnx"

DATA_PATH = PROJECT_ROOT / "data.yaml"


# =========================================================
# INFERENCE SETTINGS
# =========================================================

INPUT_SIZE = 640
MAX_IMAGE_SIZE = 640

DEFAULT_CONF_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.45


# =========================================================
# CORRECT MODEL CLASS MAPPING
#
# 0 = Bad Weld  -> DEFECT
# 1 = Good Weld -> GOOD
# 2 = Defect    -> DEFECT
# =========================================================

GOOD_IDS = {1}

DEFECTIVE_IDS = {0, 2}

CLASS_DISPLAY_NAMES = {
    0: "Bad Weld",
    1: "Good Weld",
    2: "Defect",
}


# =========================================================
# OPENCV COLORS - BGR FORMAT
# =========================================================

COLOR_GOOD = (0, 220, 100)

COLOR_DEFECTIVE = (0, 60, 230)


# =========================================================
# DEVICE
# =========================================================

DEVICE = "cpu"