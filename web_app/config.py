from pathlib import Path

WEB_APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_APP_DIR.parent

MODEL_PATH = WEB_APP_DIR / "models" / "best.onnx"

DATA_PATH = PROJECT_ROOT / "data.yaml"

DEFAULT_CONF_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.45
INPUT_SIZE = 640
MAX_IMAGE_SIZE = 640

GOOD_IDS = {1}
DEFECTIVE_IDS = {0, 2}

CLASS_DISPLAY_NAMES = {
    0: "Bad Weld",
    1: "Good Weld",
    2: "Defect",
}

COLOR_GOOD = (0, 220, 100)
COLOR_DEFECTIVE = (0, 60, 230)

DEVICE = "cpu"