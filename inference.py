import json
import gc
import cv2
import torch
from ultralytics import YOLO

# ── Memory Optimization Settings ──
torch.set_grad_enabled(False)

# Dataset class mapping:
#   0 -> Bad Weld   (defective)
#   1 -> Good Weld  (good)
#   2 -> Defect     (defective)
GOOD_IDS = {1}
DEFECTIVE_IDS = {0, 2}

# Load model
model = YOLO("runs/detect/weld_yolov8m/weights/best.pt")

image = cv2.imread("sample_weld.jpg")
if image is None:
    raise FileNotFoundError("sample_weld.jpg not found in project root.")

# Run inference with memory optimizations
with torch.inference_mode():
    results = model.predict(source=image, imgsz=640, verbose=False)

all_detections = []

# ── 1. Process and Draw All Bounding Boxes ──
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Green for good, Red for defective
        color = (0, 255, 0) if class_id in GOOD_IDS else (0, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        all_detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
            }
        )

# ── 2. Decide Overall Verdict (Outside the Box Loop) ──
defective_dets = [d for d in all_detections if d["class_id"] in DEFECTIVE_IDS]
good_dets = [d for d in all_detections if d["class_id"] in GOOD_IDS]

if defective_dets:
    best = max(defective_dets, key=lambda d: d["confidence"])
    verdict, verdict_conf = "defective", best["confidence"]
elif good_dets:
    best = max(good_dets, key=lambda d: d["confidence"])
    verdict, verdict_conf = "good", best["confidence"]
else:
    verdict, verdict_conf = "no_detection", 0.0

# ── 3. Save Prediction JSON ──
output = {"class": verdict, "confidence": round(verdict_conf, 4)}
with open("prediction.json", "w") as f:
    json.dump(output, f, indent=4)

print(f"Verdict: {verdict} | Confidence: {round(verdict_conf, 4)}")

# ── 4. Render Visualization ──
cv2.imshow("Weld Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Clean up memory
gc.collect()