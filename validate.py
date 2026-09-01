from ultralytics import YOLO
 
model = YOLO("runs/detect/weld_detection_project/weld_yolov8m/weights/best.pt")
 
metrics = model.val()
print(metrics)
