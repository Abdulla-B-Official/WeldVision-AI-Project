from ultralytics import YOLO

# Load your PyTorch trained model
model = YOLO("web_app/models/best.pt")

# Export to lightweight ONNX format
model.export(format="onnx", simplify=True)