import time
from ultralytics import YOLO
import cv2
 
model = YOLO("best.pt")          # or the full runs/... path
 
image = cv2.imread("sample.jpg")
 
start = time.time()
results = model(image)
end = time.time()
 
latency = end - start
print("Inference Time:", latency, "seconds")
