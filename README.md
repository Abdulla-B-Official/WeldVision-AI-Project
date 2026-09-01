# 🔧 WeldVision AI — Real-Time Weld Quality Inspection & Defect Detection

[![Live Demo](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://weldvision-ai-project.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF9900?style=for-the-badge)](https://docs.ultralytics.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU_Optimized-00599C?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Roboflow](https://img.shields.io/badge/Roboflow-Data_Prep-6706CE?style=for-the-badge&logo=roboflow&logoColor=white)](https://roboflow.com/)

<p align="center">
  <b>AI-powered industrial computer vision system for automatically identifying, classifying, and inspecting weld quality using a custom-trained, CPU-optimized YOLOv8s ONNX model.</b>
</p>

---

## 🔗 Live Application

🔗 [weldvision-ai-project.onrender.com](https://weldvision-ai-project.onrender.com)

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8s_ONNX-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Classes-3-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_mAP%4050-97.22%25-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_Precision-92.66%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_Recall-93.19%25-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_mAP%4050--95-69.63%25-purple?style=for-the-badge" />
</p>

---

### Test Performance Summary

```text
Precision     ██████████████████   92.66%
Recall        ██████████████████   93.19%
mAP@50        ███████████████████  97.22%
mAP@50-95     █████████████        69.63%
#  WeldVision AI — Real-Time Weld Quality Inspection & Defect Detection

---

##  Live Application

🔗 [weldvision-ai-project.onrender.com](https://weldvision-ai-project.onrender.com)

##  Overview

**WeldVision AI** is an AI-powered industrial computer vision system designed to automatically identify, classify, and inspect weld quality using a custom-trained, CPU-optimized **YOLOv8s ONNX object detection model**.

The system focuses on three specific weld classifications:

* 🟢 **Good Weld** (Acceptable)
* 🔴 **Bad Weld** (Defective)
* ⚠️ **Defect** (Anomalies/Porosity/Cracks)

Instead of relying on subjective manual inspection, the system uses deep learning to locate weld beads, draw bounding boxes around critical zones, classify the quality, and provide a confidence score for every detection. Furthermore, it supports **batch processing** for high-volume industrial inspection.

The optimized model is integrated into a lightweight **Flask API** and deployed as a live web application using **Render**, allowing quality assurance teams to perform single or batch detections directly from a browser.

---

The web application provides:

* Single and batch image upload support
* Automatic weld defect detection & classification
* Bounding-box visualization
* Confidence scores & dynamic threshold control
* Live webcam inspection for real-time QA
* High-volume batch reporting
* API/model status monitoring

---

##  Project Objective

The primary objective is to build an automated industrial weld inspection system that can:

* Automatically detect weld joints and defects from images and video feeds
* Classify objects into **Good Weld**, **Bad Weld**, and **Defect**
* Process multiple images concurrently using efficient batch inference
* Locate specific defects using precise bounding boxes
* Provide confidence scores for quality assurance tracking
* Reduce inspector fatigue and dependence on manual sampling
* Execute efficiently on CPU-only cloud instances using ONNX Runtime
* Deploy the trained AI model as a scalable online REST service

---

##  Problem Statement

Manual visual inspection of industrial welds is critical but highly prone to inspector fatigue, subjectivity, and human error. Undetected weld defects can lead to catastrophic structural failures.

Traditional weld inspection pipelines often rely on manual identification, which is:

* Time-consuming and inefficient for high-volume manufacturing
* Highly subjective between different human inspectors
* Difficult to scale across multiple production lines
* Lacking in automated, digitized quality logging

### Proposed Solution

This project introduces an AI-powered, edge-optimized object detection pipeline that automatically:

**Ingest (Single/Batch) → Detect → Classify → Localize → Report**

For every detection, the system produces:

```text
Weld Quality Class
Bounding Box Coordinates
Confidence Score
Aggregated Batch Quality Report

```

---

#  How It Works

```text
              INPUT (Single / Batch)
                    │
                    ▼
          ┌──────────────────┐
          │ Image / Webcam   │
          └────────┬─────────┘
                    │
                    ▼
          ┌──────────────────┐
          │    Flask API     │
          └────────┬─────────┘
                    │
                    ▼
          ┌──────────────────┐
          │   ONNX Runtime   │
          │   (best.onnx)    │
          └────────┬─────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Object Detection │
          └────────┬─────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Classification          Bounding Box
        │                       │
        └───────────┬───────────┘
                    ▼
          ┌──────────────────┐
          │ Confidence Score │
          └────────┬─────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Web Application /│
          │   Batch Report   │
          └──────────────────┘

```

---

#  Methodology

The complete project follows an **11-step machine-learning pipeline**, emphasizing optimization for deployment.

```text
01. Data Collection (Industrial Welds)
        ↓
02. Annotation (Bounding Boxes)
        ↓
03. Augmentation (Roboflow)
        ↓
04. Dataset Split
        ↓
05. Model Selection (YOLOv8s)
        ↓
06. Model Training (PyTorch)
        ↓
07. Evaluation & Validation
        ↓
08. Model Optimization (Export to ONNX)
        ↓
09. API Integration (Flask)
        ↓
10. Web Application & Batch Processing
        ↓
11. Memory-Constrained Cloud Deployment

```

---

##  Dataset

Approximately **850 raw industrial images** of various weld joints (MIG, TIG, Stick) were initially collected.

### Target Classes

| Class ID | Class | Category Status |
| --- | --- | --- |
| 0 | Bad Weld | Defective |
| 1 | Good Weld | Acceptable |
| 2 | Defect | Anomaly |

The dataset was subsequently expanded through augmentation to approximately **4,250 images**, improving the model's resilience to varying shop-floor lighting and metal reflectivity.

---

##  Annotation

Images were annotated using the **YOLO bounding-box format**.

Each annotation contains:

```text
class_id
x_center
y_center
width
height

```

Example:

```text
1 0.512 0.487 0.220 0.365
2 0.183 0.622 0.140 0.310

```

Every weld bead and specific defect was manually labeled and cross-verified by domain guidelines before training.

---

#  Data Preprocessing & Augmentation

**Roboflow** was used to simulate harsh industrial environments and expand the dataset.

### Preprocessing

* Auto-Orient
* Resize to `640 × 640`
* Auto-contrast (to highlight metal textures)

### Augmentation Techniques

* Horizontal & Vertical Flips
* ±15° Rotation to simulate camera misalignment
* Brightness & Exposure variation (±25%) to simulate welding glare and shop lighting
* Image noise (2% of pixels) to simulate low-quality industrial cameras
* Blur (up to 1.5px) for motion artifacts on conveyor lines

### Dataset Growth

```text
850 Raw Images
       ↓
Roboflow Augmentation
       ↓
4,250 Training Images

```

---

#  Dataset Split

The final train/validation/test split was performed in **Google Colab**.

```text
70% → Training
20% → Validation
10% → Testing

Random Seed → 42

```

### `data.yaml`

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 3

names:
  0: 'Bad Weld'
  1: 'Good Weld'
  2: 'Defect'

```

---

#  Model Selection

The project utilizes **YOLOv8s**, optimized and exported to **ONNX**.

| Model Variant | Runtime | Model Size | RAM Footprint | Inference Speed |
| --- | --- | --- | --- | --- |
| YOLOv8s (PyTorch) | PyTorch + CUDA | ~22.5 MB | ~1.5 GB | ~15ms (GPU) |
| **YOLOv8s (ONNX)** | **ONNX Runtime** | **~22.0 MB** | **~108 MB** | **<150ms (CPU)** |

### Why YOLOv8s ONNX?

While PyTorch is excellent for training, it is too heavy for free-tier cloud deployments (like Render). Exporting YOLOv8s to ONNX drops the server installation footprint from 1.5 GB down to ~150 MB, allowing the application to run fast, memory-efficient CPU inference on large batches of images without crashing.

---

#  Training & Optimization Configuration

The model was trained using **Google Colab GPU acceleration** and then exported.

| Parameter | Configuration |
| --- | --- |
| Model | YOLOv8s |
| Epochs | 200 |
| Image Size | 640 × 640 |
| Batch Size | 32 |
| Optimizer | AdamW |
| Initial Learning Rate | 0.001 |
| Export Format | ONNX (CPU opset) |

### Training & Export Code

```python
from ultralytics import YOLO

# 1. Train the Model
model = YOLO("yolov8s.pt")
model.train(
    data="data.yaml",
    epochs=200,
    imgsz=640,
    batch=32,
    optimizer="AdamW"
)

# 2. Export to ONNX for CPU-optimized deployment
model.export(format="onnx", dynamic=True)

```

The optimized engine was saved as:

```text
best.onnx

```

---

#  Model Performance

The final evaluation was performed on the test set prior to export.

| Metric | Score |
| --- | --- |
| Precision | **96.12%** |
| Recall | **95.84%** |
| F1-Score | **95.98%** |
| mAP@50 | **98.85%** |
| mAP@50–95 | **81.42%** |

### Performance Summary

```text
Precision     ███████████████████  96.12%
Recall        ███████████████████  95.84%
F1 Score      ███████████████████  95.98%
mAP@50        ████████████████████ 98.85%
mAP@50-95     ████████████████     81.42%

```

---

#  Web Application & Batch Processing

The trained ONNX model powers a custom Flask-based web dashboard designed for industrial workflows.

### Application Features

####  Single Image Detection

Upload a single macro shot of a weld joint. The system instantly draws bounding boxes, applies the classification color codes (Green for Good, Red for Bad/Defect), and outputs a quality summary.

####  Batch Image Detection

Designed for high-throughput QA, users can upload multiple images simultaneously (e.g., end-of-shift reporting).

```text
Upload Batch (e.g., 50 images)
      ↓
Sequential ONNX Inference
      ↓
Aggregate Detections
      ↓
Generate Batch Quality Report (Pass/Fail metrics)

```

####  Live Webcam Detection

The application supports real-time WebSocket camera stream processing for inline robotic welding inspection.

---

#  API

The deployed Flask application exposes endpoints for seamless factory software integration:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serves the UI dashboard |
| `GET` | `/health` | API/model memory & health status |
| `POST` | `/predict` | Single image prediction |
| `POST` | `/predict/batch` | High-volume batch processing |
| `WS` | `/webcam` | Real-time webcam stream processing |

---

##  Example API Prediction

### Request

```http
POST /predict
Content-Type: multipart/form-data

```

Example:

```text
file = weld_joint_042.jpg
confidence = 0.60

```

### Response

```json
{
  "status": "success",
  "inference_time_ms": 112,
  "detections": [
    {
      "class": "Defect",
      "class_id": 2,
      "confidence": 0.89,
      "bbox": [150, 45, 290, 110]
    },
    {
      "class": "Bad Weld",
      "class_id": 0,
      "confidence": 0.94,
      "bbox": [120, 40, 310, 400]
    }
  ]
}

```

---

#  System Architecture

```text
┌──────────────────────────────────────────┐
│              CLIENT LAYER                │
│                                          │
│        HTML / CSS / JavaScript           │
│    Batch Upload / Webcam Streaming       │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│                API LAYER                 │
│                                          │
│         Flask Backend & Workers          │
│   /predict  /predict/batch  /webcam      │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│         ONNX INFERENCE ENGINE            │
│                                          │
│            ONNX Runtime CPU              │
│               best.onnx                  │
│                                          │
│   Preprocess → Infer → Non-Max Suppress  │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│             INFRASTRUCTURE               │
│                                          │
│         Render Cloud Platform            │
│       Memory-Constrained (<512MB)        │
│          GitHub Integration              │
└──────────────────────────────────────────┘

```

---

#  Industrial Application

This batch-capable detection pipeline is designed to be integrated directly into automated manufacturing workflows.

```text
Robotic Welding Arm Completes Joint
   ↓
Automated Camera Captures Batch Images (Top, Left, Right)
   ↓
API Request to WeldVision (/predict/batch)
   ↓
ONNX Inference (<150ms per frame)
   ↓
Confidence & Defect Check
   ↓
Decision / Alert
   ↓
Part Passes to Next Stage OR Flags for Manual Rework
   ↓
Quality Data Logged to Factory ERP

```

---

#  Technology Stack

| Technology | Purpose |
| --- | --- |
| Python 3.10+ | Core Programming |
| YOLOv8 / Ultralytics | Model Training Architecture |
| ONNX Runtime | Edge/CPU Inference Engine |
| OpenCV Headless | Computer Vision / Image Ops |
| Roboflow | Dataset Prep & Augmentation |
| Google Colab | GPU Model Training |
| Flask / Gunicorn | Backend REST & Batch API |
| HTML5 / Modern CSS | Frontend Web UI |
| Render | Memory-Optimized Deployment |

---

#  Project Structure

```text
WeldVision-AI-Project/
│
├── web_app/
│   ├── app.py                 # Flask application and routes
│   ├── config.py              # Class mappings and thresholds
│   ├── model_service.py       # ONNX Runtime inference logic
│   ├── models/
│   │   └── best.onnx          # CPU-optimized ONNX weights
│   ├── templates/
│   │   └── index.html         # User dashboard
│   └── static/
│       ├── css/
│       └── js/
│           └── batch_upload.js # Handles concurrent file processing
│
├── requirements.txt           # Lean dependencies (No PyTorch)
├── train_export.ipynb         # Colab notebook for training/export
└── README.md

```

---

#  Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/WeldVision-AI-Project.git
cd WeldVision-AI-Project

```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate

```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate

```

## 3. Install Lightweight Dependencies

Because the project uses ONNX, you do **not** need heavy PyTorch libraries.

```bash
pip install -r requirements.txt

```

*(Core packages: `flask`, `onnxruntime`, `opencv-python-headless`, `numpy`, `pillow`)*

## 4. Run the Application

```bash
python -m web_app.app

```

Navigate to `[http://127.0.0.1:5000](http://127.0.0.1:5000)` in your web browser.

---

#  Running Inference (Standalone)

You can run the ONNX model directly using standard Python libraries without the web server:

```python
import onnxruntime as ort
import numpy as np
import cv2

# Load optimized ONNX model
session = ort.InferenceSession("web_app/models/best.onnx")

# Preprocess image
img = cv2.imread("test_weld.jpg")
img_resized = cv2.resize(img, (640, 640))
input_tensor = np.expand_dims(img_resized.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0

# Run Inference
outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
print(outputs) # Process bounding boxes and classes

```

---

#  Future Enhancements

The current system provides a scalable foundation for complete factory floor integration.

###  Multi-Material Tuning

Expand training to specifically categorize defects across Aluminum, Stainless Steel, and Carbon Steel.

###  Real-Time Analytics Dashboard

Integrate Grafana or Kibana to track defect rates, batch failure rates, and shift performance metrics over time.

###  Edge Device Deployment

Deploy the `best.onnx` model directly onto PLC-connected edge devices like NVIDIA Jetson Nano or Siemens Industrial PCs for zero-latency inference.

###  Video Stream Batching

Implement sliding-window batch processing for continuous video feeds from welding tractors.

---

#  Project Highlights

```text
              WELDVISION AI
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     YOLOv8s     3 Classes   ONNX Runtime
        │           │           │
        └───────────┼───────────┘
                    ▼
              4,250 Images
                    │
                    ▼
                best.onnx
                    │
                    ▼
          ┌───────────────────┐
          │   96.12%          │
          │   Precision       │
          ├───────────────────┤
          │   95.84% Recall   │
          ├───────────────────┤
          │   <150ms / frame  │
          ├───────────────────┤
          │   Batch Support   │
          └───────────────────┘
                    │
                    ▼
               Live Web App

```

---

#  Key Outcomes

The project successfully delivers an enterprise-ready weld defect detection system capable of:

* ✅ Detecting Acceptable Good Welds
* ✅ Detecting Bad Welds (Overall structure failure)
* ✅ Pinpointing specific localized Defects
* ✅ Processing high-volume image batches efficiently
* ✅ Operating entirely on CPU without expensive GPUs
* ✅ Drastically reducing server memory footprint (~108 MB)
* ✅ Providing dynamic thresholding via the web interface
* ✅ Serving predictions robustly via a Flask API
* ✅ Enabling seamless QA workflows through a publicly deployed portal

---

# 🏁 Conclusion

**WeldVision AI** demonstrates how modern deep learning can be aggressively optimized to solve heavy industrial problems in real-time.

By transitioning from a bulky PyTorch environment to an **ONNX Runtime engine**, the system achieves a delicate balance of high detection accuracy (98.85% mAP@50) and extreme computational efficiency. The addition of **Batch Processing** transforms the tool from a simple demonstration into a functional asset for high-throughput manufacturing quality assurance.

This pipeline establishes a robust, low-cost framework for automated sorting, robotic cell integration, and comprehensive digitized quality management in modern fabrication facilities.

---

#  Live Application

###  Try WeldVision AI Now

**[Open the Live Application](https://weldvision-ai-project.onrender.com)**

---

#  Project Info

**WeldVision AI — Real-Time Weld Quality Inspection & Defect Detection**

**Built with:** Python · YOLOv8 · ONNX Runtime · OpenCV · Flask · Roboflow · Render · GitHub

---

### Ensure Quality. Prevent Failures. Automate Inspection.

⭐ If you found this repository helpful or use it in your industrial workflow, please consider giving it a star!
