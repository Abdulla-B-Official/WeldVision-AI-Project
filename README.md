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

🔗 **[weldvision-ai-project.onrender.com](https://weldvision-ai-project.onrender.com)**

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8s_ONNX-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Classes-3-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_mAP%4050-97.22%25-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_Precision-92.66%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_Recall-93.19%25-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_mAP%4050--95-69.63%25-purple?style=for-the-badge" />
</p>

---

## Overview

**WeldVision AI** is an AI-powered industrial computer vision system designed to automatically identify, classify, and inspect weld quality using a custom-trained, CPU-optimized **YOLOv8s ONNX object detection model**.

The system focuses on three specific weld classifications:

* 🟢 **Good Weld** (Acceptable)
* 🔴 **Bad Weld** (Defective)
* ⚠️ **Defect** (Anomalies / Porosity / Cracks)

Instead of relying on subjective manual inspection, the system uses deep learning to locate weld beads, draw bounding boxes around critical zones, classify the quality, and provide a confidence score for every detection. Furthermore, it supports **batch processing** for high-volume industrial inspection.

The optimized model is integrated into a lightweight **Flask API** and deployed as a live web application using **Render**, allowing quality assurance teams to perform single or batch detections directly from a browser.

---

### Application Features

* Single and batch image upload support
* Automatic weld defect detection & classification
* Bounding-box visualization
* Confidence scores & dynamic threshold control
* Live webcam inspection for real-time QA
* High-volume batch reporting
* API/model status monitoring

---

## Project Objective

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

## Problem Statement

Manual visual inspection of industrial welds is critical but highly prone to inspector fatigue, subjectivity, and human error. Undetected weld defects can lead to catastrophic structural failures.

Traditional weld inspection pipelines often rely on manual identification, which is:

* Time-consuming and inefficient for high-volume manufacturing
* Highly subjective between different human inspectors
* Difficult to scale across multiple production lines
* Lacking in automated, digitized quality logging

### Proposed Solution

This project introduces an AI-powered, edge-optimized object detection pipeline that automatically:

$$\text{Ingest (Single/Batch)} \longrightarrow \text{Detect} \longrightarrow \text{Classify} \longrightarrow \text{Localize} \longrightarrow \text{Report}$$

For every detection, the system produces:

```text
Weld Quality Class
Bounding Box Coordinates
Confidence Score
Aggregated Batch Quality Report

```

---

## How It Works

```text
              INPUT (Single / Batch)
                         │
                         ▼
              ┌──────────────────┐
              │  Image / Webcam  │
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
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 Classification                   Bounding Box
        │                                 │
        └────────────────┬────────────────┘
                         │
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

## Methodology

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

## Dataset

Approximately **850 raw industrial images** of various weld joints (MIG, TIG, Stick) were initially collected.

### Target Classes

| Class ID | Class | Category Status |
| --- | --- | --- |
| **0** | Bad Weld | Defective |
| **1** | Good Weld | Acceptable |
| **2** | Defect | Anomaly |

The dataset was subsequently expanded through augmentation to approximately **4,250 images**, improving the model's resilience to varying shop-floor lighting and metal reflectivity.

---

## Annotation

Images were annotated using the **YOLO bounding-box format**. Each annotation file contains:

```text
<class_id> <x_center> <y_center> <width> <height>

```

Example:

```text
1 0.512 0.487 0.220 0.365
2 0.183 0.622 0.140 0.310

```

Every weld bead and specific defect was manually labeled and cross-verified by domain guidelines before training.

---

## Data Preprocessing & Augmentation

**Roboflow** was used to simulate harsh industrial environments and expand the dataset.

### Preprocessing

* Auto-Orient
* Resize to $640 \times 640$
* Auto-contrast (to highlight metal textures)

### Augmentation Techniques

* Horizontal & Vertical Flips
* $\pm15^\circ$ Rotation to simulate camera misalignment
* Brightness & Exposure variation ($\pm25\%$) to simulate welding glare and shop lighting
* Image noise ($2\%$ of pixels) to simulate low-quality industrial cameras
* Blur (up to 1.5px) for motion artifacts on conveyor lines

### Dataset Growth

```text
850 Raw Images  ──►  Roboflow Augmentation  ──►  4,250 Training Images

```

---

## Dataset Split

The final train/validation/test split was performed in **Google Colab** with `Random Seed = 42`:

$$\text{Training: 70\%} \quad \mid \quad \text{Validation: 20\%} \quad \mid \quad \text{Testing: 10\%}$$

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

## Model Selection

The project utilizes **YOLOv8s**, optimized and exported to **ONNX**.

| Model Variant | Runtime | Model Size | RAM Footprint | Inference Speed |
| --- | --- | --- | --- | --- |
| **YOLOv8s (PyTorch)** | PyTorch + CUDA | ~22.5 MB | ~1.5 GB | ~15ms (GPU) |
| **YOLOv8s (ONNX)** | **ONNX Runtime** | **~22.0 MB** | **~108 MB** | **<150ms (CPU)** |

### Why YOLOv8s ONNX?

While PyTorch is excellent for training, it is too heavy for free-tier cloud deployments (like Render). Exporting YOLOv8s to ONNX drops the server installation footprint from 1.5 GB down to ~150 MB, allowing the application to run fast, memory-efficient CPU inference on large batches of images without crashing.

---

## Training & Optimization Configuration

The model was trained using **Google Colab GPU acceleration** and then exported.

| Parameter | Configuration |
| --- | --- |
| **Model** | YOLOv8s |
| **Epochs** | 200 |
| **Image Size** | $640 \times 640$ |
| **Batch Size** | 32 |
| **Optimizer** | AdamW |
| **Initial Learning Rate** | 0.001 |
| **Export Format** | ONNX (CPU opset) |

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

---

## Model Performance & Evaluation Metrics

### Validation Metrics

| Metric | Score |
| --- | --- |
| **Precision** | **92.02%** |
| **Recall** | **92.81%** |
| **mAP@50** | **96.77%** |
| **mAP@50–95** | **68.77%** |

### Test Metrics (Final Evaluation)

| Metric | Score |
| --- | --- |
| **Precision** | **92.66%** |
| **Recall** | **93.19%** |
| **mAP@50** | **97.22%** |
| **mAP@50–95** | **69.63%** |

### Test Performance Summary

```text
Precision     ██████████████████   92.66%
Recall        ██████████████████   93.19%
mAP@50        ███████████████████  97.22%
mAP@50-95     █████████████        69.63%

```

---

## Web Application & Batch Processing

The trained ONNX model powers a custom Flask-based web dashboard designed for industrial workflows.

### 1. Single Image Detection

Upload a single macro shot of a weld joint. The system instantly draws bounding boxes, applies classification color codes (Green for Good, Red for Bad/Defect), and outputs a quality summary.

### 2. Batch Image Detection

Designed for high-throughput QA, users can upload multiple images simultaneously (e.g., end-of-shift reporting).

```text
Upload Batch (e.g., 50 images)  ──►  Sequential ONNX Inference  ──►  Aggregate Detections  ──►  Generate Quality Report

```

### 3. Live Webcam Detection

The application supports real-time camera stream processing for inline robotic welding inspection.

---

## 🔌 API Endpoints

The deployed Flask application exposes REST endpoints for factory software integration:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serves the UI dashboard |
| `GET` | `/health` | API/model memory & health status |
| `POST` | `/predict` | Single image prediction |
| `POST` | `/predict/batch` | High-volume batch processing |
| `WS` | `/webcam` | Real-time webcam stream processing |

### Example API Request & Response

```http
POST /predict
Content-Type: multipart/form-data

```

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

## System Architecture

```text
┌──────────────────────────────────────────┐
│               CLIENT LAYER               │
│                                          │
│         HTML / CSS / JavaScript          │
│    Batch Upload / Webcam Streaming       │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│                API LAYER                 │
│                                          │
│          Flask Backend & Workers         │
│    /predict   /predict/batch   /webcam   │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│          ONNX INFERENCE ENGINE           │
│                                          │
│             ONNX Runtime CPU             │
│                best.onnx                 │
│                                          │
│   Preprocess → Infer → Non-Max Suppress  │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│              INFRASTRUCTURE              │
│                                          │
│          Render Cloud Platform           │
│        Memory-Constrained (<512MB)       │
│           GitHub Integration             │
└──────────────────────────────────────────┘

```

---

## Industrial Application Workflow

This batch-capable detection pipeline is designed to be integrated directly into automated manufacturing workflows:

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

## Technology Stack

| Category | Technologies |
| --- | --- |
| **Language** | Python 3.10+ |
| **AI / Computer Vision** | YOLOv8 (Ultralytics), ONNX Runtime, OpenCV (Headless) |
| **Data Preparation** | Roboflow, Google Colab (GPU) |
| **Backend & Web** | Flask, Gunicorn, REST API, WebSockets |
| **Deployment** | Render Platform |

---

##  Project Structure

```text
WeldVision-AI-Project/
│
├── web_app/
│   ├── app.py                 # Flask application and REST routes
│   ├── config.py              # Class mappings and confidence thresholds
│   ├── model_service.py       # ONNX Runtime inference engine logic
│   ├── models/
│   │   └── best.onnx          # CPU-optimized ONNX weights
│   ├── templates/
│   │   └── index.html         # Web UI dashboard
│   └── static/
│       ├── css/
│       └── js/
│           └── batch_upload.js # Concurrent file processing script
│
├── requirements.txt           # Lean deployment dependencies (No PyTorch)
├── train_export.ipynb         # Colab notebook for training and export
└── README.md                  # Project documentation

```

---

## Installation & Local Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/YourUsername/WeldVision-AI-Project.git](https://github.com/YourUsername/WeldVision-AI-Project.git)
cd WeldVision-AI-Project

```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Lightweight Dependencies

Because the project uses ONNX for deployment, standard PyTorch libraries are not required:

```bash
pip install -r requirements.txt

```

### 4. Run the Application

```bash
python -m web_app.app

```

Open `http://127.0.0.1:5000` in your web browser.

---

## Standalone ONNX Inference Script

To execute model inference in Python without running the web server:

```python
import onnxruntime as ort
import numpy as np
import cv2

# 1. Load optimized ONNX session
session = ort.InferenceSession("web_app/models/best.onnx")

# 2. Preprocess image (RGB, 640x640, normalized)
img = cv2.imread("test_weld.jpg")
img_resized = cv2.resize(img, (640, 640))
input_tensor = np.expand_dims(img_resized.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0

# 3. Run Inference
outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
print("Inference Output Shape:", outputs[0].shape)

```

---

## Future Enhancements

* **Multi-Material Tuning:** Expand dataset to categorize defects specifically across Aluminum, Stainless Steel, and Carbon Steel.
* **Real-Time Analytics Dashboard:** Integrate Grafana/Kibana to track shift failure rates and defect trends over time.
* **Edge Device Deployment:** Deploy `best.onnx` directly onto edge hardware (NVIDIA Jetson Nano, Siemens Industrial PCs) via TensorRT/OpenVINO.

---

## Key Outcomes

* ✅ Automated identification of **Good Welds**, **Bad Welds**, and localized **Defects**.
* ✅ High test accuracy (**97.22% mAP@50**, **92.66% Precision**, **93.19% Recall**).
* ✅ **~108 MB RAM** runtime footprint enabling reliable deployment on CPU cloud instances.
* ✅ End-to-end REST API and batch reporting for high-volume quality inspection.

---

## Live Deployment Link

🌐 **Try the Live Application:** [weldvision-ai-project.onrender.com](https://weldvision-ai-project.onrender.com)

⭐ *If you find this repository helpful for your computer vision research or industrial workflow, please consider starring the repository!*

```

```
