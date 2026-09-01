# ⚡ WeldVision AI — Real-Time Weld Quality Inspection & Defect Detection

---

## 🌐 Live Application

🔗 [weldvision-ai-project.onrender.com](https://weldvision-ai-project.onrender.com)

## 📌 Overview

**WeldVision AI** is an industrial-grade computer vision platform designed to automate quality assurance in manufacturing and fabrication processes using custom-trained object detection deep learning models.

The system detects and categorizes welds across three specific industrial classes:

* 🟢 **Good Weld** (Class 1)
* 🔴 **Bad Weld** (Class 0)
* ⚠️ **Defect** (Class 2)

By deploying an optimized **ONNX Runtime engine**, the application delivers high-speed, sub-150ms inference on lightweight CPU servers without requiring expensive GPU infrastructure.

---

The web application provides:

* Instant single/batch image quality inspection
* Live webcam feed processing
* Dynamic bounding box generation with confidence scores
* Client-side threshold adjustment
* Lightweight server memory footprint (~108 MB)
* Direct API status monitoring

---

## 🎯 Project Objective

* Automate industrial weld quality monitoring to reduce manual inspection errors.
* Detect and pinpoint defective weld beads and joint anomalies in real time.
* Optimize heavy YOLO models into lightweight **ONNX formats** for efficient CPU-bound cloud deployments.
* Provide an interactive web dashboard for real-time monitoring and REST API integrations.

---

## 🚨 Problem Statement

Manual visual inspection of industrial welds is time-consuming, subjective, and prone to inspector fatigue, often allowing hazardous weld defects to pass through production lines.

### Proposed Solution

WeldVision AI introduces an automated end-to-end computer vision pipeline:

**Capture Frame → ONNX Inference → Non-Max Suppression → Visualize Defect & Grade Quality**

---

# 🛠️ How It Works

```text
               INPUT FRAME
                    │
                    ▼
          ┌──────────────────┐
          │ Image / Webcam   │
          └────────┬─────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Flask Web Engine │
          └────────┬─────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ ONNX Runtime Engine│
          │    (best.onnx)   │
          └────────┬─────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ NMS Postprocess  │
          └────────┬─────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Classification          Bounding Box
        │                       │
        └───────────┬───────────┘
                    ▼
          ┌──────────────────┐
          │ Quality Report   │
          │  & Overlay View  │
          └──────────────────┘

```

---

# 📑 Methodology

```text
01. Dataset Preparation & Roboflow Augmentation
        ↓
02. Model Training (YOLOv8s on PyTorch)
        ↓
03. Model Optimization (Export to ONNX Format)
        ↓
04. Backend Integration (Flask & OpenCV Headless)
        ↓
05. Memory-Constrained Deployment (Render Cloud Platform)

```

---

## 🗂️ Dataset & Classes

### Target Classes

| Class ID | Class Name | Category Status | BGR Color Code |
| --- | --- | --- | --- |
| **0** | Bad Weld | Defective | `(0, 60, 230)` Red |
| **1** | Good Weld | Acceptable | `(0, 220, 100)` Green |
| **2** | Defect | Defective | `(0, 60, 230)` Red |

---

# 🔬 Model Selection & Optimization

The model was trained on **YOLOv8s** via PyTorch and exported to **ONNX (Open Neural Network Exchange)** format for high-efficiency CPU execution.

| Model Variant | Runtime | Model Size | RAM Footprint | Target Environment |
| --- | --- | --- | --- | --- |
| YOLOv8s (PyTorch) | PyTorch + CUDA | ~22.5 MB | ~1.5 GB | Model Training (Colab) |
| **YOLOv8s (ONNX Engine)** | **ONNX Runtime (CPU)** | **~22.0 MB** | **~108 MB** | **⭐ Cloud Production (Render)** |

### Why ONNX Runtime?

* **Zero PyTorch Dependencies:** Drops server installation footprint from 1.5 GB down to ~150 MB.
* **Low RAM Overhead:** Runs within Render's free tier (512 MB memory limit) effortlessly.
* **Faster CPU Inference:** Average inference time reduced to <150ms per frame.

---

# 🔌 Web API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Serves web application dashboard |
| `GET` | `/health` | System and model health status |
| `POST` | `/predict` | Image upload and detection inference |
| `WS` | `/webcam` | Real-time WebSocket camera stream processing |

---

### Sample Response (`POST /predict`)

```json
{
  "status": "success",
  "detections": [
    {
      "class_id": 1,
      "class_name": "Good Weld",
      "confidence": 0.94,
      "bbox": [120, 45, 310, 280]
    },
    {
      "class_id": 2,
      "class_name": "Defect",
      "confidence": 0.88,
      "bbox": [320, 90, 410, 185]
    }
  ]
}

```

---

# 🧰 Technology Stack

| Domain | Technology |
| --- | --- |
| **Language** | Python 3.10+ |
| **Model Training** | PyTorch, Ultralytics YOLOv8 |
| **Inference Engine** | ONNX Runtime (CPU Engine) |
| **Computer Vision** | OpenCV Headless, Pillow, NumPy |
| **Web Framework** | Flask, Gunicorn |
| **Frontend** | HTML5, Modern CSS3, JavaScript (ES6) |
| **Deployment Platform** | Render Cloud Platform |

---

# 📁 Project Structure

```text
WeldVision-AI-Project/
│
├── web_app/
│   ├── app.py              # Flask server entrypoint
│   ├── config.py           # Paths, thresholding & class maps
│   ├── model_service.py    # ONNX Runtime inference engine
│   ├── models/
│   │   └── best.onnx       # CPU-optimized ONNX model
│   ├── templates/
│   │   └── index.html      # User interface
│   └── static/
│       ├── css/
│       └── js/
│
├── requirements.txt        # Lightweight CPU dependencies
└── README.md

```

---

# 💻 Local Setup Instructions

## 1. Clone Repository

```bash
git clone https://github.com/Abdulla-B-Official/WeldVision-AI-Project.git
cd WeldVision-AI-Project

```

## 2. Virtual Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

## 3. Install Lightweight Dependencies

```bash
pip install -r requirements.txt

```

## 4. Launch Application

```bash
python -m web_app.app

```

Navigate to `[http://127.0.0.1:5000](http://127.0.0.1:5000)` in your web browser.

---

# 🚀 Deployment Workflow

This project is configured for automated deployments on **Render**:

1. Ensure standard PyTorch dependencies are excluded from `requirements.txt` to keep build size under limits.
2. Confirm `web_app/models/best.onnx` is present.
3. Deploy as a **Web Service** with build command `pip install -r requirements.txt` and start command `gunicorn --workers 1 --threads 1 --timeout 120 web_app.app:app`.

---
