# WeldVision AI

**AI-Powered Weld Inspection** — a local web app that wraps your trained YOLOv8m model with a professional React + Flask interface.

---

## Quick Start

### 1 — Install web backend dependencies

```bash
pip install flask flask-cors pillow
# (ultralytics, opencv-python, numpy already in root requirements.txt)
```

Or use the bundled file:

```bash
pip install -r web_app/backend/requirements-web.txt
```

### 2 — Start the Flask backend

```bash
cd web_app/backend
python app.py
```

Backend runs on **http://localhost:5000**

### 3 — Start the React frontend

```bash
cd web_app/frontend
npm install
npm run dev
```

Frontend runs on **http://localhost:5173**

---

## Features

| Feature | Details |
|---|---|
| 🖼️ Image Inspection | Drag & drop JPG/PNG/WEBP, see annotated result |
| 📷 Live Webcam | Real-time YOLO detection at ~8 FPS |
| 🎯 Bounding Boxes | Color-coded: 🟢 Good Weld, 🔴 Defective |
| 📊 Detections | Class name, confidence %, bbox coords |
| ⚡ Inference Time | Displayed per detection |
| 🔌 Status Bar | Live API online/offline + model loaded/not loaded |
| 🎛️ Confidence Slider | Adjustable threshold (5%–95%) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/api/status`  | Model status, device (cuda/cpu) |
| `POST` | `/api/predict` | Upload image → JSON + annotated base64 image |
| `POST` | `/api/webcam`  | Send base64 frame → JSON + annotated base64 image |

### `GET /api/status` response
```json
{
  "status": "online",
  "model_loaded": true,
  "model_name": "best.pt",
  "device": "cuda"
}
```

### `POST /api/predict` request
- `multipart/form-data`
- Field `image`: image file
- Field `conf_threshold` (optional): float 0.01–1.0 (default 0.5)

### `POST /api/webcam` request
```json
{
  "frame": "<base64 JPEG string>",
  "conf_threshold": 0.5
}
```

### Detection response shape
```json
{
  "success": true,
  "annotated_image": "<base64 JPEG>",
  "detections": [
    {
      "class_id": 0,
      "class_name": "Bad Weld",
      "confidence": 0.87,
      "is_defective": true,
      "bbox": { "x1": 10, "y1": 20, "x2": 100, "y2": 120 }
    }
  ],
  "count": 1,
  "inference_time_ms": 42.7
}
```

---

## Project Structure

```
web_app/
├── backend/
│   ├── app.py              # Flask app — 3 API endpoints
│   ├── config.py           # Paths, thresholds, class IDs
│   ├── model_service.py    # YOLOv8 load-once + inference
│   ├── utils.py            # Image encode/decode helpers
│   └── requirements-web.txt
└── frontend/
    ├── package.json
    ├── vite.config.js       # Proxy /api → localhost:5000
    ├── index.html
    └── src/
        ├── App.jsx          # Root, status polling
        ├── App.css          # Full dark AI theme
        ├── main.jsx
        ├── components/
        │   ├── Header.jsx
        │   ├── TabSwitcher.jsx
        │   ├── ImageInspection.jsx
        │   ├── WebcamMode.jsx
        │   ├── DetectionResults.jsx
        │   └── StatusBadge.jsx
        └── services/
            └── api.js       # fetch wrappers for all 3 endpoints
```

---

## Class Mapping

| ID | Class | Type |
|---|---|---|
| 0 | Bad Weld | 🔴 Defective |
| 1 | Good Weld | 🟢 Good |
| 2 | Defect | 🔴 Defective |

---

## Notes

- The model path is resolved using `pathlib` from `config.py` — no dependency on terminal CWD.
- CUDA is used automatically when available; otherwise falls back to CPU.
- If `best.pt` is missing, the backend still starts and reports `model_loaded: false`.
- Existing project files (`inference.py`, `webcam.py`, `data.yaml`, etc.) are **not modified**.
