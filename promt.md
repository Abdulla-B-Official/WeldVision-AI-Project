Act as a senior Python and full-stack developer and Build a complete local YOLO Weld Detection web app around my EXISTING project.

IMPORTANT: DO NOT delete, rename, move, overwrite, or modify my existing YOLO files unless absolutely necessary.

Existing project:

Weld-Detection-Project/
├── datasets/
│   ├── train/
│   ├── valid/
│   └── test/
├── best.pt
├── data.yaml
├── inference.py
├── webcam.py
├── validate.py
├── latency_check.py
├── dataset_check.py
└── requirements.txt

First inspect these existing files, especially:
- inference.py
- webcam.py
- data.yaml
- requirements.txt

Reuse the existing YOLO model and inference approach. Do NOT retrain the model and do NOT create another dataset.

Create ONLY a new folder:

web_app/

Suggested structure:

web_app/
├── backend/
│   ├── app.py
│   ├── model_service.py
│   ├── config.py
│   └── utils.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── App.css
│       ├── components/
│       └── services/
└── README.md

==================================================
FUNCTIONAL REQUIREMENTS
==================================================

1. IMAGE DETECTION

Allow the user to:
- upload JPG/JPEG/PNG/WEBP
- drag and drop an image
- preview the original image
- click "Detect Weld"
- see the YOLO annotated image
- see bounding boxes
- see class names
- see confidence percentages
- see object count
- see inference time
- clear/reset results

Use the existing root model:

best.pt

Do NOT copy or move best.pt.

Calculate its path reliably from the project root using pathlib.

Use model.names so classes are NOT hard-coded.

Default confidence threshold: 0.50

==================================================
2. BACKEND
==================================================

Use Flask + Flask-CORS + Ultralytics + OpenCV/Pillow + NumPy.

Run backend on:

http://localhost:5000

Create:

GET /api/status

Return:
{
  "status": "online",
  "model_loaded": true,
  "model_name": "best.pt",
  "device": "cuda" or "cpu"
}

Create:

POST /api/predict

Accept an image, run real YOLO inference, draw bounding boxes, measure inference time, and return JSON containing:

- success
- annotated Base64 image
- detections
- count
- inference_time_ms

Each detection should contain:

class_id
class_name
confidence
bbox: x1, y1, x2, y2

Load YOLO ONCE at backend startup, not once per request.

Use CUDA automatically when available; otherwise CPU.

If best.pt is missing, backend should remain startable and /api/status should report model_loaded=false instead of crashing.

==================================================
3. WEBCAM
==================================================

Create a Live Webcam mode.

Use browser:

navigator.mediaDevices.getUserMedia({
  video: true,
  audio: false
})

Buttons:
- Start Webcam
- Stop Webcam

The browser captures frames and sends them to Flask for REAL YOLO inference.

Create:

POST /api/webcam

Return the same detection information and annotated frame.

Start with approximately 5–10 processed frames/second.

Do NOT send overlapping requests.

Do NOT save every webcam frame to disk.

When Stop Webcam is clicked:
- stop all MediaStream tracks
- stop the detection loop
- clean timers/resources
- release the camera

Show:
- Objects Detected
- Confidence
- Inference Time
- Approximate FPS

==================================================
4. FRONTEND
==================================================

Use React + Vite.

Run frontend on:

http://localhost:5173

Use Vite proxy so frontend requests /api/* go to:

http://localhost:5000

Create a modern professional UI called:

WeldVision AI

Subtitle:
AI-Powered Weld Inspection

Top status:
API ● Online/Offline
Model ● Loaded/Not Loaded

Provide two modes/tabs:

IMAGE INSPECTION
LIVE WEBCAM

Use a professional dark AI/computer-vision style with:
- responsive layout
- clean cards
- rounded corners
- subtle shadows
- good spacing
- modern buttons

Do not add fake statistics, fake metrics, or unnecessary charts.

Do NOT display training metrics such as mAP, precision, recall, or accuracy.

==================================================
5. ERROR HANDLING
==================================================

Show friendly frontend errors for:
- backend unavailable
- model missing
- invalid image
- prediction failure
- camera permission denied

Do not expose Python stack traces to the user.

==================================================
6. EXISTING FILES MUST REMAIN INTACT
==================================================

Do NOT modify these unless absolutely necessary:

best.pt
data.yaml
inference.py
webcam.py
validate.py
latency_check.py
dataset_check.py
datasets/*
requirements.txt

The new web app must be additive and use the existing project.

If an existing script already contains useful inference logic, adapt/reuse it rather than rewriting it.

Do not create a second model or second dataset.

==================================================
7. PATHS
==================================================

Because web_app is inside the existing project, determine PROJECT_ROOT correctly.

Example concept:

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "best.pt"
DATA_PATH = PROJECT_ROOT / "data.yaml"

Do not depend on the terminal's current working directory.

==================================================
8. DEPENDENCIES
==================================================

Inspect the existing requirements.txt first.

Do not unnecessarily reinstall or remove existing YOLO/PyTorch dependencies.

If web-specific dependencies are needed, create:

web_app/backend/requirements-web.txt

or clearly document additions.

==================================================
9. TEST BEFORE FINISHING
==================================================

Verify:

Backend:
- Flask starts
- best.pt loads
- /api/status works
- /api/predict works
- /api/webcam works

Frontend:
- React starts
- image upload works
- detection works
- annotated image appears
- detection details appear
- webcam starts
- live detection works
- webcam stops and releases camera
- API status works
- errors display correctly

==================================================
10. STARTUP
==================================================

Backend:

cd web_app/backend
python app.py

Frontend:

cd web_app/frontend
npm install
npm run dev

Open:

http://localhost:5173

==================================================
FINAL INSTRUCTION
==================================================

Do not blindly generate files.

FIRST inspect my current project and existing inference.py/webcam.py/data.yaml.

Then build the web app around them.

The final app must use my real best.pt and provide:
1. Image upload detection
2. Live webcam detection
3. Bounding boxes
4. Class names
5. Confidence scores
6. Object count
7. Inference time
8. API/model status

Keep my existing YOLO project intact.