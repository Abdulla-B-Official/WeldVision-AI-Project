/*
===========================================================
WeldVision AI
Frontend Controller

CORRECT CLASS MAPPING:
0 = Bad Weld  -> DEFECT
1 = Good Weld -> GOOD
2 = Good Weld -> GOOD (Aligned with backend green box)

API:
GET  /health
POST /predict

Webcam:
Browser camera -> Canvas -> /predict
===========================================================
*/

"use strict";

/* ========================================================
   GLOBAL STATE
   ======================================================== */
let selectedFile = null;
let currentMode = "image";
let cameraStream = null;
let webcamRunning = false;
let webcamTimer = null;
let webcamFrameCount = 0;
let webcamLastFpsTime = 0;
let webcamConfidence = 0.50;
let imageConfidence = 0.50;

/*
CORRECTED CLASS MAPPING:
Updated Class 2 to be a good weld (`good: true, defective: false`) 
to match the backend's green bounding box behavior.
*/
const CLASS_MAP = {
    0: { name: "Bad Weld", good: false, defective: true },
    1: { name: "Good Weld", good: true, defective: false },
    2: { name: "Good Weld", good: true, defective: false }
};

/* ========================================================
   DOM HELPER
   ======================================================== */
function $(id) {
    return document.getElementById(id);
}

/* ========================================================
   INITIALIZATION
   ======================================================== */
document.addEventListener("DOMContentLoaded", () => {
    initializeApplication();
});

function initializeApplication() {
    setupModeTabs();
    setupImageControls();
    setupWebcamControls();
    setupConfidenceSliders();
    setupDragAndDrop();
    checkServerStatus();
    
    /* Check API every 30 seconds. */
    setInterval(checkServerStatus, 30000);
}

/* ========================================================
   MODE SWITCHING
   ======================================================== */
function setupModeTabs() {
    const imageTab = $("tab-image");
    const webcamTab = $("tab-webcam");

    if (imageTab) {
        imageTab.addEventListener("click", () => {
            switchMode("image");
        });
    }

    if (webcamTab) {
        webcamTab.addEventListener("click", () => {
            switchMode("webcam");
        });
    }
}

function switchMode(mode) {
    currentMode = mode;

    const imageTab = $("tab-image");
    const webcamTab = $("tab-webcam");
    const imageControls = $("image-controls");
    const webcamControls = $("webcam-controls");
    const imagePanel = $("panel-image");
    const webcamPanel = $("panel-webcam");
    const viewerMode = $("viewer-mode-tag");
    const viewerTitle = $("viewer-title");

    if (mode === "image") {
        /* Stop camera when leaving webcam mode. */
        stopWebcam();

        imageTab?.classList.add("active");
        webcamTab?.classList.remove("active");
        imageTab?.setAttribute("aria-selected", "true");
        webcamTab?.setAttribute("aria-selected", "false");

        imageControls?.classList.remove("hidden");
        webcamControls?.classList.add("hidden");
        imagePanel?.classList.remove("hidden");
        webcamPanel?.classList.add("hidden");

        if (viewerMode) viewerMode.textContent = "IMAGE MODE";
        if (viewerTitle) viewerTitle.textContent = "Inspection Viewer";

        hideBanner($("webcam-banner"));
    } else {
        imageTab?.classList.remove("active");
        webcamTab?.classList.add("active");
        imageTab?.setAttribute("aria-selected", "false");
        webcamTab?.setAttribute("aria-selected", "true");

        imageControls?.classList.add("hidden");
        webcamControls?.classList.remove("hidden");
        imagePanel?.classList.add("hidden");
        webcamPanel?.classList.remove("hidden");

        if (viewerMode) viewerMode.textContent = "LIVE MODE";
        if (viewerTitle) viewerTitle.textContent = "Live Weld Inspection";

        hideBanner($("img-banner"));
    }
}

/* ========================================================
   IMAGE CONTROLS
   ======================================================== */
function setupImageControls() {
    const uploadButton = $("upload-btn");
    const fileInput = $("file-input");
    const detectButton = $("detect-btn");
    const clearButton = $("clear-btn");

    uploadButton?.addEventListener("click", () => {
        fileInput?.click();
    });

    fileInput?.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        if (file) handleSelectedFile(file);
    });

    detectButton?.addEventListener("click", inspectImage);
    clearButton?.addEventListener("click", clearImage);
}

/* ========================================================
   FILE HANDLING
   ======================================================== */
function handleSelectedFile(file) {
    if (!file.type || !file.type.startsWith("image/")) {
        showImageError("Please select a valid image file.");
        return;
    }

    /* Maximum 10 MB. */
    if (file.size > 10 * 1024 * 1024) {
        showImageError("Image is too large. Maximum size is 10 MB.");
        return;
    }

    selectedFile = file;
    const reader = new FileReader();

    reader.onload = (event) => {
        const image = $("original-image");
        const imageViewer = $("image-viewer-area");
        const dropZone = $("drop-zone");
        const detectButton = $("detect-btn");
        const clearButton = $("clear-btn");
        const annotated = $("annotated-image");
        const placeholder = $("detected-placeholder");

        if (image) image.src = event.target.result;

        dropZone?.classList.add("hidden");
        imageViewer?.classList.remove("hidden");

        if (annotated) {
            annotated.src = "";
            annotated.classList.add("hidden");
        }

        placeholder?.classList.remove("hidden");
        detectButton?.removeAttribute("disabled");
        clearButton?.classList.remove("hidden");

        resetResultCard();
        hideBanner($("img-banner"));
    };

    reader.readAsDataURL(file);
}

/* ========================================================
   DRAG AND DROP
   ======================================================== */
function setupDragAndDrop() {
    const dropZone = $("drop-zone");
    const fileInput = $("file-input");

    if (!dropZone) return;

    dropZone.addEventListener("click", () => {
        fileInput?.click();
    });

    dropZone.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            fileInput?.click();
        }
    });

    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropZone.classList.add("dragging");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragging");
    });

    dropZone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropZone.classList.remove("dragging");

        const files = event.dataTransfer.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });
}

/* ========================================================
   IMAGE INSPECTION
   ======================================================== */
async function inspectImage() {
    if (!selectedFile) {
        showImageError("Please upload a weld image first.");
        return;
    }

    const detectButton = $("detect-btn");
    const banner = $("img-banner");

    detectButton.disabled = true;
    detectButton.textContent = "Inspecting...";
    hideBanner(banner);

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("confidence", imageConfidence.toString());

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error("Server returned an invalid response.");
        }

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Inspection failed.");
        }

        displayImageResult(data);

    } catch (error) {
        console.error("Image inspection error:", error);
        showImageError(error.message || "Could not inspect image.");
    } finally {
        detectButton.disabled = false;
        
        /* Restore button text. */
        detectButton.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8"/>
                <path d="M21 21l-4.35-4.35"/>
            </svg>
            Inspect Weld
        `;
    }
}

/* ========================================================
   DISPLAY IMAGE RESULT
   ======================================================== */
function displayImageResult(data) {
    const annotatedImage = $("annotated-image");
    const placeholder = $("detected-placeholder");

    if (annotatedImage) {
        annotatedImage.src = data.image;
        annotatedImage.classList.remove("hidden");
    }

    placeholder?.classList.add("hidden");

    /* Normalize detections. */
    const detections = Array.isArray(data.detections) ? data.detections : [];
    const verdict = calculateVerdict(detections);

    updateVerdictCard(verdict, detections);
    updateDetectionList(detections, $("detection-cards"));
    updateStats(detections.length, data.inference_time_ms);

    /* Overlay verdict. */
    const badge = $("img-verdict-badge");
    if (badge) {
        badge.classList.remove("hidden", "good", "defective", "neutral");

        if (verdict.status === "GOOD") {
            badge.textContent = "✓ GOOD WELD";
            badge.classList.add("good");
        } else if (verdict.status === "DEFECT") {
            badge.textContent = "⚠ DEFECTIVE";
            badge.classList.add("defective");
        } else {
            badge.textContent = "○ NO WELD";
            badge.classList.add("neutral");
        }
    }
}

/* ========================================================
   CORRECT VERDICT CALCULATION
   ======================================================== */
function calculateVerdict(detections) {
    /* Only class 0 is considered a defect; classes 1 and 2 are good */
    const defective = detections.filter(detection => {
        const id = Number(detection.class_id);
        return id === 0;
    });

    const good = detections.filter(detection => {
        const id = Number(detection.class_id);
        return id === 1 || id === 2;
    });

    if (defective.length > 0) {
        const best = defective.reduce((highest, current) => {
            return Number(current.confidence) > Number(highest.confidence) ? current : highest;
        });

        return {
            status: "DEFECT",
            confidence: Number(best.confidence)
        };
    }

    if (good.length > 0) {
        const best = good.reduce((highest, current) => {
            return Number(current.confidence) > Number(highest.confidence) ? current : highest;
        });

        return {
            status: "GOOD",
            confidence: Number(best.confidence)
        };
    }

    return {
        status: "NO_WELD",
        confidence: 0
    };
}

/* ========================================================
   VERDICT CARD
   ======================================================== */
function updateVerdictCard(verdict, detections) {
    const card = $("verdict-card");
    const icon = $("verdict-icon");
    const label = $("verdict-label");
    const confidence = $("verdict-conf-display");
    const subtitle = $("verdict-sub");

    card?.classList.remove("idle", "good", "defective", "neutral");

    if (verdict.status === "GOOD") {
        card?.classList.add("good");
        if (icon) icon.textContent = "✓";
        if (label) label.textContent = "GOOD WELD";
        if (confidence) confidence.textContent = `${Math.round(verdict.confidence * 100)}% confidence`;
        if (subtitle) subtitle.textContent = "Weld condition appears good";
    } else if (verdict.status === "DEFECT") {
        card?.classList.add("defective");
        if (icon) icon.textContent = "⚠";
        if (label) label.textContent = "DEFECTIVE";
        if (confidence) confidence.textContent = `${Math.round(verdict.confidence * 100)}% confidence`;
        if (subtitle) subtitle.textContent = "Bad weld or weld defect detected";
    } else {
        card?.classList.add("neutral");
        if (icon) icon.textContent = "○";
        if (label) label.textContent = "NO WELD";
        if (confidence) confidence.textContent = "—";
        if (subtitle) subtitle.textContent = "No weld object detected";
    }
}

/* ========================================================
   DETECTION LIST
   ======================================================== */
function updateDetectionList(detections, container) {
    if (!container) return;

    container.innerHTML = "";

    if (!detections || detections.length === 0) {
        container.innerHTML = `<div class="det-empty">No detections found</div>`;
        return;
    }

    detections.forEach(detection => {
        const classId = Number(detection.class_id);
        const classInfo = CLASS_MAP[classId] || {
            name: `Class ${classId}`,
            good: false,
            defective: false
        };

        const confidence = Number(detection.confidence || 0);
        const percentage = Math.round(confidence * 100);

        const item = document.createElement("div");
        item.className = classInfo.defective ? "detection-item defective" : "detection-item good";

        item.innerHTML = `
            <div class="detection-main">
                <span class="detection-name">${escapeHTML(classInfo.name)}</span>
                <span class="detection-confidence">${percentage}%</span>
            </div>
            <div class="detection-meta">
                Class ${classId} ${classInfo.defective ? " · DEFECT" : " · GOOD"}
            </div>
        `;
        container.appendChild(item);
    });
}

/* ========================================================
   UPDATE STATISTICS
   ======================================================== */
function updateStats(count, inferenceTime) {
    const countElement = $("stat-count");
    const inferenceElement = $("stat-inference");

    if (countElement) countElement.textContent = count ?? 0;
    if (inferenceElement) {
        inferenceElement.textContent = inferenceTime != null ? Number(inferenceTime).toFixed(1) : "—";
    }
}

/* ========================================================
   CLEAR IMAGE
   ======================================================== */
function clearImage() {
    selectedFile = null;

    const fileInput = $("file-input");
    const dropZone = $("drop-zone");
    const viewer = $("image-viewer-area");
    const detectButton = $("detect-btn");
    const clearButton = $("clear-btn");
    const original = $("original-image");
    const annotated = $("annotated-image");
    const placeholder = $("detected-placeholder");
    const badge = $("img-verdict-badge");

    if (fileInput) fileInput.value = "";

    original?.removeAttribute("src");
    annotated?.removeAttribute("src");
    annotated?.classList.add("hidden");
    placeholder?.classList.remove("hidden");
    badge?.classList.add("hidden");
    viewer?.classList.add("hidden");
    dropZone?.classList.remove("hidden");
    detectButton?.setAttribute("disabled", "disabled");
    clearButton?.classList.add("hidden");

    hideBanner($("img-banner"));
    resetResultCard();
}

/* ========================================================
   RESET RESULT CARD
   ======================================================== */
function resetResultCard() {
    const card = $("verdict-card");
    const icon = $("verdict-icon");
    const label = $("verdict-label");
    const confidence = $("verdict-conf-display");
    const subtitle = $("verdict-sub");

    card?.classList.remove("good", "defective", "neutral");
    card?.classList.add("idle");

    if (icon) icon.textContent = "○";
    if (label) label.textContent = "AWAITING";
    if (confidence) confidence.textContent = "—";
    if (subtitle) subtitle.textContent = "Upload an image to inspect";

    updateStats(0, null);

    const list = $("detection-cards");
    if (list) {
        list.innerHTML = `<div class="det-empty">No detections yet</div>`;
    }
}

/* ========================================================
   CONFIDENCE SLIDERS
   ======================================================== */
function setupConfidenceSliders() {
    const imageSlider = $("img-conf-slider");
    const imageValue = $("img-conf-value");
    const webcamSlider = $("webcam-conf-slider");
    const webcamValue = $("webcam-conf-value");

    imageSlider?.addEventListener("input", () => {
        imageConfidence = Number(imageSlider.value);
        if (imageValue) imageValue.textContent = `${Math.round(imageConfidence * 100)}%`;
    });

    webcamSlider?.addEventListener("input", () => {
        webcamConfidence = Number(webcamSlider.value);
        if (webcamValue) webcamValue.textContent = `${Math.round(webcamConfidence * 100)}%`;
    });
}

/* ========================================================
   WEBCAM CONTROLS
   ======================================================== */
function setupWebcamControls() {
    const startButton = $("start-webcam-btn");
    const stopButton = $("stop-webcam-btn");

    startButton?.addEventListener("click", startWebcam);
    stopButton?.addEventListener("click", stopWebcam);
}

/* ========================================================
   START WEBCAM
   ======================================================== */
async function startWebcam() {
    if (webcamRunning) return;

    const video = $("webcam-video");
    const idle = $("webcam-idle");
    const liveBadge = $("webcam-live-badge");
    const fpsBadge = $("webcam-fps-badge");
    const startButton = $("start-webcam-btn");
    const stopButton = $("stop-webcam-btn");
    const cameraDot = $("bar-camera-dot");
    const cameraText = $("bar-camera-text");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showWebcamError("Your browser does not support webcam access.");
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        webcamRunning = true;
        video.classList.remove("hidden");
        idle?.classList.add("hidden");
        liveBadge?.classList.remove("hidden");
        fpsBadge?.classList.remove("hidden");
        startButton?.classList.add("hidden");
        stopButton?.classList.remove("hidden");
        $("webcam-slider-hint")?.classList.remove("hidden");
        
        if ($("webcam-conf-slider")) {
            $("webcam-conf-slider").disabled = true;
        }

        setDot(cameraDot, true);
        if (cameraText) cameraText.textContent = "Camera Live";

        hideBanner($("webcam-banner"));

        webcamFrameCount = 0;
        webcamLastFpsTime = performance.now();

        startWebcamDetection();
    } catch (error) {
        console.error("Camera error:", error);
        showWebcamError(getCameraErrorMessage(error));
    }
}

/* ========================================================
   CAMERA ERROR MESSAGE
   ======================================================== */
function getCameraErrorMessage(error) {
    if (error && error.name === "NotAllowedError") {
        return "Camera permission was denied. Allow camera access in your browser.";
    }
    if (error && error.name === "NotFoundError") {
        return "No camera was found on this device.";
    }
    return "Unable to start camera.";
}

/* ========================================================
   START WEBCAM AI DETECTION
   ======================================================== */
function startWebcamDetection() {
    if (webcamTimer) clearInterval(webcamTimer);
    webcamTimer = setInterval(captureAndPredictFrame, 1200);
    setTimeout(captureAndPredictFrame, 300);
}

/* ========================================================
   CAPTURE WEBCAM FRAME
   ======================================================== */
async function captureAndPredictFrame() {
    if (!webcamRunning) return;

    const video = $("webcam-video");
    const canvas = $("capture-canvas");

    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        return;
    }

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;

    const maxWidth = 960;
    const scale = Math.min(1, maxWidth / width);

    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise(resolve => {
        canvas.toBlob(resolve, "image/jpeg", 0.80);
    });

    if (!blob) return;

    const formData = new FormData();
    formData.append("file", blob, "webcam.jpg");
    formData.append("confidence", webcamConfidence.toString());

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Webcam prediction failed.");
        }

        const annotated = $("webcam-annotated");
        if (annotated) {
            annotated.src = data.image;
            annotated.classList.remove("hidden");
        }

        const detections = Array.isArray(data.detections) ? data.detections : [];
        const result = calculateVerdict(detections);

        updateWebcamVerdict(result, detections, data.inference_time_ms);

        webcamFrameCount++;
        updateFPS();

    } catch (error) {
        console.error("Webcam prediction error:", error);
    }
}

/* ========================================================
   WEBCAM VERDICT
   ======================================================== */
function updateWebcamVerdict(result, detections, inferenceTime) {
    const label = $("verdict-label");
    const icon = $("verdict-icon");
    const confidence = $("verdict-conf-display");
    const subtitle = $("verdict-sub");
    const card = $("verdict-card");

    card?.classList.remove("idle", "good", "defective", "neutral");

    if (result.status === "DEFECT") {
        card?.classList.add("defective");
        if (icon) icon.textContent = "⚠";
        if (label) label.textContent = "DEFECTIVE";
        if (confidence) confidence.textContent = `${Math.round(result.confidence * 100)}% confidence`;
        if (subtitle) subtitle.textContent = "Bad weld or defect detected";
    } else if (result.status === "GOOD") {
        card?.classList.add("good");
        if (icon) icon.textContent = "✓";
        if (label) label.textContent = "GOOD WELD";
        if (confidence) confidence.textContent = `${Math.round(result.confidence * 100)}% confidence`;
        if (subtitle) subtitle.textContent = "Good weld detected";
    } else {
        card?.classList.add("neutral");
        if (icon) icon.textContent = "○";
        if (label) label.textContent = "NO WELD";
        if (confidence) confidence.textContent = "—";
        if (subtitle) subtitle.textContent = "No weld detected";
    }

    updateStats(detections.length, inferenceTime);
    const liveSection = $("webcam-detection-cards-section");

    if (detections.length > 0) {
        liveSection?.classList.remove("hidden");
        updateDetectionList(detections, $("webcam-detection-cards"));
    } else {
        liveSection?.classList.add("hidden");
    }
}

/* ========================================================
   FPS
   ======================================================== */
function updateFPS() {
    const now = performance.now();
    const elapsed = now - webcamLastFpsTime;

    if (elapsed >= 1000) {
        const fps = (webcamFrameCount * 1000) / elapsed;
        const badge = $("webcam-fps-badge");
        if (badge) badge.textContent = `${fps.toFixed(1)} FPS`;

        webcamFrameCount = 0;
        webcamLastFpsTime = now;
    }
}

/* ========================================================
   STOP WEBCAM
   ======================================================== */
function stopWebcam() {
    webcamRunning = false;

    if (webcamTimer) {
        clearInterval(webcamTimer);
        webcamTimer = null;
    }

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    const video = $("webcam-video");
    const annotated = $("webcam-annotated");
    const idle = $("webcam-idle");
    const liveBadge = $("webcam-live-badge");
    const fpsBadge = $("webcam-fps-badge");
    const startButton = $("start-webcam-btn");
    const stopButton = $("stop-webcam-btn");
    const cameraDot = $("bar-camera-dot");
    const cameraText = $("bar-camera-text");

    if (video) {
        video.pause();
        video.srcObject = null;
        video.classList.add("hidden");
    }

    annotated?.classList.add("hidden");
    idle?.classList.remove("hidden");
    liveBadge?.classList.add("hidden");
    fpsBadge?.classList.add("hidden");
    startButton?.classList.remove("hidden");
    stopButton?.classList.add("hidden");
    $("webcam-slider-hint")?.classList.add("hidden");

    const webcamSlider = $("webcam-conf-slider");
    if (webcamSlider) webcamSlider.disabled = false;

    setDot(cameraDot, false);
    if (cameraText) cameraText.textContent = "Camera Idle";
    hideBanner($("webcam-banner"));
}

/* ========================================================
   SERVER STATUS
   ======================================================== */
async function checkServerStatus() {
    const apiBadge = $("api-badge");
    const modelBadge = $("model-badge");

    setBadge(apiBadge, "checking", "API Checking…");
    setBadge(modelBadge, "checking", "Model Checking…");

    try {
        const response = await fetch("/health", {
            method: "GET",
            cache: "no-store"
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();

        setBadge(apiBadge, "online", "API Online");
        updateSystemAPI(true);

        if (data.model_loaded) {
            setBadge(modelBadge, "online", "Model Online");
            updateSystemModel(true, data);
        } else {
            setBadge(modelBadge, "offline", "Model Error");
            updateSystemModel(false, data);
        }

        const device = data.device || "CPU";
        if ($("sys-device-label")) $("sys-device-label").textContent = `Device: ${device}`;
        if ($("perf-device")) $("perf-device").textContent = device;
        if ($("perf-model")) $("perf-model").textContent = data.model_name || "best.onnx";
        if ($("perf-status")) $("perf-status").textContent = data.model_loaded ? "Ready" : "Error";

    } catch (error) {
        console.error("Health check failed:", error);
        setBadge(apiBadge, "offline", "API Offline");
        setBadge(modelBadge, "offline", "Model Unknown");
        updateSystemAPI(false);
        updateSystemModel(false);
        if ($("perf-status")) $("perf-status").textContent = "Offline";
    }
}

/* ========================================================
   UPDATE API STATUS
   ======================================================== */
function updateSystemAPI(online) {
    const dot = $("sys-api-dot");
    const footerDot = $("bar-api-dot");
    const footerText = $("bar-api-text");

    setDot(dot, online);
    setDot(footerDot, online);

    if (footerText) {
        footerText.textContent = online ? "API Online" : "API Offline";
    }
}

/* ========================================================
   UPDATE MODEL STATUS
   ======================================================== */
function updateSystemModel(online, data = {}) {
    const dot = $("sys-model-dot");
    const footerDot = $("bar-model-dot");
    const footerText = $("bar-model-text");

    setDot(dot, online);
    setDot(footerDot, online);

    if (footerText) {
        footerText.textContent = online ? `Model: ${data.model_name || "best.onnx"}` : "Model Error";
    }
}

/* ========================================================
   STATUS BADGE
   ======================================================== */
function setBadge(badge, state, text) {
    if (!badge) return;
    badge.classList.remove("muted", "online", "offline", "checking");
    badge.classList.add(state);
    const textElement = badge.querySelector(".pill-text");
    if (textElement) textElement.textContent = text;
}

/* ========================================================
   DOT STATUS
   ======================================================== */
function setDot(dot, online) {
    if (!dot) return;
    dot.classList.remove("green", "red", "online", "offline");
    if (online) {
        dot.classList.add("green", "online");
    } else {
        dot.classList.add("red", "offline");
    }
}

/* ========================================================
   ERROR BANNERS
   ======================================================== */
function showImageError(message) {
    showBanner($("img-banner"), message);
}

function showWebcamError(message) {
    showBanner($("webcam-banner"), message);
}

function showBanner(element, message) {
    if (!element) return;
    element.textContent = message;
    element.classList.remove("hidden");
}

function hideBanner(element) {
    element?.classList.add("hidden");
}

/* ========================================================
   ESCAPE HTML
   ======================================================== */
function escapeHTML(value) {
    const div = document.createElement("div");
    div.textContent = value == null ? "" : String(value);
    return div.innerHTML;
}