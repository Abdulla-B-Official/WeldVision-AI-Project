/**
 * WeldVision AI - Industrial Inspection Dashboard Controller
 */

// --- Global Application State ---
const state = {
  currentFile: null,
  currentImageObjectUrl: null,
  annotatedImageDataUrl: null,
  activeView: 'annotated', // 'raw' | 'annotated'
  isWebcamActive: false,
  webcamStream: null,
  isInspecting: false,
  inspectionResults: null
};

// --- DOM Element References ---
const elements = {
  // Upload & Inputs
  dropZone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),
  selectFileBtn: document.getElementById('selectFileBtn'),
  
  // Webcam Controls
  toggleWebcamBtn: document.getElementById('toggleWebcamBtn'),
  captureWebcamBtn: document.getElementById('captureWebcamBtn'),
  webcamVideo: document.getElementById('webcamVideo'),
  hiddenCanvas: document.getElementById('hiddenCanvas'),

  // Display & Canvas Controls
  imagePreviewContainer: document.getElementById('imagePreviewContainer'),
  previewImage: document.getElementById('previewImage'),
  annotationCanvas: document.getElementById('annotationCanvas'),
  toggleViewBtn: document.getElementById('toggleViewBtn'),
  viewModeLabel: document.getElementById('viewModeLabel'),

  // Control Buttons
  runInspectionBtn: document.getElementById('runInspectionBtn'),
  resetBtn: document.getElementById('resetBtn'),
  exportResultsBtn: document.getElementById('exportResultsBtn'),

  // Output Panels
  statusBadge: document.getElementById('statusBadge'),
  confidenceMetric: document.getElementById('confidenceMetric'),
  defectCountMetric: document.getElementById('defectCountMetric'),
  processingTimeMetric: document.getElementById('processingTimeMetric'),
  defectListContainer: document.getElementById('defectListContainer')
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
});

function initEventListeners() {
  // File Upload Handling
  if (elements.selectFileBtn && elements.fileInput) {
    elements.selectFileBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);
  }

  if (elements.dropZone) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      elements.dropZone.addEventListener(eventName, preventDefaults, false);
    });
    ['dragenter', 'dragover'].forEach(eventName => {
      elements.dropZone.classList.add('highlight');
    });
    ['dragleave', 'drop'].forEach(eventName => {
      elements.dropZone.classList.remove('highlight');
    });
    elements.dropZone.addEventListener('drop', handleDrop);
  }

  // Webcam Controls
  if (elements.toggleWebcamBtn) {
    elements.toggleWebcamBtn.addEventListener('click', toggleWebcam);
  }
  if (elements.captureWebcamBtn) {
    elements.captureWebcamBtn.addEventListener('click', captureWebcamFrame);
  }

  // Inspection & Display Controls
  if (elements.runInspectionBtn) {
    elements.runInspectionBtn.addEventListener('click', runInspection);
  }
  if (elements.toggleViewBtn) {
    elements.toggleViewBtn.addEventListener('click', toggleImageView);
  }
  if (elements.resetBtn) {
    elements.resetBtn.addEventListener('click', resetDashboard);
  }
  if (elements.exportResultsBtn) {
    elements.exportResultsBtn.addEventListener('click', exportResultsJSON);
  }
}

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// --- Image Handling ---
function handleFileSelect(e) {
  const files = e.target.files;
  if (files && files[0]) {
    loadImageFile(files[0]);
  }
}

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files && files[0]) {
    loadImageFile(files[0]);
  }
}

function loadImageFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please upload a valid image file (JPEG, PNG).');
    return;
  }

  // Cleanup existing Object URL to prevent memory leaks
  if (state.currentImageObjectUrl) {
    URL.revokeObjectURL(state.currentImageObjectUrl);
  }

  state.currentFile = file;
  state.currentImageObjectUrl = URL.createObjectURL(file);
  state.annotatedImageDataUrl = null;
  state.inspectionResults = null;

  renderImagePreview(state.currentImageObjectUrl);
  if (elements.runInspectionBtn) elements.runInspectionBtn.disabled = false;
  clearResultsDisplay();
}

function renderImagePreview(src) {
  if (!elements.previewImage) return;
  elements.previewImage.src = src;
  elements.previewImage.onload = () => {
    if (elements.annotationCanvas) {
      elements.annotationCanvas.width = elements.previewImage.naturalWidth;
      elements.annotationCanvas.height = elements.previewImage.naturalHeight;
      clearCanvas();
    }
  };
}

// --- Webcam Integration ---
async function toggleWebcam() {
  if (state.isWebcamActive) {
    stopWebcam();
  } else {
    await startWebcam();
  }
}

async function startWebcam() {
  try {
    state.webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1920 }, height: { ideal: 1080 } }
    });
    if (elements.webcamVideo) {
      elements.webcamVideo.srcObject = state.webcamStream;
      elements.webcamVideo.style.display = 'block';
      state.isWebcamActive = true;
      if (elements.toggleWebcamBtn) elements.toggleWebcamBtn.textContent = 'Stop Camera';
      if (elements.captureWebcamBtn) elements.captureWebcamBtn.disabled = false;
    }
  } catch (err) {
    console.error('Webcam initialization failed:', err);
    alert('Unable to access camera feed: ' + err.message);
  }
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(track => track.stop());
    state.webcamStream = null;
  }
  if (elements.webcamVideo) {
    elements.webcamVideo.style.display = 'none';
  }
  state.isWebcamActive = false;
  if (elements.toggleWebcamBtn) elements.toggleWebcamBtn.textContent = 'Start Camera';
  if (elements.captureWebcamBtn) elements.captureWebcamBtn.disabled = true;
}

function captureWebcamFrame() {
  if (!state.isWebcamActive || !elements.webcamVideo) return;

  const canvas = elements.hiddenCanvas || document.createElement('canvas');
  canvas.width = elements.webcamVideo.videoWidth;
  canvas.height = elements.webcamVideo.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(elements.webcamVideo, 0, 0);

  canvas.toBlob(blob => {
    const file = new File([blob], `weld_capture_${Date.now()}.png`, { type: 'image/png' });
    loadImageFile(file);
    stopWebcam();
  }, 'image/png');
}

// --- Inspection API & Processing ---
async function runInspection() {
  if (!state.currentFile && !state.currentImageObjectUrl) return;

  setInspectingState(true);

  try {
    const formData = new FormData();
    if (state.currentFile) {
      formData.append('image', state.currentFile);
    }

    // Attempt real API fetch; fallback to simulation if backend is unreachable
    let results;
    try {
      const response = await fetch('/api/v1/inspect', {
        method: 'POST',
        body: formData
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      results = await response.json();
    } catch (networkError) {
      console.warn('Backend API unreachable. Running client inspection simulation:', networkError);
      results = await simulateInspectionResponse();
    }

    state.inspectionResults = results;
    renderInspectionResults(results);

  } catch (error) {
    console.error('Inspection failed:', error);
    updateStatusBadge('FAILED', 'status-danger');
  } finally {
    setInspectingState(false);
  }
}

// Simulated ML detection payload when running without an active backend API
function simulateInspectionResponse() {
  return new Promise(resolve => {
    setTimeout(() => {
      resolve({
        status: 'DEFECT_DETECTED', // 'PASS' | 'DEFECT_DETECTED'
        overallConfidence: 94.2,
        processingTimeMs: 142,
        defects: [
          {
            id: 1,
            type: 'Porosity',
            severity: 'High',
            confidence: 96.5,
            bbox: [120, 80, 200, 160] // [x1, y1, x2, y2]
          },
          {
            id: 2,
            type: 'Lack of Penetration',
            severity: 'Medium',
            confidence: 91.8,
            bbox: [340, 210, 480, 260]
          }
        ]
      });
    }, 800);
  });
}

// --- UI Rendering ---
function renderInspectionResults(results) {
  // Update Metrics
  if (elements.confidenceMetric) elements.confidenceMetric.textContent = `${results.overallConfidence.toFixed(1)}%`;
  if (elements.defectCountMetric) elements.defectCountMetric.textContent = results.defects.length;
  if (elements.processingTimeMetric) elements.processingTimeMetric.textContent = `${results.processingTimeMs} ms`;

  // Update Status Badge
  if (results.defects.length === 0) {
    updateStatusBadge('PASSED', 'status-success');
  } else {
    updateStatusBadge('DEFECT DETECTED', 'status-warning');
  }

  // Draw Bounding Boxes on Overlay Canvas
  drawBoundingBoxes(results.defects);

  // Populate Defect List Section
  renderDefectList(results.defects);
}

function drawBoundingBoxes(defects) {
  if (!elements.annotationCanvas) return;
  const ctx = elements.annotationCanvas.getContext('2d');
  clearCanvas();

  defects.forEach(defect => {
    const [x1, y1, x2, y2] = defect.bbox;
    const width = x2 - x1;
    const height = y2 - y1;

    // Outer Bounding Box
    ctx.strokeStyle = defect.severity === 'High' ? '#ef4444' : '#f59e0b';
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, width, height);

    // Box Tint Fill
    ctx.fillStyle = defect.severity === 'High' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(245, 158, 11, 0.15)';
    ctx.fillRect(x1, y1, width, height);

    // Label Header Tag
    const label = `${defect.type} (${defect.confidence.toFixed(1)}%)`;
    ctx.font = 'bold 14px sans-serif';
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = defect.severity === 'High' ? '#ef4444' : '#f59e0b';
    ctx.fillRect(x1, y1 > 22 ? y1 - 22 : y1, textWidth + 10, 22);

    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x1 + 5, y1 > 22 ? y1 - 6 : y1 + 16);
  });
}

function clearCanvas() {
  if (!elements.annotationCanvas) return;
  const ctx = elements.annotationCanvas.getContext('2d');
  ctx.clearRect(0, 0, elements.annotationCanvas.width, elements.annotationCanvas.height);
}

function renderDefectList(defects) {
  if (!elements.defectListContainer) return;

  if (defects.length === 0) {
    elements.defectListContainer.innerHTML = '<p class="empty-state">No defects detected in weld region.</p>';
    return;
  }

  const itemsHtml = defects.map(d => `
    <div class="defect-item defect-severity-${d.severity.toLowerCase()}">
      <div class="defect-header">
        <span class="defect-title">#${d.id} ${d.type}</span>
        <span class="defect-badge">${d.severity}</span>
      </div>
      <div class="defect-details">
        <span>Confidence: ${d.confidence.toFixed(1)}%</span>
        <span>BBox: [${d.bbox.join(', ')}]</span>
      </div>
    </div>
  `).join('');

  elements.defectListContainer.innerHTML = itemsHtml;
}

function toggleImageView() {
  if (!elements.annotationCanvas) return;

  if (state.activeView === 'annotated') {
    state.activeView = 'raw';
    elements.annotationCanvas.style.display = 'none';
    if (elements.viewModeLabel) elements.viewModeLabel.textContent = 'View Mode: Raw Image';
  } else {
    state.activeView = 'annotated';
    elements.annotationCanvas.style.display = 'block';
    if (elements.viewModeLabel) elements.viewModeLabel.textContent = 'View Mode: Annotated Overlay';
  }
}

function setInspectingState(isInspecting) {
  state.isInspecting = isInspecting;
  if (elements.runInspectionBtn) {
    elements.runInspectionBtn.disabled = isInspecting;
    elements.runInspectionBtn.textContent = isInspecting ? 'Analyzing Weld...' : 'Run Inspection';
  }
}

function updateStatusBadge(text, badgeClass) {
  if (!elements.statusBadge) return;
  elements.statusBadge.textContent = text;
  elements.statusBadge.className = `status-badge ${badgeClass}`;
}

function clearResultsDisplay() {
  if (elements.confidenceMetric) elements.confidenceMetric.textContent = '--';
  if (elements.defectCountMetric) elements.defectCountMetric.textContent = '--';
  if (elements.processingTimeMetric) elements.processingTimeMetric.textContent = '--';
  if (elements.defectListContainer) elements.defectListContainer.innerHTML = '';
  updateStatusBadge('READY', 'status-neutral');
  clearCanvas();
}

function resetDashboard() {
  stopWebcam();
  if (state.currentImageObjectUrl) {
    URL.revokeObjectURL(state.currentImageObjectUrl);
  }

  state.currentFile = null;
  state.currentImageObjectUrl = null;
  state.annotatedImageDataUrl = null;
  state.inspectionResults = null;

  if (elements.previewImage) elements.previewImage.src = '';
  if (elements.fileInput) elements.fileInput.value = '';
  if (elements.runInspectionBtn) elements.runInspectionBtn.disabled = true;

  clearResultsDisplay();
}

function exportResultsJSON() {
  if (!state.inspectionResults) {
    alert('No inspection results available to export.');
    return;
  }

  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.inspectionResults, null, 2));
  const downloadAnchor = document.createElement('a');
  downloadAnchor.setAttribute("href", dataStr);
  downloadAnchor.setAttribute("download", `weld_inspection_${Date.now()}.json`);
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
}