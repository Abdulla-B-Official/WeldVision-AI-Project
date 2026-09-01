/* ═══════════════════════════════════════════════════════════════════════════
   WeldVision AI — script.js
   Vanilla JS — industrial dashboard edition.
   All YOLO/Flask API logic preserved. UI updated for new layout.
   ═══════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── Constants ─────────────────────────────────────────────────────────────────
const STATUS_POLL_MS = 8000;
const WEBCAM_FPS     = 8;
const FRAME_INTERVAL = 1000 / WEBCAM_FPS;
const JPEG_QUALITY   = 0.75;

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  apiOnline:    false,
  modelLoaded:  false,
  activeTab:    'image',

  // Image inspection
  selectedFile: null,
  previewUrl:   null,
  annotatedUrl: null,
  detections:   [],
  count:        0,
  inferenceMs:  0,
  imgLoading:   false,
  imgConf:      0.50,
  showAnnotated: false,   // which view is active in toggle

  // Webcam
  webcamActive:     false,
  webcamStream:     null,
  webcamTimer:      null,
  webcamBusy:       false,
  webcamConf:       0.50,
  webcamAnnotated:  null,
  webcamDetections: [],
  webcamStats:      null,
  fpsFrames:        0,
  fpsTs:            Date.now(),

  // Status
  device:      '—',
  modelName:   'best.pt',
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const el = {};

// ══════════════════════════════════════════════════════════════════════════════
//  API helpers  (unchanged — same Flask endpoints)
// ══════════════════════════════════════════════════════════════════════════════

async function apiGetStatus() {
  const res = await fetch('/api/status');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function apiPredict(file, conf) {
  const form = new FormData();
  form.append('image', file);
  form.append('conf_threshold', String(conf));
  const res  = await fetch('/api/predict', { method: 'POST', body: form });
  const data = await res.json();
  if (!data.success) throw new Error(data.error || 'Prediction failed');
  return data;
}

async function apiWebcam(b64, conf) {
  const res  = await fetch('/api/webcam', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ frame: b64, conf_threshold: conf }),
  });
  const data = await res.json();
  if (!data.success) throw new Error(data.error || 'Webcam inference failed');
  return data;
}

// ══════════════════════════════════════════════════════════════════════════════
//  Status polling
// ══════════════════════════════════════════════════════════════════════════════

async function pollStatus() {
  try {
    const data      = await apiGetStatus();
    state.apiOnline   = true;
    state.modelLoaded = !!data.model_loaded;
    state.device      = data.device || '—';
    state.modelName   = data.model_name || 'best.pt';
  } catch {
    state.apiOnline   = false;
    state.modelLoaded = false;
  }
  renderStatusBadges();
  renderPerformanceCard();
}

function renderStatusBadges() {
  // ── Top-bar pills
  setPill(el.apiBadge,
    state.apiOnline ? 'green' : 'red',
    state.apiOnline ? 'API Online' : 'API Offline'
  );
  const mColor = !state.apiOnline ? 'muted' : state.modelLoaded ? 'green' : 'amber';
  const mLabel = !state.apiOnline ? 'Model Unknown' : state.modelLoaded ? 'Model Loaded' : 'Model Missing';
  setPill(el.modelBadge, mColor, mLabel);

  // ── Sidebar dots
  setDot(el.sysApiDot,   state.apiOnline   ? 'green' : 'red');
  setDot(el.sysModelDot, state.modelLoaded ? 'green' : (state.apiOnline ? 'amber' : ''));
  el.sysDeviceLabel.textContent = `Device: ${state.device.toUpperCase()}`;

  // ── Bottom status bar
  setBarDot(el.barApiDot,   state.apiOnline   ? 'green' : 'red');
  setBarDot(el.barModelDot, state.modelLoaded ? 'green' : (state.apiOnline ? 'amber' : ''));
  el.barApiText.textContent   = state.apiOnline   ? 'API Online'    : 'API Offline';
  el.barModelText.textContent = state.modelLoaded ? 'Model Loaded'  : 'Model Unknown';

  // ── Viewer status dot (idle unless active)
  if (!state.webcamActive) {
    el.viewerStatusDot.className = state.selectedFile ? 'viewer-status-dot active' : 'viewer-status-dot';
  }
}

function setPill(pillEl, colorClass, text) {
  pillEl.className = `status-pill ${colorClass}`;
  pillEl.querySelector('.pill-text').textContent = text;
}

function setDot(dotEl, colorClass) {
  dotEl.className = `sys-dot${colorClass ? ' ' + colorClass : ''}`;
}

function setBarDot(dotEl, colorClass) {
  dotEl.className = `sb-dot${colorClass ? ' ' + colorClass : ''}`;
}

function renderPerformanceCard() {
  el.perfDevice.textContent = state.device.toUpperCase();
  el.perfModel.textContent  = state.modelName;
  el.perfStatus.textContent = !state.apiOnline   ? 'Offline'
                            : state.modelLoaded  ? 'Loaded ✓'
                            :                      'Not Loaded';
  el.perfStatus.style.color = !state.apiOnline   ? 'var(--red)'
                            : state.modelLoaded  ? 'var(--green)'
                            :                      'var(--amber)';
}

// ══════════════════════════════════════════════════════════════════════════════
//  Tab switching
// ══════════════════════════════════════════════════════════════════════════════

function switchTab(tab) {
  if (tab !== 'webcam' && state.webcamActive) stopWebcam();
  state.activeTab = tab;

  const isImage  = tab === 'image';
  const isWebcam = tab === 'webcam';

  // Sidebar nav
  el.tabImage.classList.toggle('active', isImage);
  el.tabImage.setAttribute('aria-selected', isImage);
  el.tabWebcam.classList.toggle('active', isWebcam);
  el.tabWebcam.setAttribute('aria-selected', isWebcam);

  // Sidebar controls
  el.imageControls.classList.toggle('hidden', !isImage);
  el.webcamControls.classList.toggle('hidden', !isWebcam);

  // Panels
  el.panelImage.classList.toggle('hidden',  !isImage);
  el.panelWebcam.classList.toggle('hidden', !isWebcam);

  // Viewer header
  el.viewerModeTag.textContent = isImage ? 'IMAGE MODE' : 'WEBCAM MODE';

  // Right panel: reset to idle state on switch
  if (isImage) {
    renderImageVerdictAndResults();
    el.webcamStatsBody.classList.add('hidden');
    el.webcamDetectionCardsSection.classList.add('hidden');
    el.performanceCard.querySelector('.perf-list').classList.remove('hidden');
  } else {
    renderWebcamUI();
    el.performanceCard.querySelector('.perf-list').classList.add('hidden');
  }

  // Banners
  el.imgBanner.classList.add('hidden');
  el.webcamBanner.classList.add('hidden');
  renderSystemWarning();
}

// ══════════════════════════════════════════════════════════════════════════════
//  Image Inspection
// ══════════════════════════════════════════════════════════════════════════════

function handleFile(file) {
  if (!file) return;
  const allowed = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp'];
  if (!allowed.includes(file.type)) {
    showBanner(el.imgBanner, 'error', 'Unsupported format. Please upload JPG, PNG, WEBP, or BMP.');
    return;
  }
  state.selectedFile  = file;
  state.previewUrl    = URL.createObjectURL(file);
  state.annotatedUrl  = null;
  state.detections    = [];
  state.count         = 0;
  state.inferenceMs   = 0;
  state.showAnnotated = false;
  hideBanner(el.imgBanner);
  renderDropZone();
  renderImageViewer();
  renderImageVerdictAndResults();
  renderActionButtons();
}

function renderDropZone() {
  const hasFile = !!state.selectedFile;
  el.dropZone.classList.toggle('hidden', hasFile);
  el.imageViewerArea.classList.toggle('hidden', !hasFile);

  if (hasFile) {
    // Update viewer status dot
    el.viewerStatusDot.className = 'viewer-status-dot active';
  } else {
    el.viewerStatusDot.className = 'viewer-status-dot';
  }
}

function renderImageViewer() {
  if (!state.previewUrl) return;

  // Left panel: always show the original
  el.originalImage.src = state.previewUrl;

  if (state.annotatedUrl) {
    // Right panel: show annotated result, hide placeholder
    el.annotatedImage.src = state.annotatedUrl;
    el.annotatedImage.classList.remove('hidden');
    el.detectedPlaceholder.classList.add('hidden');
    el.imgVerdictBadge.classList.remove('hidden');
  } else {
    // Right panel: hide result, show placeholder
    el.annotatedImage.classList.add('hidden');
    el.detectedPlaceholder.classList.remove('hidden');
    el.imgVerdictBadge.classList.add('hidden');
  }
}

function showAnnotatedView(showAnnotated) {
  // Both images are now always visible side-by-side.
  // This function is kept for call-site compat but has no visual effect.
  state.showAnnotated = showAnnotated;
}

function renderActionButtons() {
  const canDetect = state.apiOnline && state.modelLoaded && !!state.selectedFile && !state.imgLoading;
  el.detectBtn.disabled = !canDetect;
  el.clearBtn.classList.toggle('hidden', !state.selectedFile);
}

async function runDetect() {
  if (!state.selectedFile || state.imgLoading) return;
  state.imgLoading = true;
  hideBanner(el.imgBanner);
  el.detectBtn.innerHTML = '<span class="spinner"></span> Inspecting…';
  el.detectBtn.disabled  = true;

  try {
    const result = await apiPredict(state.selectedFile, state.imgConf);
    state.annotatedUrl = `data:image/jpeg;base64,${result.annotated_image}`;
    state.detections   = result.detections || [];
    state.count        = result.count;
    state.inferenceMs  = result.inference_time_ms;
    renderImageViewer();
    renderImageVerdictAndResults();
  } catch (err) {
    showBanner(el.imgBanner, 'error', err.message || 'Detection failed. Is the backend running?');
  } finally {
    state.imgLoading = false;
    el.detectBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
      </svg> Inspect Weld`;
    renderActionButtons();
  }
}

function clearImage() {
  state.selectedFile  = null;
  state.previewUrl    = null;
  state.annotatedUrl  = null;
  state.detections    = [];
  state.count         = 0;
  state.inferenceMs   = 0;
  state.showAnnotated = false;
  el.fileInput.value  = '';
  hideBanner(el.imgBanner);
  renderDropZone();
  renderImageVerdictAndResults();
  renderActionButtons();
  el.viewerStatusDot.className = 'viewer-status-dot';
}

// ── Verdict + Results for Image Mode ─────────────────────────────────────────

function renderImageVerdictAndResults() {
  renderVerdictCard(state.detections, state.annotatedUrl, false);
  renderDetectionCards(el.detectionCards, state.detections);
  el.statCount.textContent    = state.count;
  el.statInference.textContent = state.annotatedUrl ? String(state.inferenceMs) : '—';
}

function renderVerdictCard(detections, resultUrl, isWebcam) {
  if (!resultUrl) {
    // Idle
    el.verdictCard.className        = 'verdict-card idle';
    el.verdictIcon.textContent       = '○';
    el.verdictLabel.textContent      = 'AWAITING';
    el.verdictConfDisplay.textContent = '—';
    el.verdictSub.textContent        = isWebcam ? 'Start camera to inspect' : 'Upload an image to inspect';
    el.imgVerdictBadge.textContent   = '';
    el.imgVerdictBadge.className     = 'img-verdict-badge hidden';
    return;
  }

  const hasDefective = detections.some(d => d.is_defective);
  const maxConf      = detections.length
    ? Math.max(...detections.map(d => d.confidence))
    : 0;

  if (detections.length === 0) {
    el.verdictCard.className         = 'verdict-card idle';
    el.verdictIcon.textContent        = '?';
    el.verdictLabel.textContent       = 'NO DETECTION';
    el.verdictConfDisplay.textContent = '—';
    el.verdictSub.textContent         = 'No objects above threshold';
    el.imgVerdictBadge.className      = 'img-verdict-badge hidden';
  } else if (hasDefective) {
    el.verdictCard.className         = 'verdict-card bad';
    el.verdictIcon.textContent        = '✕';
    el.verdictLabel.textContent       = 'BAD WELD';
    el.verdictConfDisplay.textContent = `${(maxConf * 100).toFixed(1)}%`;
    el.verdictSub.textContent         = `${detections.filter(d => d.is_defective).length} defect(s) detected`;
    if (!isWebcam) {
      el.imgVerdictBadge.textContent = 'BAD WELD';
      el.imgVerdictBadge.className   = 'img-verdict-badge bad';
    }
  } else {
    el.verdictCard.className         = 'verdict-card good';
    el.verdictIcon.textContent        = '✓';
    el.verdictLabel.textContent       = 'GOOD WELD';
    el.verdictConfDisplay.textContent = `${(maxConf * 100).toFixed(1)}%`;
    el.verdictSub.textContent         = `${detections.length} weld(s) verified`;
    if (!isWebcam) {
      el.imgVerdictBadge.textContent = 'GOOD WELD';
      el.imgVerdictBadge.className   = 'img-verdict-badge good';
    }
  }
}

function renderDetectionCards(container, detections) {
  if (!detections || detections.length === 0) {
    container.innerHTML = '<div class="det-empty">No detections yet</div>';
    return;
  }
  container.innerHTML = detections.map((det, i) => {
    const type    = det.is_defective ? 'bad' : 'good';
    const icon    = det.is_defective ? '🔴' : '🟢';
    const pct     = (det.confidence * 100).toFixed(1);
    const bboxStr = det.bbox
      ? `x1:${det.bbox.x1} y1:${det.bbox.y1} x2:${det.bbox.x2} y2:${det.bbox.y2}`
      : '';
    return `
      <div class="det-item ${type}">
        <span class="det-icon">${icon}</span>
        <div class="det-info">
          <div class="det-class">${det.class_name}</div>
          ${bboxStr ? `<div class="det-bbox">#${i+1} · ${bboxStr}</div>` : ''}
        </div>
        <span class="det-conf-badge ${type}">${pct}%</span>
      </div>`;
  }).join('');
}

// ── Confidence Slider ─────────────────────────────────────────────────────────

function updateImgConf(val) {
  state.imgConf = parseFloat(val);
  el.imgConfValue.textContent = `${Math.round(state.imgConf * 100)}%`;
  const pct = ((state.imgConf - 0.05) / 0.90 * 100).toFixed(1);
  el.imgConfSlider.style.setProperty('--val', `${pct}%`);
}

function updateWebcamConf(val) {
  state.webcamConf = parseFloat(val);
  el.webcamConfValue.textContent = `${Math.round(state.webcamConf * 100)}%`;
  const pct = ((state.webcamConf - 0.05) / 0.90 * 100).toFixed(1);
  el.webcamConfSlider.style.setProperty('--val', `${pct}%`);
}

// ── Banners ───────────────────────────────────────────────────────────────────

function showBanner(bannerEl, type, msg) {
  bannerEl.className = `alert-banner ${type}`;
  bannerEl.innerHTML = msg;
  bannerEl.classList.remove('hidden');
}
function hideBanner(bannerEl) { bannerEl.classList.add('hidden'); }

function renderSystemWarning() {
  const banner = state.activeTab === 'image' ? el.imgBanner : el.webcamBanner;
  if (!state.apiOnline) {
    showBanner(banner, 'warning', '🔌 Backend offline — run: <code style="font-family:monospace">python app.py</code>');
  } else if (!state.modelLoaded) {
    showBanner(banner, 'warning', '🤖 Model not loaded — check <code style="font-family:monospace">runs/weld_yolov8m/weights/best.pt</code>');
  } else {
    hideBanner(banner);
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Webcam
// ══════════════════════════════════════════════════════════════════════════════

async function startWebcam() {
  hideBanner(el.webcamBanner);
  state.webcamBusy       = false;
  state.fpsFrames        = 0;
  state.fpsTs            = Date.now();
  state.webcamStats      = null;
  state.webcamDetections = [];
  state.webcamAnnotated  = null;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    state.webcamStream = stream;
    el.webcamVideo.srcObject = stream;
    el.webcamVideo.classList.remove('hidden');
    await el.webcamVideo.play();

    state.webcamActive = true;
    renderWebcamUI();
    startDetectionLoop();
  } catch (err) {
    let msg = `Camera error: ${err.message}`;
    if (err.name === 'NotAllowedError')  msg = '⚠️ Camera permission denied. Allow access in browser settings.';
    if (err.name === 'NotFoundError')    msg = '⚠️ No camera found. Connect a camera and retry.';
    showBanner(el.webcamBanner, 'error', msg);
  }
}

function stopWebcam() {
  if (state.webcamTimer) { clearInterval(state.webcamTimer); state.webcamTimer = null; }
  if (state.webcamStream) { state.webcamStream.getTracks().forEach(t => t.stop()); state.webcamStream = null; }
  el.webcamVideo.srcObject = null;
  el.webcamVideo.classList.add('hidden');
  el.webcamAnnotated.classList.add('hidden');

  state.webcamActive     = false;
  state.webcamBusy       = false;
  state.webcamAnnotated  = null;
  state.webcamDetections = [];
  state.webcamStats      = null;

  renderWebcamUI();
  // Reset verdict to idle
  renderVerdictCard([], null, true);
}

function startDetectionLoop() {
  state.webcamTimer = setInterval(async () => {
    if (state.webcamBusy) return;
    if (!el.webcamVideo || el.webcamVideo.readyState < 2) return;

    const canvas     = el.captureCanvas;
    canvas.width     = el.webcamVideo.videoWidth  || 640;
    canvas.height    = el.webcamVideo.videoHeight || 480;
    canvas.getContext('2d').drawImage(el.webcamVideo, 0, 0);
    const b64 = canvas.toDataURL('image/jpeg', JPEG_QUALITY);

    state.webcamBusy = true;
    try {
      const result = await apiWebcam(b64, state.webcamConf);

      // FPS calc
      state.fpsFrames++;
      const elapsed = (Date.now() - state.fpsTs) / 1000;
      const fps     = elapsed > 0 ? (state.fpsFrames / elapsed).toFixed(1) : '—';
      if (elapsed > 4) { state.fpsFrames = 0; state.fpsTs = Date.now(); }

      state.webcamAnnotated  = `data:image/jpeg;base64,${result.annotated_image}`;
      state.webcamDetections = result.detections || [];
      state.webcamStats = { count: result.count, inferenceMs: result.inference_time_ms, fps };

      renderWebcamFrame();
      renderWebcamLiveResults();
    } catch { /* skip failed frame silently */ }
    finally { state.webcamBusy = false; }
  }, FRAME_INTERVAL);
}

function renderWebcamUI() {
  const active = state.webcamActive;

  // Idle overlay / live badge / fps badge
  el.webcamIdle.classList.toggle('hidden', active);
  el.webcamLiveBadge.classList.toggle('hidden', !active);
  if (!active) el.webcamFpsBadge.classList.add('hidden');

  // Buttons
  el.startWebcamBtn.classList.toggle('hidden', active);
  el.stopWebcamBtn.classList.toggle('hidden',  !active);

  // Conf slider lock
  el.webcamConfSlider.disabled = active;
  el.webcamSliderHint.classList.toggle('hidden', !active);

  // Bottom bar camera status
  setBarDot(el.barCameraDot, active ? 'green' : '');
  el.barCameraText.textContent = active ? 'Camera Live' : 'Camera Idle';

  // Viewer status dot
  el.viewerStatusDot.className = active ? 'viewer-status-dot live' : 'viewer-status-dot';

  // Right panel: show/hide sections
  if (active) {
    el.performanceCard.querySelector('.perf-list').classList.add('hidden');
    el.webcamStatsBody.classList.remove('hidden');
    renderWebcamLiveResults();
  } else {
    el.performanceCard.querySelector('.perf-list').classList.remove('hidden');
    el.webcamStatsBody.classList.add('hidden');
    el.webcamDetectionCardsSection.classList.add('hidden');
    el.webcamStatsBody.innerHTML = `
      <div class="det-empty" style="padding:16px 0">
        <span style="opacity:0.5">📡 Camera not active</span>
      </div>`;
  }
}

function renderWebcamFrame() {
  if (state.webcamAnnotated) {
    el.webcamAnnotated.src = state.webcamAnnotated;
    el.webcamAnnotated.classList.remove('hidden');
  }
  if (state.webcamStats) {
    el.webcamFpsBadge.textContent = `${state.webcamStats.fps} FPS`;
    el.webcamFpsBadge.classList.remove('hidden');
  }
}

function renderWebcamLiveResults() {
  if (!state.webcamStats) return;
  const s = state.webcamStats;

  // Verdict
  renderVerdictCard(state.webcamDetections, state.webcamAnnotated, true);

  // Stats body inside performance card
  const hasDefective  = state.webcamDetections.some(d => d.is_defective);
  const verdictColor  = hasDefective ? 'var(--red)' : 'var(--green)';
  const verdictText   = state.webcamDetections.length === 0 ? '—'
                      : hasDefective ? 'Defective' : 'Good';

  el.webcamStatsBody.innerHTML = `
    <div class="wls-row">
      <span class="wls-label">🎯 Objects</span>
      <span class="wls-value">${s.count}</span>
    </div>
    <div class="wls-row">
      <span class="wls-label">⚡ Inference</span>
      <span class="wls-value">${s.inferenceMs} ms</span>
    </div>
    <div class="wls-row">
      <span class="wls-label">🎞️ FPS</span>
      <span class="wls-value">${s.fps}</span>
    </div>
    ${state.webcamDetections.length > 0 ? `
    <div class="wls-row">
      <span class="wls-label">📋 Verdict</span>
      <span class="wls-value" style="color:${verdictColor}">${verdictText}</span>
    </div>` : ''}
  `;

  // Detection cards
  if (state.webcamDetections.length > 0) {
    el.webcamDetectionCardsSection.classList.remove('hidden');
    renderDetectionCards(el.webcamDetectionCards, state.webcamDetections);
  } else {
    el.webcamDetectionCardsSection.classList.add('hidden');
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Init
// ══════════════════════════════════════════════════════════════════════════════

function init() {
  // ── Cache DOM refs ─────────────────────────────────────────────────────────
  // Header
  el.apiBadge   = document.getElementById('api-badge');
  el.modelBadge = document.getElementById('model-badge');

  // Tabs
  el.tabImage  = document.getElementById('tab-image');
  el.tabWebcam = document.getElementById('tab-webcam');

  // Sidebar controls
  el.imageControls  = document.getElementById('image-controls');
  el.webcamControls = document.getElementById('webcam-controls');
  el.uploadBtn      = document.getElementById('upload-btn');
  el.fileInput      = document.getElementById('file-input');
  el.detectBtn      = document.getElementById('detect-btn');
  el.clearBtn       = document.getElementById('clear-btn');
  el.imgConfSlider  = document.getElementById('img-conf-slider');
  el.imgConfValue   = document.getElementById('img-conf-value');
  el.startWebcamBtn = document.getElementById('start-webcam-btn');
  el.stopWebcamBtn  = document.getElementById('stop-webcam-btn');
  el.webcamConfSlider = document.getElementById('webcam-conf-slider');
  el.webcamConfValue  = document.getElementById('webcam-conf-value');
  el.webcamSliderHint = document.getElementById('webcam-slider-hint');

  // Sidebar system
  el.sysApiDot     = document.getElementById('sys-api-dot');
  el.sysModelDot   = document.getElementById('sys-model-dot');
  el.sysDeviceLabel= document.getElementById('sys-device-label');

  // Viewer
  el.panelImage       = document.getElementById('panel-image');
  el.panelWebcam      = document.getElementById('panel-webcam');
  el.viewerStatusDot  = document.getElementById('viewer-status-dot');
  el.viewerTitle      = document.getElementById('viewer-title');
  el.viewerModeTag    = document.getElementById('viewer-mode-tag');
  el.imgBanner        = document.getElementById('img-banner');
  el.webcamBanner     = document.getElementById('webcam-banner');

  // Image viewer
  el.dropZone          = document.getElementById('drop-zone');
  el.imageViewerArea   = document.getElementById('image-viewer-area');
  el.originalImage     = document.getElementById('original-image');
  el.annotatedImage    = document.getElementById('annotated-image');
  el.detectedPlaceholder = document.getElementById('detected-placeholder');
  el.viewToggle        = document.getElementById('view-toggle');
  el.showOriginalBtn   = document.getElementById('show-original-btn');
  el.showAnnotatedBtn  = document.getElementById('show-annotated-btn');
  el.imgVerdictBadge   = document.getElementById('img-verdict-badge');

  // Webcam viewer
  el.webcamVideo      = document.getElementById('webcam-video');
  el.webcamAnnotated  = document.getElementById('webcam-annotated');
  el.webcamIdle       = document.getElementById('webcam-idle');
  el.webcamLiveBadge  = document.getElementById('webcam-live-badge');
  el.webcamFpsBadge   = document.getElementById('webcam-fps-badge');

  // Right panel
  el.verdictCard        = document.getElementById('verdict-card');
  el.verdictIcon        = document.getElementById('verdict-icon');
  el.verdictLabel       = document.getElementById('verdict-label');
  el.verdictConfDisplay = document.getElementById('verdict-conf-display');
  el.verdictSub         = document.getElementById('verdict-sub');
  el.statCount          = document.getElementById('stat-count');
  el.statInference      = document.getElementById('stat-inference');
  el.detectionCards     = document.getElementById('detection-cards');
  el.performanceCard    = document.getElementById('performance-card');
  el.perfDevice         = document.getElementById('perf-device');
  el.perfModel          = document.getElementById('perf-model');
  el.perfStatus         = document.getElementById('perf-status');
  el.webcamStatsBody              = document.getElementById('webcam-stats-body');
  el.webcamDetectionCardsSection  = document.getElementById('webcam-detection-cards-section');
  el.webcamDetectionCards         = document.getElementById('webcam-detection-cards');

  // Bottom bar
  el.barApiDot    = document.getElementById('bar-api-dot');
  el.barModelDot  = document.getElementById('bar-model-dot');
  el.barCameraDot = document.getElementById('bar-camera-dot');
  el.barApiText   = document.getElementById('bar-api-text');
  el.barModelText = document.getElementById('bar-model-text');
  el.barCameraText= document.getElementById('bar-camera-text');

  // Canvas
  el.captureCanvas = document.getElementById('capture-canvas');

  // ── Event Listeners ────────────────────────────────────────────────────────
  el.tabImage.addEventListener('click',  () => switchTab('image'));
  el.tabWebcam.addEventListener('click', () => switchTab('webcam'));

  // Upload button & file input
  el.uploadBtn.addEventListener('click', () => el.fileInput.click());
  el.fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

  // Drop zone drag & drop
  el.dropZone.addEventListener('dragover',  e => { e.preventDefault(); el.dropZone.classList.add('dragging'); });
  el.dropZone.addEventListener('dragleave', ()  => el.dropZone.classList.remove('dragging'));
  el.dropZone.addEventListener('drop', e => {
    e.preventDefault(); el.dropZone.classList.remove('dragging');
    handleFile(e.dataTransfer.files[0]);
  });
  el.dropZone.addEventListener('click',   () => el.fileInput.click());
  el.dropZone.addEventListener('keydown', e => { if (e.key === 'Enter') el.fileInput.click(); });

  // Detect / clear
  el.detectBtn.addEventListener('click', runDetect);
  el.clearBtn.addEventListener('click',  clearImage);

  // Image conf slider
  el.imgConfSlider.addEventListener('input', e => updateImgConf(e.target.value));
  updateImgConf(el.imgConfSlider.value);

  // Webcam
  el.startWebcamBtn.addEventListener('click', startWebcam);
  el.stopWebcamBtn.addEventListener('click',  stopWebcam);
  el.webcamConfSlider.addEventListener('input', e => updateWebcamConf(e.target.value));
  updateWebcamConf(el.webcamConfSlider.value);

  // View toggle (original vs annotated)
  el.showOriginalBtn.addEventListener('click',  () => showAnnotatedView(false));
  el.showAnnotatedBtn.addEventListener('click', () => showAnnotatedView(true));

  // ── Initial render ─────────────────────────────────────────────────────────
  renderDropZone();
  renderActionButtons();
  renderVerdictCard([], null, false);
  renderWebcamUI();

  // ── Status polling ─────────────────────────────────────────────────────────
  pollStatus().then(renderSystemWarning);
  setInterval(() => {
    pollStatus().then(() => {
      renderSystemWarning();
      renderActionButtons();
    });
  }, STATUS_POLL_MS);
}

document.addEventListener('DOMContentLoaded', init);
