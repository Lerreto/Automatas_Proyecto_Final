'use strict';

// ─── Constants ───────────────────────────────────────────────────────────────
const MODEL_SIZE   = 100;
const DISPLAY_SIZE = 400;
const UNDO_LIMIT   = 25;
const EPSILON      = 5e-4;  // debe coincidir con backend (solo RGB)

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const imageCanvas   = document.getElementById('imageCanvas');
const maskCanvas    = document.getElementById('maskCanvas');
const resultCanvas  = document.getElementById('resultCanvas');
const imgCtx        = imageCanvas.getContext('2d');
const maskCtx       = maskCanvas.getContext('2d');
const resultCtx     = resultCanvas.getContext('2d');

const fileInput     = document.getElementById('fileInput');
const btnLoad       = document.getElementById('btnLoad');
const btnReconstruct= document.getElementById('btnReconstruct');
const btnDownload   = document.getElementById('btnDownload');
const btnUndo       = document.getElementById('btnUndo');
const btnRedo       = document.getElementById('btnRedo');
const btnClear      = document.getElementById('btnClear');
const toolBrush     = document.getElementById('toolBrush');
const toolEraser    = document.getElementById('toolEraser');
const btnPlay       = document.getElementById('btnPlay');
const iconPlay      = document.getElementById('iconPlay');
const iconPause     = document.getElementById('iconPause');
const frameSlider   = document.getElementById('frameSlider');
const fpsSlider     = document.getElementById('fpsSlider');
const stepsSlider   = document.getElementById('stepsSlider');
const brushSize     = document.getElementById('brushSize');
const brushOpacity  = document.getElementById('brushOpacity');
const brushSizeVal  = document.getElementById('brushSizeVal');
const brushOpacityVal = document.getElementById('brushOpacityVal');
const fpsVal        = document.getElementById('fpsVal');
const stepsVal      = document.getElementById('stepsVal');
const cursorCoords  = document.getElementById('cursorCoords');
const stepCounter   = document.getElementById('stepCounter');
const statusDot     = document.getElementById('statusDot');
const metricDevice  = document.getElementById('metricDevice');
const emptyHint     = document.getElementById('emptyHint');
const resultHint    = document.getElementById('resultHint');
const logEl         = document.getElementById('log');
const vPsnr         = document.getElementById('vPsnr');
const vSsim         = document.getElementById('vSsim');
const vConvStep     = document.getElementById('vConvStep');
const btnCompare      = document.getElementById('btnCompare');
const btnCloseCompare = document.getElementById('btnCloseCompare');
const compareSection  = document.getElementById('compareSection');
const comparePeakStep = document.getElementById('comparePeakStep');
const compareMeta     = document.getElementById('compareMeta');
const originalCanvas  = document.getElementById('originalCanvas');
const peakCanvas      = document.getElementById('peakCanvas');
const originalCtx     = originalCanvas.getContext('2d');
const peakCtx         = peakCanvas.getContext('2d');

// ─── State ────────────────────────────────────────────────────────────────────
let chartConv    = null;
let chartQuality = null;

let currentTool    = 'brush';
let isDrawing      = false;
let lastX = 0, lastY = 0;
let hasImage       = false;
let undoStack      = [];
let redoStack      = [];
let frames         = [];
let currentFrame   = 0;
let isPlaying      = false;
let rafId          = null;
let lastFrameTime  = 0;
let peakData       = null;

// ─── Charts ───────────────────────────────────────────────────────────────────
function initCharts() {
  Chart.defaults.color       = '#5a6478';
  Chart.defaults.borderColor = '#1e2535';
  Chart.defaults.font.family = "'JetBrains Mono', monospace";
  Chart.defaults.font.size   = 10;

  const baseDataset = { tension: 0.3, pointRadius: 0, borderWidth: 1.5 };
  const baseScaleX  = { ticks: { maxTicksLimit: 8, color: '#5a6478' }, grid: { color: '#1e2535' } };

  chartConv = new Chart(document.getElementById('chartConvergence'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          ...baseDataset,
          label: 'convergencia ||s_t − s_{t−1}|| (izq.)',
          data: [],
          borderColor: '#f5a623',
          backgroundColor: 'rgba(245,166,35,0.07)',
          fill: true,
          yAxisID: 'y',
        },
        {
          ...baseDataset,
          label: 'células activas % (der.)',
          data: [],
          borderColor: '#5a6478',
          borderDash: [4, 3],
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { boxWidth: 10, padding: 10, font: { size: 9 } } },
      },
      scales: {
        x: baseScaleX,
        y: {
          position: 'left',
          grid: { color: '#1e2535' },
          ticks: { color: '#5a6478' },
          title: { display: true, text: 'convergencia', color: '#5a6478', font: { size: 9 } },
        },
        y1: {
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#5a6478' },
          title: { display: true, text: 'activas %', color: '#5a6478', font: { size: 9 } },
          min: 0,
          max: 100,
        },
      },
    },
  });

  chartQuality = new Chart(document.getElementById('chartQuality'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          ...baseDataset,
          label: 'PSNR zona hueco (dB) (izq.)',
          data: [],
          borderColor: '#7c6fcd',
          yAxisID: 'y',
        },
        {
          ...baseDataset,
          label: 'SSIM zona hueco ×40 (der.)',
          data: [],
          borderColor: '#5ec8c0',
          borderDash: [4, 3],
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { boxWidth: 10, padding: 10, font: { size: 9 } } },
      },
      scales: {
        x: baseScaleX,
        y: {
          position: 'left',
          grid: { color: '#1e2535' },
          ticks: { color: '#5a6478' },
          title: { display: true, text: 'PSNR (dB)', color: '#5a6478', font: { size: 9 } },
        },
        y1: {
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#5a6478' },
          title: { display: true, text: 'SSIM ×40', color: '#5a6478', font: { size: 9 } },
          min: 0,
          max: 40,
        },
      },
    },
  });
}

function updateCharts({ convergence, activity, psnr_hole, ssim_hole }, peak) {
  const labels = convergence.map((_, i) => i + 1);

  chartConv.data.labels           = labels;
  chartConv.data.datasets[0].data = convergence;
  chartConv.data.datasets[1].data = activity;
  chartConv.update('none');

  chartQuality.data.labels           = labels;
  chartQuality.data.datasets[0].data = psnr_hole;
  chartQuality.data.datasets[1].data = ssim_hole.map(v => +(v * 40).toFixed(3));
  chartQuality.update('none');

  vPsnr.textContent     = peak.psnr.toFixed(1) + ' dB';
  vSsim.textContent     = peak.ssim.toFixed(3);
  vConvStep.textContent = 'paso ' + peak.step;
}

// ─── Log helper ───────────────────────────────────────────────────────────────
function log(msg, cls = '') {
  const line = document.createElement('div');
  line.className = 'log-line' + (cls ? ' ' + cls : '');
  line.textContent = '› ' + msg;
  logEl.appendChild(line);
  // keep only last 20 lines
  while (logEl.children.length > 20) logEl.removeChild(logEl.firstChild);
  logEl.scrollTop = logEl.scrollHeight;
}

// ─── Status dot ───────────────────────────────────────────────────────────────
function setStatus(state) {
  statusDot.className = 'status-dot ' + state;
  statusDot.title = state;
}

// ─── Health check ─────────────────────────────────────────────────────────────
async function healthCheck() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error('non-200');
    const data = await res.json();
    metricDevice.textContent = data.device;
    setStatus('ready');
    log(`backend ready · device ${data.device}`, 'ok');
  } catch {
    setStatus('error');
    log('backend unreachable', 'err');
  }
}

// ─── Image load ───────────────────────────────────────────────────────────────
btnLoad.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    // Center-crop to square
    const size = Math.min(img.width, img.height);
    const sx = (img.width  - size) / 2;
    const sy = (img.height - size) / 2;
    imgCtx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    imgCtx.drawImage(img, sx, sy, size, size, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    URL.revokeObjectURL(url);
    clearMask();
    hasImage = true;
    emptyHint.classList.add('hidden');
    btnReconstruct.disabled = false;
    btnCompare.disabled = true;
    peakData = null;
    compareSection.classList.add('hidden');
    log(`specimen loaded · ${img.width}×${img.height} → center-crop ${size}×${size}`, 'ok');
    fileInput.value = '';
  };
  img.src = url;
});

// ─── Mask painting ───────────────────────────────────────────────────────────
function canvasPos(e) {
  const rect = maskCanvas.getBoundingClientRect();
  const scaleX = DISPLAY_SIZE / rect.width;
  const scaleY = DISPLAY_SIZE / rect.height;
  const cx = (e.clientX - rect.left) * scaleX;
  const cy = (e.clientY - rect.top)  * scaleY;
  return { cx, cy };
}

function setupStroke() {
  maskCtx.lineCap   = 'round';
  maskCtx.lineJoin  = 'round';
  const size = parseInt(brushSize.value);
  maskCtx.lineWidth = size;
  if (currentTool === 'brush') {
    const opacity = parseInt(brushOpacity.value) / 100;
    maskCtx.globalCompositeOperation = 'source-over';
    maskCtx.strokeStyle = `rgba(0,0,0,${opacity})`;
    maskCtx.fillStyle   = `rgba(0,0,0,${opacity})`;
  } else {
    maskCtx.globalCompositeOperation = 'destination-out';
    maskCtx.strokeStyle = 'rgba(0,0,0,1)';
    maskCtx.fillStyle   = 'rgba(0,0,0,1)';
  }
}

function drawDot(cx, cy) {
  setupStroke();
  maskCtx.beginPath();
  maskCtx.arc(cx, cy, parseInt(brushSize.value) / 2, 0, Math.PI * 2);
  maskCtx.fill();
}

function drawLine(x1, y1, x2, y2) {
  setupStroke();
  maskCtx.beginPath();
  maskCtx.moveTo(x1, y1);
  maskCtx.lineTo(x2, y2);
  maskCtx.stroke();
}

function getEventPos(e) {
  if (e.touches) {
    return { clientX: e.touches[0].clientX, clientY: e.touches[0].clientY };
  }
  return { clientX: e.clientX, clientY: e.clientY };
}

maskCanvas.addEventListener('mousedown',  startDraw);
maskCanvas.addEventListener('mousemove',  doDraw);
maskCanvas.addEventListener('mouseup',    endDraw);
maskCanvas.addEventListener('mouseleave', endDraw);
maskCanvas.addEventListener('touchstart', e => { e.preventDefault(); startDraw(e); }, { passive: false });
maskCanvas.addEventListener('touchmove',  e => { e.preventDefault(); doDraw(e); },   { passive: false });
maskCanvas.addEventListener('touchend',   e => { e.preventDefault(); endDraw(e); },  { passive: false });

maskCanvas.addEventListener('mousemove', e => {
  if (!hasImage) return;
  const { cx, cy } = canvasPos(e);
  const mx = Math.floor(cx / DISPLAY_SIZE * MODEL_SIZE);
  const my = Math.floor(cy / DISPLAY_SIZE * MODEL_SIZE);
  cursorCoords.textContent = `${mx.toString().padStart(3,' ')} · ${my.toString().padStart(3,' ')}`;
});

maskCanvas.addEventListener('mouseleave', () => {
  cursorCoords.textContent = '—';
});

function startDraw(e) {
  if (!hasImage) return;
  isDrawing = true;
  saveUndo();
  const { clientX, clientY } = getEventPos(e);
  const { cx, cy } = canvasPos({ clientX, clientY });
  lastX = cx; lastY = cy;
  drawDot(cx, cy);
}

function doDraw(e) {
  if (!isDrawing) return;
  const { clientX, clientY } = getEventPos(e);
  const { cx, cy } = canvasPos({ clientX, clientY });
  drawLine(lastX, lastY, cx, cy);
  lastX = cx; lastY = cy;
}

function endDraw() {
  isDrawing = false;
  maskCtx.globalCompositeOperation = 'source-over';
}

// ─── Undo / Redo ──────────────────────────────────────────────────────────────
function saveUndo() {
  undoStack.push(maskCtx.getImageData(0, 0, DISPLAY_SIZE, DISPLAY_SIZE));
  if (undoStack.length > UNDO_LIMIT) undoStack.shift();
  redoStack = [];
}

function undo() {
  if (!undoStack.length) return;
  redoStack.push(maskCtx.getImageData(0, 0, DISPLAY_SIZE, DISPLAY_SIZE));
  const state = undoStack.pop();
  maskCtx.putImageData(state, 0, 0);
}

function redo() {
  if (!redoStack.length) return;
  undoStack.push(maskCtx.getImageData(0, 0, DISPLAY_SIZE, DISPLAY_SIZE));
  const state = redoStack.pop();
  maskCtx.putImageData(state, 0, 0);
}

function clearMask() {
  if (undoStack.length || maskCtx.getImageData(0,0,1,1).data[3] > 0) saveUndo();
  maskCtx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
}

btnUndo.addEventListener('click', undo);
btnRedo.addEventListener('click', redo);
btnClear.addEventListener('click', clearMask);

// ─── Tool selection ───────────────────────────────────────────────────────────
function selectTool(tool) {
  currentTool = tool;
  toolBrush.classList.toggle('active', tool === 'brush');
  toolEraser.classList.toggle('active', tool === 'eraser');
}

toolBrush.addEventListener('click', () => selectTool('brush'));
toolEraser.addEventListener('click', () => selectTool('eraser'));

// ─── Slider display ───────────────────────────────────────────────────────────
brushSize.addEventListener('input',    () => brushSizeVal.textContent = brushSize.value);
brushOpacity.addEventListener('input', () => brushOpacityVal.textContent = brushOpacity.value + '%');
fpsSlider.addEventListener('input',    () => fpsVal.textContent = fpsSlider.value);
stepsSlider.addEventListener('input',  () => stepsVal.textContent = stepsSlider.value);

// ─── Extract mask & image at model resolution ─────────────────────────────────
function extractForModel() {
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = MODEL_SIZE; tmpCanvas.height = MODEL_SIZE;
  const tmpCtx = tmpCanvas.getContext('2d');

  // --- original (ground truth antes de pintar) ---
  tmpCtx.clearRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  tmpCtx.drawImage(imageCanvas, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const imgData  = tmpCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const origData = new Uint8ClampedArray(imgData.data);  // copia sin modificar

  // --- mask ---
  tmpCtx.clearRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  tmpCtx.drawImage(maskCanvas, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const maskData = tmpCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);

  // Build corrupted image (pixels painted → black) and binary mask
  const outImg  = new Uint8ClampedArray(imgData.data);
  const outMask = new Uint8ClampedArray(MODEL_SIZE * MODEL_SIZE * 4);

  for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
    const p = i * 4;
    const alpha = maskData.data[p + 3];
    if (alpha > 0) {
      outImg[p] = 0; outImg[p+1] = 0; outImg[p+2] = 0; outImg[p+3] = 255;
      outMask[p] = 0; outMask[p+1] = 0; outMask[p+2] = 0; outMask[p+3] = 255;
    } else {
      outMask[p] = 255; outMask[p+1] = 255; outMask[p+2] = 255; outMask[p+3] = 255;
    }
  }

  function toB64(imageData) {
    const c = document.createElement('canvas');
    c.width = MODEL_SIZE; c.height = MODEL_SIZE;
    c.getContext('2d').putImageData(imageData, 0, 0);
    return c.toDataURL('image/png');
  }

  return {
    image:    toB64(new ImageData(outImg,  MODEL_SIZE, MODEL_SIZE)),
    original: toB64(new ImageData(origData, MODEL_SIZE, MODEL_SIZE)),
    mask:     toB64(new ImageData(outMask, MODEL_SIZE, MODEL_SIZE)),
  };
}

// ─── Reconstruct ─────────────────────────────────────────────────────────────
btnReconstruct.addEventListener('click', async () => {
  if (!hasImage) return;
  const maskPixels = maskCtx.getImageData(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
  const hasMask = maskPixels.data.some((v, i) => i % 4 === 3 && v > 0);
  if (!hasMask) {
    log('no mask painted — draw on the image first', 'warn');
    return;
  }

  stopPlayback();
  frames = [];
  peakData = null;
  compareSection.classList.add('hidden');
  btnCompare.disabled = true;
  setStatus('working');
  btnReconstruct.disabled = true;
  btnReconstruct.classList.add('working');
  btnReconstruct.textContent = 'running…';
  resultHint.classList.add('hidden');
  log(`starting reconstruction · steps ${stepsSlider.value}`);

  const { image, original, mask } = extractForModel();
  const steps = parseInt(stepsSlider.value);
  const sample_every = Math.max(1, Math.floor(steps / 60));

  try {
    const t0 = performance.now();
    const res = await fetch('/api/reconstruct', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image, original, mask, steps, sample_every }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    const data = await res.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

    frames   = data.frames;
    peakData = data.peak;
    updateCharts(data.metrics, peakData);
    log(`${data.count} frames · ${data.steps} steps · ${elapsed}s · ${data.device}`, 'ok');
    log(`pico SSIM ${peakData.ssim.toFixed(3)} en paso ${peakData.step}`, 'ok');

    // Set up playback UI
    frameSlider.max = frames.length - 1;
    frameSlider.value = 0;
    frameSlider.disabled = false;
    btnPlay.disabled = false;
    btnDownload.disabled = false;
    btnCompare.disabled = false;
    setStatus('ready');

    showFrame(0);
    startPlayback();

  } catch (err) {
    log(`error: ${err.message}`, 'err');
    setStatus('error');
  } finally {
    btnReconstruct.disabled = false;
    btnReconstruct.classList.remove('working');
    btnReconstruct.textContent = 'reconstruct';
  }
});

// ─── Playback ─────────────────────────────────────────────────────────────────
function showFrame(idx) {
  if (!frames.length) return;
  idx = Math.max(0, Math.min(idx, frames.length - 1));
  currentFrame = idx;
  frameSlider.value = idx;
  stepCounter.textContent = `frame ${idx + 1}/${frames.length}`;

  const img = new Image();
  img.onload = () => {
    resultCtx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    resultCtx.drawImage(img, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
  };
  img.src = 'data:image/png;base64,' + frames[idx];
}

function startPlayback() {
  if (!frames.length) return;
  isPlaying = true;
  iconPlay.style.display  = 'none';
  iconPause.style.display = '';
  lastFrameTime = 0;
  rafId = requestAnimationFrame(playStep);
}

function stopPlayback() {
  isPlaying = false;
  iconPlay.style.display  = '';
  iconPause.style.display = 'none';
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
}

function playStep(ts) {
  if (!isPlaying) return;
  const fps = parseInt(fpsSlider.value);
  const interval = 1000 / fps;
  if (ts - lastFrameTime >= interval) {
    currentFrame = (currentFrame + 1) % frames.length;
    showFrame(currentFrame);
    lastFrameTime = ts;
  }
  rafId = requestAnimationFrame(playStep);
}

btnPlay.addEventListener('click', () => {
  if (isPlaying) stopPlayback();
  else startPlayback();
});

frameSlider.addEventListener('input', () => {
  stopPlayback();
  showFrame(parseInt(frameSlider.value));
});

// ─── Download ────────────────────────────────────────────────────────────────
btnDownload.addEventListener('click', () => {
  if (!frames.length) return;
  const a = document.createElement('a');
  a.href = resultCanvas.toDataURL('image/png');
  a.download = `nca_frame_${String(currentFrame + 1).padStart(3, '0')}.png`;
  a.click();
  log(`saved frame ${currentFrame + 1}`, 'ok');
});

// ─── Keyboard shortcuts ───────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  const tag = document.activeElement.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;

  if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) { e.preventDefault(); undo(); }
  if ((e.key === 'Z' || e.key === 'z') && (e.ctrlKey || e.metaKey) && e.shiftKey) { e.preventDefault(); redo(); }
  if (e.key === 'y' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); redo(); }
  if (e.key === 'Delete' || e.key === 'Backspace') clearMask();
  if (e.key === 'b' || e.key === 'B') selectTool('brush');
  if (e.key === 'e' || e.key === 'E') selectTool('eraser');
  if (e.key === 'l' || e.key === 'L') fileInput.click();
  if (e.key === ' ') {
    e.preventDefault();
    if (frames.length) {
      if (isPlaying) stopPlayback(); else startPlayback();
    }
  }
});

// ─── Compare panel ───────────────────────────────────────────────────────────
function openCompare() {
  if (!peakData || !peakData.frame) return;

  // Downscale original to 100×100 then upscale with nearest-neighbor,
  // matching the backend's NEAREST upscale so both sides are at model resolution.
  const tmp = document.createElement('canvas');
  tmp.width = MODEL_SIZE; tmp.height = MODEL_SIZE;
  tmp.getContext('2d').drawImage(imageCanvas, 0, 0, MODEL_SIZE, MODEL_SIZE);

  originalCtx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
  originalCtx.imageSmoothingEnabled = false;
  originalCtx.drawImage(tmp, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);

  const img = new Image();
  img.onload = () => {
    peakCtx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    peakCtx.imageSmoothingEnabled = false;
    peakCtx.drawImage(img, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
  };
  img.src = 'data:image/png;base64,' + peakData.frame;

  comparePeakStep.textContent = peakData.step;
  compareMeta.textContent = `PSNR ${peakData.psnr.toFixed(1)} dB · SSIM ${peakData.ssim.toFixed(3)}`;

  compareSection.classList.remove('hidden');
  compareSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function closeCompare() {
  compareSection.classList.add('hidden');
}

btnCompare.addEventListener('click', openCompare);
btnCloseCompare.addEventListener('click', closeCompare);

// ─── Init ────────────────────────────────────────────────────────────────────
initCharts();
healthCheck();
