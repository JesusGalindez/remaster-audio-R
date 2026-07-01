const API_BASE = window.location.origin.includes('tauri') || window.location.protocol === 'file:' || !window.location.origin.includes('8000')
  ? 'http://127.0.0.1:8000' 
  : '';

// Element selectors
const videoDropZone = document.getElementById('drop-zone-video');
const audioDropZone = document.getElementById('drop-zone-audio');
const fileVideoInput = document.getElementById('file-video');
const fileAudioInput = document.getElementById('file-audio');
const videoFileName = document.getElementById('video-file-name');
const audioFileName = document.getElementById('audio-file-name');

const localPathInput = document.getElementById('local-path');
const syncSlider = document.getElementById('sync-slider');
const syncValText = document.getElementById('sync-val');

const toggleAiSync = document.getElementById('toggle-ai-sync');
const manualSyncContainer = document.getElementById('manual-sync-container');
const aiSyncInfoContainer = document.getElementById('ai-sync-info-container');

const btnPreview = document.getElementById('btn-preview');
const btnProcess = document.getElementById('btn-process');

const stateIdle = document.getElementById('state-idle');
const stateProcessing = document.getElementById('state-processing');
const stateCompleted = document.getElementById('state-completed');
const stateFailed = document.getElementById('state-failed');

const progressPercent = document.getElementById('progress-percent');
const progressTitle = document.getElementById('progress-title');
const progressMsg = document.getElementById('progress-msg');
const progressFill = document.getElementById('progress-fill');

const previewPlayer = document.getElementById('preview-player');
const btnDownload = document.getElementById('btn-download');
const errorMsgText = document.getElementById('error-msg');

let selectedVideoFile = null;
let selectedAudioFile = null;

// --- Drag & Drop handlers ---
function setupDragAndDrop(zone, input, callback) {
    zone.addEventListener('click', () => input.click());
    
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    ['dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, () => zone.classList.remove('dragover'));
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            callback(e.dataTransfer.files[0]);
        }
    });

    input.addEventListener('change', () => {
        if (input.files.length) {
            callback(input.files[0]);
        }
    });
}

setupDragAndDrop(videoDropZone, fileVideoInput, (file) => {
    selectedVideoFile = file;
    videoFileName.textContent = file.name;
});

setupDragAndDrop(audioDropZone, fileAudioInput, (file) => {
    selectedAudioFile = file;
    audioFileName.textContent = file.name;
});

// --- Lip-Sync Handlers ---
syncSlider.addEventListener('input', () => {
    syncValText.textContent = `${syncSlider.value} ms`;
});

function adjustSync(amount) {
    let newVal = parseInt(syncSlider.value) + amount;
    newVal = Math.max(-1000, Math.min(1000, newVal));
    syncSlider.value = newVal;
    syncValText.textContent = `${newVal} ms`;
}

function resetSync() {
    syncSlider.value = 0;
    syncValText.textContent = `0 ms`;
}

// Toggle manual vs AI sync visual containers
toggleAiSync.addEventListener('change', () => {
    if (toggleAiSync.checked) {
        manualSyncContainer.classList.add('hidden');
        aiSyncInfoContainer.classList.remove('hidden');
    } else {
        manualSyncContainer.classList.remove('hidden');
        aiSyncInfoContainer.classList.add('hidden');
    }
});

// --- State switches ---
function showState(state) {
    [stateIdle, stateProcessing, stateCompleted, stateFailed].forEach(s => s.classList.add('hidden'));
    state.classList.remove('hidden');
}

function showIdleState() {
    showState(stateIdle);
}

// --- API Request Processing ---
async function startProcessing(preview = false) {
    const localPath = localPathInput.value.trim();
    const syncMs = syncSlider.value;

    // Check if input is valid
    if (!selectedVideoFile && !localPath) {
        alert("Please drop a video file or enter a local file path.");
        return;
    }

    const formData = new FormData();
    formData.append("sync_ms", syncMs);
    formData.append("auto_sync_lips", toggleAiSync.checked);
    
    const aiStartSec = document.getElementById("ai-start-sec").value;
    formData.append("ai_start_sec", aiStartSec);
    
    formData.append("preview", preview);

    if (localPath) {
        formData.append("local_path", localPath);
    } else {
        formData.append("video", selectedVideoFile);
    }

    if (selectedAudioFile) {
        formData.append("sync_ref", selectedAudioFile);
    }

    showState(stateProcessing);
    updateProgress("Starting...", "Sending file details to server...", 0);

    try {
        const response = await fetch(`${API_BASE}/api/remaster`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Request failed");
        }

        const data = await response.json();
        listenToTaskProgress(data.task_id);
    } catch (err) {
        errorMsgText.textContent = err.message;
        showState(stateFailed);
    }
}

btnPreview.addEventListener('click', () => startProcessing(true));
btnProcess.addEventListener('click', () => startProcessing(false));

// --- SSE Event Handler ---
function listenToTaskProgress(taskId) {
    const eventSource = new EventSource(`${API_BASE}/api/stream-status/${taskId}`);

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.status === "processing" || data.status === "starting") {
            updateProgress(
                data.status === "starting" ? "Bootstrapping..." : "Remastering...", 
                data.message, 
                data.percent
            );
        } else if (data.status === "completed") {
            eventSource.close();
            // Configure download & preview buttons
            const fileUrl = `${API_BASE}/api/download/${data.output_file}`;
            previewPlayer.src = fileUrl;
            btnDownload.href = fileUrl;
            btnDownload.download = data.output_file;
            showState(stateCompleted);
        } else if (data.status === "failed") {
            eventSource.close();
            errorMsgText.textContent = data.message;
            showState(stateFailed);
        }
    };

    eventSource.onerror = (e) => {
        console.error("SSE connection error", e);
        eventSource.close();
    };
}

function updateProgress(title, message, percent) {
    progressTitle.textContent = title;
    progressMsg.textContent = message;
    progressPercent.textContent = `${percent}%`;
    progressFill.style.width = `${percent}%`;
}
