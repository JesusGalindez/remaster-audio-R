import os
import sys
import time
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add local bin/ folder to PATH so any subprocess/library finds our downloaded FFmpeg
bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin')
if bin_dir not in os.environ["PATH"]:
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]

import remasterer

app = FastAPI(title="YouTube Audio Remasterer")

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
WEB_DIR = os.path.join(BASE_DIR, "web")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

# Global task state dictionary
tasks = {}

def update_task_progress(task_id, message, percent):
    tasks[task_id] = {
        "status": "processing",
        "message": message,
        "percent": percent,
        "updated_at": time.time()
    }

async def run_remaster_task(task_id, input_path, output_path, sync_ms, sync_ref=None, auto_sync_lips=False, ai_start_sec=5, preview=False):
    try:
        def progress_cb(message, percent):
            update_task_progress(task_id, message, percent)
            
        # Run synchronous process_file in executor to avoid blocking async event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            remasterer.process_file, 
            input_path, 
            output_path, 
            sync_ms, 
            sync_ref, 
            auto_sync_lips, 
            ai_start_sec,
            preview, 
            progress_cb
        )
        
        tasks[task_id] = {
            "status": "completed",
            "message": "Processing finished successfully!",
            "percent": 100,
            "output_file": os.path.basename(output_path),
            "updated_at": time.time()
        }
    except Exception as e:
        tasks[task_id] = {
            "status": "failed",
            "message": str(e),
            "percent": 100,
            "updated_at": time.time()
        }

@app.post("/api/remaster")
async def start_remaster(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(None),
    local_path: str = Form(None),
    sync_ms: int = Form(0),
    sync_ref: UploadFile = File(None),
    auto_sync_lips: bool = Form(False),
    ai_start_sec: int = Form(5),
    preview: bool = Form(False)
):
    # Resolve input video
    if local_path:
        if not os.path.exists(local_path):
            raise HTTPException(status_code=400, detail="Local file path does not exist.")
        input_path = local_path
        filename = os.path.basename(local_path)
    elif video:
        input_path = os.path.join(UPLOAD_DIR, video.filename)
        with open(input_path, "wb") as buffer:
            buffer.write(await video.read())
        filename = video.filename
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded video file or a local_path.")

    # Resolve reference audio if provided
    ref_path = None
    if sync_ref:
        ref_path = os.path.join(UPLOAD_DIR, sync_ref.filename)
        with open(ref_path, "wb") as buffer:
            buffer.write(await sync_ref.read())

    # Set up output path
    task_id = f"task_{int(time.time() * 1000)}"
    prefix = "preview_" if preview else "remastered_"
    output_filename = f"{prefix}{filename}"
    output_path = os.path.join(EXPORT_DIR, output_filename)
    
    # Initialize task state
    tasks[task_id] = {
        "status": "starting",
        "message": "Initializing processing engine...",
        "percent": 0,
        "updated_at": time.time()
    }
    
    # Queue task in background
    background_tasks.add_task(
        run_remaster_task, 
        task_id, 
        input_path, 
        output_path, 
        sync_ms, 
        ref_path, 
        auto_sync_lips, 
        ai_start_sec,
        preview
    )
    
    return {"task_id": task_id, "output_file": output_filename}

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    return tasks[task_id]

@app.get("/api/stream-status/{task_id}")
async def stream_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
        
    async def event_generator():
        while True:
            task = tasks.get(task_id)
            if not task:
                break
            
            yield f"data: {json.dumps(task)}\n\n"
            
            if task["status"] in ("completed", "failed"):
                break
                
            await asyncio.sleep(0.5)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

# Serve web frontend
@app.get("/")
async def get_index():
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        return {"error": "Web interface is building. Please refresh in a moment."}
    return FileResponse(index_path)

# Static files mount (must be mounted last to avoid overriding API routes)
if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR), name="static")

if __name__ == "__main__":
    # Download FFmpeg and RNNoise model at startup to guarantee dependencies are ready
    try:
        remasterer.get_ffmpeg_binary()
        remasterer.get_rnnoise_model()
    except Exception as e:
        print(f"Failed to bootstrap dependencies: {e}")
        sys.exit(1)
        
    print("Dependencies validated. Launching dashboard server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
