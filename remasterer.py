import sys
import os
import urllib.request
import gzip
import shutil
import stat
import platform
import subprocess
import time
import argparse

# Add local bin/ folder to PATH so any subprocess/library finds our downloaded FFmpeg
bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin')
if bin_dir not in os.environ["PATH"]:
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]


# Global configurations
DEFAULT_MODEL_URL = "https://raw.githubusercontent.com/richardpl/arnndn-models/master/cb.rnnn"
DEFAULT_FFMPEG_VERSION = "b5.0.1"
BASE_FFMPEG_URL = f"https://github.com/eugeneware/ffmpeg-static/releases/download/{DEFAULT_FFMPEG_VERSION}/"

def get_ffmpeg_binary():
    """Detects system platform and returns local or system FFmpeg binary path."""
    system = sys.platform
    arch = platform.machine().lower()
    
    bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin')
    os.makedirs(bin_dir, exist_ok=True)
    
    ffmpeg_name = 'ffmpeg.exe' if system == 'win32' else 'ffmpeg'
    ffmpeg_path = os.path.join(bin_dir, ffmpeg_name)
    
    # Check if local binary exists
    if os.path.exists(ffmpeg_path):
        return ffmpeg_path
        
    # Check if system-wide FFmpeg is installed
    system_ffmpeg = shutil.which('ffmpeg')
    if system_ffmpeg:
        return system_ffmpeg
        
    # Download static build from github releases
    print("Local FFmpeg not found. Downloading static build from GitHub releases...")
    if system == 'darwin':
        if 'arm' in arch or 'm1' in arch or 'm2' in arch or 'm3' in arch:
            url = BASE_FFMPEG_URL + "darwin-arm64.gz"
        else:
            url = BASE_FFMPEG_URL + "darwin-x64.gz"
    elif system == 'win32':
        url = BASE_FFMPEG_URL + "win32-x64.gz"
    elif system.startswith('linux'):
        url = BASE_FFMPEG_URL + "linux-x64.gz"
    else:
        raise OSError(f"Unsupported platform: {system}")
        
    gz_path = ffmpeg_path + ".gz"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(gz_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
        with gzip.open(gz_path, 'rb') as f_in:
            with open(ffmpeg_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        os.remove(gz_path)
        
        # Add executable permissions
        if system != 'win32':
            st = os.stat(ffmpeg_path)
            os.chmod(ffmpeg_path, st.st_mode | stat.S_IEXEC)
            # Remove macOS quarantine if needed
            if system == 'darwin':
                try:
                    subprocess.run(["xattr", "-dr", "com.apple.quarantine", ffmpeg_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
            
        print("FFmpeg configured successfully!")
        return ffmpeg_path
    except Exception as e:
        if os.path.exists(gz_path):
            os.remove(gz_path)
        if os.path.exists(ffmpeg_path):
            os.remove(ffmpeg_path)
        raise RuntimeError(f"Failed to auto-configure FFmpeg binary: {e}")

def get_rnnoise_model():
    """Downloads the vocal RNNoise model to models/cb.rnnn if not present."""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'cb.rnnn')
    
    if os.path.exists(model_path):
        return model_path
        
    print("Downloading pre-trained RNNoise model (cb.rnnn)...")
    try:
        req = urllib.request.Request(DEFAULT_MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Model downloaded successfully!")
        return model_path
    except Exception as e:
        if os.path.exists(model_path):
            os.remove(model_path)
        raise RuntimeError(f"Failed to fetch RNNoise model: {e}")

def find_audio_offset(video_audio_path, ref_audio_path):
    """Computes synchronization offset between two audio tracks using cross-correlation."""
    try:
        import numpy as np
        import scipy.signal
        import soundfile as sf
    except ImportError:
        print("Warning: numpy, scipy, or soundfile not installed. Auto-sync is disabled.")
        return 0

    print("Analyzing audio files for synchronization...")
    data1, sr1 = sf.read(video_audio_path)
    data2, sr2 = sf.read(ref_audio_path)
    
    if len(data1.shape) > 1:
        data1 = np.mean(data1, axis=1)
    if len(data2.shape) > 1:
        data2 = np.mean(data2, axis=1)
        
    target_sr = 8000
    num_samples1 = int(len(data1) * target_sr / sr1)
    data1_resampled = scipy.signal.resample(data1, num_samples1)
    
    num_samples2 = int(len(data2) * target_sr / sr2)
    data2_resampled = scipy.signal.resample(data2, num_samples2)
    
    data1_resampled = (data1_resampled - np.mean(data1_resampled)) / (np.std(data1_resampled) + 1e-8)
    data2_resampled = (data2_resampled - np.mean(data2_resampled)) / (np.std(data2_resampled) + 1e-8)
    
    correlation = scipy.signal.correlate(data1_resampled, data2_resampled, mode='full')
    lag = np.argmax(correlation) - (len(data2_resampled) - 1)
    
    offset_seconds = lag / target_sr
    offset_ms = int(offset_seconds * 1000)
    print(f"Calculated offset: {offset_ms} ms")
    return offset_ms

# --- AI Lip Sync Features (SyncNet) ---

def ensure_syncnet_installed():
    """Verifies that torch, torchvision, opencv and syncnet-python are installed, or auto-installs them."""
    try:
        import torch
        import torchvision
        import syncnet_python
        import cv2
    except ImportError:
        print("AI Lip-Sync dependencies (torch, torchvision, syncnet-python, opencv-python) not found.")
        print("Installing dynamically (this might take a few minutes)...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "syncnet-python", "opencv-python"
        ], check=True)
        print("Dependencies installed successfully!")

def download_syncnet_weights():
    """Downloads weights for s3fd face detector and SyncNet from Hugging Face if not present."""
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    sfd_path = os.path.join(weights_dir, 'sfd_face.pth')
    syncnet_path = os.path.join(weights_dir, 'syncnet_v2.model')
    
    urls = {
        sfd_path: "https://huggingface.co/lithiumice/syncnet/resolve/main/sfd_face.pth",
        syncnet_path: "https://huggingface.co/lithiumice/syncnet/resolve/main/syncnet_v2.model"
    }
    
    for path, url in urls.items():
        if not os.path.exists(path):
            print(f"Downloading SyncNet weights file: {os.path.basename(path)} (~50MB-70MB)...")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                print(f"Downloaded {os.path.basename(path)}")
            except Exception as e:
                if os.path.exists(path):
                    os.remove(path)
                raise RuntimeError(f"Failed to download weight file {os.path.basename(path)}: {e}")
                
    return sfd_path, syncnet_path

def get_video_duration(video_path, ffmpeg_bin):
    """Retrieves duration of a video in seconds."""
    cmd = [ffmpeg_bin, "-i", video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stderr
    for line in output.split('\n'):
        if "Duration:" in line:
            try:
                parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            except Exception:
                pass
    return 0.0

def get_video_fps(video_path, ffmpeg_bin):
    """Retrieves framerate (FPS) of a video."""
    cmd = [ffmpeg_bin, "-i", video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stderr
    for line in output.split('\n'):
        if "Video:" in line and "fps" in line:
            try:
                parts = line.split("fps")[0].split(",")
                fps_str = parts[-1].strip()
                return float(fps_str)
            except Exception:
                pass
    return 30.0

def run_ai_lip_sync(video_path, ai_start_sec=5, progress_callback=None):
    """Runs SyncNet inference on an optimized 15s video snippet starting at ai_start_sec to detect the lip-sync delay offset."""
    ensure_syncnet_installed()
    sfd_path, syncnet_path = download_syncnet_weights()
    
    import torch
    import numpy as np
    from syncnet_python import SyncNetPipeline
    
    ffmpeg = get_ffmpeg_binary()
    
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    snippet_path = os.path.join(temp_dir, f"temp_sync_snippet_{int(time.time())}.mp4")
    
    try:
        # Extract a 15-second snippet (skip to user specified start time, bound by duration)
        # Downsample resolution to 320x240 and frame rate to 15fps using FFmpeg to speed up face detection by ~90%
        duration = get_video_duration(video_path, ffmpeg)
        start_sec = max(0, min(int(duration) - 15, ai_start_sec))
        duration_sec = min(15, int(duration))
        
        if progress_callback:
            progress_callback(f"Extracting optimized 15s snippet starting at {start_sec}s for AI analysis...", 15)
            
        print(f"Extracting {duration_sec}s video snippet starting at {start_sec}s scaled to 320x240 @ 15fps for fast AI lip-sync analysis...")
        cmd_snippet = [
            ffmpeg, "-y", "-ss", str(start_sec), "-i", video_path, 
            "-t", str(duration_sec), 
            "-vf", "fps=15,scale=320:240", 
            "-c:a", "aac", snippet_path
        ]
        subprocess.run(cmd_snippet, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(snippet_path) or os.path.getsize(snippet_path) == 0:
            print("Warning: Snippet extraction failed. Using original file directly.")
            snippet_path = video_path
            
        # Select device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        if progress_callback:
            progress_callback("Running SyncNet AI face and audio alignment analysis...", 20)
            
        print(f"Initializing SyncNet AI pipeline on {device}...")
        pipeline = SyncNetPipeline(
            s3fd_weights=sfd_path,
            syncnet_weights=syncnet_path,
            device=device
        )
        
        print("Running AI SyncNet inference...")
        results = pipeline.inference(video_path=snippet_path)
        offset_list, confidence_list, min_dist_list, best_confidence, best_min_dist, detections_json, success = results
        
        if not success or not offset_list:
            print("AI Warning: Face detection or SyncNet inference failed (no speaker face found). Offset set to 0.")
            return 0
            
        # Select the offset that corresponds to the highest confidence track
        best_idx = np.argmax(confidence_list)
        best_offset_frames = offset_list[best_idx]
        best_conf = confidence_list[best_idx]
        
        # Enforce confidence threshold of 5.0 for reliable alignment
        if best_conf < 5.0:
            msg = f"AI Warning: Low confidence sync score ({best_conf:.2f} < 5.00). Face tracking might be blurry or silence/music detected. Skipping auto-sync adjustment (fallback to 0ms)."
            print(msg)
            if progress_callback:
                progress_callback(f"⚠️ AI Low Confidence ({best_conf:.2f}). Fallback to 0ms.", 25)
            return 0
            
        # Calculate offset in ms based on snippet's 15 fps
        fps = get_video_fps(snippet_path, ffmpeg)
        offset_ms = int((best_offset_frames / fps) * 1000)
        
        print(f"SyncNet AI result - Best offset: {best_offset_frames} frames ({offset_ms} ms) with confidence {best_conf:.2f}")
        return offset_ms
    except Exception as e:
        print(f"AI Lip-Sync error: {e}. Falling back to 0ms sync.")
        return 0
    finally:
        if os.path.exists(snippet_path) and snippet_path != video_path:
            os.remove(snippet_path)

# --- Core Remasterer Process ---

def process_file(input_path, output_path, sync_ms=0, sync_ref=None, auto_sync_lips=False, ai_start_sec=5, preview=False, progress_callback=None):
    """Executes the remastering pipeline on a single file."""
    ffmpeg = get_ffmpeg_binary()
    model_path = get_rnnoise_model()
    
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_audio = os.path.join(temp_dir, f"temp_master_{int(time.time())}.wav")
    
    try:
        audio_input = input_path
        actual_sync = sync_ms
        
        # Handle automated AI Lip-Sync detection
        if auto_sync_lips:
            if progress_callback:
                progress_callback("Bootstrapping Lip-Sync AI models...", 5)
            # This calls SyncNet on a 15s snippet starting at ai_start_sec
            actual_sync = run_ai_lip_sync(input_path, ai_start_sec, progress_callback)
            
        # Handle automatic audio alignment using high-quality reference track
        elif sync_ref and os.path.exists(sync_ref):
            if progress_callback:
                progress_callback("Analyzing audio alignment...", 10)
            
            temp_scratch = os.path.join(temp_dir, f"temp_scratch_{int(time.time())}.wav")
            subprocess.run([
                ffmpeg, "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", "16000", temp_scratch
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            actual_sync = find_audio_offset(temp_scratch, sync_ref)
            audio_input = sync_ref
            
            if os.path.exists(temp_scratch):
                os.remove(temp_scratch)
        
        if progress_callback:
            progress_callback("Applying de-noising and normalising loudness...", 30)
            
        # Apply filters (de-noising, highpass, loudness norm)
        filter_str = f"arnndn=m='{model_path}',highpass=f=80,loudnorm=i=-14:tp=-1:lra=7"
        
        cmd_audio = [ffmpeg, "-y", "-i", audio_input]
        if preview:
            cmd_audio.extend(["-t", "15"])
            
        cmd_audio.extend(["-af", filter_str, "-c:a", "pcm_s24le", "-ar", "48000", temp_audio])
        
        result = subprocess.run(cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio processing failed:\n{result.stderr}")
            
        if progress_callback:
            progress_callback("Muxing audio back to video stream...", 70)
            
        # Re-mux processed audio with video using lossless video copy
        cmd_mux = [ffmpeg, "-y"]
        
        # Handle offset shifting dynamically without re-encoding
        if actual_sync > 0:
            sync_sec = actual_sync / 1000.0
            cmd_mux.extend(["-itsoffset", f"{sync_sec:.3f}", "-i", input_path, "-i", temp_audio])
        elif actual_sync < 0:
            sync_sec = abs(actual_sync) / 1000.0
            cmd_mux.extend(["-i", input_path, "-itsoffset", f"{sync_sec:.3f}", "-i", temp_audio])
        else:
            cmd_mux.extend(["-i", input_path, "-i", temp_audio])
            
        if preview:
            cmd_mux.extend(["-t", "15"])
            
        cmd_mux.extend([
            "-map", "0:v:0", "-map", "1:a:0", 
            "-c:v", "copy", "-c:a", "aac", "-b:a", "384k", "-ar", "48000", 
            "-shortest", output_path
        ])
        
        result_mux = subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result_mux.returncode != 0:
            raise RuntimeError(f"FFmpeg video muxing failed:\n{result_mux.stderr}")
            
        if progress_callback:
            progress_callback("Remastering completed successfully!", 100)
            
        print(f"Processed successfully: {output_path} (Sync: {actual_sync}ms)")
        return True
    finally:
        # Clean up temporary files
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

def batch_process(input_dir, output_dir, sync_ms=0, auto_sync_lips=False):
    """Processes all videos in a directory."""
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = ('.mp4', '.mkv', '.mov', '.avi')
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_exts)]
    if not files:
        print("No supported video files found.")
        return
        
    print(f"Starting batch process of {len(files)} files...")
    for idx, filename in enumerate(files):
        print(f"\n[{idx+1}/{len(files)}] Processing: {filename}")
        in_file = os.path.join(input_dir, filename)
        out_file = os.path.join(output_dir, f"remastered_{filename}")
        try:
            process_file(in_file, out_file, sync_ms=sync_ms, auto_sync_lips=auto_sync_lips)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

def watch_directory(watch_dir, output_dir, sync_ms=0, auto_sync_lips=False):
    """Monitors a folder and processes new files automatically."""
    print(f"Watching directory '{watch_dir}' for new videos. Outputs will go to '{output_dir}'...")
    os.makedirs(watch_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    supported_exts = ('.mp4', '.mkv', '.mov', '.avi')
    processed_files = set()
    
    for filename in os.listdir(watch_dir):
        if filename.lower().endswith(supported_exts):
            processed_files.add(filename)
            
    try:
        while True:
            current_files = os.listdir(watch_dir)
            for filename in current_files:
                if filename.lower().endswith(supported_exts) and filename not in processed_files:
                    in_file = os.path.join(watch_dir, filename)
                    out_file = os.path.join(output_dir, f"remastered_{filename}")
                    
                    initial_size = os.path.getsize(in_file)
                    time.sleep(2)
                    if os.path.getsize(in_file) != initial_size:
                        continue
                        
                    print(f"\nNew file detected: {filename}. Processing...")
                    try:
                        process_file(in_file, out_file, sync_ms=sync_ms, auto_sync_lips=auto_sync_lips)
                        processed_files.add(filename)
                    except Exception as e:
                        print(f"Error processing watched file {filename}: {e}")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopped watching directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Audio Remasterer CLI Utility")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    
    # Process command
    p_parser = subparsers.add_parser("process", help="Process a single video")
    p_parser.add_argument("input", help="Path to input video")
    p_parser.add_argument("output", help="Path to output remastered video")
    p_parser.add_argument("--sync-ms", type=int, default=0, help="Manual audio offset delay in milliseconds")
    p_parser.add_argument("--sync-ref", default=None, help="Path to high-quality external microphone audio recording")
    p_parser.add_argument("--auto-sync-lips", action="store_true", help="100% automated AI Lip-Sync detection using SyncNet")
    p_parser.add_argument("--ai-start-sec", type=int, default=5, help="Start time in seconds of the video snippet for AI analysis")
    p_parser.add_argument("--preview", action="store_true", help="Generate a fast 15-second preview")
    
    # Batch command
    b_parser = subparsers.add_parser("batch", help="Batch process videos in a folder")
    b_parser.add_argument("input_dir", help="Directory containing original videos")
    b_parser.add_argument("output_dir", help="Directory where remastered videos will be saved")
    b_parser.add_argument("--sync-ms", type=int, default=0, help="Audio sync shift in milliseconds")
    b_parser.add_argument("--auto-sync-lips", action="store_true", help="Use AI Lip-Sync detection")
    
    # Watch command
    w_parser = subparsers.add_parser("watch", help="Monitor folder and remaster new files")
    w_parser.add_argument("watch_dir", help="Folder to monitor")
    w_parser.add_argument("output_dir", help="Folder for output files")
    w_parser.add_argument("--sync-ms", type=int, default=0, help="Audio sync shift in milliseconds")
    w_parser.add_argument("--auto-sync-lips", action="store_true", help="Use AI Lip-Sync detection")
    
    args = parser.parse_args()
    
    try:
        get_ffmpeg_binary()
        get_rnnoise_model()
    except Exception as e:
        print(f"Setup Error: {e}")
        sys.exit(1)
        
    if args.command == "process":
        process_file(
            args.input, args.output, 
            sync_ms=args.sync_ms, 
            sync_ref=args.sync_ref, 
            auto_sync_lips=args.auto_sync_lips, 
            ai_start_sec=args.ai_start_sec,
            preview=args.preview
        )
    elif args.command == "batch":
        batch_process(args.input_dir, args.output_dir, args.sync_ms, args.auto_sync_lips)
    elif args.command == "watch":
        watch_directory(args.watch_dir, args.output_dir, args.sync_ms, args.auto_sync_lips)
    else:
        parser.print_help()
