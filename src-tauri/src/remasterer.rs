use std::path::PathBuf;
use std::fs::{self, File};
use std::io::{Write, Read};
use std::process::{Command, Stdio};
use std::env;
use flate2::read::GzDecoder;

use crate::dsp;

const DEFAULT_MODEL_URL: &str = "https://raw.githubusercontent.com/richardpl/arnndn-models/master/cb.rnnn";
const BASE_FFMPEG_URL: &str = "https://github.com/eugeneware/ffmpeg-static/releases/download/b5.0.1/";

/// Helper to get the absolute workspace root path
pub fn get_workspace_dir() -> PathBuf {
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Dynamic platform detection for FFmpeg download
pub fn get_ffmpeg_binary() -> Result<String, String> {
    let workspace = get_workspace_dir();
    let bin_dir = workspace.join("bin");
    fs::create_dir_all(&bin_dir).map_err(|e| format!("Failed to create bin dir: {}", e))?;

    let is_windows = env::consts::OS == "windows";
    let exe_ext = if is_windows { ".exe" } else { "" };
    let ffmpeg_path = bin_dir.join(format!("ffmpeg{}", exe_ext));

    if ffmpeg_path.exists() {
        return Ok(ffmpeg_path.to_string_lossy().into_owned());
    }

    // Determine download URL
    let target_arch = env::consts::ARCH;
    let target_os = env::consts::OS;

    let filename = match (target_os, target_arch) {
        ("macos", "aarch64") => "darwin-arm64.gz",
        ("macos", "x86_64") => "darwin-x64.gz",
        ("linux", "x86_64") => "linux-x64.gz",
        ("windows", "x86_64") => "win32-x64.gz",
        _ => return Err(format!("Unsupported platform: {}-{}", target_os, target_arch)),
    };

    let download_url = format!("{}{}", BASE_FFMPEG_URL, filename);
    println!("Downloading FFmpeg static build for your platform from {}...", download_url);

    // Download compressed bytes
    let response = reqwest::blocking::get(&download_url)
        .map_err(|e| format!("Failed to download FFmpeg: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("FFmpeg download returned status: {}", response.status()));
    }

    let compressed_bytes = response.bytes()
        .map_err(|e| format!("Failed to read FFmpeg response bytes: {}", e))?;

    // Decompress gzip on the fly
    println!("Decompressing FFmpeg binary...");
    let mut decoder = GzDecoder::new(&compressed_bytes[..]);
    let mut decompressed_bytes = Vec::new();
    decoder.read_to_end(&mut decompressed_bytes)
        .map_err(|e| format!("Failed to decompress FFmpeg Gzip stream: {}", e))?;

    // Write file
    let mut file = File::create(&ffmpeg_path)
        .map_err(|e| format!("Failed to create FFmpeg file: {}", e))?;
    file.write_all(&decompressed_bytes)
        .map_err(|e| format!("Failed to write FFmpeg binary: {}", e))?;

    // Set executable permissions on macOS/Linux
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&ffmpeg_path).map_err(|e| e.to_string())?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&ffmpeg_path, perms).map_err(|e| e.to_string())?;
    }

    println!("FFmpeg successfully installed at: {:?}", ffmpeg_path);
    Ok(ffmpeg_path.to_string_lossy().into_owned())
}

/// Downloads and returns the RNNoise vocal cleaning model path
pub fn get_rnnoise_model() -> Result<String, String> {
    let workspace = get_workspace_dir();
    let models_dir = workspace.join("models");
    fs::create_dir_all(&models_dir).map_err(|e| format!("Failed to create models dir: {}", e))?;

    let model_path = models_dir.join("cb.rnnn");
    if model_path.exists() {
        return Ok(model_path.to_string_lossy().into_owned());
    }

    println!("Downloading RNNoise neural vocal model...");
    let response = reqwest::blocking::get(DEFAULT_MODEL_URL)
        .map_err(|e| format!("Failed to download vocal model: {}", e))?;
    
    let bytes = response.bytes().map_err(|e| format!("Failed to read model bytes: {}", e))?;
    let mut file = File::create(&model_path).map_err(|e| format!("Failed to create model file: {}", e))?;
    file.write_all(&bytes).map_err(|e| format!("Failed to write model file: {}", e))?;

    Ok(model_path.to_string_lossy().into_owned())
}

/// Spawns a background python thread to run Lip-Sync estimation using SyncNet
pub fn run_ai_lip_sync<F>(video_path: &str, ai_start_sec: i32, progress_cb: &F) -> i32
where
    F: Fn(&str, u32),
{
    progress_cb("Bootstrapping Python AI Lip-Sync context...", 10);
    
    // Build python inline command
    let py_cmd = format!(
        "import sys; sys.path.insert(0, '{}'); import remasterer; print(remasterer.run_ai_lip_sync(r'{}', {}))",
        get_workspace_dir().to_string_lossy(),
        video_path,
        ai_start_sec
    );

    println!("Spawning SyncNet AI analysis via Python subprocess...");
    let result = Command::new("python3")
        .arg("-c")
        .arg(&py_cmd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    match result {
        Ok(output) => {
            let out_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let err_str = String::from_utf8_lossy(&output.stderr).to_string();
            
            if !output.status.success() {
                println!("Python sub-process failed with error:\n{}", err_str);
                return 0;
            }

            if let Ok(offset_ms) = out_str.parse::<i32>() {
                println!("AI Lip-Sync computed offset from Python: {} ms", offset_ms);
                offset_ms
            } else {
                println!("Failed to parse Python output offset: '{}'", out_str);
                0
            }
        }
        Err(e) => {
            println!("Error executing Python subprocess for AI Sync: {}", e);
            0
        }
    }
}

/// Core DSP remastering function
pub fn process_file<F>(
    input_path: &str,
    output_path: &str,
    sync_ms: i32,
    sync_ref: Option<&str>,
    auto_sync_lips: bool,
    ai_start_sec: i32,
    preview: bool,
    progress_cb: F,
) -> Result<(), String>
where
    F: Fn(&str, u32),
{
    let ffmpeg = get_ffmpeg_binary()?;
    let model_path = get_rnnoise_model()?;
    
    // Inject bin/ path into system PATH dynamically for child sub-processes
    let workspace = get_workspace_dir();
    let bin_path = workspace.join("bin");
    if let Some(path_env) = env::var_os("PATH") {
        let mut paths = env::split_paths(&path_env).collect::<Vec<_>>();
        if !paths.contains(&bin_path) {
            paths.insert(0, bin_path);
            let new_path_env = env::join_paths(paths).map_err(|e| e.to_string())?;
            env::set_var("PATH", new_path_env);
        }
    }

    let temp_dir = workspace.join("temp");
    fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
    
    let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let temp_audio = temp_dir.join(format!("temp_master_{}.wav", timestamp));
    let temp_audio_str = temp_audio.to_string_lossy().into_owned();

    let mut actual_sync = sync_ms;
    let mut audio_input = input_path.to_string();

    // 1. Resolve Sync Delay Shift
    if auto_sync_lips {
        actual_sync = run_ai_lip_sync(input_path, ai_start_sec, &progress_cb);
    } else if let Some(ref_path) = sync_ref {
        progress_cb("Running FFT cross-correlation in Rust...", 15);
        actual_sync = dsp::find_audio_offset(&ffmpeg, input_path, ref_path)?;
        audio_input = ref_path.to_string();
    }

    // 2. Process Audio (RNNoise + Highpass + Loudnorm)
    progress_cb("Applying vocal cleaning and loudness normalization...", 35);
    
    // Escape model path spaces for FFmpeg syntax
    let escaped_model = model_path.replace("\\", "/").replace(":", "\\:");
    let filter_str = format!("arnndn=m='{}',highpass=f=80,loudnorm=i=-14:tp=-1:lra=7", escaped_model);

    let mut cmd_audio = Command::new(&ffmpeg);
    cmd_audio.arg("-y").arg("-i").arg(&audio_input);
    
    if preview {
        cmd_audio.arg("-t").arg("15");
    }
    
    cmd_audio.arg("-af").arg(&filter_str)
             .arg("-c:a").arg("pcm_s24le")
             .arg("-ar").arg("48000")
             .arg(&temp_audio_str);

    let result_audio = cmd_audio.stdout(Stdio::null())
                                .stderr(Stdio::piped())
                                .output()
                                .map_err(|e| format!("FFmpeg audio command failed: {}", e))?;

    if !result_audio.status.success() {
        let err_msg = String::from_utf8_lossy(&result_audio.stderr);
        let _ = fs::remove_file(&temp_audio);
        return Err(format!("FFmpeg audio processing failed:\n{}", err_msg));
    }

    // 3. Mux back Audio and Video streams
    progress_cb("Multiplexing high-fidelity track back to video...", 75);
    
    let mut cmd_mux = Command::new(&ffmpeg);
    cmd_mux.arg("-y");

    // Correct delay offset application: positive offset delays video, negative delays audio
    if actual_sync > 0 {
        let delay_sec = (actual_sync as f64) / 1000.0;
        cmd_mux.arg("-itsoffset").arg(format!("{:.3}", delay_sec))
               .arg("-i").arg(input_path)
               .arg("-i").arg(&temp_audio_str);
    } else if actual_sync < 0 {
        let delay_sec = (actual_sync.abs() as f64) / 1000.0;
        cmd_mux.arg("-i").arg(input_path)
               .arg("-itsoffset").arg(format!("{:.3}", delay_sec))
               .arg("-i").arg(&temp_audio_str);
    } else {
        cmd_mux.arg("-i").arg(input_path)
               .arg("-i").arg(&temp_audio_str);
    }

    if preview {
        cmd_mux.arg("-t").arg("15");
    }

    cmd_mux.arg("-map").arg("0:v:0")
           .arg("-map").arg("1:a:0")
           .arg("-c:v").arg("copy")
           .arg("-c:a").arg("aac")
           .arg("-b:a").arg("384k")
           .arg("-ar").arg("48000")
           .arg("-shortest")
           .arg(output_path);

    let result_mux = cmd_mux.stdout(Stdio::null())
                            .stderr(Stdio::piped())
                            .output()
                            .map_err(|e| format!("FFmpeg video muxing failed: {}", e))?;

    // Cleanup temp wav file
    let _ = fs::remove_file(&temp_audio);

    if !result_mux.status.success() {
        let err_msg = String::from_utf8_lossy(&result_mux.stderr);
        return Err(format!("FFmpeg video muxing failed:\n{}", err_msg));
    }

    progress_cb("Finished successfully!", 100);
    Ok(())
}
