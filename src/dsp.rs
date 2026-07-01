use std::process::{Command, Stdio};
use std::io::Read;
use rustfft::{FftPlanner, num_complex::Complex};

/// Extracts raw 32-bit float mono audio samples at 16kHz from a file using FFmpeg.
pub fn get_audio_samples(ffmpeg_bin: &str, file_path: &str, start_sec: f64, duration_sec: f64) -> Result<Vec<f32>, String> {
    let mut cmd = Command::new(ffmpeg_bin);
    cmd.arg("-y");
    
    // Seek to start time
    if start_sec > 0.0 {
        cmd.arg("-ss").arg(format!("{:.3}", start_sec));
    }
    
    cmd.arg("-i").arg(file_path)
       .arg("-t").arg(format!("{:.3}", duration_sec))
       .arg("-f").arg("f32le")
       .arg("-ac").arg("1")
       .arg("-ar").arg("16000")
       .arg("pipe:1")
       .stdout(Stdio::piped())
       .stderr(Stdio::null());

    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn FFmpeg: {}", e))?;
    let mut stdout = child.stdout.take().ok_or("Failed to open FFmpeg stdout pipe")?;
    
    let mut buffer = Vec::new();
    stdout.read_to_end(&mut buffer).map_err(|e| format!("Failed to read stdout: {}", e))?;
    let _ = child.wait(); // Clean up process

    if buffer.is_empty() {
        return Err("No audio samples returned from FFmpeg".to_string());
    }

    // Convert bytes (f32 little endian) to f32 slice
    let num_samples = buffer.len() / 4;
    let mut samples = Vec::with_capacity(num_samples);
    for chunk in buffer.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        samples.push(val);
    }
    
    Ok(samples)
}

/// Helper to pad vector to the next power of two
fn next_power_of_two(n: usize) -> usize {
    let mut m = 1;
    while m < n {
        m <<= 1;
    }
    m
}

/// Finds the millisecond alignment offset of audio2 relative to audio1 using FFT cross-correlation.
/// Returns the offset in milliseconds. A positive offset means audio2 lags audio1.
pub fn find_audio_offset(ffmpeg_bin: &str, path1: &str, path2: &str) -> Result<i32, String> {
    // Extract 30 seconds of samples starting from second 5 to avoid intros
    let samples1 = get_audio_samples(ffmpeg_bin, path1, 5.0, 30.0)?;
    let samples2 = get_audio_samples(ffmpeg_bin, path2, 5.0, 30.0)?;

    let n = samples1.len();
    let m = samples2.len();
    let pad_len = next_power_of_two(n + m - 1);

    // Prepare complex vectors padded with zeros
    let mut in1: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); pad_len];
    let mut in2: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); pad_len];

    for i in 0..n {
        in1[i] = Complex::new(samples1[i], 0.0);
    }
    for i in 0..m {
        // Reverse array 2 for cross-correlation via convolution theorem
        in2[i] = Complex::new(samples2[m - 1 - i], 0.0);
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(pad_len);
    
    let mut out1 = in1.clone();
    let mut out2 = in2.clone();
    
    fft.process(&mut out1);
    fft.process(&mut out2);

    // Element-wise multiply the two spectra
    let mut spectral_prod: Vec<Complex<f32>> = out1.iter()
        .zip(out2.iter())
        .map(|(a, b)| a * b)
        .collect();

    // Compute inverse FFT
    let ifft = planner.plan_fft_inverse(pad_len);
    ifft.process(&mut spectral_prod);

    // Find the index of the maximum absolute correlation value
    let mut max_val = -1.0;
    let mut max_idx = 0;
    for i in 0..pad_len {
        let abs_val = (spectral_prod[i].re * spectral_prod[i].re + spectral_prod[i].im * spectral_prod[i].im).sqrt();
        if abs_val > max_val {
            max_val = abs_val;
            max_idx = i;
        }
    }

    // Map circular convolution index back to linear correlation shift
    let shift = if max_idx < pad_len / 2 {
        max_idx as i32 - (m as i32 - 1)
    } else {
        (max_idx as i32 - pad_len as i32) - (m as i32 - 1)
    };

    // Convert sample shift at 16kHz to milliseconds
    let offset_ms = ((shift as f64) / 16000.0 * 1000.0).round() as i32;
    
    Ok(-offset_ms) // Invert to match offset convention: positive = audio2 lags
}
