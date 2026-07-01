use axum::{
    extract::{Multipart, Path, State},
    http::{header, HeaderValue, StatusCode},
    response::{
        sse::{Event, Sse},
        Html, IntoResponse, Response,
    },
    routing::{get, post},
    Router,
};
use futures_util::stream::Stream;
use rust_embed::RustEmbed;
use serde::Serialize;
use std::{
    collections::HashMap,
    convert::Infallible,
    fs,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;

mod dsp;
mod remasterer;

#[derive(RustEmbed)]
#[folder = "web/"]
struct Assets;

#[derive(Clone, Serialize)]
struct TaskStatus {
    status: String,
    message: String,
    percent: u32,
    output_file: Option<String>,
    updated_at: f64,
}

type TaskMap = Arc<RwLock<HashMap<String, TaskStatus>>>;

struct AppState {
    tasks: TaskMap,
}

#[tokio::main]
async fn main() {
    // Check local dependencies
    if let Err(e) = remasterer::get_ffmpeg_binary() {
        eprintln!("Dependency warning: {}", e);
    }
    if let Err(e) = remasterer::get_rnnoise_model() {
        eprintln!("Dependency warning: {}", e);
    }

    let state = Arc::new(AppState {
        tasks: Arc::new(RwLock::new(HashMap::new())),
    });

    let app = Router::new()
        // Serve static web pages
        .route("/", get(serve_index))
        .route("/:filename", get(serve_static))
        // API routes
        .route("/api/remaster", post(api_remaster))
        .route("/api/status/:task_id", get(api_status))
        .route("/api/stream-status/:task_id", get(api_stream_status))
        .route("/api/download/:filename", get(api_download))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8000").await.unwrap();
    println!("Server running on http://127.0.0.1:8000");
    axum::serve(listener, app).await.unwrap();
}

// --- Static File Handlers (Embedded) ---

async fn serve_index() -> impl IntoResponse {
    match Assets::get("index.html") {
        Some(content) => {
            let html = String::from_utf8_lossy(&content.data).into_owned();
            Html(html).into_response()
        }
        None => (StatusCode::NOT_FOUND, "Index HTML not found").into_response(),
    }
}

async fn serve_static(Path(filename): Path<String>) -> impl IntoResponse {
    match Assets::get(&filename) {
        Some(content) => {
            let mime = mime_guess::from_path(&filename).first_or_octet_stream();
            (
                [(header::CONTENT_TYPE, HeaderValue::from_str(mime.as_ref()).unwrap())],
                content.data.into_owned(),
            )
                .into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

// --- API Route Handlers ---

async fn api_status(
    State(state): State<Arc<AppState>>,
    Path(task_id): Path<String>,
) -> impl IntoResponse {
    let tasks = state.tasks.read().await;
    match tasks.get(&task_id) {
        Some(status) => axum::Json(status.clone()).into_response(),
        None => (StatusCode::NOT_FOUND, "Task not found").into_response(),
    }
}

async fn api_stream_status(
    State(state): State<Arc<AppState>>,
    Path(task_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            let tasks = state.tasks.read().await;
            if let Some(status) = tasks.get(&task_id) {
                let serialized = serde_json::to_string(status).unwrap();
                yield Ok(Event::default().data(serialized));
                if status.status == "completed" || status.status == "failed" {
                    break;
                }
            } else {
                yield Ok(Event::default().data(r#"{"status":"failed","message":"Task disappeared"}"#));
                break;
            }
        }
    };
    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default())
}

async fn api_download(Path(filename): Path<String>) -> Response {
    let workspace = remasterer::get_workspace_dir();
    let file_path = workspace.join("exports").join(&filename);
    
    if !file_path.exists() {
        return (StatusCode::NOT_FOUND, "File not found").into_response();
    }

    match fs::read(&file_path) {
        Ok(bytes) => {
            let mime = mime_guess::from_path(&filename).first_or_octet_stream();
            (
                [
                    (header::CONTENT_TYPE, HeaderValue::from_str(mime.as_ref()).unwrap()),
                    (
                        header::CONTENT_DISPOSITION,
                        HeaderValue::from_str(&format!("attachment; filename=\"{}\"", filename)).unwrap(),
                    ),
                ],
                bytes,
            )
                .into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read file: {}", e)).into_response(),
    }
}

async fn api_remaster(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut sync_ms = 0;
    let mut auto_sync_lips = false;
    let mut ai_start_sec = 5;
    let mut preview = false;
    let mut local_path = String::new();

    let mut video_data: Option<Vec<u8>> = None;
    let mut video_name = String::new();
    let mut ref_data: Option<Vec<u8>> = None;
    let mut ref_name = String::new();

    // Parse Multipart upload data
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "sync_ms" {
            if let Ok(val) = field.text().await {
                sync_ms = val.parse::<i32>().unwrap_or(0);
            }
        } else if name == "auto_sync_lips" {
            if let Ok(val) = field.text().await {
                auto_sync_lips = val == "true";
            }
        } else if name == "ai_start_sec" {
            if let Ok(val) = field.text().await {
                ai_start_sec = val.parse::<i32>().unwrap_or(5);
            }
        } else if name == "preview" {
            if let Ok(val) = field.text().await {
                preview = val == "true";
            }
        } else if name == "local_path" {
            if let Ok(val) = field.text().await {
                local_path = val.trim().to_string();
            }
        } else if name == "video" {
            video_name = field.file_name().unwrap_or("video.mp4").to_string();
            if let Ok(bytes) = field.bytes().await {
                video_data = Some(bytes.to_vec());
            }
        } else if name == "sync_ref" {
            ref_name = field.file_name().unwrap_or("ref.wav").to_string();
            if let Ok(bytes) = field.bytes().await {
                ref_data = Some(bytes.to_vec());
            }
        }
    }

    let workspace = remasterer::get_workspace_dir();
    let upload_dir = workspace.join("uploads");
    let export_dir = workspace.join("exports");
    
    let _ = fs::create_dir_all(&upload_dir);
    let _ = fs::create_dir_all(&export_dir);

    // Resolve input video path
    let input_path = if !local_path.is_empty() {
        if !PathBuf::from(&local_path).exists() {
            return (StatusCode::BAD_REQUEST, "Local path does not exist").into_response();
        }
        local_path.clone()
    } else if let Some(data) = video_data {
        let dest = upload_dir.join(&video_name);
        if let Err(e) = fs::write(&dest, data) {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to save uploaded video: {}", e)).into_response();
        }
        dest.to_string_lossy().into_owned()
    } else {
        return (StatusCode::BAD_REQUEST, "No input video provided").into_response();
    };

    let filename = PathBuf::from(&input_path).file_name().unwrap_or_default().to_string_lossy().into_owned();

    // Resolve reference audio path
    let ref_path = if let Some(data) = ref_data {
        let dest = upload_dir.join(&ref_name);
        if let Err(e) = fs::write(&dest, data) {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to save reference audio: {}", e)).into_response();
        }
        Some(dest.to_string_lossy().into_owned())
    } else {
        None
    };

    // Prepare outputs
    let task_id = format!("task_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis());
    let prefix = if preview { "preview_" } else { "remastered_" };
    let output_filename = format!("{}{}", prefix, filename);
    let output_path = export_dir.join(&output_filename).to_string_lossy().into_owned();

    // Initialize state
    let task_status = TaskStatus {
        status: "starting".to_string(),
        message: "Initializing worker thread...".to_string(),
        percent: 0,
        output_file: None,
        updated_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
    };
    
    state.tasks.write().await.insert(task_id.clone(), task_status);

    // Spawn background task
    let tasks_clone = state.tasks.clone();
    let task_id_clone = task_id.clone();
    let output_filename_clone = output_filename.clone();

    tokio::spawn(async move {
        let tasks_for_cb = tasks_clone.clone();
        let task_id_for_cb = task_id_clone.clone();

        let cb = move |msg: &str, percent: u32| {
            let mut lock = tasks_for_cb.blocking_write();
            if let Some(t) = lock.get_mut(&task_id_for_cb) {
                t.status = "processing".to_string();
                t.message = msg.to_string();
                t.percent = percent;
                t.updated_at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
            }
        };

        let result = remasterer::process_file(
            &input_path,
            &output_path,
            sync_ms,
            ref_path.as_deref(),
            auto_sync_lips,
            ai_start_sec,
            preview,
            cb,
        );

        let mut lock = tasks_clone.write().await;
        if let Some(t) = lock.get_mut(&task_id_clone) {
            match result {
                Ok(_) => {
                    t.status = "completed".to_string();
                    t.message = "Processing finished successfully!".to_string();
                    t.percent = 100;
                    t.output_file = Some(output_filename_clone);
                }
                Err(e) => {
                    t.status = "failed".to_string();
                    t.message = e;
                    t.percent = 100;
                }
            }
            t.updated_at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        }
    });

    axum::Json(serde_json::json!({
        "task_id": task_id,
        "output_file": output_filename
    }))
    .into_response()
}
