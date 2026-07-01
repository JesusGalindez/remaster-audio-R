mod server;
mod dsp;
mod remasterer;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      
      // Start Axum web server in the background using Tauri's async runtime
      tauri::async_runtime::spawn(async move {
          server::start_server().await;
      });
      
      // Give the server 200ms to bind to the port before webview opens
      std::thread::sleep(std::time::Duration::from_millis(200));
      
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
