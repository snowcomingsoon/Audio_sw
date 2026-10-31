#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

use eframe::egui::{
    self, epaint::Mesh, Align, Align2, Color32, CornerRadius, CursorIcon, Event, FontId, Frame, Id,
    Key, Layout, Margin, Order, Pos2, Rect, RichText, ScrollArea, Sense, Shape, Stroke, StrokeKind,
    Ui, Vec2,
};

use crate::ui_widgets::{brighten, pill_button, StudioPalette};
use crate::wizard::GuidedWizardChoices;

/// Opzioni Studio AI passate a `demucs_pedalboard_master.py` tramite `ai_pipeline_options.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AiPipelineOptions {
    pub autotune_mode: &'static str,
    pub drum_mode: &'static str,
    pub bass_mode: &'static str,
    pub target_bpm: Option<f32>,
    pub force_reseparate: bool,
}

impl Default for AiPipelineOptions {
    fn default() -> Self {
        Self {
            autotune_mode: "off",
            drum_mode: "natural",
            bass_mode: "natural",
            target_bpm: None,
            force_reseparate: false,
        }
    }
}

impl AiPipelineOptions {
    pub fn from_wizard_choices(c: &GuidedWizardChoices) -> Self {
        Self {
            autotune_mode: c.autotune_mode.as_json_str(),
            drum_mode: c.drum_mode.as_json_str(),
            bass_mode: c.bass_mode.as_json_str(),
            target_bpm: c.target_bpm,
            force_reseparate: c.force_reseparate,
        }
    }

    pub fn to_json_pretty(&self) -> String {
        let bpm_txt = match self.target_bpm {
            Some(b) if b.is_finite() && b >= 45.0 => format!("{b:.2}"),
            _ => "null".to_string(),
        };
        let force_txt = if self.force_reseparate {
            "true"
        } else {
            "false"
        };
        format!(
            "{{\n  \"autotune_mode\": \"{}\",\n  \"drum_mode\": \"{}\",\n  \"bass_mode\": \"{}\",\n  \"target_bpm\": {},\n  \"force_reseparate\": {}\n}}\n",
            self.autotune_mode, self.drum_mode, self.bass_mode, bpm_txt, force_txt
        )
    }
}

/// Report telemetrico generato a fine pipeline in `<out_dir>/ai_master_report.json`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AiMasterReport {
    pub detected_key: Option<String>,
    pub effective_bpm: Option<f32>,
    pub kicks: Option<u32>,
    pub snares: Option<u32>,
    pub toms: Option<u32>,
    pub double_pedal_strokes: Option<u32>,
    pub fast_roll_strokes: Option<u32>,
    pub quality_score: Option<f32>,
}

impl AiMasterReport {
    pub fn format_italian_badge(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(ref key) = self.detected_key {
            if !key.trim().is_empty() {
                parts.push(format!("Tonalità: {}", key.trim()));
            }
        }
        if let Some(bpm) = self.effective_bpm {
            if bpm.is_finite() && bpm >= 40.0 {
                parts.push(format!("{bpm:.1} BPM"));
            }
        }
        match (self.kicks, self.snares, self.toms) {
            (Some(k), Some(s), Some(t)) if t > 0 => {
                parts.push(format!("Cassa: {k} | Rullante: {s} | Tom: {t}"));
            }
            (Some(k), Some(s), _) if k > 0 || s > 0 => {
                parts.push(format!("Cassa: {k} | Rullante: {s}"));
            }
            _ => {}
        }
        if let Some(dp) = self.double_pedal_strokes {
            if dp > 0 {
                parts.push(format!("Doppio Pedale: {dp}"));
            }
        }
        if let Some(q) = self.quality_score {
            if q.is_finite() && q > 0.0 {
                parts.push(format!("Qualità AI: {q:.0}/100"));
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("  •  "))
        }
    }
}

pub fn parse_ai_master_report(json_text: &str) -> AiMasterReport {
    let detected_key = extract_json_str(json_text, "detected_key");
    let effective_bpm = extract_json_f32(json_text, "effective_bpm")
        .or_else(|| extract_json_f32(json_text, "detected_bpm"));
    let kicks = extract_json_f32(json_text, "kicks").map(|v| v.max(0.0).round() as u32);
    let snares = extract_json_f32(json_text, "snares").map(|v| v.max(0.0).round() as u32);
    let toms = extract_json_f32(json_text, "toms").map(|v| v.max(0.0).round() as u32);
    let double_pedal_strokes =
        extract_json_f32(json_text, "double_pedal_strokes").map(|v| v.max(0.0).round() as u32);
    let fast_roll_strokes =
        extract_json_f32(json_text, "fast_roll_strokes").map(|v| v.max(0.0).round() as u32);
    let quality_score = extract_json_f32(json_text, "score_0_100");
    AiMasterReport {
        detected_key,
        effective_bpm,
        kicks,
        snares,
        toms,
        double_pedal_strokes,
        fast_roll_strokes,
        quality_score,
    }
}

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

const MAX_LOG_LINES: usize = 18;
const AUTO_CLOSE_DELAY_SECS: f32 = 1.5_f32;
const CORAL_ACCENT: Color32 = Color32::from_rgb(244, 63, 94);

const PIPELINE_STEPS: [&str; 5] = [
    "1. Decodifica & Preparazione Audio 44.1 kHz",
    "2. Separazione Reti Neurali Hybrid Transformer (htdemucs_ft)",
    "3. De-Noise Spettrale & Channel Strip Stems (Voce, Batteria, Basso, Chitarre)",
    "4. Calibrazione Loudness Master ITU-R BS.1770-4 (-11.2 LUFS)",
    "5. Caricamento Automatico 4 Tracce in Timeline",
];

#[derive(Clone, Debug, PartialEq, Eq)]
enum RunOutcome {
    Idle,
    Running,
    Success,
    Cancelled,
    Failed(String),
}

struct SharedAiState {
    outcome: RunOutcome,
    pct: f32,
    active_step: usize,
    saw_zero_based_stage: bool,
    stage_title: String,
    detail: String,
    gpu_badge: String,
    gpu_is_cuda: bool,
    sub_model_idx: u8,
    sub_pct: f32,
    last_tqdm_pct: f32,
    speed_txt: String,
    eta_txt: String,
    started_at: Option<Instant>,
    finished_elapsed_secs: Option<u64>,
    completed_at: Option<Instant>,
    auto_close_armed: bool,
    input_file_label: String,
    logs: VecDeque<String>,
    last_error_line: Option<String>,
    pending_summary: Option<String>,
    latest_report: Option<AiMasterReport>,
}

impl SharedAiState {
    fn new() -> Self {
        Self {
            outcome: RunOutcome::Idle,
            pct: 0.0_f32,
            active_step: 1,
            saw_zero_based_stage: false,
            stage_title: "Pronto per Separazione AI".to_string(),
            detail: "In attesa di avvio...".to_string(),
            gpu_badge: "Rilevamento GPU...".to_string(),
            gpu_is_cuda: false,
            sub_model_idx: 0,
            sub_pct: 0.0_f32,
            last_tqdm_pct: -1.0_f32,
            speed_txt: String::new(),
            eta_txt: String::new(),
            started_at: None,
            finished_elapsed_secs: None,
            completed_at: None,
            auto_close_armed: false,
            input_file_label: String::new(),
            logs: VecDeque::with_capacity(MAX_LOG_LINES + 2),
            last_error_line: None,
            pending_summary: None,
            latest_report: None,
        }
    }

    fn elapsed_secs(&self) -> u64 {
        if let Some(done_s) = self.finished_elapsed_secs {
            done_s
        } else if let Some(t0) = self.started_at {
            t0.elapsed().as_secs()
        } else {
            0
        }
    }
}

/// Controller thread-safe per l'esecuzione nascosta della pipeline AI Python (Demucs v4 + Pedalboard)
/// e il monitoraggio telemetrico real-time dentro l'interfaccia nativa `egui`.
pub struct AiPipelineController {
    running: Arc<AtomicBool>,
    cancel_flag: Arc<AtomicBool>,
    completed_flag: Arc<AtomicBool>,
    run_generation: Arc<AtomicU64>,
    child_proc: Arc<Mutex<Option<Child>>>,
    state: Arc<Mutex<SharedAiState>>,
    modal_open: bool,
    show_tech_log: bool,
}

impl Default for AiPipelineController {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AiPipelineController {
    fn drop(&mut self) {
        if self.is_running() {
            self.cancel();
        }
    }
}

#[allow(dead_code)]
impl AiPipelineController {
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            completed_flag: Arc::new(AtomicBool::new(false)),
            run_generation: Arc::new(AtomicU64::new(0)),
            child_proc: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(SharedAiState::new())),
            modal_open: false,
            show_tech_log: false,
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Restituisce `true` una sola volta al termine di una separazione completata con successo,
    /// consentendo a `main.rs` di ricaricare automaticamente i 4 Stems nella Timeline.
    pub fn take_completed_flag(&self) -> bool {
        self.completed_flag.swap(false, Ordering::SeqCst)
    }

    /// Restituisce l'ultimo riepilogo di stato disponibile per aggiornare `status_msg` nel footer.
    pub fn latest_summary(&self) -> Option<String> {
        self.state
            .lock()
            .ok()
            .and_then(|mut st| st.pending_summary.take())
    }

    /// Restituisce una sola volta il report telemetrico JSON (`Tonalità`, `BPM`, `Cassa/Rullante`) a fine corsa.
    pub fn take_latest_report(&self) -> Option<AiMasterReport> {
        self.state
            .lock()
            .ok()
            .and_then(|mut st| st.latest_report.take())
    }

    pub fn set_modal_open(&mut self, open: bool) {
        self.modal_open = open;
        if open {
            if let Ok(mut st) = self.state.lock() {
                if st.outcome != RunOutcome::Running {
                    st.auto_close_armed = false;
                }
            }
        }
    }

    pub fn is_modal_open(&self) -> bool {
        self.modal_open
    }

    pub fn start(
        &mut self,
        input_file: PathBuf,
        out_dir: PathBuf,
        options: AiPipelineOptions,
        ctx: egui::Context,
    ) {
        if self.is_running() {
            self.modal_open = true;
            return;
        }

        let my_gen = self.run_generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.cancel_flag.store(false, Ordering::SeqCst);
        self.completed_flag.store(false, Ordering::SeqCst);
        self.running.store(true, Ordering::SeqCst);
        self.modal_open = true;
        self.show_tech_log = false;

        let file_label = input_file
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("traccia_audio")
            .to_string();

        if let Ok(mut st) = self.state.lock() {
            st.outcome = RunOutcome::Running;
            st.pct = 2.0_f32;
            st.active_step = 1;
            st.saw_zero_based_stage = false;
            st.stage_title = "Decodifica & Preparazione Audio 44.1 kHz".to_string();
            st.detail = "Avvio motore Python AI (htdemucs_ft) in background...".to_string();
            st.gpu_badge = "Rilevamento GPU...".to_string();
            st.gpu_is_cuda = false;
            st.sub_model_idx = 0;
            st.sub_pct = 0.0_f32;
            st.last_tqdm_pct = -1.0_f32;
            st.speed_txt.clear();
            st.eta_txt.clear();
            st.started_at = Some(Instant::now());
            st.finished_elapsed_secs = None;
            st.completed_at = None;
            st.auto_close_armed = false;
            st.input_file_label = file_label.clone();
            st.logs.clear();
            st.last_error_line = None;
            st.latest_report = None;
            push_log_line(
                &mut st.logs,
                format!("[Avvio] Sorgente: {file_label}  •  Pipeline 44.1 kHz / 24-bit"),
            );
            st.pending_summary = Some(
                "Separazione AI Demucs v4 avviata  •  Monitoraggio real-time attivo".to_string(),
            );
        }
        ctx.request_repaint();

        let running_arc = Arc::clone(&self.running);
        let cancel_arc = Arc::clone(&self.cancel_flag);
        let completed_arc = Arc::clone(&self.completed_flag);
        let gen_arc = Arc::clone(&self.run_generation);
        let child_arc = Arc::clone(&self.child_proc);
        let state_arc = Arc::clone(&self.state);

        std::thread::spawn(move || {
            let py_exe = resolve_python_executable();
            let script_path = resolve_pipeline_script();

            let _ = std::fs::create_dir_all(&out_dir);
            let cfg_path = out_dir.join("ai_pipeline_options.json");
            let _ = std::fs::write(&cfg_path, options.to_json_pretty());

            // Rimuove la cache raw/drumsep/6s solo quando l'utente forza esplicitamente la ri-separazione AI
            if options.force_reseparate {
                let _ = std::fs::remove_dir_all(out_dir.join("stems_raw"));
                let _ = std::fs::remove_dir_all(out_dir.join("stems_drumsep"));
                let _ = std::fs::remove_dir_all(out_dir.join("stems_6s"));
            }

            let enriched_path = build_enriched_path(&py_exe);

            let mut cmd = Command::new(&py_exe);
            cmd.arg("-u")
                .arg(&script_path)
                .arg(&input_file)
                .arg(&out_dir)
                .arg("--config-file")
                .arg(&cfg_path)
                .env("PATH", enriched_path)
                .env("PYTHONUNBUFFERED", "1")
                .env("PYTHONDONTWRITEBYTECODE", "1")
                .env("PYTHONNOUSERSITE", "1")
                .env("PYTHONIOENCODING", "utf-8")
                .env("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
                .env("TORCH_BLAS_PREFER_HIPBLASLT", "0")
                .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());

            // Installer standalone (Windows & macOS Full Setup): modelli Demucs pre-inclusi offline.
            if let Some(app_root) = script_path.parent() {
                let bundled_hf_home = app_root.join("models");
                if bundled_hf_home.join("hub").exists() {
                    if bundled_hf_home
                        .join("hub")
                        .join("models--adefossez--HTDemucs-ft")
                        .exists()
                    {
                        cmd.env("HF_HUB_OFFLINE", "1");
                    }
                    cmd.env("HF_HOME", bundled_hf_home);
                }
            }

            #[cfg(target_os = "windows")]
            {
                cmd.creation_flags(CREATE_NO_WINDOW);
            }

            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(err) => {
                    if gen_arc.load(Ordering::SeqCst) == my_gen {
                        #[cfg(target_os = "macos")]
                        let msg = format!(
                            "Impossibile avviare Python AI ({err}). Esegui 'Attiva Motore AI (Apple M1).command' dal DMG."
                        );
                        #[cfg(not(target_os = "macos"))]
                        let msg = format!("Impossibile avviare Python AI: {err}");
                        if let Ok(mut st) = state_arc.lock() {
                            st.outcome = RunOutcome::Failed(msg.clone());
                            st.detail = msg.clone();
                            st.finished_elapsed_secs = Some(st.elapsed_secs());
                            push_log_line(&mut st.logs, format!("[Errore] {msg}"));
                            st.pending_summary = Some(format!("Errore avvio Separazione AI: {err}"));
                        }
                        running_arc.store(false, Ordering::SeqCst);
                        ctx.request_repaint();
                    }
                    return;
                }
            };

            let stdout_pipe = child.stdout.take();
            let stderr_pipe = child.stderr.take();

            if let Ok(mut slot) = child_arc.lock() {
                *slot = Some(child);
            }

            let out_handle = stdout_pipe.map(|pipe| {
                let st_cl = Arc::clone(&state_arc);
                let cancel_cl = Arc::clone(&cancel_arc);
                let gen_cl = Arc::clone(&gen_arc);
                let ctx_cl = ctx.clone();
                std::thread::spawn(move || {
                    read_stream_splitting_cr_lf(pipe, false, my_gen, &gen_cl, &cancel_cl, &st_cl, &ctx_cl);
                })
            });

            let err_handle = stderr_pipe.map(|pipe| {
                let st_cl = Arc::clone(&state_arc);
                let cancel_cl = Arc::clone(&cancel_arc);
                let gen_cl = Arc::clone(&gen_arc);
                let ctx_cl = ctx.clone();
                std::thread::spawn(move || {
                    read_stream_splitting_cr_lf(pipe, true, my_gen, &gen_cl, &cancel_cl, &st_cl, &ctx_cl);
                })
            });

            // Attende la terminazione senza trattenere il Mutex (così cancel() è istantaneo)
            let wait_result = loop {
                if cancel_arc.load(Ordering::SeqCst) {
                    if let Ok(mut slot) = child_arc.lock() {
                        if let Some(ref mut c) = *slot {
                            terminate_child_tree(c);
                        }
                    }
                }

                let polled = if let Ok(mut slot) = child_arc.lock() {
                    if let Some(ref mut c) = *slot {
                        match c.try_wait() {
                            Ok(Some(status)) => Some(Ok(status)),
                            Ok(None) => None,
                            Err(e) => Some(Err(e)),
                        }
                    } else {
                        break None;
                    }
                } else {
                    None
                };

                if let Some(done) = polled {
                    break Some(done);
                }

                // Richiede repaint periodico per mantenere fluido il timer dei secondi trascorsi
                ctx.request_repaint();
                std::thread::sleep(Duration::from_millis(35));
            };

            if let Some(h) = out_handle {
                let _ = h.join();
            }
            if let Some(h) = err_handle {
                let _ = h.join();
            }

            if let Ok(mut slot) = child_arc.lock() {
                *slot = None;
            }

            if gen_arc.load(Ordering::SeqCst) != my_gen {
                return;
            }

            let was_cancelled = cancel_arc.load(Ordering::SeqCst);
            if was_cancelled {
                if let Ok(mut st) = state_arc.lock() {
                    if st.outcome != RunOutcome::Cancelled {
                        st.outcome = RunOutcome::Cancelled;
                        st.detail = "Elaborazione interrotta dall'utente.".to_string();
                        st.finished_elapsed_secs = Some(st.elapsed_secs());
                        st.speed_txt.clear();
                        st.eta_txt.clear();
                        push_log_line(
                            &mut st.logs,
                            "[Sistema] Separazione AI annullata dall'utente.".to_string(),
                        );
                        st.pending_summary = Some("Separazione AI annullata.".to_string());
                    }
                }
                running_arc.store(false, Ordering::SeqCst);
                ctx.request_repaint();
                return;
            }

            match wait_result {
                Some(Ok(status)) if status.success() => {
                    let parsed_report = std::fs::read_to_string(out_dir.join("ai_master_report.json"))
                        .ok()
                        .map(|txt| parse_ai_master_report(&txt));
                    if let Ok(mut st) = state_arc.lock() {
                        st.outcome = RunOutcome::Success;
                        st.pct = 100.0_f32;
                        st.active_step = 5;
                        st.sub_model_idx = 4;
                        st.sub_pct = 100.0_f32;
                        st.stage_title = "Separazione AI & Mastering Completati".to_string();
                        st.detail = "4 Tracce Studio WAV 24-bit pronte nella Timeline".to_string();
                        st.speed_txt.clear();
                        st.eta_txt.clear();
                        st.finished_elapsed_secs = Some(st.elapsed_secs());
                        st.completed_at = Some(Instant::now());
                        st.auto_close_armed = true;
                        st.latest_report = parsed_report;
                        push_log_line(
                            &mut st.logs,
                            "[Completato] 4 Stems WAV 24-bit + Master -11.2 LUFS pronti in Timeline."
                                .to_string(),
                        );
                        st.pending_summary = Some(
                            "Separazione AI completata! Caricamento 4 tracce Studio in Timeline..."
                                .to_string(),
                        );
                    }
                    completed_arc.store(true, Ordering::SeqCst);
                    running_arc.store(false, Ordering::SeqCst);
                }
                Some(Ok(status)) => {
                    if let Ok(mut st) = state_arc.lock() {
                        let code_txt = status
                            .code()
                            .map(|c| format!("codice {c}"))
                            .unwrap_or_else(|| "terminato".to_string());
                        let base_reason = st
                            .last_error_line
                            .clone()
                            .unwrap_or_else(|| format!("Processo AI terminato ({code_txt})"));
                        #[cfg(target_os = "macos")]
                        let reason = if base_reason.contains("ModuleNotFoundError")
                            || base_reason.contains("No module named")
                        {
                            format!(
                                "{base_reason} — Apri 'Attiva Motore AI (Apple M1).command' nel DMG per completare il setup AI."
                            )
                        } else {
                            base_reason
                        };
                        #[cfg(not(target_os = "macos"))]
                        let reason = base_reason;
                        st.outcome = RunOutcome::Failed(reason.clone());
                        st.detail = reason.clone();
                        st.finished_elapsed_secs = Some(st.elapsed_secs());
                        st.speed_txt.clear();
                        st.eta_txt.clear();
                        push_log_line(&mut st.logs, format!("[Errore] {reason}"));
                        st.pending_summary = Some(format!("Errore Separazione AI: {reason}"));
                    }
                    running_arc.store(false, Ordering::SeqCst);
                }
                Some(Err(err)) => {
                    if let Ok(mut st) = state_arc.lock() {
                        let reason = format!("Errore monitoraggio processo AI: {err}");
                        st.outcome = RunOutcome::Failed(reason.clone());
                        st.detail = reason.clone();
                        st.finished_elapsed_secs = Some(st.elapsed_secs());
                        push_log_line(&mut st.logs, format!("[Errore] {reason}"));
                        st.pending_summary = Some(reason);
                    }
                    running_arc.store(false, Ordering::SeqCst);
                }
                None => {
                    running_arc.store(false, Ordering::SeqCst);
                }
            }

            ctx.request_repaint();
        });
    }

    pub fn cancel(&mut self) {
        self.cancel_flag.store(true, Ordering::SeqCst);

        if let Ok(mut slot) = self.child_proc.lock() {
            if let Some(ref mut child) = *slot {
                terminate_child_tree(child);
            }
        }

        if self.running.swap(false, Ordering::SeqCst) {
            if let Ok(mut st) = self.state.lock() {
                st.outcome = RunOutcome::Cancelled;
                st.detail = "Elaborazione interrotta dall'utente.".to_string();
                st.finished_elapsed_secs = Some(st.elapsed_secs());
                st.speed_txt.clear();
                st.eta_txt.clear();
                push_log_line(
                    &mut st.logs,
                    "[Sistema] Separazione AI annullata dall'utente.".to_string(),
                );
                st.pending_summary = Some("Separazione AI annullata.".to_string());
            }
        }
    }
}

fn candidate_search_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let push_ancestors = |start: PathBuf, out: &mut Vec<PathBuf>| {
        let mut cur: Option<&Path> = Some(start.as_path());
        for _ in 0..5 {
            if let Some(p) = cur {
                if !out.iter().any(|existing| existing == p) {
                    out.push(p.to_path_buf());
                }
                cur = p.parent();
            } else {
                break;
            }
        }
    };

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            // Supporto nativo macOS App Bundle (`StudioEQ Master Pro.app/Contents/MacOS` -> `Contents/Resources`)
            if let Some(contents_dir) = dir.parent() {
                let mac_resources = contents_dir.join("Resources");
                if mac_resources.is_dir() {
                    roots.push(mac_resources);
                }
            }
            push_ancestors(dir.to_path_buf(), &mut roots);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        push_ancestors(cwd, &mut roots);
    }
    roots
}

fn is_windows_store_stub(path: &Path) -> bool {
    path.to_string_lossy()
        .to_ascii_lowercase()
        .contains("windowsapps")
}

fn resolve_python_executable() -> PathBuf {
    #[cfg(not(target_os = "windows"))]
    {
        let sys_standalone_py =
            PathBuf::from("/Library/Application Support/StudioEQ_Master_Pro/python/bin/python3");
        if sys_standalone_py.exists() {
            return sys_standalone_py;
        }
        if let Ok(home) = std::env::var("HOME") {
            let home_dir = PathBuf::from(&home);
            let mac_venv_candidates = [
                home_dir
                    .join("Library")
                    .join("Application Support")
                    .join("StudioEQ_Master_Pro")
                    .join("python")
                    .join("bin")
                    .join("python3"),
                home_dir
                    .join("Library")
                    .join("Application Support")
                    .join("StudioEQ_Master_Pro")
                    .join(".venv")
                    .join("bin")
                    .join("python3"),
                home_dir
                    .join("Library")
                    .join("Application Support")
                    .join("StudioEQ_Master_Pro")
                    .join(".venv")
                    .join("bin")
                    .join("python"),
                home_dir.join(".studioeq_venv").join("bin").join("python3"),
            ];
            for cand in mac_venv_candidates {
                if cand.exists() {
                    return cand;
                }
            }
        }
    }

    for root in candidate_search_roots() {
        #[cfg(target_os = "windows")]
        {
            let venv_py = root.join(".venv").join("Scripts").join("python.exe");
            if venv_py.exists() && !is_windows_store_stub(&venv_py) {
                return venv_py;
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            for bin_name in ["python3", "python"] {
                let venv_py = root.join(".venv").join("bin").join(bin_name);
                if venv_py.exists() && !is_windows_store_stub(&venv_py) {
                    return venv_py;
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(local_app) = std::env::var("LOCALAPPDATA") {
            let py_root = PathBuf::from(&local_app).join("Programs").join("Python");
            if let Ok(entries) = std::fs::read_dir(py_root) {
                let mut found: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path().join("python.exe"))
                    .filter(|p| p.exists() && !is_windows_store_stub(p))
                    .collect();
                found.sort();
                if let Some(best) = found.pop() {
                    return best;
                }
            }
        }

        let mut where_cmd = Command::new("where");
        where_cmd
            .arg("python")
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::null());
        if let Ok(out) = where_cmd.output() {
            if out.status.success() {
                for line in String::from_utf8_lossy(&out.stdout).lines() {
                    let p = PathBuf::from(line.trim());
                    if p.exists() && !is_windows_store_stub(&p) {
                        return p;
                    }
                }
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        for sys_py in [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ] {
            let p = PathBuf::from(sys_py);
            if p.exists() {
                return p;
            }
        }
        return PathBuf::from("python3");
    }

    #[cfg(target_os = "windows")]
    PathBuf::from("python")
}

fn resolve_pipeline_script() -> PathBuf {
    #[cfg(not(target_os = "windows"))]
    {
        let sys_bundle_root = PathBuf::from("/Library/Application Support/StudioEQ_Master_Pro");
        let sys_script = sys_bundle_root.join("demucs_pedalboard_master.py");
        if sys_script.exists() && sys_bundle_root.join("models").join("hub").exists() {
            return sys_script;
        }
        if let Ok(home) = std::env::var("HOME") {
            let user_bundle_root = PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("StudioEQ_Master_Pro");
            let user_script = user_bundle_root.join("demucs_pedalboard_master.py");
            if user_script.exists() && user_bundle_root.join("models").join("hub").exists() {
                return user_script;
            }
        }
    }
    for root in candidate_search_roots() {
        let candidate = root.join("demucs_pedalboard_master.py");
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("demucs_pedalboard_master.py")
}

fn build_enriched_path(py_exe: &Path) -> String {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(py_dir) = py_exe.parent() {
        dirs.push(py_dir.to_path_buf());
    }
    #[cfg(target_os = "windows")]
    if let Ok(local_app) = std::env::var("LOCALAPPDATA") {
        let winget_links = PathBuf::from(&local_app)
            .join("Microsoft")
            .join("WinGet")
            .join("Links");
        if winget_links.exists() {
            dirs.push(winget_links);
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let sys_ffmpeg = PathBuf::from("/Library/Application Support/StudioEQ_Master_Pro/ffmpeg/bin");
        if sys_ffmpeg.is_dir() {
            dirs.push(sys_ffmpeg);
        }
        if let Ok(home) = std::env::var("HOME") {
            let home_support = PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("StudioEQ_Master_Pro");
            for sub in [home_support.join("ffmpeg").join("bin"), home_support.join("bin")] {
                if sub.is_dir() {
                    dirs.push(sub);
                }
            }
        }
        for mac_bin in ["/opt/homebrew/bin", "/usr/local/bin"] {
            let p = PathBuf::from(mac_bin);
            if p.is_dir() {
                dirs.push(p);
            }
        }
    }
    let sep = if cfg!(target_os = "windows") { ";" } else { ":" };
    let mut parts: Vec<String> = dirs
        .into_iter()
        .map(|d| d.to_string_lossy().into_owned())
        .collect();
    if let Ok(existing) = std::env::var("PATH") {
        if !existing.is_empty() {
            parts.push(existing);
        }
    }
    parts.join(sep)
}

fn terminate_child_tree(child: &mut Child) {
    #[cfg(target_os = "windows")]
    {
        let pid = child.id();
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
}

/// Legge uno stream `Read` dividendo sia su `\n` che su `\r` (fondamentale per le barre `tqdm` su `stderr`).
fn read_stream_splitting_cr_lf<R: Read>(
    mut reader: R,
    is_stderr: bool,
    my_gen: u64,
    gen_arc: &AtomicU64,
    cancel_arc: &AtomicBool,
    state_arc: &Mutex<SharedAiState>,
    ctx: &egui::Context,
) {
    let mut chunk = [0u8; 512];
    let mut line_buf: Vec<u8> = Vec::with_capacity(256);

    loop {
        if cancel_arc.load(Ordering::Relaxed) || gen_arc.load(Ordering::Relaxed) != my_gen {
            break;
        }
        match reader.read(&mut chunk) {
            Ok(0) => {
                if !line_buf.is_empty() && gen_arc.load(Ordering::SeqCst) == my_gen {
                    let s = String::from_utf8_lossy(&line_buf);
                    handle_raw_stream_segment(&s, is_stderr, state_arc, ctx);
                }
                break;
            }
            Ok(n) => {
                for &b in &chunk[..n] {
                    if b == b'\n' || b == b'\r' {
                        if !line_buf.is_empty() {
                            if gen_arc.load(Ordering::SeqCst) == my_gen
                                && !cancel_arc.load(Ordering::Relaxed)
                            {
                                let s = String::from_utf8_lossy(&line_buf);
                                handle_raw_stream_segment(&s, is_stderr, state_arc, ctx);
                            }
                            line_buf.clear();
                        }
                    } else if line_buf.len() < 4096 {
                        line_buf.push(b);
                    }
                }
            }
            Err(_) => break,
        }
    }
}

fn strip_ansi_codes(input: &str) -> String {
    if !input.contains('\x1b') {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            if chars.peek() == Some(&'[') {
                let _ = chars.next();
                while let Some(&c) = chars.peek() {
                    let _ = chars.next();
                    if ('@'..='~').contains(&c) {
                        break;
                    }
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn is_ignorable_noise_line(line: &str) -> bool {
    if line.is_empty() {
        return true;
    }
    let lower = line.to_ascii_lowercase();
    if lower.contains("hf_token")
        || lower.contains("unauthenticated requests")
        || lower.contains("hf hub")
        || lower.contains("huggingface_hub")
        || lower.contains("faster downloads")
        || lower.contains("userwarning")
        || lower.contains("futurewarning")
        || lower.contains("deprecationwarning")
        || lower.contains("warnings.warn(")
        || lower.starts_with("warning:")
    {
        return true;
    }

    // Filtra le righe del report JSON grezzo finale stampato su stdout
    let trimmed = line.trim();
    matches!(trimmed, "{" | "}" | "},")
        || trimmed.starts_with("\"original_lufs\"")
        || trimmed.starts_with("\"master_ai\"")
        || trimmed.starts_with("\"lufs\"")
        || trimmed.starts_with("\"rms_db\"")
        || trimmed.starts_with("\"peak_db\"")
        || trimmed.starts_with("\"output_wav\"")
        || trimmed.starts_with("\"output_mp3\"")
        || trimmed.starts_with("\"stems_dir\"")
        || trimmed.starts_with("\"detected_bpm\"")
        || trimmed.starts_with("\"effective_bpm\"")
        || trimmed.starts_with("\"detected_key\"")
        || trimmed.starts_with("\"drum_hits\"")
        || trimmed.starts_with("\"kicks\"")
        || trimmed.starts_with("\"snares\"")
        || trimmed.starts_with("\"autotune_mode\"")
        || trimmed.starts_with("\"drum_mode\"")
        || trimmed.starts_with("\"bass_mode\"")
}

fn push_log_line(logs: &mut VecDeque<String>, entry: String) {
    if entry.trim().is_empty() {
        return;
    }
    if logs.back().map(|last| last == &entry).unwrap_or(false) {
        return;
    }
    logs.push_back(entry);
    while logs.len() > MAX_LOG_LINES {
        logs.pop_front();
    }
}

fn normalize_gpu_badge(raw_gpu: &str, is_cuda_hint: bool) -> (String, bool) {
    let trimmed = raw_gpu.trim();
    let upper = trimmed.to_ascii_uppercase();
    if upper.contains("CPU") && !upper.contains("CUDA") && !upper.contains("ROCM") && !upper.contains("RADEON") {
        let label = if upper.starts_with("CPU") {
            trimmed.replace(" - ", " • ").replace(": ", " • ")
        } else {
            format!("CPU • {trimmed}")
        };
        (label, false)
    } else if upper.contains("ROCM")
        || upper.contains("AMD")
        || upper.contains("RADEON")
        || upper.contains("RX ")
    {
        let stripped = if upper.starts_with("ROCM") || upper.starts_with("CUDA") {
            trimmed[4..]
                .trim_start_matches(|c: char| c == ':' || c == '-' || c == '•' || c.is_whitespace())
        } else {
            trimmed
        };
        if stripped.is_empty() {
            ("ROCm • GPU AMD Attiva".to_string(), true)
        } else {
            (format!("ROCm • {stripped}"), true)
        }
    } else if is_cuda_hint
        || upper.contains("CUDA")
        || upper.contains("NVIDIA")
        || upper.contains("RTX")
        || upper.contains("GTX")
        || upper.contains("GEFORCE")
    {
        if upper.starts_with("CUDA") {
            let rest = trimmed[4..]
                .trim_start_matches(|c: char| c == ':' || c == '-' || c == '•' || c.is_whitespace());
            if rest.is_empty() {
                ("CUDA • GPU Attiva".to_string(), true)
            } else {
                (format!("CUDA • {rest}"), true)
            }
        } else {
            (format!("CUDA • {trimmed}"), true)
        }
    } else if upper.contains("MPS") || upper.contains("METAL") || upper.contains("APPLE") {
        let cleaned = trimmed
            .replace("(Metal MPS)", "")
            .replace("(MPS)", "")
            .trim()
            .to_string();
        if upper.starts_with("METAL") {
            (trimmed.to_string(), true)
        } else if cleaned.is_empty() {
            ("Metal MPS • Apple Silicon GPU".to_string(), true)
        } else {
            (format!("Metal MPS • {cleaned}"), true)
        }
    } else {
        (trimmed.to_string(), false)
    }
}

/// Estrae un campo stringa da un oggetto JSON piatto senza dipendenze esterne, decodificando `\uXXXX` e sequenze escape.
fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start_key = json.find(&needle)?;
    let after_key = json[start_key + needle.len()..].trim_start();
    let after_colon = after_key.strip_prefix(':')?.trim_start();
    if after_colon.starts_with("null") {
        return None;
    }
    let content = after_colon.strip_prefix('"')?;

    let mut out = String::new();
    let mut chars = content.chars();
    while let Some(c) = chars.next() {
        if c == '"' {
            return Some(out);
        }
        if c == '\\' {
            match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'n' => out.push(' '),
                'r' => {}
                't' => out.push(' '),
                'u' => {
                    let mut hex = String::with_capacity(4);
                    for _ in 0..4 {
                        hex.push(chars.next()?);
                    }
                    if let Ok(code) = u32::from_str_radix(&hex, 16) {
                        if let Some(decoded) = char::from_u32(code) {
                            out.push(decoded);
                        }
                    }
                }
                other => out.push(other),
            }
        } else {
            out.push(c);
        }
    }
    None
}

/// Estrae un campo numerico (`f32`) da un oggetto JSON piatto senza `serde_json`.
fn extract_json_f32(json: &str, key: &str) -> Option<f32> {
    let needle = format!("\"{key}\"");
    let start_key = json.find(&needle)?;
    let after_key = json[start_key + needle.len()..].trim_start();
    let after_colon = after_key.strip_prefix(':')?.trim_start();
    let num_end = after_colon
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+'))
        .unwrap_or(after_colon.len());
    if num_end == 0 {
        return None;
    }
    after_colon[..num_end]
        .parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())
}

struct TqdmMetrics {
    sub_pct: f32,
    speed_txt: Option<String>,
    eta_txt: Option<String>,
}

/// Analizza una riga `tqdm` di Demucs, es. `38%|████   | 105.3/274.95 [00:05<00:07, 23.74seconds/s]`.
fn parse_tqdm_line(line: &str) -> Option<TqdmMetrics> {
    let pct_pos = line.find("%|")?;
    let prefix = line[..pct_pos].trim();
    let num_start = prefix
        .char_indices()
        .rev()
        .take_while(|&(_, c)| c.is_ascii_digit() || c == '.')
        .last()
        .map(|(idx, _)| idx)?;
    let sub_pct = prefix[num_start..]
        .parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())?
        .clamp(0.0_f32, 100.0_f32);

    let rest = &line[pct_pos + 2..];
    let mut speed_txt = None;
    let mut eta_txt = None;

    if let (Some(open_b), Some(close_b)) = (rest.rfind('['), rest.rfind(']')) {
        if close_b > open_b + 1 {
            let bracket = &rest[open_b + 1..close_b];
            let mut parts = bracket.split(',');
            if let Some(time_part) = parts.next() {
                if let Some((_, eta_raw)) = time_part.split_once('<') {
                    let eta_clean = eta_raw.trim();
                    if !eta_clean.is_empty() && !eta_clean.contains('?') {
                        eta_txt = Some(format!("ETA {eta_clean}"));
                    }
                }
            }
            if let Some(rate_part) = parts.next() {
                let rate_clean = rate_part.trim();
                if let Some(sec_s) = rate_clean.strip_suffix("seconds/s") {
                    if let Ok(val) = sec_s.trim().parse::<f32>() {
                        if val.is_finite() && val > 0.0_f32 {
                            speed_txt = Some(format!("{val:.1}x tempo reale"));
                        }
                    }
                } else if let Some(inv_s) = rate_clean.strip_suffix("s/seconds") {
                    if let Ok(val) = inv_s.trim().parse::<f32>() {
                        if val.is_finite() && val > 0.001_f32 {
                            speed_txt = Some(format!("{:.2}x tempo reale", 1.0_f32 / val));
                        }
                    }
                } else if rate_clean.ends_with("it/s") && !rate_clean.contains('?') {
                    speed_txt = Some(rate_clean.to_string());
                }
            }
        }
    }

    Some(TqdmMetrics {
        sub_pct,
        speed_txt,
        eta_txt,
    })
}

fn handle_raw_stream_segment(
    raw: &str,
    is_stderr: bool,
    state_arc: &Mutex<SharedAiState>,
    ctx: &egui::Context,
) {
    let stripped = strip_ansi_codes(raw);
    let clean = stripped.trim();
    if is_ignorable_noise_line(clean) {
        return;
    }

    // 1. Telemetria strutturata @@AI_PROGRESS@@ {...}
    if let Some(marker_idx) = clean.find("@@AI_PROGRESS@@") {
        let json_part = clean[marker_idx + "@@AI_PROGRESS@@".len()..].trim();
        let pct_opt = extract_json_f32(json_part, "pct");
        let stage_idx_opt = extract_json_f32(json_part, "stage_idx")
            .or_else(|| extract_json_f32(json_part, "step"));
        let stage_opt = extract_json_str(json_part, "stage");
        let detail_opt = extract_json_str(json_part, "detail");
        let gpu_opt = extract_json_str(json_part, "gpu");

        if let Ok(mut st) = state_arc.lock() {
            if let Some(gpu_raw) = gpu_opt {
                if !gpu_raw.trim().is_empty() {
                    let (badge, is_cuda) = normalize_gpu_badge(&gpu_raw, true);
                    st.gpu_badge = badge;
                    st.gpu_is_cuda = is_cuda;
                }
            }

            if let Some(raw_stage) = stage_idx_opt {
                let idx_i = raw_stage.round() as i32;
                if idx_i <= 0 {
                    st.saw_zero_based_stage = true;
                    st.active_step = 1;
                } else if st.saw_zero_based_stage {
                    st.active_step = ((idx_i + 1) as usize).clamp(1, 5);
                } else {
                    let stage_lower = stage_opt
                        .as_deref()
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    if idx_i == 1
                        && (stage_lower.contains("demucs")
                            || stage_lower.contains("separazione")
                            || stage_lower.contains("transformer"))
                    {
                        st.saw_zero_based_stage = true;
                        st.active_step = 2;
                    } else {
                        st.active_step = (idx_i as usize).clamp(1, 5);
                    }
                }
            }

            if let Some(p) = pct_opt {
                st.pct = st.pct.max(p.clamp(0.0_f32, 100.0_f32));
                if stage_idx_opt.is_none() {
                    st.active_step = if st.pct < 10.0_f32 {
                        1
                    } else if st.pct < 77.0_f32 {
                        2
                    } else if st.pct < 93.0_f32 {
                        3
                    } else if st.pct < 99.0_f32 {
                        4
                    } else {
                        5
                    };
                }
            }

            if st.active_step >= 3 {
                st.sub_model_idx = 4;
                st.sub_pct = 100.0_f32;
                st.speed_txt.clear();
                st.eta_txt.clear();
            }

            if let Some(ref s_txt) = stage_opt {
                if !s_txt.trim().is_empty() {
                    st.stage_title = s_txt.trim().to_string();
                }
            }
            if let Some(ref d_txt) = detail_opt {
                if !d_txt.trim().is_empty() {
                    st.detail = d_txt.trim().to_string();
                }
            }

            let summary_detail = if !st.detail.is_empty() {
                st.detail.clone()
            } else {
                st.stage_title.clone()
            };
            st.pending_summary = Some(format!(
                "Separazione AI ({:.0}%)  •  {}",
                st.pct, summary_detail
            ));

            let log_msg = match (&stage_opt, &detail_opt) {
                (Some(s), Some(d)) if !s.is_empty() && !d.is_empty() => {
                    format!("[{:>3.0}%] {} — {}", st.pct, s.trim(), d.trim())
                }
                (Some(s), _) if !s.is_empty() => format!("[{:>3.0}%] {}", st.pct, s.trim()),
                (_, Some(d)) if !d.is_empty() => format!("[{:>3.0}%] {}", st.pct, d.trim()),
                _ => String::new(),
            };
            push_log_line(&mut st.logs, log_msg);
        }

        ctx.request_repaint();
        return;
    }

    // 2. Avanzamento tqdm real-time (4 passate Transformer htdemucs_ft su stderr)
    if let Some(tqdm) = parse_tqdm_line(clean) {
        if let Ok(mut st) = state_arc.lock() {
            st.active_step = st.active_step.max(2);
            if st.active_step == 2 {
                st.stage_title =
                    "Separazione Reti Neurali Hybrid Transformer (htdemucs_ft)".to_string();
            }

            if st.sub_model_idx == 0 {
                st.sub_model_idx = 1;
            } else if (st.last_tqdm_pct >= 95.0_f32 && tqdm.sub_pct < 20.0_f32)
                || (st.last_tqdm_pct >= 85.0_f32 && tqdm.sub_pct + 40.0_f32 < st.last_tqdm_pct)
            {
                st.sub_model_idx = (st.sub_model_idx + 1).min(4);
            }
            st.last_tqdm_pct = tqdm.sub_pct;
            st.sub_pct = tqdm.sub_pct;

            let mapped_pct = 12.0_f32
                + ((st.sub_model_idx.saturating_sub(1)) as f32 * 16.0_f32)
                + (tqdm.sub_pct * 0.16_f32);
            st.pct = st.pct.max(mapped_pct.clamp(12.0_f32, 76.0_f32));
            st.detail = format!(
                "Rete Neurale Transformer {}/4  •  {:.0}%",
                st.sub_model_idx, tqdm.sub_pct
            );

            if let Some(sp) = tqdm.speed_txt {
                st.speed_txt = sp;
            }
            if let Some(eta) = tqdm.eta_txt {
                st.eta_txt = eta;
            }

            st.pending_summary = Some(format!(
                "Separazione AI ({:.0}%)  •  Modello {}/4 ({:.0}%)",
                st.pct, st.sub_model_idx, tqdm.sub_pct
            ));

            // Aggiorna in-place la riga del modello attivo nella Console Tecnica senza flood
            let tag = format!("[Modello {}/4]", st.sub_model_idx);
            let live_entry = if tqdm.sub_pct >= 99.9_f32 {
                if st.speed_txt.is_empty() {
                    format!("{tag} Passata Transformer completata (100%)")
                } else {
                    format!(
                        "{tag} Passata Transformer completata (100%)  •  {}",
                        st.speed_txt
                    )
                }
            } else {
                let mut parts = format!("{tag} Avanzamento {:.0}%", tqdm.sub_pct);
                if !st.speed_txt.is_empty() {
                    parts.push_str(&format!("  •  {}", st.speed_txt));
                }
                if !st.eta_txt.is_empty() {
                    parts.push_str(&format!("  •  {}", st.eta_txt));
                }
                parts
            };

            let replace_back = st
                .logs
                .back()
                .map(|last| last.starts_with(&tag))
                .unwrap_or(false);
            if replace_back {
                if let Some(back) = st.logs.back_mut() {
                    *back = live_entry;
                }
            } else {
                push_log_line(&mut st.logs, live_entry);
            }
        }

        ctx.request_repaint();
        return;
    }

    // 3. Righe testuali informative di stdout / stderr (Demucs, Pedalboard, LUFS, Errori)
    if let Ok(mut st) = state_arc.lock() {
        if let Some(idx) = clean.find("GPU rilevata:") {
            let gpu_name = clean[idx + "GPU rilevata:".len()..].trim();
            if !gpu_name.is_empty() {
                let (badge, is_cuda) = normalize_gpu_badge(gpu_name, true);
                st.gpu_badge = badge;
                st.gpu_is_cuda = is_cuda;
            }
        } else if clean.contains("device=CPU") {
            st.gpu_badge = "CPU • Elaborazione Multi-Core".to_string();
            st.gpu_is_cuda = false;
        } else if clean.contains("device=CUDA") && !st.gpu_is_cuda {
            st.gpu_badge = "CUDA • Accelerazione GPU Attiva".to_string();
            st.gpu_is_cuda = true;
        }

        if clean.contains("[Demucs AI] Caricamento modello") {
            st.active_step = st.active_step.max(2);
            st.pct = st.pct.max(8.0_f32);
            st.stage_title =
                "Separazione Reti Neurali Hybrid Transformer (htdemucs_ft)".to_string();
            st.detail = "Caricamento pesi neurali htdemucs_ft in memoria GPU...".to_string();
        } else if clean.contains("Separazione 4-Stems completata") {
            st.active_step = st.active_step.max(2);
            st.sub_model_idx = 4;
            st.sub_pct = 100.0_f32;
            st.pct = st.pct.max(76.0_f32);
            st.speed_txt.clear();
            st.eta_txt.clear();
            st.detail = clean.replace("[Demucs AI]", "").trim().to_string();
        } else if clean.contains("[Stem 1/4") {
            st.active_step = st.active_step.max(3);
            st.sub_model_idx = 4;
            st.sub_pct = 100.0_f32;
            st.speed_txt.clear();
            st.eta_txt.clear();
            st.pct = st.pct.max(80.0_f32);
            st.stage_title = "De-Noise Spettrale & Channel Strip Stems".to_string();
            st.detail = "Trattamento Stem 1/4: Voce (Spectral De-Noise & Vocal Strip)".to_string();
        } else if clean.contains("[Stem 2/4") {
            st.active_step = st.active_step.max(3);
            st.pct = st.pct.max(84.0_f32);
            st.stage_title = "De-Noise Spettrale & Channel Strip Stems".to_string();
            st.detail = "Trattamento Stem 2/4: Batteria (Drum Punch & Anti-Box)".to_string();
        } else if clean.contains("[Stem 3/4") {
            st.active_step = st.active_step.max(3);
            st.pct = st.pct.max(88.0_f32);
            st.stage_title = "De-Noise Spettrale & Channel Strip Stems".to_string();
            st.detail = "Trattamento Stem 3/4: Basso (Mono Sub & Anti-Resonance)".to_string();
        } else if clean.contains("[Stem 4/4") {
            st.active_step = st.active_step.max(3);
            st.pct = st.pct.max(92.0_f32);
            st.stage_title = "De-Noise Spettrale & Channel Strip Stems".to_string();
            st.detail =
                "Trattamento Stem 4/4: Chitarre/Altro (De-Buzz & Stereo Widening)".to_string();
        } else if clean.contains("[MASTER BUS]") {
            st.active_step = st.active_step.max(4);
            st.pct = st.pct.max(96.0_f32);
            st.stage_title =
                "Calibrazione Loudness Master ITU-R BS.1770-4 (-11.2 LUFS)".to_string();
            st.detail = "Bus Glue Compressor + True-Peak Limiter (-0.5 dBTP)...".to_string();
        } else if clean.contains("COMPLETATO CON SUCCESSO") {
            st.active_step = 5;
            st.pct = st.pct.max(99.0_f32);
            st.stage_title = "Caricamento Automatico 4 Tracce in Timeline".to_string();
            st.detail = "Finalizzazione WAV 24-bit completata".to_string();
        }

        if is_stderr
            && (clean.contains("Error")
                || clean.contains("Exception")
                || clean.contains("CUDA out of memory")
                || clean.contains("Impossibile"))
        {
            st.last_error_line = Some(clean.to_string());
        }

        push_log_line(&mut st.logs, clean.to_string());
    }

    ctx.request_repaint();
}

// =============================================================================
// COMPONENTI UI MODERN STUDIO (HEADER PILL + MODALE GLASS/OBSIDIAN)
// =============================================================================

/// Pulsante nell'header principale:
/// - Quando inattivo: renderizza `"Separa AI"` e restituisce `true` se cliccato per avviare una nuova separazione.
/// - Quando attivo: renderizza una pillola Viola luminosa con avanzamento live (es. `"AI 64%"`) che riapre la modale.
pub fn draw_ai_header_button(ui: &mut Ui, ctrl: &mut AiPipelineController) -> bool {
    if !ctrl.is_running() {
        let clicked = pill_button(
            ui,
            "Separa AI",
            true,
            StudioPalette::STUDIO_VIOLET,
            Vec2::new(96.0_f32, 34.0_f32),
        )
        .on_hover_text("Separa il brano in Voce, Batteria, Basso e Chitarre con Demucs AI su GPU")
        .clicked();
        return clicked;
    }

    let pct = ctrl
        .state
        .lock()
        .map(|s| s.pct.clamp(0.0_f32, 100.0_f32))
        .unwrap_or(0.0_f32);

    ui.ctx()
        .request_repaint_after(Duration::from_millis(33));

    let label = format!("AI {:.0}%", pct);
    let font = FontId::proportional(13.2_f32);
    let galley = ui
        .painter()
        .layout_no_wrap(label.clone(), font.clone(), StudioPalette::CRISP_TEXT);
    let size = Vec2::new((galley.size().x + 38.0_f32).max(102.0_f32), 34.0_f32);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());

    if ui.is_rect_visible(rect) {
        let time_s = ui.input(|i| i.time) as f32;
        let pulse = 0.5_f32 + 0.5_f32 * (time_s * 3.4_f32).sin();
        let hover_t = ui
            .ctx()
            .animate_bool_responsive(resp.id.with("ai_hdr_hover"), resp.hovered());

        let radius_u8 = (rect.height() * 0.5_f32).round().clamp(2.0_f32, 60.0_f32) as u8;
        let radius = CornerRadius::same(radius_u8);
        let painter = ui.painter();

        // Alone luminoso Viola pulsante
        let glow_alpha = (34.0_f32 + 28.0_f32 * pulse + 22.0_f32 * hover_t)
            .clamp(0.0_f32, 110.0_f32)
            .round() as u8;
        painter.rect_filled(
            rect.expand(2.6_f32 + 0.8_f32 * pulse),
            radius,
            with_alpha(StudioPalette::STUDIO_VIOLET, glow_alpha),
        );

        // Sfondo scuro satinato + riempimento progressivo interno
        painter.rect_filled(rect, radius, Color32::from_rgb(24, 20, 44));

        let fill_frac = (pct / 100.0_f32).clamp(0.06_f32, 1.0_f32);
        let fill_w = rect.width() * fill_frac;
        let fill_rect = Rect::from_min_size(rect.min, Vec2::new(fill_w, rect.height()));
        let fill_clip = ui.clip_rect().intersect(fill_rect);
        if fill_clip.is_positive() {
            let fill_col = lerp_color(
                StudioPalette::STUDIO_VIOLET,
                StudioPalette::ELECTRIC_CYAN,
                fill_frac * 0.65_f32,
            );
            painter
                .with_clip_rect(fill_clip)
                .rect_filled(rect, radius, with_alpha(fill_col, 175));
        }

        let border_col = lerp_color(
            brighten(StudioPalette::STUDIO_VIOLET, 24),
            StudioPalette::ELECTRIC_CYAN,
            0.35_f32 * pulse + 0.35_f32 * hover_t,
        );
        painter.rect_stroke(
            rect,
            radius,
            Stroke::new(1.2_f32, border_col),
            StrokeKind::Inside,
        );

        // Punto LED attivo a sinistra + etichetta nitida
        let dot_pos = Pos2::new(rect.left() + 13.5_f32, rect.center().y);
        painter.circle_filled(
            dot_pos,
            5.2_f32,
            with_alpha(StudioPalette::ELECTRIC_CYAN, (70.0_f32 + 70.0_f32 * pulse) as u8),
        );
        painter.circle_filled(dot_pos, 2.8_f32, StudioPalette::CRISP_TEXT);

        painter.text(
            Pos2::new(rect.center().x + 5.5_f32, rect.center().y),
            Align2::CENTER_CENTER,
            &label,
            font,
            StudioPalette::CRISP_TEXT,
        );
    }

    if resp.hovered() {
        ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
    }

    if resp
        .on_hover_text("Separazione AI in esecuzione — Clicca per aprire il pannello di avanzamento")
        .clicked()
    {
        ctrl.set_modal_open(true);
    }

    false
}

/// Finestra modale centrata Obsidian/Glass (`560px`) con telemetria GPU, barra Hero Violet->Cyan,
/// 4 sub-passate Transformer, checklist a 5 step e console tecnica espandibile.
pub fn draw_ai_pipeline_modal(ctx: &egui::Context, ctrl: &mut AiPipelineController) {
    if !ctrl.is_modal_open() {
        return;
    }

    // Snapshot veloce dello stato condiviso per minimizzare il tempo di lock
    let (
        outcome,
        pct,
        active_step,
        stage_title,
        detail,
        gpu_badge,
        gpu_is_cuda,
        sub_model_idx,
        sub_pct,
        speed_txt,
        eta_txt,
        elapsed_s,
        completed_elapsed_f32,
        auto_close_armed,
        file_label,
        logs_snapshot,
    ) = if let Ok(st) = ctrl.state.lock() {
        (
            st.outcome.clone(),
            st.pct.clamp(0.0_f32, 100.0_f32),
            st.active_step.clamp(1, 5),
            st.stage_title.clone(),
            st.detail.clone(),
            st.gpu_badge.clone(),
            st.gpu_is_cuda,
            st.sub_model_idx,
            st.sub_pct.clamp(0.0_f32, 100.0_f32),
            st.speed_txt.clone(),
            st.eta_txt.clone(),
            st.elapsed_secs(),
            st.completed_at.map(|t| t.elapsed().as_secs_f32()),
            st.auto_close_armed,
            st.input_file_label.clone(),
            st.logs.iter().cloned().collect::<Vec<String>>(),
        )
    } else {
        return;
    };

    let is_running = outcome == RunOutcome::Running;

    // Auto-espande il log tecnico se si verifica un errore così l'utente vede subito la causa
    if matches!(outcome, RunOutcome::Failed(_)) && !ctrl.show_tech_log && !logs_snapshot.is_empty()
    {
        ctrl.show_tech_log = true;
    }

    // Chiusura automatica morbida 1.5s dopo il completamento (a meno che l'utente non stia consultando il log)
    if outcome == RunOutcome::Success && auto_close_armed {
        if ctrl.show_tech_log {
            // L'utente sta ispezionando il log tecnico: sospende l'auto-close
            if let Ok(mut st) = ctrl.state.lock() {
                st.completed_at = Some(Instant::now());
            }
        } else if let Some(done_dt) = completed_elapsed_f32 {
            if done_dt >= AUTO_CLOSE_DELAY_SECS {
                if let Ok(mut st) = ctrl.state.lock() {
                    st.auto_close_armed = false;
                }
                ctrl.set_modal_open(false);
                return;
            }
            ctx.request_repaint_after(Duration::from_millis(40));
        }
    }

    if is_running {
        ctx.request_repaint_after(Duration::from_millis(33));
    }

    // Tasto Esc: riduce in background se attivo, altrimenti chiude la modale
    if ctx.input_mut(|i| consume_escape_single_press(i)) {
        ctrl.set_modal_open(false);
        return;
    }

    let screen = ctx.screen_rect();
    let mut backdrop_clicked = false;
    let mut action_hide_modal = false;
    let mut action_cancel = false;

    // 1. Backdrop oscurato morbido
    egui::Area::new(Id::new("studio_eq_ai_pipeline_backdrop"))
        .order(Order::Middle)
        .fixed_pos(screen.min)
        .show(ctx, |ui| {
            let (rect, resp) = ui.allocate_exact_size(screen.size(), Sense::click());
            ui.painter().rect_filled(
                rect,
                CornerRadius::ZERO,
                Color32::from_rgba_unmultiplied(5, 8, 15, 184),
            );
            if resp.clicked() {
                backdrop_clicked = true;
            }
        });

    let card_width = 560.0_f32.min((screen.width() - 28.0_f32).max(340.0_f32));
    let mut card_rect_drawn = Rect::NOTHING;

    // 2. Card centrale Obsidian / Glass (560px, rounded 14px, 1px STUDIO_VIOLET / SUBTLE_HAIRLINE)
    egui::Area::new(Id::new("studio_eq_ai_pipeline_modal_card"))
        .order(Order::Foreground)
        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
        .show(ctx, |ui| {
            let border_accent = match outcome {
                RunOutcome::Success => {
                    lerp_color(StudioPalette::SUBTLE_HAIRLINE, StudioPalette::EMERALD, 0.62_f32)
                }
                RunOutcome::Failed(_) => {
                    lerp_color(StudioPalette::SUBTLE_HAIRLINE, CORAL_ACCENT, 0.62_f32)
                }
                RunOutcome::Cancelled => {
                    lerp_color(StudioPalette::SUBTLE_HAIRLINE, StudioPalette::WARM_AMBER, 0.52_f32)
                }
                _ => lerp_color(
                    StudioPalette::SUBTLE_HAIRLINE,
                    StudioPalette::STUDIO_VIOLET,
                    0.58_f32,
                ),
            };

            let inner_resp = Frame::NONE
                .fill(Color32::from_rgb(15, 20, 31))
                .corner_radius(CornerRadius::same(14))
                .stroke(Stroke::new(1.0_f32, border_accent))
                .inner_margin(Margin::symmetric(20, 18))
                .show(ui, |ui| {
                    let inner_w = (card_width - 40.0_f32).max(300.0_f32);
                    ui.set_min_width(inner_w);
                    ui.set_max_width(inner_w);

                    // =========================================================
                    // 1. TOP ROW: TITOLO + BADGE HARDWARE GPU + PULSANTE RIDUCI
                    // =========================================================
                    ui.horizontal(|ui| {
                        let status_dot_col = match outcome {
                            RunOutcome::Success => StudioPalette::EMERALD,
                            RunOutcome::Failed(_) => CORAL_ACCENT,
                            RunOutcome::Cancelled => StudioPalette::WARM_AMBER,
                            _ => StudioPalette::STUDIO_VIOLET,
                        };
                        draw_glowing_dot(ui, status_dot_col, is_running);

                        ui.label(
                            RichText::new("SEPARAZIONE AI & MASTERING STUDIO")
                                .size(14.8_f32)
                                .strong()
                                .color(StudioPalette::CRISP_TEXT),
                        );

                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let top_btn_lbl = if is_running { "Riduci" } else { "Chiudi" };
                            if pill_button(
                                ui,
                                top_btn_lbl,
                                false,
                                StudioPalette::ELECTRIC_CYAN,
                                Vec2::new(68.0_f32, 28.0_f32),
                            )
                            .on_hover_text(if is_running {
                                "Nascondi la finestra in background continuando l'elaborazione"
                            } else {
                                "Chiudi pannello"
                            })
                            .clicked()
                            {
                                action_hide_modal = true;
                            }
                        });
                    });

                    ui.add_space(6.0_f32);

                    // Riga Hardware Pill Badge GPU + nome file sorgente
                    ui.horizontal(|ui| {
                        draw_gpu_hardware_pill(ui, &gpu_badge, gpu_is_cuda, is_running);

                        if !file_label.is_empty() {
                            ui.add_space(4.0_f32);
                            ui.label(
                                RichText::new(format!("•  {}", truncate_middle(&file_label, 30)))
                                    .size(11.5_f32)
                                    .color(StudioPalette::MUTED_LABEL),
                            );
                        }
                    });

                    ui.add_space(10.0_f32);
                    draw_hairline_separator(ui);
                    ui.add_space(10.0_f32);

                    // =========================================================
                    // 2. HERO PROGRESS BAR (0% - 100%) + TELEMETRIA VELOCITÀ/ETA
                    // =========================================================
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.set_max_width((inner_w - 110.0_f32).max(180.0_f32));
                            ui.label(
                                RichText::new(&stage_title)
                                    .size(13.2_f32)
                                    .strong()
                                    .color(StudioPalette::CRISP_TEXT),
                            );
                            ui.add_space(2.0_f32);
                            ui.label(
                                RichText::new(&detail)
                                    .size(11.8_f32)
                                    .color(StudioPalette::MUTED_LABEL),
                            );
                        });

                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let pct_col = match outcome {
                                RunOutcome::Success => StudioPalette::EMERALD,
                                RunOutcome::Failed(_) => CORAL_ACCENT,
                                RunOutcome::Cancelled => StudioPalette::WARM_AMBER,
                                _ => StudioPalette::CRISP_TEXT,
                            };
                            ui.label(
                                RichText::new(format!("{pct:>5.1}%"))
                                    .monospace()
                                    .size(23.0_f32)
                                    .strong()
                                    .color(pct_col),
                            );
                        });
                    });

                    ui.add_space(8.0_f32);
                    draw_hero_gradient_progress_bar(ui, inner_w, 18.0_f32, pct, &outcome);
                    ui.add_space(6.0_f32);

                    // Strip telemetrica inferiore: Velocità x tempo reale + ETA + Timer trascorso
                    ui.horizontal(|ui| {
                        let telemetry_left = match (&speed_txt.is_empty(), &eta_txt.is_empty()) {
                            (false, false) => format!("{speed_txt}  •  {eta_txt}"),
                            (false, true) => speed_txt.clone(),
                            (true, false) => eta_txt.clone(),
                            (true, true) => match outcome {
                                RunOutcome::Success => {
                                    "Elaborazione completata a 24-bit / 44.1 kHz".to_string()
                                }
                                RunOutcome::Cancelled => "Sessione AI interrotta".to_string(),
                                RunOutcome::Failed(_) => "Elaborazione interrotta per errore".to_string(),
                                _ => "Elaborazione neurale / DSP attiva...".to_string(),
                            },
                        };

                        let telem_col = if !speed_txt.is_empty() {
                            StudioPalette::ELECTRIC_CYAN
                        } else {
                            StudioPalette::MUTED_LABEL
                        };
                        ui.label(
                            RichText::new(telemetry_left)
                                .monospace()
                                .size(11.3_f32)
                                .color(telem_col),
                        );

                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let mins = elapsed_s / 60;
                            let secs = elapsed_s % 60;
                            ui.label(
                                RichText::new(format!("Trascorso {mins:02}:{secs:02}"))
                                    .monospace()
                                    .size(11.3_f32)
                                    .color(StudioPalette::MUTED_LABEL),
                            );
                        });
                    });

                    // =========================================================
                    // 3. SUB-MODEL NEURAL PASS BAR (Mostrata durante Stage 2 Demucs)
                    // =========================================================
                    if active_step == 2 && is_running {
                        ui.add_space(10.0_f32);
                        draw_submodel_pass_bar(ui, inner_w, sub_model_idx, sub_pct);
                    }

                    ui.add_space(12.0_f32);

                    // =========================================================
                    // 4. CHECKLIST VISUALE DELLA PIPELINE A 5 STEP
                    // =========================================================
                    draw_five_step_checklist(ui, inner_w, active_step, &outcome);

                    ui.add_space(10.0_f32);

                    // =========================================================
                    // 5. TOGGLE COLLASSABILE "Mostra Log Tecnico Live"
                    // =========================================================
                    draw_collapsible_tech_console(
                        ui,
                        inner_w,
                        &mut ctrl.show_tech_log,
                        &logs_snapshot,
                    );

                    ui.add_space(12.0_f32);
                    draw_hairline_separator(ui);
                    ui.add_space(10.0_f32);

                    // =========================================================
                    // 6. RIGA AZIONI INFERIORE (BACKGROUND / ANNULLA / COMPLETATO)
                    // =========================================================
                    match outcome {
                        RunOutcome::Running | RunOutcome::Idle => {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new("L'audio resta utilizzabile in background")
                                        .size(11.4_f32)
                                        .color(StudioPalette::MUTED_LABEL),
                                );

                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if pill_button(
                                        ui,
                                        "Annulla",
                                        true,
                                        CORAL_ACCENT,
                                        Vec2::new(86.0_f32, 32.0_f32),
                                    )
                                    .on_hover_text("Interrompi immediatamente il processo Python AI")
                                    .clicked()
                                    {
                                        action_cancel = true;
                                    }

                                    if pill_button(
                                        ui,
                                        "Nascondi in Background",
                                        false,
                                        StudioPalette::STUDIO_VIOLET,
                                        Vec2::new(164.0_f32, 32.0_f32),
                                    )
                                    .on_hover_text(
                                        "Riduci questa finestra: vedrai la percentuale live nel pulsante in alto",
                                    )
                                    .clicked()
                                    {
                                        action_hide_modal = true;
                                    }
                                });
                            });
                        }
                        RunOutcome::Success => {
                            Frame::NONE
                                .fill(with_alpha(StudioPalette::EMERALD, 26))
                                .corner_radius(CornerRadius::same(10))
                                .stroke(Stroke::new(
                                    1.0_f32,
                                    with_alpha(StudioPalette::EMERALD, 135),
                                ))
                                .inner_margin(Margin::symmetric(12, 9))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.horizontal(|ui| {
                                        draw_check_circle_icon(ui, StudioPalette::EMERALD);
                                        ui.label(
                                            RichText::new(
                                                "Separazione Completata! Le 4 tracce sono pronte nella Timeline.",
                                            )
                                            .size(12.2_f32)
                                            .strong()
                                            .color(StudioPalette::CRISP_TEXT),
                                        );

                                        ui.with_layout(
                                            Layout::right_to_left(Align::Center),
                                            |ui| {
                                                if pill_button(
                                                    ui,
                                                    "Chiudi e Ascolta",
                                                    true,
                                                    StudioPalette::EMERALD,
                                                    Vec2::new(128.0_f32, 32.0_f32),
                                                )
                                                .clicked()
                                                {
                                                    action_hide_modal = true;
                                                }
                                            },
                                        );
                                    });
                                });
                        }
                        RunOutcome::Cancelled => {
                            Frame::NONE
                                .fill(with_alpha(StudioPalette::WARM_AMBER, 22))
                                .corner_radius(CornerRadius::same(10))
                                .stroke(Stroke::new(
                                    1.0_f32,
                                    with_alpha(StudioPalette::WARM_AMBER, 120),
                                ))
                                .inner_margin(Margin::symmetric(12, 9))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new(
                                                "Separazione AI interrotta. La traccia attuale resta intatta.",
                                            )
                                            .size(12.2_f32)
                                            .color(StudioPalette::CRISP_TEXT),
                                        );
                                        ui.with_layout(
                                            Layout::right_to_left(Align::Center),
                                            |ui| {
                                                if pill_button(
                                                    ui,
                                                    "Chiudi",
                                                    false,
                                                    StudioPalette::WARM_AMBER,
                                                    Vec2::new(88.0_f32, 30.0_f32),
                                                )
                                                .clicked()
                                                {
                                                    action_hide_modal = true;
                                                }
                                            },
                                        );
                                    });
                                });
                        }
                        RunOutcome::Failed(ref err_msg) => {
                            Frame::NONE
                                .fill(with_alpha(CORAL_ACCENT, 24))
                                .corner_radius(CornerRadius::same(10))
                                .stroke(Stroke::new(1.0_f32, with_alpha(CORAL_ACCENT, 135)))
                                .inner_margin(Margin::symmetric(12, 9))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.horizontal(|ui| {
                                        ui.vertical(|ui| {
                                            ui.set_max_width((inner_w - 125.0_f32).max(160.0_f32));
                                            ui.label(
                                                RichText::new("Errore durante l'elaborazione AI")
                                                    .size(12.2_f32)
                                                    .strong()
                                                    .color(StudioPalette::CRISP_TEXT),
                                            );
                                            ui.label(
                                                RichText::new(truncate_middle(err_msg, 62))
                                                    .size(11.2_f32)
                                                    .color(Color32::from_rgb(253, 164, 175)),
                                            );
                                        });
                                        ui.with_layout(
                                            Layout::right_to_left(Align::Center),
                                            |ui| {
                                                if pill_button(
                                                    ui,
                                                    "Chiudi",
                                                    true,
                                                    CORAL_ACCENT,
                                                    Vec2::new(88.0_f32, 30.0_f32),
                                                )
                                                .clicked()
                                                {
                                                    action_hide_modal = true;
                                                }
                                            },
                                        );
                                    });
                                });
                        }
                    }
                });

            card_rect_drawn = inner_resp.response.rect;

            // Alone perimetrale morbido esterno + riflesso speculare superiore sulla card Glass
            let halo_col = match outcome {
                RunOutcome::Success => with_alpha(StudioPalette::EMERALD, 26),
                RunOutcome::Failed(_) => with_alpha(CORAL_ACCENT, 26),
                RunOutcome::Cancelled => with_alpha(StudioPalette::WARM_AMBER, 20),
                _ => with_alpha(StudioPalette::STUDIO_VIOLET, 26),
            };
            ui.painter().rect_stroke(
                card_rect_drawn,
                CornerRadius::same(14),
                Stroke::new(3.5_f32, halo_col),
                StrokeKind::Outside,
            );

            let top_spec = Rect::from_min_max(
                Pos2::new(card_rect_drawn.left() + 12.0_f32, card_rect_drawn.top() + 1.0_f32),
                Pos2::new(card_rect_drawn.right() - 12.0_f32, card_rect_drawn.top() + 2.0_f32),
            );
            ui.painter().line_segment(
                [top_spec.left_center(), top_spec.right_center()],
                Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 28)),
            );
        });

    if action_cancel {
        ctrl.cancel();
    } else if action_hide_modal {
        ctrl.set_modal_open(false);
    } else if backdrop_clicked {
        let pointer_pos = ctx.input(|i| i.pointer.interact_pos());
        let clicked_outside = pointer_pos
            .map(|p| !card_rect_drawn.contains(p))
            .unwrap_or(true);
        if clicked_outside {
            ctrl.set_modal_open(false);
        }
    }
}

// =============================================================================
// SOTTO-WIDGET PITTORICI PER LA MODALE AI
// =============================================================================

fn draw_glowing_dot(ui: &mut Ui, accent: Color32, animate: bool) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(12.0_f32), Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    let pulse = if animate {
        let t = ui.input(|i| i.time) as f32;
        0.5_f32 + 0.5_f32 * (t * 3.6_f32).sin()
    } else {
        0.75_f32
    };
    let c = rect.center();
    ui.painter().circle_filled(
        c,
        5.8_f32,
        with_alpha(accent, (42.0_f32 + 54.0_f32 * pulse) as u8),
    );
    ui.painter().circle_filled(c, 3.3_f32, brighten(accent, 30));
}

fn draw_gpu_hardware_pill(ui: &mut Ui, badge_text: &str, is_cuda: bool, is_running: bool) {
    let accent = if is_cuda {
        StudioPalette::EMERALD
    } else if badge_text.starts_with("CPU") {
        StudioPalette::WARM_AMBER
    } else {
        StudioPalette::ELECTRIC_CYAN
    };

    let border_col = if is_cuda {
        lerp_color(StudioPalette::EMERALD, StudioPalette::ELECTRIC_CYAN, 0.38_f32)
    } else {
        with_alpha(accent, 130)
    };

    Frame::NONE
        .fill(with_alpha(accent, 24))
        .corner_radius(CornerRadius::same(11))
        .stroke(Stroke::new(1.0_f32, border_col))
        .inner_margin(Margin::symmetric(10, 3))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0_f32;
                let (dot_r, _) = ui.allocate_exact_size(Vec2::splat(8.0_f32), Sense::hover());
                let pulse = if is_running {
                    let t = ui.input(|i| i.time) as f32;
                    0.5_f32 + 0.5_f32 * (t * 3.0_f32).sin()
                } else {
                    0.6_f32
                };
                ui.painter().circle_filled(
                    dot_r.center(),
                    4.4_f32,
                    with_alpha(accent, (50.0_f32 + 55.0_f32 * pulse) as u8),
                );
                ui.painter()
                    .circle_filled(dot_r.center(), 2.5_f32, brighten(accent, 28));

                let text_col = if is_cuda {
                    Color32::from_rgb(209, 250, 229)
                } else {
                    StudioPalette::CRISP_TEXT
                };
                ui.label(
                    RichText::new(badge_text)
                        .monospace()
                        .size(11.2_f32)
                        .strong()
                        .color(text_col),
                );
            });
        });
}

fn draw_hairline_separator(ui: &mut Ui) {
    let (_, rect) = ui.allocate_space(Vec2::new(ui.available_width(), 1.0_f32));
    ui.painter().line_segment(
        [rect.left_center(), rect.right_center()],
        Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE),
    );
}

/// Barra di avanzamento Hero (`height: 18px`) con scanalatura incassata, riempimento sfumato GPU
/// `STUDIO_VIOLET -> ELECTRIC_CYAN` e highlight speculare superiore di `1px`.
fn draw_hero_gradient_progress_bar(
    ui: &mut Ui,
    width: f32,
    height: f32,
    pct: f32,
    outcome: &RunOutcome,
) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }

    let radius_u8 = (height * 0.5_f32).round().clamp(2.0_f32, 40.0_f32) as u8;
    let radius = CornerRadius::same(radius_u8);
    let painter = ui.painter();

    // Scanalatura incassata scura
    painter.rect_filled(rect, radius, StudioPalette::DEEP_OBSIDIAN);
    painter.rect_stroke(
        rect,
        radius,
        Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE),
        StrokeKind::Inside,
    );
    painter.line_segment(
        [
            Pos2::new(rect.left() + 6.0_f32, rect.top() + 1.0_f32),
            Pos2::new(rect.right() - 6.0_f32, rect.top() + 1.0_f32),
        ],
        Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(0, 0, 0, 110)),
    );

    let norm = (pct / 100.0_f32).clamp(0.0_f32, 1.0_f32);
    if norm <= 0.002_f32 {
        return;
    }

    let fill_w = (rect.width() * norm).max(height);
    let fill_right = (rect.left() + fill_w).min(rect.right());
    let fill_rect = Rect::from_min_max(rect.min, Pos2::new(fill_right, rect.max.y));

    let (col_start, col_end) = match outcome {
        RunOutcome::Success => (
            lerp_color(StudioPalette::ELECTRIC_CYAN, StudioPalette::EMERALD, 0.45_f32),
            StudioPalette::EMERALD,
        ),
        RunOutcome::Failed(_) => (StudioPalette::STUDIO_VIOLET, CORAL_ACCENT),
        RunOutcome::Cancelled => (StudioPalette::STUDIO_VIOLET, StudioPalette::WARM_AMBER),
        _ => (
            StudioPalette::STUDIO_VIOLET,
            lerp_color(
                StudioPalette::STUDIO_VIOLET,
                StudioPalette::ELECTRIC_CYAN,
                (0.35_f32 + 0.65_f32 * norm).clamp(0.0_f32, 1.0_f32),
            ),
        ),
    };

    // Alone esterno soffuso attorno alla barra attiva
    painter.rect_filled(
        fill_rect.expand(1.8_f32),
        CornerRadius::same(radius_u8.saturating_add(2)),
        with_alpha(col_end, 36),
    );

    // Capsula sfumata continua (cerchio sx + cerchio dx + Mesh centrale)
    let r = height * 0.5_f32;
    if fill_rect.width() <= height + 0.5_f32 {
        painter.rect_filled(fill_rect, radius, col_start);
    } else {
        let left_c = Pos2::new(fill_rect.left() + r, fill_rect.center().y);
        let right_c = Pos2::new(fill_rect.right() - r, fill_rect.center().y);
        painter.circle_filled(left_c, r, col_start);
        painter.circle_filled(right_c, r, col_end);

        let mut mesh = Mesh::default();
        mesh.colored_vertex(Pos2::new(left_c.x, fill_rect.top()), col_start);
        mesh.colored_vertex(Pos2::new(left_c.x, fill_rect.bottom()), col_start);
        mesh.colored_vertex(Pos2::new(right_c.x, fill_rect.top()), col_end);
        mesh.colored_vertex(Pos2::new(right_c.x, fill_rect.bottom()), col_end);
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(1, 3, 2);
        painter.add(Shape::mesh(mesh));
    }

    // Riflesso animato leggero quando in esecuzione
    if *outcome == RunOutcome::Running && fill_rect.width() > 36.0_f32 {
        let t = ui.input(|i| i.time) as f32;
        let phase = (t * 0.65_f32).fract();
        let sheen_cx = fill_rect.left() + phase * fill_rect.width();
        let sheen_half = 24.0_f32;
        let sx0 = (sheen_cx - sheen_half).max(fill_rect.left() + r);
        let sx1 = (sheen_cx + sheen_half).min(fill_rect.right() - r);
        if sx1 > sx0 + 2.0_f32 {
            let mut sheen = Mesh::default();
            let transparent = Color32::from_rgba_unmultiplied(255, 255, 255, 0);
            let peak = Color32::from_rgba_unmultiplied(255, 255, 255, 34);
            let smid = 0.5_f32 * (sx0 + sx1);
            sheen.colored_vertex(Pos2::new(sx0, fill_rect.top()), transparent);
            sheen.colored_vertex(Pos2::new(sx0, fill_rect.bottom()), transparent);
            sheen.colored_vertex(Pos2::new(smid, fill_rect.top()), peak);
            sheen.colored_vertex(Pos2::new(smid, fill_rect.bottom()), peak);
            sheen.colored_vertex(Pos2::new(sx1, fill_rect.top()), transparent);
            sheen.colored_vertex(Pos2::new(sx1, fill_rect.bottom()), transparent);
            sheen.add_triangle(0, 1, 2);
            sheen.add_triangle(1, 3, 2);
            sheen.add_triangle(2, 3, 4);
            sheen.add_triangle(3, 5, 4);
            painter.add(Shape::mesh(sheen));
        }
    }

    // Top 1px specular highlight lungo la cresta superiore del riempimento
    if fill_rect.width() > 14.0_f32 {
        painter.line_segment(
            [
                Pos2::new(fill_rect.left() + 6.5_f32, fill_rect.top() + 1.5_f32),
                Pos2::new(fill_rect.right() - 6.5_f32, fill_rect.top() + 1.5_f32),
            ],
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 72)),
        );
    }
}

/// Barra a 4 segmenti per le 4 reti Transformer del bag-of-models `htdemucs_ft`.
fn draw_submodel_pass_bar(ui: &mut Ui, total_w: f32, sub_model_idx: u8, sub_pct: f32) {
    let gap = 8.0_f32;
    let pill_w = ((total_w - gap * 3.0_f32) / 4.0_f32).max(56.0_f32);
    let pill_h = 26.0_f32;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = gap;
        for pass in 1u8..=4u8 {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(pill_w, pill_h), Sense::hover());
            if !ui.is_rect_visible(rect) {
                continue;
            }
            let painter = ui.painter();
            let radius = CornerRadius::same(7);

            let is_done =
                pass < sub_model_idx || (pass == sub_model_idx && sub_pct >= 99.9_f32);
            let is_current = pass == sub_model_idx && !is_done;

            if is_done {
                painter.rect_filled(rect, radius, with_alpha(StudioPalette::EMERALD, 32));
                painter.rect_stroke(
                    rect,
                    radius,
                    Stroke::new(1.0_f32, with_alpha(StudioPalette::EMERALD, 155)),
                    StrokeKind::Inside,
                );
                let icon_c = Pos2::new(rect.left() + 12.0_f32, rect.center().y);
                draw_mini_checkmark(painter, icon_c, StudioPalette::EMERALD);
                painter.text(
                    Pos2::new(rect.center().x + 5.0_f32, rect.center().y),
                    Align2::CENTER_CENTER,
                    format!("Modello {pass}/4"),
                    FontId::proportional(11.3_f32),
                    Color32::from_rgb(209, 250, 229),
                );
            } else if is_current {
                painter.rect_filled(rect, radius, Color32::from_rgb(20, 18, 38));

                let frac = (sub_pct / 100.0_f32).clamp(0.04_f32, 1.0_f32);
                let sub_fill =
                    Rect::from_min_size(rect.min, Vec2::new(rect.width() * frac, rect.height()));
                let clip = ui.clip_rect().intersect(sub_fill);
                if clip.is_positive() {
                    painter.with_clip_rect(clip).rect_filled(
                        rect,
                        radius,
                        with_alpha(StudioPalette::STUDIO_VIOLET, 96),
                    );
                }

                painter.rect_stroke(
                    rect,
                    radius,
                    Stroke::new(1.1_f32, brighten(StudioPalette::STUDIO_VIOLET, 24)),
                    StrokeKind::Inside,
                );

                let dot_c = Pos2::new(rect.left() + 11.0_f32, rect.center().y);
                painter.circle_filled(dot_c, 3.0_f32, brighten(StudioPalette::STUDIO_VIOLET, 40));

                painter.text(
                    Pos2::new(rect.center().x + 5.0_f32, rect.center().y),
                    Align2::CENTER_CENTER,
                    format!("Modello {pass}/4"),
                    FontId::proportional(11.3_f32),
                    StudioPalette::CRISP_TEXT,
                );
            } else {
                painter.rect_filled(rect, radius, StudioPalette::DEEP_OBSIDIAN);
                painter.rect_stroke(
                    rect,
                    radius,
                    Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE),
                    StrokeKind::Inside,
                );
                painter.text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    format!("Modello {pass}/4"),
                    FontId::proportional(11.2_f32),
                    StudioPalette::MUTED_LABEL,
                );
            }
        }
    });
}

fn draw_five_step_checklist(
    ui: &mut Ui,
    total_w: f32,
    active_step: usize,
    outcome: &RunOutcome,
) {
    Frame::NONE
        .fill(Color32::from_rgb(11, 15, 24))
        .corner_radius(CornerRadius::same(10))
        .stroke(Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE))
        .inner_margin(Margin::symmetric(12, 9))
        .show(ui, |ui| {
            ui.set_width((total_w - 24.0_f32).max(240.0_f32));
            ui.spacing_mut().item_spacing.y = 4.0_f32;

            for (idx, step_label) in PIPELINE_STEPS.iter().enumerate() {
                let step_num = idx + 1;
                let is_completed = *outcome == RunOutcome::Success || step_num < active_step;
                let is_active = *outcome == RunOutcome::Running && step_num == active_step;
                let is_failed_here =
                    matches!(outcome, RunOutcome::Failed(_)) && step_num == active_step;

                let row_bg = if is_active {
                    with_alpha(StudioPalette::STUDIO_VIOLET, 24)
                } else if is_failed_here {
                    with_alpha(CORAL_ACCENT, 22)
                } else {
                    Color32::TRANSPARENT
                };

                Frame::NONE
                    .fill(row_bg)
                    .corner_radius(CornerRadius::same(6))
                    .inner_margin(Margin::symmetric(6, 3))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            let (icon_rect, _) =
                                ui.allocate_exact_size(Vec2::splat(16.0_f32), Sense::hover());
                            let painter = ui.painter();
                            let c = icon_rect.center();

                            if is_completed {
                                painter.circle_filled(
                                    c,
                                    7.0_f32,
                                    with_alpha(StudioPalette::EMERALD, 42),
                                );
                                painter.circle_stroke(
                                    c,
                                    7.0_f32,
                                    Stroke::new(1.1_f32, StudioPalette::EMERALD),
                                );
                                draw_mini_checkmark(painter, c, StudioPalette::EMERALD);
                            } else if is_active {
                                let t = ui.input(|i| i.time) as f32;
                                let pulse = 0.5_f32 + 0.5_f32 * (t * 4.0_f32).sin();
                                painter.circle_filled(
                                    c,
                                    7.4_f32,
                                    with_alpha(
                                        StudioPalette::STUDIO_VIOLET,
                                        (48.0_f32 + 56.0_f32 * pulse) as u8,
                                    ),
                                );
                                painter.circle_filled(
                                    c,
                                    3.6_f32,
                                    brighten(StudioPalette::STUDIO_VIOLET, 38),
                                );
                            } else if is_failed_here {
                                painter.circle_filled(c, 6.8_f32, with_alpha(CORAL_ACCENT, 48));
                                painter.circle_filled(c, 3.2_f32, CORAL_ACCENT);
                            } else {
                                painter.circle_stroke(
                                    c,
                                    5.5_f32,
                                    Stroke::new(1.1_f32, Color32::from_rgb(56, 70, 96)),
                                );
                            }

                            let text_col = if is_active {
                                StudioPalette::CRISP_TEXT
                            } else if is_completed {
                                Color32::from_rgb(206, 224, 218)
                            } else if is_failed_here {
                                Color32::from_rgb(253, 164, 175)
                            } else {
                                StudioPalette::MUTED_LABEL
                            };

                            let rt = RichText::new(*step_label).size(12.0_f32).color(text_col);
                            ui.label(if is_active { rt.strong() } else { rt });
                        });
                    });
            }
        });
}

fn draw_collapsible_tech_console(
    ui: &mut Ui,
    total_w: f32,
    expanded: &mut bool,
    logs: &[String],
) {
    let toggle_h = 26.0_f32;
    let (hdr_rect, hdr_resp) =
        ui.allocate_exact_size(Vec2::new(total_w, toggle_h), Sense::click());
    if hdr_resp.clicked() {
        *expanded = !*expanded;
    }
    if hdr_resp.hovered() {
        ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
    }

    if ui.is_rect_visible(hdr_rect) {
        let hover_t = ui
            .ctx()
            .animate_bool_responsive(hdr_resp.id.with("log_toggle_hover"), hdr_resp.hovered());
        let painter = ui.painter();
        if hover_t > 0.01_f32 {
            painter.rect_filled(
                hdr_rect,
                CornerRadius::same(6),
                with_alpha(StudioPalette::CARD_GLASS, (150.0_f32 * hover_t) as u8),
            );
        }

        // Icona vettoriale Chevron (▸ chiuso / ▾ aperto)
        let ch_c = Pos2::new(hdr_rect.left() + 9.0_f32, hdr_rect.center().y);
        let ch_col = lerp_color(
            StudioPalette::MUTED_LABEL,
            StudioPalette::ELECTRIC_CYAN,
            hover_t,
        );
        if *expanded {
            painter.line_segment(
                [
                    Pos2::new(ch_c.x - 4.0_f32, ch_c.y - 2.0_f32),
                    Pos2::new(ch_c.x, ch_c.y + 2.5_f32),
                ],
                Stroke::new(1.5_f32, ch_col),
            );
            painter.line_segment(
                [
                    Pos2::new(ch_c.x, ch_c.y + 2.5_f32),
                    Pos2::new(ch_c.x + 4.0_f32, ch_c.y - 2.0_f32),
                ],
                Stroke::new(1.5_f32, ch_col),
            );
        } else {
            painter.line_segment(
                [
                    Pos2::new(ch_c.x - 2.0_f32, ch_c.y - 4.0_f32),
                    Pos2::new(ch_c.x + 2.5_f32, ch_c.y),
                ],
                Stroke::new(1.5_f32, ch_col),
            );
            painter.line_segment(
                [
                    Pos2::new(ch_c.x + 2.5_f32, ch_c.y),
                    Pos2::new(ch_c.x - 2.0_f32, ch_c.y + 4.0_f32),
                ],
                Stroke::new(1.5_f32, ch_col),
            );
        }

        let toggle_txt = if *expanded {
            "Nascondi Log Tecnico Live"
        } else {
            "Mostra Log Tecnico Live"
        };
        painter.text(
            Pos2::new(hdr_rect.left() + 21.0_f32, hdr_rect.center().y),
            Align2::LEFT_CENTER,
            toggle_txt,
            FontId::proportional(11.8_f32),
            lerp_color(StudioPalette::MUTED_LABEL, StudioPalette::CRISP_TEXT, hover_t),
        );

        painter.text(
            Pos2::new(hdr_rect.right() - 6.0_f32, hdr_rect.center().y),
            Align2::RIGHT_CENTER,
            format!("Console Tecnica ({})", logs.len()),
            FontId::monospace(10.6_f32),
            Color32::from_rgb(100, 116, 139),
        );
    }

    if *expanded {
        ui.add_space(4.0_f32);
        Frame::NONE
            .fill(Color32::from_rgb(7, 10, 16))
            .corner_radius(CornerRadius::same(8))
            .stroke(Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE))
            .inner_margin(Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.set_width((total_w - 20.0_f32).max(220.0_f32));
                ui.set_min_height(94.0_f32);
                ui.set_max_height(94.0_f32);

                ScrollArea::vertical()
                    .max_height(94.0_f32)
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 3.0_f32;
                        if logs.is_empty() {
                            ui.label(
                                RichText::new("In attesa di output dal motore AI...")
                                    .monospace()
                                    .size(11.0_f32)
                                    .color(StudioPalette::MUTED_LABEL),
                            );
                        } else {
                            for line in logs {
                                let col = if line.starts_with("[Completato]")
                                    || line.contains("COMPLETATO")
                                {
                                    StudioPalette::EMERALD
                                } else if line.starts_with("[Errore]") || line.contains("Error") {
                                    CORAL_ACCENT
                                } else if line.starts_with("[Modello") {
                                    StudioPalette::ELECTRIC_CYAN
                                } else {
                                    Color32::from_rgb(186, 198, 216)
                                };
                                ui.label(RichText::new(line).monospace().size(11.0_f32).color(col));
                            }
                        }
                    });
            });
    }
}

fn draw_check_circle_icon(ui: &mut Ui, accent: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(18.0_f32), Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    let painter = ui.painter();
    let c = rect.center();
    painter.circle_filled(c, 8.0_f32, with_alpha(accent, 48));
    painter.circle_stroke(c, 8.0_f32, Stroke::new(1.2_f32, accent));
    draw_mini_checkmark(painter, c, brighten(accent, 35));
}

fn draw_mini_checkmark(painter: &egui::Painter, center: Pos2, color: Color32) {
    let p1 = Pos2::new(center.x - 3.3_f32, center.y + 0.1_f32);
    let p2 = Pos2::new(center.x - 0.8_f32, center.y + 2.7_f32);
    let p3 = Pos2::new(center.x + 3.6_f32, center.y - 2.5_f32);
    let stroke = Stroke::new(1.7_f32, color);
    painter.line_segment([p1, p2], stroke);
    painter.line_segment([p2, p3], stroke);
}

fn consume_escape_single_press(i: &mut egui::InputState) -> bool {
    let mut triggered = false;
    i.events.retain(|ev| match ev {
        Event::Key {
            key,
            physical_key,
            pressed: true,
            repeat,
            modifiers,
        } if (*key == Key::Escape || *physical_key == Some(Key::Escape))
            && !modifiers.command
            && !modifiers.ctrl
            && !modifiers.alt =>
        {
            if !*repeat {
                triggered = true;
            }
            false
        }
        _ => true,
    });
    triggered
}

fn truncate_middle(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars || max_chars < 10 {
        return text.to_string();
    }
    let head = (max_chars - 3) / 2;
    let tail = max_chars - 3 - head;
    let prefix: String = text.chars().take(head).collect();
    let suffix: String = text.chars().skip(count - tail).collect();
    format!("{prefix}...{suffix}")
}

#[inline]
fn with_alpha(c: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha)
}

#[inline]
fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0_f32, 1.0_f32);
    let inv = 1.0_f32 - t;
    let r = (a.r() as f32 * inv + b.r() as f32 * t).round() as u8;
    let g = (a.g() as f32 * inv + b.g() as f32 * t).round() as u8;
    let bl = (a.b() as f32 * inv + b.b() as f32 * t).round() as u8;
    let al = (a.a() as f32 * inv + b.a() as f32 * t).round() as u8;
    Color32::from_rgba_unmultiplied(r, g, bl, al)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wizard::{AutoTuneMode, BassEnhanceMode, DrumSampleMode};

    #[test]
    fn test_ai_pipeline_options_json_serialization() {
        let mut choices = GuidedWizardChoices::default();
        let def_opts = AiPipelineOptions::from_wizard_choices(&choices);
        let json_default = def_opts.to_json_pretty();
        assert!(json_default.contains("\"autotune_mode\": \"off\""));
        assert!(json_default.contains("\"drum_mode\": \"natural\""));
        assert!(json_default.contains("\"bass_mode\": \"natural\""));
        assert!(json_default.contains("\"target_bpm\": null"));
        assert!(json_default.contains("\"force_reseparate\": false"));

        choices.autotune_mode = AutoTuneMode::Moderno;
        choices.drum_mode = DrumSampleMode::StudioPunch;
        choices.bass_mode = BassEnhanceMode::StudioReAmp;
        choices.target_bpm = Some(128.5);
        choices.force_reseparate = true;

        let custom_opts = AiPipelineOptions::from_wizard_choices(&choices);
        let json_custom = custom_opts.to_json_pretty();
        assert!(json_custom.contains("\"autotune_mode\": \"modern\""));
        assert!(json_custom.contains("\"drum_mode\": \"punch\""));
        assert!(json_custom.contains("\"bass_mode\": \"reamp\""));
        assert!(json_custom.contains("\"target_bpm\": 128.50"));
        assert!(json_custom.contains("\"force_reseparate\": true"));
    }

    #[test]
    fn test_parse_ai_master_report_and_italian_badge() {
        let raw_json = r#"{
            "original_lufs": -18.4,
            "detected_key": "La Minore \u2022 A min",
            "detected_bpm": 118.0,
            "effective_bpm": 122.4,
            "drum_hits": {
                "kicks": 142,
                "snares": 96
            }
        }"#;

        let rep = parse_ai_master_report(raw_json);
        assert_eq!(rep.detected_key.as_deref(), Some("La Minore • A min"));
        assert!((rep.effective_bpm.unwrap() - 122.4).abs() < 1e-3);
        assert_eq!(rep.kicks, Some(142));
        assert_eq!(rep.snares, Some(96));

        let badge = rep.format_italian_badge().expect("badge expected");
        assert!(badge.contains("Tonalità: La Minore • A min"));
        assert!(badge.contains("122.4 BPM"));
        assert!(badge.contains("Cassa: 142 | Rullante: 96"));
    }

    #[test]
    fn test_tqdm_and_gpu_badge_parsing() {
        let tqdm_line = " 42%|████▏     | 115.2/274.0 [00:04<00:06, 25.40seconds/s]";
        let m = parse_tqdm_line(tqdm_line).expect("tqdm expected");
        assert!((m.sub_pct - 42.0).abs() < 1e-3);
        assert_eq!(m.eta_txt.as_deref(), Some("ETA 00:06"));
        assert_eq!(m.speed_txt.as_deref(), Some("25.4x tempo reale"));

        let (cuda_lbl, cuda_on) = normalize_gpu_badge("NVIDIA GeForce RTX 4070", true);
        assert!(cuda_on && cuda_lbl.starts_with("CUDA •"));

        let (rocm_lbl, rocm_on) = normalize_gpu_badge("AMD Radeon RX 7900 XT", false);
        assert!(rocm_on && rocm_lbl.starts_with("ROCm •"));

        let (mps_lbl, mps_on) = normalize_gpu_badge("Apple M1 (Metal MPS)", false);
        assert!(mps_on && mps_lbl == "Metal MPS • Apple M1");

        let (cpu_m1_lbl, cpu_m1_on) =
            normalize_gpu_badge("CPU Multi-Core Apple Silicon (8 Threads)", false);
        assert!(!cpu_m1_on && cpu_m1_lbl.starts_with("CPU"));

        let (cpu_lbl, cpu_on) = normalize_gpu_badge("CPU - 16 Threads", false);
        assert!(!cpu_on && cpu_lbl.starts_with("CPU •"));
    }
}
