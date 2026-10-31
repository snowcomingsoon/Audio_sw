#![forbid(unsafe_code)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ai_pipeline;
mod audio_editor;
mod audio_io;
mod auto_eq;
mod bpm_stretch;
mod dsp;
mod eq_drawer;
mod icon;
mod project_io;
mod shortcuts_and_meters;
mod timeline_view;
mod ui_widgets;
mod wizard;

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Pos2,
    Rect, RichText, Stroke, StrokeKind, Vec2,
};
use rustfft::{num_complex::Complex, FftPlanner};

use ai_pipeline::{
    draw_ai_header_button, draw_ai_pipeline_modal, parse_ai_master_report, AiPipelineController,
    AiPipelineOptions,
};
use audio_editor::{
    apply_fade_in, apply_fade_out, boost_or_normalize_selection, copy_selection, crop_to_selection,
    cut_selection, delete_selection, duplicate_selection, paste_at_playhead, silence_selection,
    AudioClipboard, AudioEditHistory, GainEditOp,
};
use audio_io::{
    decode_audio_file, export_session, PlaybackSourceMode, RealtimeAudioEngine, StereoBuffer,
};
use auto_eq::analyze_and_compute_auto_eq;
use bpm_stretch::{detect_bpm_fast, OriginalTempoRefs, MAX_USER_BPM, MIN_USER_BPM};
use dsp::DspParams;
use eq_drawer::{draw_eq_and_master_drawer, EqDrawerState, EqDrawerTarget};
use project_io::{
    draw_drop_zone_overlay, export_all_stems_wav, extract_dropped_audio_path,
    load_preset_file_with_stems, save_preset_file_with_stems,
};
use shortcuts_and_meters::{draw_shortcuts_modal, poll_daw_shortcuts, UndoStack};
use timeline_view::{
    draw_multitrack_timeline, TimelineEditAction, TimelineEditContext, TimelineState,
};
use ui_widgets::{pill_button, transport_play_button, StudioPalette};
use wizard::{draw_guided_wizard_modal, GuidedWizardChoices, GuidedWizardState, SourceProfile};

struct StudioEqApp {
    engine: Option<RealtimeAudioEngine>,
    loaded_file_path: Option<PathBuf>,
    loaded_file_name: String,
    loaded_dir: PathBuf,
    status_msg: String,

    show_eq_drawer: bool,
    show_help_modal: bool,
    timeline_ui: TimelineState,
    eq_ui: EqDrawerState,
    undo_stack: UndoStack,

    edit_history: AudioEditHistory,
    clipboard: AudioClipboard,
    tempo_refs: OriginalTempoRefs,
    ui_target_bpm: f32,
    ai_badge_info: Option<String>,
    tempo_refresh_ready: Arc<AtomicBool>,

    ai_ctrl: AiPipelineController,
    wizard_ui: GuidedWizardState,

    loading_busy: Arc<AtomicBool>,
    bg_status: Arc<Mutex<Option<String>>>,

    screenshot_target: Option<PathBuf>,
    screenshot_focus_stem: Option<usize>,
    screenshot_requested: bool,
    ready_frames: u32,

    fft_planner: FftPlanner<f32>,
    hann_window: Vec<f32>,
    smoothed_spectrum_db: Vec<f32>,
}

fn save_color_image_bmp(img: &egui::ColorImage, path: &Path) -> std::io::Result<()> {
    let w = img.size[0] as u32;
    let h = img.size[1] as u32;
    let pixel_bytes = w * h * 4;
    let file_size = 54 + pixel_bytes;
    let mut out = Vec::with_capacity(file_size as usize);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(-(h as i32)).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&pixel_bytes.to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for px in &img.pixels {
        let [r, g, b, a] = px.to_srgba_unmultiplied();
        out.extend_from_slice(&[b, g, r, a]);
    }
    std::fs::write(path, out)
}

impl StudioEqApp {
    fn configure_modern_fonts_and_theme(ctx: &egui::Context) {
        let mut fonts = FontDefinitions::default();

        let proportional_candidates = [
            r"C:\Windows\Fonts\segoeui.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ];
        for cand in proportional_candidates {
            if let Ok(regular_bytes) = std::fs::read(cand) {
                fonts.font_data.insert(
                    "studio_sans".to_owned(),
                    Arc::new(FontData::from_owned(regular_bytes)),
                );
                fonts
                    .families
                    .entry(FontFamily::Proportional)
                    .or_default()
                    .insert(0, "studio_sans".to_owned());
                break;
            }
        }

        let symbol_candidates = [
            r"C:\Windows\Fonts\seguisym.ttf",
            "/System/Library/Fonts/Apple Symbols.ttf",
        ];
        for sym_cand in symbol_candidates {
            if let Ok(sym_bytes) = std::fs::read(sym_cand) {
                fonts.font_data.insert(
                    "studio_symbols".to_owned(),
                    Arc::new(FontData::from_owned(sym_bytes)),
                );
                let prop = fonts.families.entry(FontFamily::Proportional).or_default();
                let insert_pos = prop.len().min(1);
                prop.insert(insert_pos, "studio_symbols".to_owned());
                fonts
                    .families
                    .entry(FontFamily::Monospace)
                    .or_default()
                    .push("studio_symbols".to_owned());
                break;
            }
        }

        let mono_candidates = [
            r"C:\Windows\Fonts\consola.ttf",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
        ];
        for cand in mono_candidates {
            if let Ok(mono_bytes) = std::fs::read(cand) {
                fonts.font_data.insert(
                    "studio_mono".to_owned(),
                    Arc::new(FontData::from_owned(mono_bytes)),
                );
                fonts
                    .families
                    .entry(FontFamily::Monospace)
                    .or_default()
                    .insert(0, "studio_mono".to_owned());
                break;
            }
        }
        ctx.set_fonts(fonts);

        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(7.0_f32, 7.0_f32);
        style.spacing.button_padding = Vec2::new(10.0_f32, 6.0_f32);
        style.visuals = egui::Visuals::dark();
        style.visuals.panel_fill = StudioPalette::DEEP_OBSIDIAN;
        style.visuals.window_fill = StudioPalette::ELEVATED_SURFACE;
        style.visuals.extreme_bg_color = Color32::from_rgb(7, 10, 15);
        style.visuals.widgets.noninteractive.bg_stroke =
            Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE);
        ctx.set_style(style);
    }

    fn new(
        cc: &eframe::CreationContext<'_>,
        screenshot_target: Option<PathBuf>,
        screenshot_focus_stem: Option<usize>,
    ) -> Self {
        Self::configure_modern_fonts_and_theme(&cc.egui_ctx);

        let engine = RealtimeAudioEngine::start().ok();

        let n_fft = 4096usize;
        let hann_window: Vec<f32> = (0..n_fft)
            .map(|i| {
                0.5_f32 * (1.0_f32 - (2.0_f32 * PI as f32 * i as f32 / (n_fft - 1) as f32).cos())
            })
            .collect();

        let default_out_dir = Self::locate_default_output_dir();
        let orig_candidate = default_out_dir.join("traccia_originale_sala.mp3");
        let initial_path = if orig_candidate.exists() {
            Some(orig_candidate)
        } else {
            Self::find_first_example_audio(&default_out_dir)
        };
        let initial_name = initial_path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "Sessione Vuota".to_string());

        let initial_report = std::fs::read_to_string(default_out_dir.join("ai_master_report.json"))
            .ok()
            .map(|txt| parse_ai_master_report(&txt));
        let initial_bpm = initial_report
            .as_ref()
            .and_then(|r| r.effective_bpm)
            .map(|b| b.clamp(MIN_USER_BPM, MAX_USER_BPM))
            .unwrap_or(120.0_f32);
        let initial_badge = initial_report.and_then(|r| r.format_italian_badge());

        let mut app = Self {
            engine,
            loaded_file_path: initial_path,
            loaded_file_name: initial_name,
            loaded_dir: default_out_dir.clone(),
            status_msg: "Avvio rapido  •  Caricamento tracce audio in background...".to_string(),
            show_eq_drawer: true,
            show_help_modal: false,
            timeline_ui: TimelineState::default(),
            eq_ui: EqDrawerState::default(),
            undo_stack: UndoStack::default(),
            edit_history: AudioEditHistory::default(),
            clipboard: AudioClipboard::default(),
            tempo_refs: OriginalTempoRefs::default(),
            ui_target_bpm: initial_bpm,
            ai_badge_info: initial_badge,
            tempo_refresh_ready: Arc::new(AtomicBool::new(false)),
            ai_ctrl: AiPipelineController::new(),
            wizard_ui: GuidedWizardState::default(),
            loading_busy: Arc::new(AtomicBool::new(false)),
            bg_status: Arc::new(Mutex::new(None)),
            screenshot_target,
            screenshot_focus_stem,
            screenshot_requested: false,
            ready_frames: 0,
            fft_planner: FftPlanner::new(),
            hann_window,
            smoothed_spectrum_db: vec![-90.0_f32; n_fft / 2],
        };

        app.spawn_autoload_workspace(default_out_dir, None, cc.egui_ctx.clone());
        app
    }

    fn find_first_example_audio(out_dir: &Path) -> Option<PathBuf> {
        let parent = out_dir.parent()?;
        let ex_dir = parent.join("_esempi");
        let entries = std::fs::read_dir(ex_dir).ok()?;
        let mut found: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(|ext| {
                        matches!(
                            ext.to_ascii_lowercase().as_str(),
                            "mp3" | "mpeg" | "wav" | "flac" | "ogg" | "m4a"
                        )
                    })
                    .unwrap_or(false)
            })
            .collect();
        found.sort();
        found.into_iter().next()
    }

    fn locate_default_output_dir() -> PathBuf {
        let mut roots: Vec<PathBuf> = Vec::new();
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
                push_ancestors(dir.to_path_buf(), &mut roots);
            }
        }
        if let Ok(cwd) = std::env::current_dir() {
            push_ancestors(cwd, &mut roots);
        }

        // 1. Preferisce la cartella del progetto che contiene lo script AI o un `output/` già popolato
        //    (escludendo bundle `.app` e volumi sola-lettura `/Volumes/` su macOS)
        for r in &roots {
            let r_str = r.to_string_lossy();
            if r_str.contains(".app") || r_str.starts_with("/Volumes/") || r_str == "/Applications" || r_str == "/" {
                continue;
            }
            let out_cand = r.join("output");
            if out_cand.exists() {
                return out_cand;
            }
            if r.join("demucs_pedalboard_master.py").exists() || r.join("Avvia_StudioEQ.bat").exists()
            {
                let _ = std::fs::create_dir_all(&out_cand);
                return out_cand;
            }
        }

        #[cfg(target_os = "macos")]
        if let Ok(home) = std::env::var("HOME") {
            let user_out = PathBuf::from(home)
                .join("Music")
                .join("StudioEQ_Master_Pro")
                .join("output");
            let _ = std::fs::create_dir_all(&user_out);
            return user_out;
        }

        let fallback = roots
            .first()
            .cloned()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("output");
        let _ = std::fs::create_dir_all(&fallback);
        fallback
    }

    fn spawn_autoload_workspace(
        &mut self,
        out_dir: PathBuf,
        wizard_choices: Option<GuidedWizardChoices>,
        ctx: egui::Context,
    ) {
        let Some(ref eng) = self.engine else {
            self.status_msg = "Attenzione: nessun dispositivo audio rilevato".to_string();
            return;
        };
        let state_arc = Arc::clone(&eng.state);
        let busy = Arc::clone(&self.loading_busy);
        let bg_msg = Arc::clone(&self.bg_status);
        let tempo_ready = Arc::clone(&self.tempo_refresh_ready);
        let initial_fallback = self.loaded_file_path.clone();
        busy.store(true, Ordering::SeqCst);

        std::thread::spawn(move || {
            let sr = state_arc.lock().map(|s| s.device_sr).unwrap_or(44100);

            let orig = out_dir.join("traccia_originale_sala.mp3");
            let main_source = orig
                .exists()
                .then_some(orig.clone())
                .or(initial_fallback);
            if wizard_choices.is_none() {
                if let Some(ref src_path) = main_source {
                    if let Ok(buf) = decode_audio_file(src_path) {
                        let resampled = buf.resample_to(sr);
                        let initial_params = DspParams::preset_sala_prove();
                        if let Ok(mut st) = state_arc.lock() {
                            if st.main_track.is_none() {
                                st.main_track = Some(resampled);
                                st.mode = PlaybackSourceMode::OriginalWithLiveDsp;
                                st.params = initial_params;
                                st.params_dirty = true;
                            }
                        }
                        if let Ok(mut m) = bg_msg.lock() {
                            *m = Some(
                                "Traccia principale pronta  •  Controllo 4 Stems AI...".to_string(),
                            );
                        }
                        ctx.request_repaint();
                    }
                }
            }

            let ai_wav = out_dir.join("Master_AI_Demucs_Pedalboard.wav");
            let ai_mp3 = out_dir.join("Master_AI_Demucs_Pedalboard.mp3");
            if let Some(p) = ai_wav
                .exists()
                .then_some(ai_wav)
                .or_else(|| ai_mp3.exists().then_some(ai_mp3))
            {
                if let Ok(buf) = decode_audio_file(&p) {
                    let resampled = buf.resample_to(sr);
                    if let Ok(mut st) = state_arc.lock() {
                        st.ai_master_track = Some(resampled);
                    }
                }
            }

            let stems_dir = out_dir.join("stems_studio");
            let stem_names = [
                "01_Voce_StudioEQ",
                "02_Batteria_StudioEQ",
                "03_Basso_StudioEQ",
                "04_Chitarre_Altro_StudioEQ",
            ];

            let mut decoded_stems: Vec<StereoBuffer> = Vec::new();
            for name in stem_names {
                let w = stems_dir.join(format!("{name}.wav"));
                let m = stems_dir.join(format!("{name}.mp3"));
                if let Some(tp) = w.exists().then_some(w).or_else(|| m.exists().then_some(m)) {
                    if let Ok(b) = decode_audio_file(&tp) {
                        decoded_stems.push(b.resample_to(sr));
                    }
                }
            }

            if decoded_stems.len() == 4 {
                if let Ok(arr) = decoded_stems.try_into() {
                    let (target_params, stem_gains) = if let Some(ch) = wizard_choices {
                        (ch.build_stem_master_params(), ch.stem_gains_db())
                    } else {
                        (DspParams::preset_stem_polish(), [0.0_f32; 4])
                    };

                    if let Ok(mut st) = state_arc.lock() {
                        st.stems = Some(arr);
                        st.mode = PlaybackSourceMode::FourStemMixer;
                        st.params = target_params;
                        for (i, &g) in stem_gains.iter().enumerate() {
                            st.stem_channels[i].gain_db = g;
                        }
                        st.params_dirty = true;
                    }
                    if let Ok(mut m) = bg_msg.lock() {
                        *m = Some(
                            "4 Tracce AI Separate attive  •  Regola Bassi / Medi / Alti su ogni traccia o clicca EQ"
                                .to_string(),
                        );
                    }
                }
            } else if main_source.is_some() {
                if let Ok(mut m) = bg_msg.lock() {
                    *m = Some(
                        "Traccia pronta  •  Clicca Guidata, Auto-EQ o Separa AI per personalizzare il suono"
                            .to_string(),
                    );
                }
            } else if let Ok(mut m) = bg_msg.lock() {
                *m = Some(
                    "Pronto  •  Trascina un file audio nella finestra o premi Apri Audio"
                        .to_string(),
                );
            }

            tempo_ready.store(true, Ordering::SeqCst);
            busy.store(false, Ordering::SeqCst);
            ctx.request_repaint();
        });
    }

    fn refresh_tempo_refs_from_engine(&mut self) {
        let Some(ref eng) = self.engine else { return };
        let snap = eng.state.lock().ok().map(|st| {
            (
                st.main_track.clone(),
                st.ai_master_track.clone(),
                st.stems.clone(),
            )
        });
        let Some((main_t, ai_t, stems_t)) = snap else {
            return;
        };

        let report_bpm = std::fs::read_to_string(self.loaded_dir.join("ai_master_report.json"))
            .ok()
            .map(|txt| parse_ai_master_report(&txt))
            .and_then(|r| {
                if self.ai_badge_info.is_none() {
                    self.ai_badge_info = r.format_italian_badge();
                }
                r.effective_bpm
            });

        let detected = report_bpm.unwrap_or_else(|| {
            if let Some(ref arr) = stems_t {
                // La traccia Batteria (indice 1) offre i transienti più netti per il rilevamento BPM
                detect_bpm_fast(&arr[1])
            } else if let Some(ref m) = main_t {
                detect_bpm_fast(m)
            } else if let Some(ref a) = ai_t {
                detect_bpm_fast(a)
            } else {
                120.0_f32
            }
        });

        self.tempo_refs.store_loaded(main_t, ai_t, stems_t, detected);
        self.ui_target_bpm = self.tempo_refs.current_bpm;
        self.edit_history.clear();
    }

    fn apply_live_target_bpm(&mut self, target_bpm: f32) {
        let Some(ref eng) = self.engine else { return };
        let clamped = target_bpm.clamp(MIN_USER_BPM, MAX_USER_BPM);
        let (new_main, new_ai, new_stems) = self.tempo_refs.render_for_target_bpm(clamped);
        if new_main.is_none() && new_ai.is_none() && new_stems.is_none() {
            self.status_msg = "Nessuna traccia audio attiva per il cambio BPM.".to_string();
            return;
        }

        if let Ok(mut st) = eng.state.lock() {
            let old_len = st.active_length().max(1);
            self.edit_history.push_before_edit(&st);
            st.main_track = new_main;
            st.ai_master_track = new_ai;
            st.stems = new_stems;
            let new_len = st.active_length();
            if new_len > 0 {
                let scale = (new_len as f64) / (old_len as f64);
                st.position_frame =
                    ((st.position_frame as f64 * scale).round() as usize).min(new_len - 1);
                if let Some((a, b)) = st.selection_range {
                    let na = ((a as f64 * scale).round() as usize).min(new_len);
                    let nb = ((b as f64 * scale).round() as usize).min(new_len);
                    if nb > na + 16 {
                        st.selection_range = Some((na, nb));
                    } else {
                        st.selection_range = None;
                        st.loop_enabled = false;
                    }
                }
            } else {
                st.position_frame = 0;
                st.selection_range = None;
                st.loop_enabled = false;
            }
        }
        self.tempo_refs.current_bpm = clamped;
        self.ui_target_bpm = clamped;
        self.status_msg = format!(
            "BPM aggiornato senza distorsione: {:.1} BPM (Originale {:.1} BPM  •  Ctrl+Z per annullare)",
            clamped, self.tempo_refs.detected_bpm
        );
    }

    fn execute_timeline_edit(&mut self, action: TimelineEditAction) {
        let Some(ref eng) = self.engine else { return };
        let mut mutated = false;
        let result: Result<String, String> = if let Ok(mut st) = eng.state.lock() {
            match action {
                TimelineEditAction::CutSelection => {
                    let r = cut_selection(&mut st, &mut self.edit_history, &mut self.clipboard);
                    mutated = r.is_ok();
                    r
                }
                TimelineEditAction::DeleteSelection => {
                    let r = delete_selection(&mut st, &mut self.edit_history);
                    mutated = r.is_ok();
                    r
                }
                TimelineEditAction::CropSelection => {
                    let r = crop_to_selection(&mut st, &mut self.edit_history);
                    mutated = r.is_ok();
                    r
                }
                TimelineEditAction::SilenceSelection => {
                    let r = silence_selection(&mut st, &mut self.edit_history);
                    mutated = r.is_ok();
                    r
                }
                TimelineEditAction::CopySelection => copy_selection(&st, &mut self.clipboard),
                TimelineEditAction::PasteAtPlayhead => {
                    let r = paste_at_playhead(&mut st, &mut self.edit_history, &self.clipboard);
                    mutated = r.is_ok();
                    r
                }
                TimelineEditAction::DuplicateSelection => {
                    let r = duplicate_selection(&mut st, &mut self.edit_history);
                    mutated = r.is_ok();
                    r
                }
                TimelineEditAction::FadeIn => {
                    let r = apply_fade_in(&mut st, &mut self.edit_history);
                    mutated = r.is_ok();
                    r
                }
                TimelineEditAction::FadeOut => {
                    let r = apply_fade_out(&mut st, &mut self.edit_history);
                    mutated = r.is_ok();
                    r
                }
                TimelineEditAction::NormalizeSelection => {
                    let r = boost_or_normalize_selection(
                        &mut st,
                        &mut self.edit_history,
                        GainEditOp::NORMALIZE_MINUS_1DB,
                    );
                    mutated = r.is_ok();
                    r
                }
                TimelineEditAction::UndoAudioEdit => {
                    let r = self.edit_history.undo(&mut st);
                    mutated = r.is_ok();
                    r
                }
                TimelineEditAction::RedoAudioEdit => {
                    let r = self.edit_history.redo(&mut st);
                    mutated = r.is_ok();
                    r
                }
                TimelineEditAction::ClearSelection => {
                    st.selection_range = None;
                    st.loop_enabled = false;
                    Ok("Selezione azzerata".to_string())
                }
                TimelineEditAction::FocusStemEq(idx) => {
                    let s_idx = idx.min(3);
                    self.show_eq_drawer = true;
                    self.eq_ui.active_target = EqDrawerTarget::Stem(s_idx);
                    st.mode = PlaybackSourceMode::FourStemMixer;
                    let names = ["VOCE", "BATTERIA", "BASSO", "CHITARRE & ALTRO"];
                    Ok(format!(
                        "EQ 3-Bande attivo su {}  •  Regola Bassi, Medi e Alti (-12 .. +12 dB)",
                        names[s_idx]
                    ))
                }
            }
        } else {
            return;
        };

        if mutated {
            if let Ok(st) = eng.state.lock() {
                self.tempo_refs.sync_after_audio_edit(&st);
            }
        }

        match result {
            Ok(msg) => self.status_msg = msg,
            Err(err) => self.status_msg = err,
        }
    }

    fn load_main_file(&mut self, path: &Path) -> Result<(), String> {
        let buf = decode_audio_file(path)?;
        let Some(ref eng) = self.engine else {
            return Err("Motore audio non disponibile".to_string());
        };
        let sr = eng.state.lock().map(|s| s.device_sr).unwrap_or(44100);
        let resampled = buf.resample_to(sr);
        let detected_bpm = detect_bpm_fast(&resampled);
        let initial_params = DspParams::preset_sala_prove();

        if let Ok(mut st) = eng.state.lock() {
            st.playing = false;
            st.position_frame = 0;
            st.selection_range = None;
            st.loop_enabled = false;
            st.main_track = Some(resampled.clone());
            st.stems = None;
            st.ai_master_track = None;
            st.mode = PlaybackSourceMode::OriginalWithLiveDsp;
            st.params = initial_params.clone();
            for ch in &mut st.stem_channels {
                ch.gain_db = 0.0_f32;
                ch.pan = 0.0_f32;
                ch.muted = false;
                ch.solo = false;
                ch.eq.reset();
            }
            st.params_dirty = true;
        }
        self.eq_ui.active_target = EqDrawerTarget::MasterBus;
        self.undo_stack.reset(initial_params);
        self.edit_history.clear();
        self.tempo_refs
            .store_loaded(Some(resampled), None, None, detected_bpm);
        self.ui_target_bpm = self.tempo_refs.current_bpm;
        self.ai_badge_info = None;
        self.loaded_file_path = Some(path.to_path_buf());
        self.loaded_file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("traccia_audio")
            .to_string();

        // Apre automaticamente il Menu Guidato precompilando il BPM rilevato
        self.wizard_ui
            .open_for_track_with_bpm(&self.loaded_file_name, detected_bpm);

        self.status_msg = format!(
            "Caricato: {} ({:.1} BPM)  •  Scegli le opzioni nella Procedura Guidata",
            self.loaded_file_name, detected_bpm
        );
        Ok(())
    }

    fn apply_wizard_choices(&mut self, choices: GuidedWizardChoices, ctx: &egui::Context) {
        let Some(ref eng) = self.engine else { return };

        let base_auto_eq = if choices.source_profile == SourceProfile::AutoEqIntelligente {
            let track_opt = eng.state.lock().ok().and_then(|st| st.main_track.clone());
            track_opt.map(|buf| analyze_and_compute_auto_eq(&buf).0)
        } else {
            None
        };

        let outcome = choices.build_outcome(base_auto_eq);
        let [m_clean, m_vocal, m_air, m_punch] = outcome.macro_knobs_pct;
        self.eq_ui.macro_clean = m_clean;
        self.eq_ui.macro_vocal = m_vocal;
        self.eq_ui.macro_air = m_air;
        self.eq_ui.macro_punch = m_punch;

        if let Ok(mut st) = eng.state.lock() {
            if st.stems.is_some() && !outcome.run_ai_split {
                st.params = choices.build_stem_master_params();
                st.mode = PlaybackSourceMode::FourStemMixer;
            } else {
                st.params = outcome.params.clone();
                if st.main_track.is_some() {
                    st.mode = PlaybackSourceMode::OriginalWithLiveDsp;
                }
            }
            st.params.bypass = false;
            for (i, &g) in outcome.stem_gains_db.iter().enumerate() {
                st.stem_channels[i].gain_db = g;
            }
            st.params_dirty = true;
            self.undo_stack.push_if_changed(&st.params);
        }

        self.status_msg = outcome.summary_msg;

        if outcome.run_ai_split {
            if let Some(in_file) = self.loaded_file_path.clone() {
                let ai_opts = AiPipelineOptions::from_wizard_choices(&choices);
                self.ai_ctrl
                    .start(in_file, self.loaded_dir.clone(), ai_opts, ctx.clone());
            } else {
                self.status_msg =
                    "Apri prima un file audio per avviare la separazione AI in 4 tracce".to_string();
            }
        } else if let Some(tbpm) = choices.target_bpm {
            self.apply_live_target_bpm(tbpm);
        }
    }

    fn trigger_auto_eq(&mut self) {
        let Some(ref eng) = self.engine else { return };
        let track_opt = eng.state.lock().ok().and_then(|st| {
            st.main_track
                .clone()
                .or_else(|| st.ai_master_track.clone())
        });
        if let Some(buf) = track_opt {
            let (computed, summary) = analyze_and_compute_auto_eq(&buf);
            if let Ok(mut st) = eng.state.lock() {
                if st.stems.is_none() {
                    st.mode = PlaybackSourceMode::OriginalWithLiveDsp;
                }
                st.params = computed.clone();
                st.params.bypass = false;
                st.params_dirty = true;
            }
            self.undo_stack.push_if_changed(&computed);
            self.show_eq_drawer = true;
            self.status_msg = summary;
        }
    }

    fn update_fft(&mut self, samples: &[f32]) {
        let n = 4096usize;
        if samples.len() < n {
            for v in &mut self.smoothed_spectrum_db {
                *v = (*v * 0.90_f32 + (-92.0_f32) * 0.10_f32).max(-95.0_f32);
            }
            return;
        }
        let mut buf: Vec<Complex<f32>> = samples[samples.len() - n..]
            .iter()
            .zip(self.hann_window.iter())
            .map(|(&s, &w)| Complex {
                re: s * w,
                im: 0.0_f32,
            })
            .collect();

        let fft = self.fft_planner.plan_fft_forward(n);
        fft.process(&mut buf);

        let scale = 2.0_f32 / (n as f32);
        for (i, c) in buf[..n / 2].iter().enumerate() {
            let mag = (c.re * c.re + c.im * c.im).sqrt() * scale;
            let db = 20.0_f32 * (mag + 1e-6_f32).log10();
            self.smoothed_spectrum_db[i] = self.smoothed_spectrum_db[i] * 0.76_f32 + db * 0.24_f32;
        }
    }

    fn fmt_time(secs: f64) -> String {
        let tot_ms = (secs.max(0.0) * 10.0) as u64;
        let mins = (tot_ms / 10) / 60;
        let s = (tot_ms / 10) % 60;
        let ds = tot_ms % 10;
        format!("{:02}:{:02}.{}", mins, s, ds)
    }

    fn draw_stereo_header_meter(ui: &mut egui::Ui, peak_l: f32, peak_r: f32, rms_db: f32) {
        let desired = Vec2::new(108.0_f32, 34.0_f32);
        let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
        if !ui.is_rect_visible(rect) {
            return;
        }
        let p = ui.painter();
        p.rect_filled(rect, 8.0_f32, Color32::from_rgb(10, 14, 21));
        p.rect_stroke(
            rect,
            8.0_f32,
            Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE),
            StrokeKind::Inside,
        );

        let to_norm = |lin: f32| -> f32 {
            let db = 20.0_f32 * (lin.max(1e-4_f32)).log10();
            ((db + 48.0_f32) / 48.0_f32).clamp(0.0_f32, 1.0_f32)
        };

        for (ch_idx, (lbl, pk)) in [("L", peak_l), ("R", peak_r)].into_iter().enumerate() {
            let y = rect.top() + 6.0_f32 + ch_idx as f32 * 13.0_f32;
            p.text(
                Pos2::new(rect.left() + 6.0_f32, y + 4.5_f32),
                Align2::LEFT_CENTER,
                lbl,
                FontId::monospace(9.5_f32),
                StudioPalette::MUTED_LABEL,
            );

            let bar_bg = Rect::from_min_size(
                Pos2::new(rect.left() + 17.0_f32, y + 1.0_f32),
                Vec2::new(50.0_f32, 7.0_f32),
            );
            p.rect_filled(bar_bg, 3.0_f32, Color32::from_rgb(22, 29, 42));

            let n = to_norm(pk);
            if n > 0.01_f32 {
                let w = bar_bg.width() * n;
                let fill_col = if n > 0.92_f32 {
                    StudioPalette::WARM_AMBER
                } else if n > 0.72_f32 {
                    StudioPalette::ELECTRIC_CYAN
                } else {
                    StudioPalette::EMERALD
                };
                p.rect_filled(
                    Rect::from_min_size(bar_bg.min, Vec2::new(w, bar_bg.height())),
                    3.0_f32,
                    fill_col,
                );
            }
        }

        p.text(
            Pos2::new(rect.right() - 6.0_f32, rect.center().y),
            Align2::RIGHT_CENTER,
            format!("{:+.0}", rms_db.max(-60.0_f32)),
            FontId::monospace(10.0_f32),
            StudioPalette::MUTED_LABEL,
        );
    }
}

impl eframe::App for StudioEqApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(16));

        if let Ok(mut guard) = self.bg_status.lock() {
            if let Some(msg) = guard.take() {
                self.status_msg = msg;
            }
        }

        if let Some(ai_msg) = self.ai_ctrl.latest_summary() {
            self.status_msg = ai_msg;
        }

        if let Some(rep) = self.ai_ctrl.take_latest_report() {
            self.ai_badge_info = rep.format_italian_badge();
            if let Some(b) = rep.effective_bpm {
                self.ui_target_bpm = b.clamp(MIN_USER_BPM, MAX_USER_BPM);
            }
        }

        if self.ai_ctrl.take_completed_flag() {
            let dir = self.loaded_dir.clone();
            let applied = self.wizard_ui.last_applied;
            self.spawn_autoload_workspace(dir, applied, ctx.clone());
        }

        if self.tempo_refresh_ready.swap(false, Ordering::SeqCst) {
            self.refresh_tempo_refs_from_engine();
        }

        // Gestione screenshot GPU per verifica automatica UI (--screenshot <path.bmp>)
        if let Some(ref target_bmp) = self.screenshot_target {
            let saved_opt = ctx.input(|i| {
                for ev in &i.events {
                    if let egui::Event::Screenshot { image, .. } = ev {
                        return Some(Arc::clone(image));
                    }
                }
                None
            });
            if let Some(img) = saved_opt {
                let _ = save_color_image_bmp(&img, target_bmp);
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }
            if !self.loading_busy.load(Ordering::SeqCst) {
                if self.ready_frames == 1 {
                    if let Some(s_idx) = self.screenshot_focus_stem {
                        let s = s_idx.min(3);
                        self.eq_ui.active_target = EqDrawerTarget::Stem(s);
                        if let Some(ref eng) = self.engine {
                            if let Ok(mut st) = eng.state.lock() {
                                st.stem_channels[s].eq.low_db = 3.8_f32;
                                st.stem_channels[s].eq.mid_db = -2.4_f32;
                                st.stem_channels[s].eq.high_db = 3.2_f32;
                            }
                        }
                    }
                }
                self.ready_frames += 1;
                if self.ready_frames >= 5 && !self.screenshot_requested {
                    self.screenshot_requested = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                        egui::UserData::default(),
                    ));
                }
            }
        }

        // Gestione Drag & Drop file audio direttamente nella finestra (apre anche la Procedura Guidata!)
        if let Some(dropped) = extract_dropped_audio_path(ctx) {
            if let Err(e) = self.load_main_file(&dropped) {
                self.status_msg = format!("Errore apertura file: {e}");
            }
        }

        // Scorciatoie da tastiera DAW + Editing Audio Non-Distruttivo + Undo/Redo
        if !self.wizard_ui.open {
            let sc = poll_daw_shortcuts(ctx);
            if sc.toggle_eq_drawer {
                self.show_eq_drawer = !self.show_eq_drawer;
            }
            if sc.toggle_help {
                self.show_help_modal = !self.show_help_modal;
            }
            if sc.run_auto_eq {
                self.trigger_auto_eq();
            }

            let mut shortcut_edit_action: Option<TimelineEditAction> = None;
            if sc.cut_selection {
                shortcut_edit_action = Some(TimelineEditAction::CutSelection);
            } else if sc.copy_selection {
                shortcut_edit_action = Some(TimelineEditAction::CopySelection);
            } else if sc.paste_selection {
                shortcut_edit_action = Some(TimelineEditAction::PasteAtPlayhead);
            } else if sc.delete_selection {
                shortcut_edit_action = Some(TimelineEditAction::DeleteSelection);
            } else if sc.crop_selection {
                shortcut_edit_action = Some(TimelineEditAction::CropSelection);
            } else if sc.duplicate_selection {
                shortcut_edit_action = Some(TimelineEditAction::DuplicateSelection);
            } else if sc.clear_selection {
                shortcut_edit_action = Some(TimelineEditAction::ClearSelection);
            }

            if let Some(act) = shortcut_edit_action {
                self.execute_timeline_edit(act);
            }

            if sc.undo {
                if self.edit_history.can_undo() {
                    self.execute_timeline_edit(TimelineEditAction::UndoAudioEdit);
                } else if let Some(ref eng) = self.engine {
                    if let Ok(mut st) = eng.state.lock() {
                        if let Some(prev) = self.undo_stack.undo() {
                            st.params = prev;
                            st.params_dirty = true;
                            self.status_msg = "Annullata modifica EQ/Master".to_string();
                        }
                    }
                }
            }
            if sc.redo {
                if self.edit_history.can_redo() {
                    self.execute_timeline_edit(TimelineEditAction::RedoAudioEdit);
                } else if let Some(ref eng) = self.engine {
                    if let Ok(mut st) = eng.state.lock() {
                        if let Some(next) = self.undo_stack.redo() {
                            st.params = next;
                            st.params_dirty = true;
                            self.status_msg = "Ripristinata modifica EQ/Master".to_string();
                        }
                    }
                }
            }

            if let Some(ref eng) = self.engine {
                if let Ok(mut st) = eng.state.lock() {
                    if sc.toggle_play {
                        st.playing = !st.playing;
                    }
                    if sc.rewind {
                        st.position_frame = 0;
                    }
                    if sc.toggle_loop && st.selection_range.is_some() {
                        st.loop_enabled = !st.loop_enabled;
                    }
                    if sc.toggle_ab_bypass {
                        st.params.bypass = !st.params.bypass;
                        if st.params.bypass {
                            st.mode = PlaybackSourceMode::OriginalWithLiveDsp;
                        }
                        st.params_dirty = true;
                    }
                    if let Some(stem_idx) = sc.solo_stem {
                        if st.stems.is_some() && stem_idx < 4 {
                            st.stem_channels[stem_idx].solo = !st.stem_channels[stem_idx].solo;
                            st.mode = PlaybackSourceMode::FourStemMixer;
                        }
                    }
                }
            }
        }

        let mut fft_snapshot = Vec::new();
        let mut device_sr = 44100u32;
        let mut is_playing = false;
        let mut pos_frame = 0usize;
        let mut total_frames = 0usize;
        let mut peak_l = 0.0_f32;
        let mut peak_r = 0.0_f32;
        let mut rms_db = -60.0_f32;
        let mut has_stems = false;

        if let Some(ref eng) = self.engine {
            if let Ok(st) = eng.state.lock() {
                fft_snapshot.extend(st.fft_ring.iter().copied());
                device_sr = st.device_sr;
                is_playing = st.playing;
                pos_frame = st.position_frame;
                total_frames = st.active_length();
                peak_l = st.peak_l;
                peak_r = st.peak_r;
                rms_db = st.rms_db;
                has_stems = st.stems.is_some();
            }
        }

        if is_playing {
            self.update_fft(&fft_snapshot);
        } else {
            self.update_fft(&[]);
        }

        let cyan = StudioPalette::ELECTRIC_CYAN;
        let violet = StudioPalette::STUDIO_VIOLET;
        let emerald = StudioPalette::EMERALD;
        let amber = StudioPalette::WARM_AMBER;

        // =========================================================================
        // 1. TOP MODERN DAW HEADER BAR (Zero-Overlap Responsive Layout @ 100%-150% DPI)
        // =========================================================================
        egui::TopBottomPanel::top("modern_daw_header")
            .frame(
                egui::Frame::NONE
                    .fill(StudioPalette::ELEVATED_SURFACE)
                    .stroke(Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE))
                    .inner_margin(egui::Margin::symmetric(10, 7)),
            )
            .show(ctx, |ui| {
                let header_w = ui.available_width();
                ui.spacing_mut().item_spacing.x = if header_w < 1220.0_f32 {
                    4.5_f32
                } else {
                    5.5_f32
                };
                ui.horizontal(|ui| {
                    if transport_play_button(ui, is_playing, cyan)
                        .on_hover_text("Play / Pausa (Barra Spaziatrice)")
                        .clicked()
                    {
                        if let Some(ref eng) = self.engine {
                            if let Ok(mut st) = eng.state.lock() {
                                st.playing = !st.playing;
                            }
                        }
                    }

                    if pill_button(ui, "|<", false, cyan, Vec2::new(32.0_f32, 34.0_f32))
                        .on_hover_text("Torna all'inizio (Tasto Home)")
                        .clicked()
                    {
                        if let Some(ref eng) = self.engine {
                            if let Ok(mut st) = eng.state.lock() {
                                st.position_frame = 0;
                            }
                        }
                    }

                    if header_w >= 1260.0_f32 {
                        let cur_s = pos_frame as f64 / device_sr.max(1) as f64;
                        let tot_s = total_frames as f64 / device_sr.max(1) as f64;
                        egui::Frame::NONE
                            .fill(StudioPalette::DEEP_OBSIDIAN)
                            .stroke(Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE))
                            .corner_radius(CornerRadius::same(8))
                            .inner_margin(egui::Margin::symmetric(9, 7))
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(format!(
                                        "{} / {}",
                                        Self::fmt_time(cur_s),
                                        Self::fmt_time(tot_s)
                                    ))
                                    .monospace()
                                    .size(13.0_f32)
                                    .color(StudioPalette::CRISP_TEXT),
                                );
                            });
                    }

                    if let Some(ref eng) = self.engine {
                        if let Ok(mut st) = eng.state.lock() {
                            if st.selection_range.is_some()
                                && pill_button(
                                    ui,
                                    "Loop",
                                    st.loop_enabled,
                                    amber,
                                    Vec2::new(48.0_f32, 34.0_f32),
                                )
                                .on_hover_text("Ripeti in loop la selezione (Tasto L)")
                                .clicked()
                            {
                                st.loop_enabled = !st.loop_enabled;
                            }

                            let in_stems = st.mode == PlaybackSourceMode::FourStemMixer;
                            let in_master =
                                st.mode != PlaybackSourceMode::FourStemMixer && !st.params.bypass;
                            let in_orig = st.mode == PlaybackSourceMode::OriginalWithLiveDsp
                                && st.params.bypass;

                            if has_stems
                                && pill_button(
                                    ui,
                                    "4 Tracce AI",
                                    in_stems,
                                    violet,
                                    Vec2::new(84.0_f32, 34.0_f32),
                                )
                                .on_hover_text(
                                    "Ascolta e mixa singolarmente Voce, Batteria, Basso e Chitarre",
                                )
                                .clicked()
                            {
                                st.mode = PlaybackSourceMode::FourStemMixer;
                                st.params.bypass = false;
                                st.params_dirty = true;
                            }

                            if pill_button(
                                ui,
                                "Master EQ",
                                in_master,
                                cyan,
                                Vec2::new(80.0_f32, 34.0_f32),
                            )
                            .on_hover_text("Ascolta il Master equalizzato e potenziato")
                            .clicked()
                            {
                                if st.ai_master_track.is_some() {
                                    st.mode = PlaybackSourceMode::PreRenderedAiMaster;
                                    st.params = DspParams::preset_stem_polish();
                                } else {
                                    st.mode = PlaybackSourceMode::OriginalWithLiveDsp;
                                    st.params = DspParams::preset_sala_prove();
                                }
                                st.params.bypass = false;
                                st.params_dirty = true;
                                self.eq_ui.active_target = EqDrawerTarget::MasterBus;
                            }

                            if pill_button(
                                ui,
                                "Originale",
                                in_orig,
                                amber,
                                Vec2::new(74.0_f32, 34.0_f32),
                            )
                            .on_hover_text("Confronta con l'audio grezzo originale (Tasto B)")
                            .clicked()
                            {
                                st.mode = PlaybackSourceMode::OriginalWithLiveDsp;
                                st.params.bypass = true;
                                st.params_dirty = true;
                            }
                        }
                    }

                    if pill_button(ui, "Guidata", self.wizard_ui.open, amber, Vec2::new(68.0_f32, 34.0_f32))
                        .on_hover_text("Apri la Procedura Guidata 4-Step: Separazione AI, Bass Boost, Auto-Tune, Campioni Batteria/Basso e BPM")
                        .clicked()
                    {
                        let name = self.loaded_file_name.clone();
                        self.wizard_ui
                            .open_for_track_with_bpm(&name, self.tempo_refs.detected_bpm);
                    }

                    if pill_button(ui, "Auto-EQ", false, cyan, Vec2::new(64.0_f32, 34.0_f32))
                        .on_hover_text("Analizza lo spettro e calcola l'EQ ottimale in automatico (Tasto A)")
                        .clicked()
                    {
                        self.trigger_auto_eq();
                    }

                    // Controllo BPM Time-Stretch senza distorsione né cambio tonalità
                    if total_frames > 0 && header_w >= 1120.0_f32 {
                        let bpm_changed =
                            (self.ui_target_bpm - self.tempo_refs.current_bpm).abs() >= 0.15_f32;
                        let bpm_modified_from_orig =
                            (self.tempo_refs.current_bpm - self.tempo_refs.detected_bpm).abs()
                                >= 0.25_f32;
                        let mut trigger_bpm_apply: Option<f32> = None;

                        egui::Frame::NONE
                            .fill(StudioPalette::DEEP_OBSIDIAN)
                            .stroke(Stroke::new(
                                1.0_f32,
                                if bpm_modified_from_orig || bpm_changed {
                                    violet
                                } else {
                                    StudioPalette::SUBTLE_HAIRLINE
                                },
                            ))
                            .corner_radius(CornerRadius::same(8))
                            .inner_margin(egui::Margin::symmetric(6, 4))
                            .show(ui, |ui| {
                                ui.spacing_mut().item_spacing.x = 3.0_f32;
                                if pill_button(ui, "-", false, cyan, Vec2::new(20.0_f32, 24.0_f32))
                                    .on_hover_text("-1 BPM")
                                    .clicked()
                                {
                                    self.ui_target_bpm =
                                        (self.ui_target_bpm - 1.0_f32).clamp(MIN_USER_BPM, MAX_USER_BPM);
                                }
                                let dv = ui.add(
                                    egui::DragValue::new(&mut self.ui_target_bpm)
                                        .range(MIN_USER_BPM..=MAX_USER_BPM)
                                        .speed(0.25)
                                        .fixed_decimals(1)
                                        .suffix(" BPM"),
                                );
                                if dv
                                    .on_hover_text(format!(
                                        "Time-Stretch WSOLA Stereo Phase-Locked senza cambio tonalità\nBPM rilevato: {:.1} BPM (Invio o clicca Applica)",
                                        self.tempo_refs.detected_bpm
                                    ))
                                    .lost_focus()
                                    && ctx.input(|i| i.key_pressed(egui::Key::Enter))
                                {
                                    trigger_bpm_apply = Some(self.ui_target_bpm);
                                }
                                if pill_button(ui, "+", false, cyan, Vec2::new(20.0_f32, 24.0_f32))
                                    .on_hover_text("+1 BPM")
                                    .clicked()
                                {
                                    self.ui_target_bpm =
                                        (self.ui_target_bpm + 1.0_f32).clamp(MIN_USER_BPM, MAX_USER_BPM);
                                }
                                if bpm_changed {
                                    if pill_button(
                                        ui,
                                        "Applica",
                                        true,
                                        violet,
                                        Vec2::new(52.0_f32, 24.0_f32),
                                    )
                                    .on_hover_text("Applica subito il nuovo BPM mantenendo intatti pitch e transienti")
                                    .clicked()
                                    {
                                        trigger_bpm_apply = Some(self.ui_target_bpm);
                                    }
                                } else if bpm_modified_from_orig
                                    && pill_button(
                                        ui,
                                        "Orig",
                                        false,
                                        amber,
                                        Vec2::new(36.0_f32, 24.0_f32),
                                    )
                                    .on_hover_text(format!(
                                        "Ripristina il BPM originale rilevato ({:.1} BPM)",
                                        self.tempo_refs.detected_bpm
                                    ))
                                    .clicked()
                                {
                                    trigger_bpm_apply = Some(self.tempo_refs.detected_bpm);
                                }
                            });

                        if let Some(target_b) = trigger_bpm_apply {
                            self.apply_live_target_bpm(target_b);
                        }
                    }

                    if header_w >= 1460.0_f32 {
                        Self::draw_stereo_header_meter(ui, peak_l, peak_r, rms_db);
                    }

                    // Azioni a destra: ?, Esporta, Salva Stems, EQ & Bassi, Separa AI (In-App), Apri Audio
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = if header_w < 1220.0_f32 {
                            4.5_f32
                        } else {
                            5.5_f32
                        };
                        if pill_button(
                            ui,
                            "?",
                            self.show_help_modal,
                            cyan,
                            Vec2::new(28.0_f32, 34.0_f32),
                        )
                        .on_hover_text("Guida rapida e scorciatoie da tastiera (F1)")
                        .clicked()
                        {
                            self.show_help_modal = !self.show_help_modal;
                        }

                        if pill_button(ui, "Esporta", true, emerald, Vec2::new(74.0_f32, 34.0_f32))
                            .on_hover_text("Salva il Master finale in WAV 24-bit da studio o MP3 320 kbps")
                            .clicked()
                        {
                            if let Some(out_path) = rfd::FileDialog::new()
                                .set_file_name("Master_StudioEQ_24bit.wav")
                                .add_filter("WAV 24-Bit Studio", &["wav"])
                                .add_filter("MP3 320 kbps", &["mp3"])
                                .save_file()
                            {
                                if let Some(ref eng) = self.engine {
                                    if let Ok(st) = eng.state.lock() {
                                        let res = export_session(
                                            st.mode,
                                            st.main_track.as_ref(),
                                            st.ai_master_track.as_ref(),
                                            st.stems.as_ref(),
                                            &st.stem_channels,
                                            &st.params,
                                            &out_path,
                                        );
                                        match res {
                                            Ok(_) => {
                                                self.status_msg = format!(
                                                    "Esportato con successo: {}",
                                                    out_path
                                                        .file_name()
                                                        .and_then(|s| s.to_str())
                                                        .unwrap_or("file")
                                                )
                                            }
                                            Err(e) => self.status_msg = format!("Errore export: {e}"),
                                        }
                                    }
                                }
                            }
                        }

                        if has_stems
                            && pill_button(
                                ui,
                                "Salva Stems",
                                false,
                                emerald,
                                Vec2::new(88.0_f32, 34.0_f32),
                            )
                            .on_hover_text("Esporta le 4 tracce separate (Voce, Batteria, Basso, Chitarre) in WAV 24-bit")
                            .clicked()
                        {
                            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                                if let Some(ref eng) = self.engine {
                                    if let Ok(st) = eng.state.lock() {
                                        if let Some(ref stems_arr) = st.stems {
                                            match export_all_stems_wav(
                                                stems_arr,
                                                &st.stem_channels,
                                                &folder,
                                            ) {
                                                Ok(n) => {
                                                    self.status_msg = format!(
                                                        "Esportati {n} Stems WAV 24-bit in {}",
                                                        folder.display()
                                                    )
                                                }
                                                Err(e) => {
                                                    self.status_msg = format!("Errore export stems: {e}")
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if pill_button(
                            ui,
                            "EQ & Bassi",
                            self.show_eq_drawer,
                            cyan,
                            Vec2::new(86.0_f32, 34.0_f32),
                        )
                        .on_hover_text("Mostra o nascondi il pannello Equalizzatore e Manopole Smart (Tasto E)")
                        .clicked()
                        {
                            self.show_eq_drawer = !self.show_eq_drawer;
                        }

                        if draw_ai_header_button(ui, &mut self.ai_ctrl) {
                            if let Some(in_file) = self.loaded_file_path.clone() {
                                let last_ch = self.wizard_ui.last_applied.unwrap_or_default();
                                let ai_opts = AiPipelineOptions::from_wizard_choices(&last_ch);
                                self.ai_ctrl.start(
                                    in_file,
                                    self.loaded_dir.clone(),
                                    ai_opts,
                                    ctx.clone(),
                                );
                            } else {
                                self.status_msg =
                                    "Apri prima un file audio per separare gli strumenti con l'AI"
                                        .to_string();
                            }
                        }

                        if pill_button(ui, "Apri Audio", false, cyan, Vec2::new(82.0_f32, 34.0_f32))
                            .on_hover_text("Apri un file audio e avvia la Procedura Guidata")
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Audio", &["mp3", "mpeg", "wav", "flac", "ogg", "m4a"])
                                .pick_file()
                            {
                                if let Err(e) = self.load_main_file(&path) {
                                    self.status_msg = format!("Errore: {e}");
                                }
                            }
                        }
                    });
                });
            });

        // =========================================================================
        // 2. BOTTOM STATUS BAR + GESTIONE PRESET (.seqpreset)
        // =========================================================================
        egui::TopBottomPanel::bottom("modern_status_footer")
            .frame(
                egui::Frame::NONE
                    .fill(Color32::from_rgb(11, 15, 22))
                    .stroke(Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE))
                    .inner_margin(egui::Margin::symmetric(14, 5)),
            )
            .show(ctx, |ui| {
                let footer_w = ui.available_width();
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(&self.status_msg)
                            .size(12.0_f32)
                            .color(StudioPalette::MUTED_LABEL),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if pill_button(
                            ui,
                            "Carica Preset",
                            false,
                            cyan,
                            Vec2::new(88.0_f32, 22.0_f32),
                        )
                        .on_hover_text("Carica un preset di equalizzazione Master + Tracce Isolate (.seqpreset)")
                        .clicked()
                        {
                            if let Some(p) = rfd::FileDialog::new()
                                .add_filter("Preset StudioEQ", &["seqpreset"])
                                .pick_file()
                            {
                                match load_preset_file_with_stems(&p) {
                                    Ok((loaded_p, stem_opt)) => {
                                        if let Some(ref eng) = self.engine {
                                            if let Ok(mut st) = eng.state.lock() {
                                                st.params = loaded_p.clone();
                                                if let Some((stem_eqs, stem_mix)) = stem_opt {
                                                    for i in 0..4 {
                                                        st.stem_channels[i].eq = stem_eqs[i];
                                                        st.stem_channels[i].gain_db = stem_mix[i].0;
                                                        st.stem_channels[i].pan = stem_mix[i].1;
                                                    }
                                                }
                                                st.params_dirty = true;
                                            }
                                        }
                                        self.undo_stack.push_if_changed(&loaded_p);
                                        self.status_msg =
                                            "Preset caricato (Master + EQ Tracce Isolate)".to_string();
                                    }
                                    Err(e) => self.status_msg = format!("Errore preset: {e}"),
                                }
                            }
                        }
                        if pill_button(
                            ui,
                            "Salva Preset",
                            false,
                            cyan,
                            Vec2::new(82.0_f32, 22.0_f32),
                        )
                        .on_hover_text("Salva la curva EQ Master e le impostazioni Bassi/Medi/Alti delle tracce isolate (.seqpreset)")
                        .clicked()
                        {
                            if let Some(p) = rfd::FileDialog::new()
                                .set_file_name("MioPreset_Sala.seqpreset")
                                .add_filter("Preset StudioEQ", &["seqpreset"])
                                .save_file()
                            {
                                if let Some(ref eng) = self.engine {
                                    if let Ok(st) = eng.state.lock() {
                                        if let Err(e) = save_preset_file_with_stems(
                                            &st.params,
                                            Some(&st.stem_channels),
                                            &p,
                                        ) {
                                            self.status_msg = format!("Errore salvataggio: {e}");
                                        } else {
                                            self.status_msg =
                                                "Preset salvato (Master + Tracce Isolate)".to_string();
                                        }
                                    }
                                }
                            }
                        }

                        let rem_w = ui.available_width();
                        if let Some(ref badge) = self.ai_badge_info {
                            if footer_w >= 960.0_f32 && rem_w >= 140.0_f32 {
                                let b_font = FontId::monospace(11.1_f32);
                                let full_w = ui
                                    .painter()
                                    .layout_no_wrap(
                                        badge.clone(),
                                        b_font.clone(),
                                        StudioPalette::ELECTRIC_CYAN,
                                    )
                                    .size()
                                    .x;
                                let shown_opt = if full_w + 26.0_f32 <= rem_w {
                                    Some(badge.clone())
                                } else {
                                    let segs: Vec<&str> = badge.split("  •  ").collect();
                                    let compact = if segs.len() >= 3 {
                                        format!("{}  •  {}  •  {}", segs[0], segs[1], segs[segs.len() - 1])
                                    } else {
                                        segs.iter().take(2).copied().collect::<Vec<_>>().join("  •  ")
                                    };
                                    let comp_w = ui
                                        .painter()
                                        .layout_no_wrap(
                                            compact.clone(),
                                            b_font,
                                            StudioPalette::ELECTRIC_CYAN,
                                        )
                                        .size()
                                        .x;
                                    (comp_w + 26.0_f32 <= rem_w).then_some(compact)
                                };
                                if let Some(shown_txt) = shown_opt {
                                    ui.separator();
                                    ui.label(
                                        RichText::new(shown_txt)
                                            .monospace()
                                            .size(11.1_f32)
                                            .color(StudioPalette::ELECTRIC_CYAN),
                                    )
                                    .on_hover_text(badge);
                                }
                            }
                        } else if footer_w >= 1180.0_f32 && rem_w >= 470.0_f32 {
                            ui.separator();
                            ui.label(
                                RichText::new(format!(
                                    "{} Hz  •  Spazio: Play  •  Canc/Ctrl+X: Taglia  •  Ctrl+T: Ritaglia  •  1-4: Solo",
                                    device_sr
                                ))
                                .size(11.5_f32)
                                .color(Color32::from_rgb(100, 116, 139)),
                            );
                        }
                    });
                });
            });

        // =========================================================================
        // 3. DRAWER EQUALIZZATORE & 4 MANOPOLE SMART (Richiudibile)
        // =========================================================================
        if self.show_eq_drawer {
            egui::TopBottomPanel::bottom("eq_smart_drawer")
                .resizable(false)
                .frame(
                    egui::Frame::NONE
                        .fill(StudioPalette::ELEVATED_SURFACE)
                        .stroke(Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE))
                        .inner_margin(egui::Margin::symmetric(14, 10)),
                )
                .show(ctx, |ui| {
                    if let Some(ref eng) = self.engine {
                        if let Ok(mut st) = eng.state.lock() {
                            draw_eq_and_master_drawer(
                                ui,
                                &mut st,
                                &mut self.eq_ui,
                                &self.smoothed_spectrum_db,
                            );
                            if !ctx.input(|i| i.pointer.any_down()) {
                                self.undo_stack.push_if_changed(&st.params);
                            }
                        }
                    }
                });
        }

        // =========================================================================
        // 4. AREA CENTRALE: TIMELINE MULTI-TRACCIA CON BARRA EDITING & FORME D'ONDA
        // =========================================================================
        let mut pending_tl_action: Option<TimelineEditAction> = None;
        egui::CentralPanel::default()
            .frame(
                egui::Frame::NONE
                    .fill(StudioPalette::DEEP_OBSIDIAN)
                    .inner_margin(egui::Margin::symmetric(12, 8)),
            )
            .show(ctx, |ui| {
                let active_stem_eq_idx = match (self.show_eq_drawer, self.eq_ui.active_target) {
                    (true, EqDrawerTarget::Stem(i)) => Some(i),
                    _ => None,
                };
                let edit_ctx = TimelineEditContext {
                    has_clipboard: self.clipboard.has_content(),
                    can_undo_edit: self.edit_history.can_undo(),
                    can_redo_edit: self.edit_history.can_redo(),
                    active_stem_eq_idx,
                };
                if let Some(ref eng) = self.engine {
                    if let Ok(mut st) = eng.state.lock() {
                        pending_tl_action = draw_multitrack_timeline(
                            ui,
                            &mut st,
                            &mut self.timeline_ui,
                            edit_ctx,
                        );
                    }
                }
            });
        if let Some(act) = pending_tl_action {
            self.execute_timeline_edit(act);
        }

        // Modali In-App: Procedura Guidata 3-Step, Monitor Progresso AI GPU, Guida F1 e Drag&Drop
        if let Some(chosen) = draw_guided_wizard_modal(ctx, &mut self.wizard_ui) {
            self.apply_wizard_choices(chosen, ctx);
        }
        draw_ai_pipeline_modal(ctx, &mut self.ai_ctrl);
        draw_shortcuts_modal(ctx, &mut self.show_help_modal);
        draw_drop_zone_overlay(ctx);
    }
}

fn main() -> eframe::Result<()> {
    let mut screenshot_target: Option<PathBuf> = None;
    let mut screenshot_focus_stem: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--screenshot" {
            if let Some(p) = args.next() {
                screenshot_target = Some(PathBuf::from(p));
            }
        } else if arg == "--focus-stem" {
            if let Some(v) = args.next() {
                screenshot_focus_stem = v.parse::<usize>().ok();
            }
        }
    }

    let icon = icon::build_app_icon();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("StudioEQ Master Pro — Multi-Track AI Stem Mixer & Equalizer")
            .with_inner_size([1380.0_f32, 850.0_f32])
            .with_min_inner_size([1060.0_f32, 660.0_f32])
            .with_drag_and_drop(true)
            .with_icon(icon),
        ..Default::default()
    };

    eframe::run_native(
        "StudioEQ Master Pro",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(StudioEqApp::new(
                cc,
                screenshot_target,
                screenshot_focus_stem,
            )))
        }),
    )
}
