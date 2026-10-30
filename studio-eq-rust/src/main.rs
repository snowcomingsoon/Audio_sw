#![forbid(unsafe_code)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ai_pipeline;
mod audio_io;
mod auto_eq;
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

use ai_pipeline::{draw_ai_header_button, draw_ai_pipeline_modal, AiPipelineController};
use audio_io::{
    decode_audio_file, export_session, PlaybackSourceMode, RealtimeAudioEngine, StereoBuffer,
};
use auto_eq::analyze_and_compute_auto_eq;
use dsp::DspParams;
use eq_drawer::{draw_eq_and_master_drawer, EqDrawerState};
use project_io::{
    draw_drop_zone_overlay, export_all_stems_wav, extract_dropped_audio_path, load_preset_file,
    save_preset_file,
};
use shortcuts_and_meters::{draw_shortcuts_modal, poll_daw_shortcuts, UndoStack};
use timeline_view::{draw_multitrack_timeline, TimelineState};
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

    ai_ctrl: AiPipelineController,
    wizard_ui: GuidedWizardState,

    loading_busy: Arc<AtomicBool>,
    bg_status: Arc<Mutex<Option<String>>>,

    screenshot_target: Option<PathBuf>,
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

    fn new(cc: &eframe::CreationContext<'_>, screenshot_target: Option<PathBuf>) -> Self {
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
        let initial_path = orig_candidate.exists().then_some(orig_candidate);
        let initial_name = if initial_path.is_some() {
            "traccia_originale_sala.mp3".to_string()
        } else {
            "Sessione Vuota".to_string()
        };

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
            ai_ctrl: AiPipelineController::new(),
            wizard_ui: GuidedWizardState::default(),
            loading_busy: Arc::new(AtomicBool::new(false)),
            bg_status: Arc::new(Mutex::new(None)),
            screenshot_target,
            screenshot_requested: false,
            ready_frames: 0,
            fft_planner: FftPlanner::new(),
            hann_window,
            smoothed_spectrum_db: vec![-90.0_f32; n_fft / 2],
        };

        app.spawn_autoload_workspace(default_out_dir, None, cc.egui_ctx.clone());
        app
    }

    fn locate_default_output_dir() -> PathBuf {
        let mut candidates = vec![PathBuf::from(r"c:\_gestione ufficio\Audio SW\output")];
        if let Ok(exe) = std::env::current_exe() {
            if let Some(p) = exe.parent() {
                candidates.push(p.join("output"));
                if let Some(pp) = p.parent() {
                    candidates.push(pp.join("output"));
                    candidates.push(pp.join("Resources").join("output"));
                }
            }
        }
        for c in &candidates {
            if c.exists() {
                return c.clone();
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
        PathBuf::from(r"c:\_gestione ufficio\Audio SW\output")
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
        busy.store(true, Ordering::SeqCst);

        std::thread::spawn(move || {
            let sr = state_arc.lock().map(|s| s.device_sr).unwrap_or(44100);

            let orig = out_dir.join("traccia_originale_sala.mp3");
            if wizard_choices.is_none() && orig.exists() {
                if let Ok(buf) = decode_audio_file(&orig) {
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
                            "Traccia principale pronta  •  Caricamento 4 Stems AI...".to_string(),
                        );
                    }
                    ctx.request_repaint();
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
                            "4 Tracce AI Separate attive in Timeline  •  Usa M / S / Volume / Pan o clicca Guidata per cambiare stile"
                                .to_string(),
                        );
                    }
                }
            } else if orig.exists() {
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

            busy.store(false, Ordering::SeqCst);
            ctx.request_repaint();
        });
    }

    fn load_main_file(&mut self, path: &Path) -> Result<(), String> {
        let buf = decode_audio_file(path)?;
        let Some(ref eng) = self.engine else {
            return Err("Motore audio non disponibile".to_string());
        };
        let sr = eng.state.lock().map(|s| s.device_sr).unwrap_or(44100);
        let resampled = buf.resample_to(sr);
        let initial_params = DspParams::preset_sala_prove();

        if let Ok(mut st) = eng.state.lock() {
            st.playing = false;
            st.position_frame = 0;
            st.selection_range = None;
            st.main_track = Some(resampled);
            st.stems = None;
            st.ai_master_track = None;
            st.mode = PlaybackSourceMode::OriginalWithLiveDsp;
            st.params = initial_params.clone();
            for ch in &mut st.stem_channels {
                ch.gain_db = 0.0_f32;
                ch.pan = 0.0_f32;
                ch.muted = false;
                ch.solo = false;
            }
            st.params_dirty = true;
        }
        self.undo_stack.reset(initial_params);
        self.loaded_file_path = Some(path.to_path_buf());
        self.loaded_file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("traccia_audio")
            .to_string();

        // Apre automaticamente il Menu Guidato (Step 1: Dividi Tracce? -> Step 2: Bass Boost -> Step 3: Tocco Finale)
        self.wizard_ui.open_for_track(&self.loaded_file_name);

        self.status_msg = format!(
            "Caricato: {}  •  Scegli le opzioni nella Procedura Guidata",
            self.loaded_file_name
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
                self.ai_ctrl
                    .start(in_file, self.loaded_dir.clone(), ctx.clone());
            } else {
                self.status_msg =
                    "Apri prima un file audio per avviare la separazione AI in 4 tracce".to_string();
            }
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

        if self.ai_ctrl.take_completed_flag() {
            let dir = self.loaded_dir.clone();
            let applied = self.wizard_ui.last_applied;
            self.spawn_autoload_workspace(dir, applied, ctx.clone());
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

        // Scorciatoie da tastiera DAW + Undo/Redo (disattivate mentre il Menu Guidato è in primo piano)
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
                    if sc.undo {
                        if let Some(prev) = self.undo_stack.undo() {
                            st.params = prev;
                            st.params_dirty = true;
                        }
                    }
                    if sc.redo {
                        if let Some(next) = self.undo_stack.redo() {
                            st.params = next;
                            st.params_dirty = true;
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

                    if pill_button(ui, "Guidata", self.wizard_ui.open, amber, Vec2::new(72.0_f32, 34.0_f32))
                        .on_hover_text("Apri la Procedura Guidata semplice: Divisione tracce, Boost dei Bassi, Voce e Spinta Master")
                        .clicked()
                    {
                        let name = self.loaded_file_name.clone();
                        self.wizard_ui.open_for_track(&name);
                    }

                    if pill_button(ui, "Auto-EQ", false, cyan, Vec2::new(68.0_f32, 34.0_f32))
                        .on_hover_text("Analizza lo spettro e calcola l'EQ ottimale in automatico (Tasto A)")
                        .clicked()
                    {
                        self.trigger_auto_eq();
                    }

                    if header_w >= 1340.0_f32 {
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
                                self.ai_ctrl
                                    .start(in_file, self.loaded_dir.clone(), ctx.clone());
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
                        .on_hover_text("Carica un preset di equalizzazione (.seqpreset)")
                        .clicked()
                        {
                            if let Some(p) = rfd::FileDialog::new()
                                .add_filter("Preset StudioEQ", &["seqpreset"])
                                .pick_file()
                            {
                                match load_preset_file(&p) {
                                    Ok(loaded_p) => {
                                        if let Some(ref eng) = self.engine {
                                            if let Ok(mut st) = eng.state.lock() {
                                                st.params = loaded_p.clone();
                                                st.params_dirty = true;
                                            }
                                        }
                                        self.undo_stack.push_if_changed(&loaded_p);
                                        self.status_msg = "Preset caricato con successo".to_string();
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
                        .on_hover_text("Salva la curva EQ attuale su disco (.seqpreset)")
                        .clicked()
                        {
                            if let Some(p) = rfd::FileDialog::new()
                                .set_file_name("MioPreset_Sala.seqpreset")
                                .add_filter("Preset StudioEQ", &["seqpreset"])
                                .save_file()
                            {
                                if let Some(ref eng) = self.engine {
                                    if let Ok(st) = eng.state.lock() {
                                        if let Err(e) = save_preset_file(&st.params, &p) {
                                            self.status_msg = format!("Errore salvataggio: {e}");
                                        } else {
                                            self.status_msg = "Preset salvato".to_string();
                                        }
                                    }
                                }
                            }
                        }

                        if footer_w >= 1180.0_f32 {
                            ui.separator();
                            ui.label(
                                RichText::new(format!(
                                    "{} Hz  •  Spazio: Play  •  B: Confronta A/B  •  1-4: Solo Strumenti  •  Ctrl+Z: Annulla",
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
        // 4. AREA CENTRALE: TIMELINE MULTI-TRACCIA CON FORME D'ONDA SEPARATE
        // =========================================================================
        egui::CentralPanel::default()
            .frame(
                egui::Frame::NONE
                    .fill(StudioPalette::DEEP_OBSIDIAN)
                    .inner_margin(egui::Margin::symmetric(12, 8)),
            )
            .show(ctx, |ui| {
                if let Some(ref eng) = self.engine {
                    if let Ok(mut st) = eng.state.lock() {
                        draw_multitrack_timeline(ui, &mut st, &mut self.timeline_ui);
                    }
                }
            });

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
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--screenshot" {
            if let Some(p) = args.next() {
                screenshot_target = Some(PathBuf::from(p));
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
        Box::new(move |cc| Ok(Box::new(StudioEqApp::new(cc, screenshot_target)))),
    )
}
