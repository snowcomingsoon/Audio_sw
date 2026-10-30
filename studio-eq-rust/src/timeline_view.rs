use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Margin, PointerButton, Pos2, Rect, ScrollArea,
    Sense, Stroke, StrokeKind, Ui, Vec2,
};

use crate::audio_io::{PlaybackSourceMode, SharedAudioState, StemChannel, StereoBuffer};
use crate::ui_widgets::{brighten, sleek_progress_or_fader};

const HEADER_WIDTH: f32 = 215.0_f32;
const RULER_HEIGHT: f32 = 26.0_f32;
const COL_GAP: f32 = 6.0_f32;
const LANE_GAP: f32 = 4.5_f32;
const MIN_ZOOM: f32 = 1.0_f32;
const MAX_ZOOM: f32 = 24.0_f32;

const STEM_LABELS: [&str; 4] = ["VOCE", "BATTERIA", "BASSO", "CHITARRE & ALTRO"];
const STEM_SUBTITLES_IT: [&str; 4] = [
    "Pulizia Voce + Presenza",
    "Spinta Cassa + Anti-Scatola",
    "Sub Centrato + Anti-Rimbombo",
    "Corpo Medie + Apertura Stereo",
];
const STEM_TOOLTIPS_IT: [&str; 4] = [
    "Traccia 1 — VOCE (Tasto Solo rapido: 1)\nFiltro anti-rientri sala + presenza vocale nitida",
    "Traccia 2 — BATTERIA (Tasto Solo rapido: 2)\nSpinta cassa 62 Hz + pulizia risonanze scatolari",
    "Traccia 3 — BASSO (Tasto Solo rapido: 3)\nBasse frequenze centrate in mono + anti-rimbombo",
    "Traccia 4 — CHITARRE & ALTRO (Tasto Solo rapido: 4)\nCorpo sulle medie 1.4 kHz + apertura panoramica stereo",
];

fn stem_accent_color(idx: usize) -> Color32 {
    match idx {
        0 => Color32::from_rgb(56, 189, 248),  // Voce: ciano elettrico
        1 => Color32::from_rgb(245, 158, 11),  // Batteria: ambra vibrante
        2 => Color32::from_rgb(168, 85, 247),  // Basso: viola neon
        _ => Color32::from_rgb(34, 197, 94),   // Chitarre & Altro: verde smeraldo
    }
}

/// Azioni immediate emesse dalla Barra Strumenti Editing Audio in cima alla Timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimelineEditAction {
    CutSelection,
    DeleteSelection,
    CropSelection,
    SilenceSelection,
    CopySelection,
    PasteAtPlayhead,
    DuplicateSelection,
    FadeIn,
    FadeOut,
    NormalizeSelection,
    UndoAudioEdit,
    RedoAudioEdit,
    ClearSelection,
}

/// Stato live passato dal controller principale alla Barra Strumenti Editing Audio.
#[derive(Clone, Copy, Debug, Default)]
pub struct TimelineEditContext {
    pub has_clipboard: bool,
    pub can_undo_edit: bool,
    pub can_redo_edit: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionDragKind {
    Creating { anchor: usize },
    AdjustLeft { fixed_right: usize },
    AdjustRight { fixed_left: usize },
}

/// Stato visuale della Timeline Multi-Traccia (zoom, scorrimento orizzontale e selezione regione).
#[derive(Clone, Debug)]
pub struct TimelineState {
    pub zoom: f32,
    pub scroll_norm: f32,
    #[allow(dead_code)]
    pub drag_select_start: Option<usize>,
    #[allow(dead_code)]
    pub ruler_drag_start: Option<usize>,
    pub lane_drag: Option<SelectionDragKind>,
    pub ruler_drag: Option<SelectionDragKind>,
}

impl Default for TimelineState {
    fn default() -> Self {
        Self {
            zoom: 1.0_f32,
            scroll_norm: 0.0_f32,
            drag_select_start: None,
            ruler_drag_start: None,
            lane_drag: None,
            ruler_drag: None,
        }
    }
}

fn fmt_mm_ss_tenths(sec: f64) -> String {
    let total_tenths = (sec.max(0.0) * 10.0).round() as u64;
    let mins = total_tenths / 600;
    let secs = (total_tenths / 10) % 60;
    let tenths = total_tenths % 10;
    format!("{mins:02}:{secs:02}.{tenths}")
}

fn toolbar_action_pill(
    ui: &mut Ui,
    label: &str,
    enabled: bool,
    accent: Color32,
    height: f32,
) -> egui::Response {
    let font = FontId::proportional(11.1_f32);
    let text_col = if enabled {
        Color32::from_rgb(236, 242, 250)
    } else {
        Color32::from_rgb(92, 106, 128)
    };
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font.clone(), text_col);
    let width = (galley.size().x + 15.0_f32).ceil().max(42.0_f32);
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(width, height), sense);

    if ui.is_rect_visible(rect) {
        let hovered = enabled && resp.hovered();
        let pressed = enabled && resp.is_pointer_button_down_on();
        let bg = if !enabled {
            Color32::from_rgb(15, 20, 31)
        } else if pressed {
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 68)
        } else if hovered {
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 38)
        } else {
            Color32::from_rgb(20, 27, 42)
        };
        let border = if !enabled {
            Color32::from_rgb(27, 36, 52)
        } else if hovered || pressed {
            accent
        } else {
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 105)
        };

        let r = CornerRadius::same(6);
        ui.painter().rect_filled(rect, r, bg);
        ui.painter()
            .rect_stroke(rect, r, Stroke::new(1.0_f32, border), StrokeKind::Inside);
        ui.painter()
            .text(rect.center(), Align2::CENTER_CENTER, label, font, text_col);
    }

    if enabled && resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

fn draw_selection_status_strip(
    ui: &mut Ui,
    selection_range: Option<(usize, usize)>,
    sample_rate: u32,
    solo_stem_active: bool,
    out_action: &mut Option<TimelineEditAction>,
) {
    let cyan = Color32::from_rgb(56, 189, 248);
    let amber = Color32::from_rgb(245, 158, 11);

    if let Some((a, b)) = selection_range.filter(|&(x, y)| x.max(y) > x.min(y) + 16) {
        let s0 = a.min(b);
        let s1 = a.max(b);
        let sr = sample_rate.max(1) as f64;
        let t0 = s0 as f64 / sr;
        let t1 = s1 as f64 / sr;
        let dur = (s1 - s0) as f64 / sr;
        let solo_suffix = if solo_stem_active {
            "  •  Solo attivo"
        } else {
            ""
        };
        let readout = format!(
            "Selezione: {} – {} (durata {:.1}s){}",
            fmt_mm_ss_tenths(t0),
            fmt_mm_ss_tenths(t1),
            dur,
            solo_suffix
        );

        ui.label(
            egui::RichText::new(readout)
                .size(11.3_f32)
                .strong()
                .color(if solo_stem_active { amber } else { cyan }),
        );

        if toolbar_action_pill(
            ui,
            "✕ Deseleziona",
            true,
            Color32::from_rgb(244, 63, 94),
            22.0_f32,
        )
        .on_hover_text("Azzera la selezione corrente e disattiva il Loop (Tasto Esc)")
        .clicked()
        {
            *out_action = Some(TimelineEditAction::ClearSelection);
        }
    } else {
        ui.label(
            egui::RichText::new(
                "Trascina sulle tracce per selezionare una parte da tagliare o sfumare",
            )
            .size(11.2_f32)
            .color(Color32::from_rgb(132, 148, 174)),
        );
    }
}

fn draw_audio_editing_toolbar(
    ui: &mut Ui,
    state: &SharedAudioState,
    sample_rate: u32,
    edit_ctx: TimelineEditContext,
) -> Option<TimelineEditAction> {
    let mut action = None;
    let has_sel = state
        .selection_range
        .map(|(a, b)| a.max(b) > a.min(b) + 16)
        .unwrap_or(false);
    let solo_stem_active = state.mode == PlaybackSourceMode::FourStemMixer
        && state.stems.is_some()
        && state.stem_channels.iter().any(|c| c.solo);

    let cyan = Color32::from_rgb(56, 189, 248);
    let amber = Color32::from_rgb(245, 158, 11);
    let emerald = Color32::from_rgb(52, 211, 153);
    let violet = Color32::from_rgb(192, 132, 252);
    let coral = Color32::from_rgb(251, 113, 133);

    egui::Frame::NONE
        .fill(Color32::from_rgb(14, 19, 30))
        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(31, 43, 65)))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(8, 5))
        .show(ui, |ui| {
            let bar_w = ui.available_width();
            let single_row = bar_w >= 1640.0_f32;
            let btn_h = 24.0_f32;

            let draw_buttons = |ui: &mut Ui, act: &mut Option<TimelineEditAction>| {
                ui.spacing_mut().item_spacing = Vec2::new(4.5_f32, 4.0_f32);

                if toolbar_action_pill(ui, "✂ Taglia (Canc)", has_sel, coral, btn_h)
                    .on_hover_text(
                        "Taglia via la zona selezionata negli appunti con micro-crossfade da 8 ms\n\
                         Scorciatoie: Canc / Backspace (elimina) oppure Ctrl+X (taglia negli appunti)",
                    )
                    .clicked()
                {
                    *act = Some(TimelineEditAction::CutSelection);
                }

                if toolbar_action_pill(
                    ui,
                    "◰ Ritaglia Inizio/Fine (Ctrl+T)",
                    has_sel,
                    emerald,
                    btn_h,
                )
                .on_hover_text(
                    "Mantiene SOLO la parte selezionata eliminando chiacchiere prima e dopo il brano (Ctrl+T)\n\
                     Applica automaticamente micro-fade anti-click da 5 ms su inizio e fine",
                )
                .clicked()
                {
                    *act = Some(TimelineEditAction::CropSelection);
                }

                if toolbar_action_pill(ui, "🔇 Silenzia Zona", has_sel, amber, btn_h)
                    .on_hover_text(
                        "Azzera morbidamente il volume nella selezione (rampe cosinusoidali 10 ms).\n\
                         Funzione Smart Sala Prove: se uno Stem è in Solo, silenzia SOLO quello Stem!",
                    )
                    .clicked()
                {
                    *act = Some(TimelineEditAction::SilenceSelection);
                }

                if toolbar_action_pill(ui, "📋 Copia", has_sel, cyan, btn_h)
                    .on_hover_text("Copia la zona selezionata negli appunti Studio (Ctrl+C)")
                    .clicked()
                {
                    *act = Some(TimelineEditAction::CopySelection);
                }

                if toolbar_action_pill(ui, "📥 Incolla", edit_ctx.has_clipboard, cyan, btn_h)
                    .on_hover_text(
                        "Incolla gli appunti alla posizione attuale della testina con crossfade da 8 ms (Ctrl+V)\n\
                         Suggerimento: usa Ctrl+D per duplicare al volo la selezione!",
                    )
                    .clicked()
                {
                    *act = Some(TimelineEditAction::PasteAtPlayhead);
                }

                if toolbar_action_pill(ui, "◢ Fade In", true, violet, btn_h)
                    .on_hover_text(
                        "Applica una sfumatura d'ingresso S-curve musicale sulla selezione,\n\
                         oppure sui primi 2.0 secondi del brano se non hai selezionato nulla",
                    )
                    .clicked()
                {
                    *act = Some(TimelineEditAction::FadeIn);
                }

                if toolbar_action_pill(ui, "◣ Fade Out", true, violet, btn_h)
                    .on_hover_text(
                        "Applica una sfumatura d'uscita S-curve musicale sulla selezione,\n\
                         oppure sugli ultimi 3.0 secondi del brano se non hai selezionato nulla",
                    )
                    .clicked()
                {
                    *act = Some(TimelineEditAction::FadeOut);
                }

                if toolbar_action_pill(ui, "Normalizza", true, emerald, btn_h)
                    .on_hover_text(
                        "Normalizza a -1.0 dBFS la selezione o la traccia attiva\n\
                         (utilissimo dopo tagli o silenziamenti; se uno Stem è in Solo, normalizza solo quello Stem!)",
                    )
                    .clicked()
                {
                    *act = Some(TimelineEditAction::NormalizeSelection);
                }

                if toolbar_action_pill(ui, "↶ Annulla Edit", edit_ctx.can_undo_edit, cyan, btn_h)
                    .on_hover_text("Annulla l'ultima modifica audio sulla timeline (Ctrl+Z)")
                    .clicked()
                {
                    *act = Some(TimelineEditAction::UndoAudioEdit);
                }

                if toolbar_action_pill(
                    ui,
                    "↷ Ripristina Edit",
                    edit_ctx.can_redo_edit,
                    cyan,
                    btn_h,
                )
                .on_hover_text("Ripristina la modifica audio annullata (Ctrl+Y)")
                .clicked()
                {
                    *act = Some(TimelineEditAction::RedoAudioEdit);
                }
            };

            if single_row {
                ui.horizontal(|ui| {
                    draw_buttons(ui, &mut action);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 6.0_f32;
                        if has_sel {
                            if toolbar_action_pill(
                                ui,
                                "✕ Deseleziona",
                                true,
                                Color32::from_rgb(244, 63, 94),
                                22.0_f32,
                            )
                            .on_hover_text("Azzera la selezione corrente e disattiva il Loop (Esc)")
                            .clicked()
                            {
                                action = Some(TimelineEditAction::ClearSelection);
                            }
                            if let Some((a, b)) = state.selection_range {
                                let s0 = a.min(b);
                                let s1 = a.max(b);
                                let sr = sample_rate.max(1) as f64;
                                let solo_suffix = if solo_stem_active {
                                    " • Solo attivo"
                                } else {
                                    ""
                                };
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Selezione: {} – {} (durata {:.1}s){}",
                                        fmt_mm_ss_tenths(s0 as f64 / sr),
                                        fmt_mm_ss_tenths(s1 as f64 / sr),
                                        (s1 - s0) as f64 / sr,
                                        solo_suffix
                                    ))
                                    .size(11.3_f32)
                                    .strong()
                                    .color(if solo_stem_active { amber } else { cyan }),
                                );
                            }
                        } else {
                            ui.label(
                                egui::RichText::new(
                                    "Trascina sulle tracce per selezionare una parte da tagliare o sfumare",
                                )
                                .size(11.2_f32)
                                .color(Color32::from_rgb(132, 148, 174)),
                            );
                        }
                    });
                });
            } else {
                ui.horizontal_wrapped(|ui| {
                    draw_buttons(ui, &mut action);
                });
                ui.add_space(3.0_f32);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0_f32;
                    draw_selection_status_strip(
                        ui,
                        state.selection_range,
                        sample_rate,
                        solo_stem_active,
                        &mut action,
                    );
                });
            }
        });

    action
}

fn near_selection_edge_px(
    hover_x: f32,
    rect: Rect,
    start_frame: usize,
    vis_frames: usize,
    selection_range: Option<(usize, usize)>,
) -> Option<bool> {
    let (a, b) = selection_range?;
    let s0 = a.min(b);
    let s1 = a.max(b);
    if s1 <= s0 {
        return None;
    }
    let end_frame = start_frame + vis_frames;
    let tol = 7.5_f32;
    if s0 >= start_frame && s0 <= end_frame {
        let sx0 = frame_to_x(s0, rect, start_frame, vis_frames);
        if (hover_x - sx0).abs() <= tol {
            return Some(true); // Bordo sinistro
        }
    }
    if s1 >= start_frame && s1 <= end_frame {
        let sx1 = frame_to_x(s1, rect, start_frame, vis_frames);
        if (hover_x - sx1).abs() <= tol {
            return Some(false); // Bordo destro
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn handle_range_selection_interaction(
    ui: &mut Ui,
    resp: &egui::Response,
    rect: Rect,
    start_frame: usize,
    vis_frames: usize,
    total_frames: usize,
    activate_loop_on_create: bool,
    position_frame: &mut usize,
    selection_range: &mut Option<(usize, usize)>,
    loop_enabled: &mut bool,
    drag_state: &mut Option<SelectionDragKind>,
    default_hover_cursor: egui::CursorIcon,
) {
    let near_edge = ui
        .input(|i| i.pointer.hover_pos())
        .filter(|p| rect.expand2(Vec2::new(6.0_f32, 0.0_f32)).contains(*p))
        .and_then(|p| near_selection_edge_px(p.x, rect, start_frame, vis_frames, *selection_range));

    if resp.double_clicked() {
        *selection_range = None;
        *loop_enabled = false;
        *drag_state = None;
        if let Some(pos) = resp.interact_pointer_pos() {
            *position_frame = x_to_frame(pos.x, rect, start_frame, vis_frames, total_frames);
        }
    } else {
        if resp.drag_started() {
            let origin = ui
                .input(|i| i.pointer.press_origin())
                .or_else(|| resp.interact_pointer_pos());
            if let Some(pos) = origin {
                let edge_hit =
                    near_selection_edge_px(pos.x, rect, start_frame, vis_frames, *selection_range);
                match (edge_hit, *selection_range) {
                    (Some(true), Some((a, b))) => {
                        *drag_state = Some(SelectionDragKind::AdjustLeft {
                            fixed_right: a.max(b),
                        });
                    }
                    (Some(false), Some((a, b))) => {
                        *drag_state = Some(SelectionDragKind::AdjustRight {
                            fixed_left: a.min(b),
                        });
                    }
                    _ => {
                        let anchor = x_to_frame(pos.x, rect, start_frame, vis_frames, total_frames);
                        *drag_state = Some(SelectionDragKind::Creating { anchor });
                    }
                }
            }
        }

        if resp.dragged() {
            if let (Some(kind), Some(pos)) = (*drag_state, resp.interact_pointer_pos()) {
                let curr = x_to_frame(pos.x, rect, start_frame, vis_frames, total_frames);
                let min_span = (vis_frames / 300).max(32);
                match kind {
                    SelectionDragKind::Creating { anchor } => {
                        let lo = anchor.min(curr);
                        let hi = anchor.max(curr);
                        if hi >= lo + min_span {
                            *selection_range = Some((lo, hi));
                            if activate_loop_on_create {
                                *loop_enabled = true;
                            }
                            *position_frame = lo;
                        } else if activate_loop_on_create {
                            *position_frame = curr;
                        }
                    }
                    SelectionDragKind::AdjustLeft { fixed_right } => {
                        let lo = curr.min(fixed_right.saturating_sub(min_span));
                        let hi = fixed_right.max(lo + min_span).min(total_frames);
                        *selection_range = Some((lo, hi));
                        *position_frame = lo;
                    }
                    SelectionDragKind::AdjustRight { fixed_left } => {
                        let lo = fixed_left.min(total_frames.saturating_sub(min_span));
                        let hi = curr.max(lo + min_span).min(total_frames);
                        *selection_range = Some((lo, hi));
                    }
                }
            }
        }

        if resp.drag_stopped() {
            *drag_state = None;
        }

        if resp.clicked() && !resp.double_clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let clicked = x_to_frame(pos.x, rect, start_frame, vis_frames, total_frames);
                let shift_held = ui.input(|i| i.modifiers.shift);
                if shift_held {
                    let (lo, hi) = if let Some((a, b)) = *selection_range {
                        let s0 = a.min(b);
                        let s1 = a.max(b);
                        if clicked < s0 {
                            (clicked, s1)
                        } else {
                            (s0, clicked)
                        }
                    } else {
                        ((*position_frame).min(clicked), (*position_frame).max(clicked))
                    };
                    if hi > lo + 16 {
                        *selection_range = Some((lo, hi));
                        *position_frame = lo;
                    }
                } else {
                    *position_frame = clicked;
                }
                *drag_state = None;
            }
        }
    }

    if resp.dragged() || (resp.hovered() && near_edge.is_some()) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
    } else if resp.hovered() {
        ui.ctx().set_cursor_icon(default_hover_cursor);
    }
}

/// Disegna la Timeline Multi-Traccia stile Audacity 4 / Logic Pro / Moises.ai con Barra Editing Audio,
/// righello adattivo, corsie separate per i 4 Stems Demucs (o traccia Stereo principale), controlli M/S/Volume/Pan,
/// mini LED meter live, forme d'onda Peak + RMS, maniglie di selezione e playhead sincronizzato.
pub fn draw_multitrack_timeline(
    ui: &mut egui::Ui,
    state: &mut SharedAudioState,
    tl: &mut TimelineState,
    edit_ctx: TimelineEditContext,
) -> Option<TimelineEditAction> {
    tl.zoom = tl.zoom.clamp(MIN_ZOOM, MAX_ZOOM);
    tl.scroll_norm = tl.scroll_norm.clamp(0.0_f32, 1.0_f32);

    let (total_frames, sample_rate) = active_timeline_metrics(state);
    let mut triggered_action: Option<TimelineEditAction> = None;

    egui::Frame::NONE
        .fill(Color32::from_rgb(11, 15, 24))
        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(28, 37, 54)))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::symmetric(8, 6))
        .show(ui, |ui| {
            let total_w = ui.available_width().max(380.0_f32);
            let wave_w = (total_w - HEADER_WIDTH - COL_GAP).max(140.0_f32);

            if total_frames == 0 {
                draw_empty_timeline_placeholder(ui, wave_w);
                return;
            }

            // 0. BARRA STRUMENTI EDITING AUDIO NON-DISTRUTTIVO
            if let Some(act) = draw_audio_editing_toolbar(ui, state, sample_rate, edit_ctx) {
                triggered_action = Some(act);
            }
            ui.add_space(4.5_f32);

            // Gestione input Zoom (Ctrl + rotellina / pinch) e Scroll orizzontale (Shift + rotellina / trackpad X)
            let timeline_hover_rect = Rect::from_min_size(
                ui.cursor().min,
                Vec2::new(total_w, ui.available_height().max(200.0_f32)),
            );
            let pointer_in_timeline = ui
                .input(|i| i.pointer.hover_pos())
                .map(|p| timeline_hover_rect.contains(p))
                .unwrap_or(false);

            let wave_x0_est = ui.cursor().min.x + HEADER_WIDTH + COL_GAP;
            if pointer_in_timeline {
                handle_zoom_and_wheel_scroll(
                    ui,
                    tl,
                    wave_x0_est,
                    wave_w,
                    state.position_frame,
                    total_frames,
                );
            }

            // Auto-follow dolce del playhead durante la riproduzione quando zoom > 1.0
            let pointer_down = ui.input(|i| i.pointer.primary_down());
            if state.playing && tl.zoom > 1.005_f32 && !pointer_down {
                let vis_frac = (1.0_f32 / tl.zoom).clamp(1.0_f32 / MAX_ZOOM, 1.0_f32);
                let max_start_frac = (1.0_f32 - vis_frac).max(1e-5_f32);
                let cur_start_frac = tl.scroll_norm * max_start_frac;
                let play_norm =
                    (state.position_frame as f32 / total_frames as f32).clamp(0.0_f32, 1.0_f32);

                if play_norm > cur_start_frac + 0.74_f32 * vis_frac {
                    let target_start =
                        (play_norm - 0.74_f32 * vis_frac).clamp(0.0_f32, max_start_frac);
                    let target_norm = target_start / max_start_frac;
                    tl.scroll_norm += (target_norm - tl.scroll_norm) * 0.22_f32;
                } else if play_norm < cur_start_frac {
                    let target_start =
                        (play_norm - 0.12_f32 * vis_frac).clamp(0.0_f32, max_start_frac);
                    let target_norm = target_start / max_start_frac;
                    tl.scroll_norm += (target_norm - tl.scroll_norm) * 0.22_f32;
                }
                tl.scroll_norm = tl.scroll_norm.clamp(0.0_f32, 1.0_f32);
            }

            if state.playing {
                ui.ctx().request_repaint();
            }

            // Calcolo finestra di campioni visibile [start_frame .. end_frame]
            let vis_frac = (1.0_f32 / tl.zoom).clamp(1.0_f32 / MAX_ZOOM, 1.0_f32);
            let vis_frames = ((total_frames as f32 * vis_frac).round() as usize)
                .clamp(64.min(total_frames), total_frames);
            let max_start = total_frames.saturating_sub(vis_frames);
            let start_frame = ((max_start as f32) * tl.scroll_norm).round() as usize;
            let end_frame = (start_frame + vis_frames).min(total_frames);

            let mut ruler_wave_rect = Rect::NOTHING;
            let mut last_wave_bottom = 0.0_f32;
            let mut major_tick_xs: Vec<f32> = Vec::new();

            ui.scope(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(COL_GAP, LANE_GAP);

                // 1. RIGHELLO TEMPORALE + Angolo Timecode & Pulsanti Zoom (- / 1.0x / +)
                ui.horizontal(|ui| {
                    draw_ruler_corner_box(
                        ui,
                        state.position_frame,
                        total_frames,
                        sample_rate,
                        tl,
                    );

                    let (r_rect, r_resp) = ui.allocate_exact_size(
                        Vec2::new(wave_w, RULER_HEIGHT),
                        Sense::click_and_drag(),
                    );
                    ruler_wave_rect = r_rect;
                    last_wave_bottom = r_rect.bottom();

                    handle_range_selection_interaction(
                        ui,
                        &r_resp,
                        r_rect,
                        start_frame,
                        vis_frames,
                        total_frames,
                        true,
                        &mut state.position_frame,
                        &mut state.selection_range,
                        &mut state.loop_enabled,
                        &mut tl.ruler_drag,
                        egui::CursorIcon::PointingHand,
                    );

                    let loop_active = state.loop_enabled;
                    major_tick_xs = draw_time_ruler(
                        ui.painter(),
                        r_rect,
                        start_frame,
                        vis_frames,
                        sample_rate,
                        state.selection_range,
                        loop_active,
                    );

                    if !r_resp.dragged() {
                        r_resp.on_hover_text(
                            "Righello Temporale\n\
                             • Clic singolo: sposta il cursore in quel punto\n\
                             • Shift + Clic: estende la selezione fino al punto cliccato\n\
                             • Trascina i bordi sinistro/destro: rifinisce al millimetro la selezione\n\
                             • Trascina sul righello: seleziona un intervallo e attiva il Loop (Tasto L)\n\
                             • Doppio clic: azzera il Loop",
                        );
                    }
                });

                // 2. CORSIE TRACCE ORIZZONTALI IMPILATE (4 Stems oppure Traccia Stereo Principale)
                if state.stems.is_some() {
                    let scroll_reserve = if tl.zoom > 1.005_f32 { 17.0_f32 } else { 0.0_f32 };
                    let avail_h = if ui.available_height().is_finite() {
                        (ui.available_height() - scroll_reserve).max(120.0_f32)
                    } else {
                        360.0_f32
                    };
                    let raw_lane_h = ((avail_h - 3.0_f32 * LANE_GAP) / 4.0_f32).floor();
                    let needs_vscroll = raw_lane_h < 66.0_f32;
                    let lane_h = if needs_vscroll {
                        68.0_f32
                    } else {
                        raw_lane_h.clamp(66.0_f32, 114.0_f32)
                    };

                    if needs_vscroll {
                        let viewport_bottom_limit = ruler_wave_rect.bottom() + LANE_GAP + avail_h;
                        ScrollArea::vertical()
                            .id_salt("stems_timeline_vscroll")
                            .max_height(avail_h)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                draw_four_stem_lanes(
                                    ui,
                                    state,
                                    tl,
                                    wave_w,
                                    lane_h,
                                    start_frame,
                                    vis_frames,
                                    total_frames,
                                    &major_tick_xs,
                                    &mut last_wave_bottom,
                                );
                            });
                        last_wave_bottom = last_wave_bottom.min(viewport_bottom_limit);
                    } else {
                        draw_four_stem_lanes(
                            ui,
                            state,
                            tl,
                            wave_w,
                            lane_h,
                            start_frame,
                            vis_frames,
                            total_frames,
                            &major_tick_xs,
                            &mut last_wave_bottom,
                        );
                    }
                } else {
                    let avail_single = if ui.available_height().is_finite() {
                        (ui.available_height() - 18.0_f32).clamp(116.0_f32, 168.0_f32)
                    } else {
                        148.0_f32
                    };
                    let lane_h = avail_single;
                    let warm_amber = Color32::from_rgb(245, 158, 11);

                    ui.horizontal(|ui| {
                        let is_ai = state.mode == PlaybackSourceMode::PreRenderedAiMaster
                            && state.ai_master_track.is_some();

                        draw_main_stereo_header(
                            ui,
                            lane_h,
                            warm_amber,
                            is_ai,
                            sample_rate,
                            state.peak_l,
                            state.peak_r,
                            &mut state.loop_enabled,
                            &mut state.params.bypass,
                            &mut state.params.comp_drive_db,
                            &mut state.params.stereo_width,
                            &mut state.params_dirty,
                        );

                        let active_buf = if is_ai {
                            state.ai_master_track.as_ref().or(state.main_track.as_ref())
                        } else {
                            state.main_track.as_ref().or(state.ai_master_track.as_ref())
                        };

                        if let Some(buf) = active_buf {
                            let w_rect = draw_waveform_lane(
                                ui,
                                wave_w,
                                lane_h,
                                buf,
                                warm_amber,
                                false,
                                1.0_f32,
                                start_frame,
                                vis_frames,
                                total_frames,
                                &major_tick_xs,
                                &mut state.position_frame,
                                &mut state.selection_range,
                                tl,
                            );
                            last_wave_bottom = w_rect.bottom();
                        }
                    });
                }

                // 3. PLAYHEAD VERTICALE UNIFICATO (attraversa righello e tutte le corsie)
                if ruler_wave_rect.width() > 1.0_f32
                    && last_wave_bottom > ruler_wave_rect.top()
                    && state.position_frame >= start_frame
                    && state.position_frame <= end_frame
                {
                    let norm_x = (state.position_frame.saturating_sub(start_frame)) as f32
                        / (vis_frames.max(1) as f32);
                    let px = (ruler_wave_rect.left() + norm_x * ruler_wave_rect.width())
                        .clamp(ruler_wave_rect.left(), ruler_wave_rect.right());
                    draw_playhead_overlay(
                        ui.painter(),
                        px,
                        ruler_wave_rect.top(),
                        last_wave_bottom,
                    );
                }

                // 4. BARRA DI SCORRIMENTO SOTTILE (visibile quando zoom > 1.0) + scorciatoie discrete
                if tl.zoom > 1.005_f32 {
                    ui.horizontal(|ui| {
                        let (hint_rect, hint_resp) = ui
                            .allocate_exact_size(Vec2::new(HEADER_WIDTH, 13.0_f32), Sense::hover());
                        let hint_txt = if state.selection_range.is_some() {
                            "2xClic onda: azzera Loop"
                        } else {
                            "Shift + Rotellina: scorri"
                        };
                        ui.painter().text(
                            Pos2::new(hint_rect.left() + 4.0_f32, hint_rect.center().y),
                            Align2::LEFT_CENTER,
                            hint_txt,
                            FontId::proportional(10.2_f32),
                            Color32::from_rgb(100, 116, 139),
                        );
                        hint_resp.on_hover_text(
                            "Scorri orizzontalmente il brano con la barra a destra o con Shift + Rotellina del mouse",
                        );

                        draw_thin_scrollbar(ui, wave_w, vis_frac, &mut tl.scroll_norm);
                    });
                }
            });
        });

    triggered_action
}

#[allow(clippy::too_many_arguments)]
fn draw_four_stem_lanes(
    ui: &mut Ui,
    state: &mut SharedAudioState,
    tl: &mut TimelineState,
    wave_w: f32,
    lane_h: f32,
    start_frame: usize,
    vis_frames: usize,
    total_frames: usize,
    major_tick_xs: &[f32],
    last_wave_bottom: &mut f32,
) {
    for idx in 0..4 {
        let accent = stem_accent_color(idx);
        let label = STEM_LABELS[idx];

        ui.horizontal(|ui| {
            let any_solo_before = state.stem_channels.iter().any(|c| c.solo);
            let silenced_before = if any_solo_before {
                !state.stem_channels[idx].solo
            } else {
                state.stem_channels[idx].muted
            };

            let touched = draw_stem_track_header(
                ui,
                idx,
                lane_h,
                label,
                accent,
                silenced_before,
                &mut state.stem_channels[idx],
            );
            if touched {
                state.mode = PlaybackSourceMode::FourStemMixer;
            }

            let any_solo = state.stem_channels.iter().any(|c| c.solo);
            let is_silenced = if any_solo {
                !state.stem_channels[idx].solo
            } else {
                state.stem_channels[idx].muted
            };
            let visual_gain = 10.0_f32.powf(state.stem_channels[idx].gain_db / 20.0_f32);

            if let Some(ref stems) = state.stems {
                let w_rect = draw_waveform_lane(
                    ui,
                    wave_w,
                    lane_h,
                    &stems[idx],
                    accent,
                    is_silenced,
                    visual_gain,
                    start_frame,
                    vis_frames,
                    total_frames,
                    major_tick_xs,
                    &mut state.position_frame,
                    &mut state.selection_range,
                    tl,
                );
                *last_wave_bottom = w_rect.bottom();
            }
        });
    }
}

fn active_timeline_metrics(state: &SharedAudioState) -> (usize, u32) {
    if let Some(ref stems) = state.stems {
        let len = stems.iter().map(|s| s.len()).max().unwrap_or(0);
        let sr = stems[0].sample_rate.max(1);
        (len, sr)
    } else if let Some(ref main) = state.main_track {
        (main.len(), main.sample_rate.max(1))
    } else if let Some(ref ai) = state.ai_master_track {
        (ai.len(), ai.sample_rate.max(1))
    } else {
        (0, state.device_sr.max(44100))
    }
}

fn handle_zoom_and_wheel_scroll(
    ui: &egui::Ui,
    tl: &mut TimelineState,
    wave_x0: f32,
    wave_w: f32,
    position_frame: usize,
    total_frames: usize,
) {
    let (zoom_delta, scroll_delta, shift_held, ctrl_held, hover_pos) = ui.input(|i| {
        (
            i.zoom_delta(),
            i.smooth_scroll_delta,
            i.modifiers.shift,
            i.modifiers.ctrl || i.modifiers.command,
            i.pointer.hover_pos(),
        )
    });

    let zoom_factor = if (zoom_delta - 1.0_f32).abs() > 0.001_f32 {
        Some(zoom_delta)
    } else if ctrl_held && scroll_delta.y.abs() > 0.1_f32 {
        Some((scroll_delta.y * 0.0065_f32).exp())
    } else {
        None
    };

    if let Some(factor) = zoom_factor {
        ui.ctx().input_mut(|i| {
            i.smooth_scroll_delta = Vec2::ZERO;
        });
        let old_zoom = tl.zoom;
        let new_zoom = (old_zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        if (new_zoom - old_zoom).abs() > 1e-4_f32 {
            let old_vis = 1.0_f32 / old_zoom;
            let new_vis = 1.0_f32 / new_zoom;
            let old_start = tl.scroll_norm * (1.0_f32 - old_vis);

            let anchor_t = hover_pos
                .filter(|p| p.x >= wave_x0 && p.x <= wave_x0 + wave_w)
                .map(|p| ((p.x - wave_x0) / wave_w.max(1.0_f32)).clamp(0.0_f32, 1.0_f32))
                .unwrap_or_else(|| {
                    let play_norm = position_frame as f32 / total_frames.max(1) as f32;
                    ((play_norm - old_start) / old_vis.max(1e-5_f32)).clamp(0.0_f32, 1.0_f32)
                });

            let anchor_song_norm = old_start + anchor_t * old_vis;
            let max_new_start = (1.0_f32 - new_vis).max(0.0_f32);
            let new_start = (anchor_song_norm - anchor_t * new_vis).clamp(0.0_f32, max_new_start);

            tl.zoom = new_zoom;
            tl.scroll_norm = if max_new_start > 1e-5_f32 {
                (new_start / max_new_start).clamp(0.0_f32, 1.0_f32)
            } else {
                0.0_f32
            };
        }
        return;
    }

    // Shift + rotellina oppure scorrimento orizzontale nativo del trackpad
    let pan_px = if shift_held {
        -scroll_delta.y
    } else {
        -scroll_delta.x
    };
    if pan_px.abs() > 0.1_f32 && tl.zoom > 1.005_f32 {
        ui.ctx().input_mut(|i| {
            if shift_held {
                i.smooth_scroll_delta.y = 0.0_f32;
            } else {
                i.smooth_scroll_delta.x = 0.0_f32;
            }
        });
        let vis_frac = 1.0_f32 / tl.zoom;
        let max_start_frac = (1.0_f32 - vis_frac).max(1e-4_f32);
        let d_norm = (pan_px / wave_w.max(1.0_f32)) * vis_frac / max_start_frac;
        tl.scroll_norm = (tl.scroll_norm + d_norm).clamp(0.0_f32, 1.0_f32);
    }
}

fn apply_centered_zoom(
    tl: &mut TimelineState,
    target_zoom: f32,
    position_frame: usize,
    total_frames: usize,
) {
    let new_zoom = target_zoom.clamp(MIN_ZOOM, MAX_ZOOM);
    if new_zoom <= 1.01_f32 {
        tl.zoom = 1.0_f32;
        tl.scroll_norm = 0.0_f32;
        return;
    }
    let new_vis = 1.0_f32 / new_zoom;
    let play_norm = (position_frame as f32 / total_frames.max(1) as f32).clamp(0.0_f32, 1.0_f32);
    let max_start = (1.0_f32 - new_vis).max(1e-5_f32);
    let new_start = (play_norm - 0.35_f32 * new_vis).clamp(0.0_f32, max_start);
    tl.zoom = new_zoom;
    tl.scroll_norm = (new_start / max_start).clamp(0.0_f32, 1.0_f32);
}

fn draw_ruler_corner_box(
    ui: &mut Ui,
    position_frame: usize,
    total_frames: usize,
    sample_rate: u32,
    tl: &mut TimelineState,
) {
    ui.allocate_ui(Vec2::new(HEADER_WIDTH, RULER_HEIGHT), |ui| {
        ui.set_min_size(Vec2::new(HEADER_WIDTH, RULER_HEIGHT));
        ui.set_max_size(Vec2::new(HEADER_WIDTH, RULER_HEIGHT));
        ui.spacing_mut().item_spacing = Vec2::new(3.0_f32, 0.0_f32);

        let rect = Rect::from_min_size(ui.max_rect().min, Vec2::new(HEADER_WIDTH, RULER_HEIGHT));
        if ui.is_rect_visible(rect) {
            ui.painter()
                .rect_filled(rect, CornerRadius::same(6), Color32::from_rgb(14, 19, 29));
            ui.painter().rect_stroke(
                rect,
                CornerRadius::same(6),
                Stroke::new(1.0_f32, Color32::from_rgb(30, 41, 59)),
                StrokeKind::Inside,
            );
        }

        let cur_s = position_frame as f32 / sample_rate.max(1) as f32;
        let tot_s = total_frames as f32 / sample_rate.max(1) as f32;
        let time_str = format!(
            "{:02}:{:04.1}/{:02}:{:02}",
            (cur_s as u32) / 60,
            cur_s % 60.0_f32,
            (tot_s as u32) / 60,
            (tot_s as u32) % 60,
        );

        let v_pad = ((RULER_HEIGHT - 18.0_f32) * 0.5_f32).max(1.0_f32);
        ui.add_space(v_pad);
        ui.horizontal(|ui| {
            ui.add_space(7.0_f32);
            let (tc_rect, tc_resp) =
                ui.allocate_exact_size(Vec2::new(108.0_f32, 18.0_f32), Sense::hover());
            ui.painter().text(
                Pos2::new(tc_rect.left(), tc_rect.center().y),
                Align2::LEFT_CENTER,
                time_str,
                FontId::monospace(10.3_f32),
                Color32::from_rgb(226, 232, 240),
            );
            tc_resp.on_hover_text("Posizione attuale / Durata totale del brano");

            let cyan = Color32::from_rgb(56, 189, 248);
            if compact_pill_toggle(ui, "-", false, cyan, Vec2::new(20.0_f32, 18.0_f32))
                .on_hover_text("Riduci Zoom forme d'onda (oppure Ctrl + Rotellina giù)")
                .clicked()
            {
                apply_centered_zoom(tl, tl.zoom / 1.5_f32, position_frame, total_frames);
            }

            let zoomed = tl.zoom > 1.05_f32;
            if compact_pill_toggle(
                ui,
                &format!("{:.1}x", tl.zoom),
                zoomed,
                cyan,
                Vec2::new(40.0_f32, 18.0_f32),
            )
            .on_hover_text("Livello di Zoom attuale  •  Clicca per vedere l'intero brano (1.0x)")
            .clicked()
            {
                tl.zoom = 1.0_f32;
                tl.scroll_norm = 0.0_f32;
            }

            if compact_pill_toggle(ui, "+", false, cyan, Vec2::new(20.0_f32, 18.0_f32))
                .on_hover_text("Ingrandisci forme d'onda (oppure Ctrl + Rotellina su)")
                .clicked()
            {
                apply_centered_zoom(tl, tl.zoom * 1.5_f32, position_frame, total_frames);
            }
        });
    });
}

fn draw_time_ruler(
    painter: &egui::Painter,
    rect: Rect,
    start_frame: usize,
    vis_frames: usize,
    sample_rate: u32,
    selection_range: Option<(usize, usize)>,
    loop_enabled: bool,
) -> Vec<f32> {
    let mut major_xs = Vec::new();
    let radius = CornerRadius::same(6);

    // Sfondo scuro elegante #0E131D
    painter.rect_filled(rect, radius, Color32::from_rgb(14, 19, 29));
    painter.rect_stroke(
        rect,
        radius,
        Stroke::new(1.0_f32, Color32::from_rgb(30, 41, 59)),
        StrokeKind::Inside,
    );

    // Fascia luminosa traslucida sulla regione selezionata (ambra se Loop attivo, ciano se selezione)
    if let Some((a, b)) = selection_range {
        let s0 = a.min(b);
        let s1 = a.max(b);
        let end_frame = start_frame + vis_frames;
        if s1 > s0 && s1 > start_frame && s0 < end_frame {
            let sx0 = frame_to_x(s0.max(start_frame), rect, start_frame, vis_frames);
            let sx1 = frame_to_x(s1.min(end_frame), rect, start_frame, vis_frames);
            if sx1 > sx0 {
                let (fill_col, bar_col, edge_col) = if loop_enabled {
                    (
                        Color32::from_rgba_unmultiplied(245, 158, 11, 58),
                        Color32::from_rgb(245, 158, 11),
                        Color32::from_rgb(251, 191, 36),
                    )
                } else {
                    (
                        Color32::from_rgba_unmultiplied(56, 189, 248, 56),
                        Color32::from_rgb(56, 189, 248),
                        Color32::from_rgb(125, 211, 252),
                    )
                };
                let sel_rect = Rect::from_min_max(
                    Pos2::new(sx0, rect.top() + 1.0_f32),
                    Pos2::new(sx1, rect.bottom() - 1.0_f32),
                );
                painter.rect_filled(sel_rect, CornerRadius::same(2), fill_col);
                let top_strip = Rect::from_min_max(
                    Pos2::new(sx0, rect.top() + 1.0_f32),
                    Pos2::new(sx1, rect.top() + 3.5_f32),
                );
                painter.rect_filled(top_strip, CornerRadius::ZERO, bar_col);
                painter.line_segment(
                    [Pos2::new(sx0, rect.top()), Pos2::new(sx0, rect.bottom())],
                    Stroke::new(1.5_f32, edge_col),
                );
                painter.line_segment(
                    [Pos2::new(sx1, rect.top()), Pos2::new(sx1, rect.bottom())],
                    Stroke::new(1.5_f32, edge_col),
                );
                // Maniglie laterali visibili sul righello per trascinare inizio/fine selezione
                let grip_l = Rect::from_min_max(
                    Pos2::new(sx0, rect.top() + 2.0_f32),
                    Pos2::new((sx0 + 4.5_f32).min(sx1), rect.top() + 12.0_f32),
                );
                let grip_r = Rect::from_min_max(
                    Pos2::new((sx1 - 4.5_f32).max(sx0), rect.top() + 2.0_f32),
                    Pos2::new(sx1, rect.top() + 12.0_f32),
                );
                painter.rect_filled(grip_l, CornerRadius::same(2), edge_col);
                painter.rect_filled(grip_r, CornerRadius::same(2), edge_col);
                if loop_enabled && (sx1 - sx0) > 46.0_f32 {
                    painter.text(
                        Pos2::new((sx0 + sx1) * 0.5_f32, rect.bottom() - 6.5_f32),
                        Align2::CENTER_CENTER,
                        "LOOP",
                        FontId::proportional(9.2_f32),
                        edge_col,
                    );
                }
            }
        }
    }

    let sr = sample_rate.max(1) as f32;
    let start_sec = start_frame as f32 / sr;
    let vis_sec = (vis_frames.max(1) as f32) / sr;
    let end_sec = start_sec + vis_sec;

    let target_step = vis_sec * (105.0_f32 / rect.width().max(120.0_f32));
    const STEPS: [f32; 13] = [
        0.1_f32, 0.25_f32, 0.5_f32, 1.0_f32, 2.0_f32, 5.0_f32, 10.0_f32, 15.0_f32, 30.0_f32,
        60.0_f32, 120.0_f32, 300.0_f32, 600.0_f32,
    ];
    let major_step = STEPS
        .iter()
        .copied()
        .find(|&s| s >= target_step)
        .unwrap_or(600.0_f32);

    let sub_divs: u32 = if (major_step - 15.0_f32).abs() < 0.01_f32
        || (major_step - 30.0_f32).abs() < 0.01_f32
    {
        3
    } else if (major_step - 2.0_f32).abs() < 0.01_f32 || (major_step - 0.25_f32).abs() < 0.01_f32 {
        4
    } else {
        5
    };
    let minor_step = major_step / (sub_divs as f32);

    let first_idx = (start_sec / minor_step).floor() as i64;
    let last_idx = (end_sec / minor_step).ceil() as i64;

    for k in first_idx..=last_idx {
        if k < 0 {
            continue;
        }
        let t = (k as f32) * minor_step;
        if t < start_sec - 1e-4_f32 || t > end_sec + 1e-4_f32 {
            continue;
        }
        let x = rect.left() + ((t - start_sec) / vis_sec) * rect.width();
        if x < rect.left() + 1.0_f32 || x > rect.right() - 1.0_f32 {
            continue;
        }

        let is_major = (k as u64) % (sub_divs as u64) == 0;
        if is_major {
            major_xs.push(x);
            painter.line_segment(
                [Pos2::new(x, rect.bottom() - 8.5_f32), Pos2::new(x, rect.bottom())],
                Stroke::new(1.2_f32, Color32::from_rgb(100, 116, 139)),
            );

            let label = format_ruler_time(t, major_step < 1.0_f32);
            if x + 36.0_f32 <= rect.right() {
                painter.text(
                    Pos2::new(x + 4.0_f32, rect.top() + 10.0_f32),
                    Align2::LEFT_CENTER,
                    label,
                    FontId::proportional(10.8_f32),
                    Color32::from_rgb(186, 198, 216),
                );
            }
        } else {
            painter.line_segment(
                [Pos2::new(x, rect.bottom() - 4.0_f32), Pos2::new(x, rect.bottom())],
                Stroke::new(1.0_f32, Color32::from_rgb(51, 65, 85)),
            );
        }
    }

    major_xs
}

fn format_ruler_time(sec: f32, sub_second: bool) -> String {
    let total_tenths = (sec * 10.0_f32).round() as u32;
    let mins = total_tenths / 600;
    let secs = (total_tenths / 10) % 60;
    let tenths = total_tenths % 10;
    if sub_second {
        format!("{mins}:{secs:02}.{tenths}")
    } else {
        format!("{mins}:{secs:02}")
    }
}

/// Intestazione sinistra di uno Stem (`215px` larghezza, card `#151C2B`, altezza adattiva senza clipping).
/// Restituisce `true` se l'utente ha modificato Mute, Solo, Volume o Pan.
fn draw_stem_track_header(
    ui: &mut Ui,
    idx: usize,
    lane_h: f32,
    title: &str,
    accent: Color32,
    is_silenced: bool,
    ch: &mut StemChannel,
) -> bool {
    let mut touched = false;

    ui.allocate_ui(Vec2::new(HEADER_WIDTH, lane_h), |ui| {
        ui.set_min_size(Vec2::new(HEADER_WIDTH, lane_h));
        ui.set_max_size(Vec2::new(HEADER_WIDTH, lane_h));
        ui.spacing_mut().item_spacing = Vec2::new(4.0_f32, 0.0_f32);

        let card_rect = Rect::from_min_size(ui.max_rect().min, Vec2::new(HEADER_WIDTH, lane_h));
        let radius = CornerRadius::same(8);

        // Sfondo card #151C2B + barra verticale colorata a sinistra (4px)
        ui.painter()
            .rect_filled(card_rect, radius, Color32::from_rgb(21, 28, 43));
        ui.painter().rect_stroke(
            card_rect,
            radius,
            Stroke::new(1.0_f32, Color32::from_rgb(35, 46, 68)),
            StrokeKind::Inside,
        );

        let bar_col = if is_silenced {
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 75)
        } else {
            accent
        };
        let left_bar = Rect::from_min_max(
            card_rect.min,
            Pos2::new(card_rect.left() + 4.0_f32, card_rect.bottom()),
        );
        ui.painter().rect_filled(
            left_bar,
            CornerRadius {
                nw: 8,
                sw: 8,
                ne: 0,
                se: 0,
            },
            bar_col,
        );

        // Mini LED Meter verticale live sul bordo destro della card
        let meter_pad_y = if lane_h < 76.0_f32 { 6.5_f32 } else { 8.5_f32 };
        let meter_rect = Rect::from_min_max(
            Pos2::new(card_rect.right() - 13.0_f32, card_rect.top() + meter_pad_y),
            Pos2::new(card_rect.right() - 6.0_f32, card_rect.bottom() - meter_pad_y),
        );
        let shown_peak = if is_silenced { 0.0_f32 } else { ch.live_peak };
        draw_vertical_led_meter(ui.painter(), meter_rect, shown_peak);

        // Centratura verticale adattiva (nasconde il sottotitolo secondario sotto 84px per evitare qualsiasi taglio)
        let show_subtitle = lane_h >= 84.0_f32;
        let (row_gap, top_pad) = if show_subtitle {
            let gap = ((lane_h - 73.0_f32) / 3.0_f32).clamp(1.5_f32, 4.5_f32);
            let used = 18.0_f32 + 11.0_f32 + 20.0_f32 + 16.0_f32 + 3.0_f32 * gap;
            (gap, ((lane_h - used) * 0.5_f32).max(3.0_f32))
        } else {
            let gap = ((lane_h - 60.0_f32) / 2.0_f32).clamp(1.5_f32, 4.0_f32);
            let used = 18.0_f32 + 20.0_f32 + 16.0_f32 + 2.0_f32 * gap;
            (gap, ((lane_h - used) * 0.5_f32).max(2.5_f32))
        };

        let inner_w = HEADER_WIDTH - 11.0_f32 - 18.0_f32;
        let track_tip = STEM_TOOLTIPS_IT[idx.min(3)];

        ui.add_space(top_pad);
        ui.horizontal(|ui| {
            ui.add_space(10.0_f32);
            ui.vertical(|ui| {
                ui.set_max_width(inner_w);
                ui.spacing_mut().item_spacing = Vec2::new(4.0_f32, row_gap);

                // Riga 1: Nome traccia + pillole compatte M e S
                ui.horizontal(|ui| {
                    let title_w = inner_w - 58.0_f32;
                    let (t_rect, t_resp) =
                        ui.allocate_exact_size(Vec2::new(title_w, 18.0_f32), Sense::hover());
                    let title_col = if is_silenced {
                        Color32::from_rgb(132, 146, 170)
                    } else {
                        Color32::from_rgb(241, 245, 249)
                    };
                    ui.painter().text(
                        Pos2::new(t_rect.left(), t_rect.center().y),
                        Align2::LEFT_CENTER,
                        title,
                        FontId::proportional(11.6_f32),
                        title_col,
                    );
                    t_resp.on_hover_text(track_tip);

                    if compact_pill_toggle(
                        ui,
                        "M",
                        ch.muted,
                        Color32::from_rgb(239, 68, 68),
                        Vec2::new(25.0_f32, 18.0_f32),
                    )
                    .on_hover_text("Muto: silenzia questo strumento nel mix")
                    .clicked()
                    {
                        ch.muted = !ch.muted;
                        if ch.muted {
                            ch.solo = false;
                        }
                        touched = true;
                    }

                    if compact_pill_toggle(
                        ui,
                        "S",
                        ch.solo,
                        Color32::from_rgb(245, 158, 11),
                        Vec2::new(25.0_f32, 18.0_f32),
                    )
                    .on_hover_text(format!(
                        "Solo: ascolta unicamente {} isolato (Tasto rapido {})",
                        title,
                        idx + 1
                    ))
                    .clicked()
                    {
                        ch.solo = !ch.solo;
                        if ch.solo {
                            ch.muted = false;
                        }
                        touched = true;
                    }
                });

                // Sottotitolo in italiano dello stem (visibile quando l'altezza lo consente)
                if show_subtitle {
                    let (sub_rect, sub_resp) =
                        ui.allocate_exact_size(Vec2::new(inner_w, 11.0_f32), Sense::hover());
                    ui.painter().text(
                        Pos2::new(sub_rect.left(), sub_rect.center().y),
                        Align2::LEFT_CENTER,
                        STEM_SUBTITLES_IT[idx.min(3)],
                        FontId::proportional(9.8_f32),
                        Color32::from_rgb(104, 119, 145),
                    );
                    sub_resp.on_hover_text(track_tip);
                }

                // Riga 2: Fader Volume (-24 .. +8 dB) + etichetta dB (doppio clic -> 0.0 dB)
                let vol_row = ui.horizontal(|ui| {
                    let fader_col = if is_silenced {
                        Color32::from_rgb(100, 116, 139)
                    } else {
                        accent
                    };
                    let fader_min = ui.cursor().min;
                    if sleek_progress_or_fader(
                        ui,
                        &mut ch.gain_db,
                        -24.0_f32,
                        8.0_f32,
                        fader_col,
                        120.0_f32,
                        20.0_f32,
                    ) {
                        touched = true;
                    }
                    let fader_rect = Rect::from_min_size(fader_min, Vec2::new(120.0_f32, 20.0_f32));
                    if ui.rect_contains_pointer(fader_rect)
                        && ui.input(|i| i.pointer.button_double_clicked(PointerButton::Primary))
                    {
                        ch.gain_db = 0.0_f32;
                        touched = true;
                    }

                    let (db_rect, db_resp) =
                        ui.allocate_exact_size(Vec2::new(58.0_f32, 20.0_f32), Sense::click());
                    if db_resp.double_clicked() {
                        ch.gain_db = 0.0_f32;
                        touched = true;
                    }
                    ui.painter().text(
                        Pos2::new(db_rect.right(), db_rect.center().y),
                        Align2::RIGHT_CENTER,
                        format!("{:+.1} dB", ch.gain_db),
                        FontId::monospace(10.5_f32),
                        Color32::from_rgb(191, 203, 222),
                    );
                    db_resp.on_hover_text("Doppio clic per riportare il volume a 0.0 dB");
                });
                if !ui.ctx().is_using_pointer() {
                    vol_row.response.on_hover_text(
                        "Volume Strumento (-24 .. +8 dB)\n\
                         • Trascina per alzare o abbassare il volume\n\
                         • Doppio clic per riportare esattamente a 0.0 dB",
                    );
                }

                // Riga 3: Mini controllo Pan bipolare (L .. R) + etichetta Pan
                ui.horizontal(|ui| {
                    let pan_col = if is_silenced {
                        Color32::from_rgb(100, 116, 139)
                    } else {
                        accent
                    };
                    if sleek_bipolar_pan(
                        ui,
                        &mut ch.pan,
                        pan_col,
                        120.0_f32,
                        16.0_f32,
                        "Panoramica Stereo Sinistra / Destra (L / R)\n\
                         • Trascina per posizionare lo strumento nello spazio stereo\n\
                         • Doppio clic per riportare al Centro (PAN C)",
                    ) {
                        touched = true;
                    }

                    let pan_txt = format_pan_label(ch.pan);
                    let (p_rect, p_resp) =
                        ui.allocate_exact_size(Vec2::new(58.0_f32, 16.0_f32), Sense::click());
                    if p_resp.double_clicked() {
                        ch.pan = 0.0_f32;
                        touched = true;
                    }
                    ui.painter().text(
                        Pos2::new(p_rect.right(), p_rect.center().y),
                        Align2::RIGHT_CENTER,
                        pan_txt,
                        FontId::monospace(10.0_f32),
                        Color32::from_rgb(148, 163, 184),
                    );
                    p_resp.on_hover_text("Doppio clic per centrare il Pan (PAN C)");
                });
            });
        });
    });

    touched
}

/// Intestazione sinistra per la Traccia Stereo Principale (`state.stems.is_none()`).
#[allow(clippy::too_many_arguments)]
fn draw_main_stereo_header(
    ui: &mut Ui,
    lane_h: f32,
    accent: Color32,
    is_ai_master: bool,
    sample_rate: u32,
    peak_l: f32,
    peak_r: f32,
    loop_enabled: &mut bool,
    bypass_dsp: &mut bool,
    comp_drive_db: &mut f64,
    stereo_width: &mut f64,
    params_dirty: &mut bool,
) {
    ui.allocate_ui(Vec2::new(HEADER_WIDTH, lane_h), |ui| {
        ui.set_min_size(Vec2::new(HEADER_WIDTH, lane_h));
        ui.set_max_size(Vec2::new(HEADER_WIDTH, lane_h));
        ui.spacing_mut().item_spacing = Vec2::new(4.0_f32, 0.0_f32);

        let card_rect = Rect::from_min_size(ui.max_rect().min, Vec2::new(HEADER_WIDTH, lane_h));
        let radius = CornerRadius::same(8);

        ui.painter()
            .rect_filled(card_rect, radius, Color32::from_rgb(21, 28, 43));
        ui.painter().rect_stroke(
            card_rect,
            radius,
            Stroke::new(1.0_f32, Color32::from_rgb(35, 46, 68)),
            StrokeKind::Inside,
        );

        let left_bar = Rect::from_min_max(
            card_rect.min,
            Pos2::new(card_rect.left() + 4.0_f32, card_rect.bottom()),
        );
        ui.painter().rect_filled(
            left_bar,
            CornerRadius {
                nw: 8,
                sw: 8,
                ne: 0,
                se: 0,
            },
            accent,
        );

        // Doppio meter LED Stereo L / R sul bordo destro
        let meter_l = Rect::from_min_max(
            Pos2::new(card_rect.right() - 18.0_f32, card_rect.top() + 10.0_f32),
            Pos2::new(card_rect.right() - 12.5_f32, card_rect.bottom() - 10.0_f32),
        );
        let meter_r = Rect::from_min_max(
            Pos2::new(card_rect.right() - 10.5_f32, card_rect.top() + 10.0_f32),
            Pos2::new(card_rect.right() - 5.0_f32, card_rect.bottom() - 10.0_f32),
        );
        draw_vertical_led_meter(ui.painter(), meter_l, peak_l);
        draw_vertical_led_meter(ui.painter(), meter_r, peak_r);

        let inner_w = HEADER_WIDTH - 12.0_f32 - 24.0_f32;
        let row_gap = ((lane_h - 86.0_f32) / 4.0_f32).clamp(3.0_f32, 7.0_f32);
        let used_h = 20.0_f32 + 15.0_f32 + 20.0_f32 + 18.0_f32 + 3.0_f32 * row_gap;
        let top_pad = ((lane_h - used_h) * 0.5_f32).max(6.0_f32);

        ui.add_space(top_pad);
        ui.horizontal(|ui| {
            ui.add_space(11.0_f32);
            ui.vertical(|ui| {
                ui.set_max_width(inner_w);
                ui.spacing_mut().item_spacing = Vec2::new(4.0_f32, row_gap);

                // Riga 1: Titolo traccia + pulsanti DSP / LOOP
                ui.horizontal(|ui| {
                    let (t_rect, t_resp) = ui.allocate_exact_size(
                        Vec2::new(inner_w - 72.0_f32, 20.0_f32),
                        Sense::hover(),
                    );
                    ui.painter().text(
                        Pos2::new(t_rect.left(), t_rect.center().y),
                        Align2::LEFT_CENTER,
                        "TRACCIA AUDIO",
                        FontId::proportional(12.2_f32),
                        Color32::from_rgb(241, 245, 249),
                    );
                    t_resp.on_hover_text(
                        "Traccia Stereo Principale\nPremi 'Separa AI' in alto per dividerla in Voce, Batteria, Basso e Chitarre",
                    );

                    if compact_pill_toggle(
                        ui,
                        "FX",
                        !*bypass_dsp,
                        Color32::from_rgb(34, 197, 94),
                        Vec2::new(31.0_f32, 19.0_f32),
                    )
                    .on_hover_text("Attiva / Disattiva Equalizzatore e Mastering in tempo reale (Tasto B)")
                    .clicked()
                    {
                        *bypass_dsp = !*bypass_dsp;
                        *params_dirty = true;
                    }

                    if compact_pill_toggle(
                        ui,
                        "LP",
                        *loop_enabled,
                        Color32::from_rgb(56, 189, 248),
                        Vec2::new(31.0_f32, 19.0_f32),
                    )
                    .on_hover_text("Attiva / Disattiva ripetizione a ciclo Loop (Tasto L)")
                    .clicked()
                    {
                        *loop_enabled = !*loop_enabled;
                    }
                });

                // Sottotitolo informativo stato catena
                let sub = if is_ai_master {
                    format!("Master AI Stereo • {:.1} kHz", sample_rate as f32 / 1000.0_f32)
                } else if *bypass_dsp {
                    format!("Originale Puro • {:.1} kHz", sample_rate as f32 / 1000.0_f32)
                } else {
                    format!("Stereo + Master EQ • {:.1} kHz", sample_rate as f32 / 1000.0_f32)
                };
                let (s_rect, _) =
                    ui.allocate_exact_size(Vec2::new(inner_w, 15.0_f32), Sense::hover());
                ui.painter().text(
                    Pos2::new(s_rect.left(), s_rect.center().y),
                    Align2::LEFT_CENTER,
                    sub,
                    FontId::proportional(10.8_f32),
                    Color32::from_rgb(120, 136, 162),
                );

                // Drive / Gain Master (-6 .. +12 dB)
                let drive_row = ui.horizontal(|ui| {
                    let mut drive_f32 = *comp_drive_db as f32;
                    let fader_min = ui.cursor().min;
                    if sleek_progress_or_fader(
                        ui,
                        &mut drive_f32,
                        -6.0_f32,
                        12.0_f32,
                        accent,
                        114.0_f32,
                        20.0_f32,
                    ) {
                        *comp_drive_db = drive_f32 as f64;
                        *params_dirty = true;
                    }
                    let fader_rect = Rect::from_min_size(fader_min, Vec2::new(114.0_f32, 20.0_f32));
                    if ui.rect_contains_pointer(fader_rect)
                        && ui.input(|i| i.pointer.button_double_clicked(PointerButton::Primary))
                    {
                        *comp_drive_db = 3.8;
                        *params_dirty = true;
                    }
                    let (d_rect, d_resp) =
                        ui.allocate_exact_size(Vec2::new(58.0_f32, 20.0_f32), Sense::click());
                    if d_resp.double_clicked() {
                        *comp_drive_db = 3.8;
                        *params_dirty = true;
                    }
                    ui.painter().text(
                        Pos2::new(d_rect.right(), d_rect.center().y),
                        Align2::RIGHT_CENTER,
                        format!("{:+.1} dB", *comp_drive_db),
                        FontId::monospace(10.5_f32),
                        Color32::from_rgb(191, 203, 222),
                    );
                    d_resp.on_hover_text("Doppio clic per ripristinare la spinta consigliata (+3.8 dB)");
                });
                if !ui.ctx().is_using_pointer() {
                    drive_row.response.on_hover_text(
                        "Spinta Compressore Master (-6 .. +12 dB)\n• Doppio clic per ripristinare +3.8 dB",
                    );
                }

                // Ampiezza Stereo (Width 0% .. 200%)
                ui.horizontal(|ui| {
                    let mut width_norm = (*stereo_width as f32 - 1.0_f32).clamp(-1.0_f32, 1.0_f32);
                    if sleek_bipolar_pan(
                        ui,
                        &mut width_norm,
                        accent,
                        114.0_f32,
                        18.0_f32,
                        "Ampiezza Immagine Stereo (0% Mono .. 200% Ampio)\n• Doppio clic per ripristinare 122%",
                    ) {
                        *stereo_width = (1.0_f32 + width_norm).clamp(0.0_f32, 2.0_f32) as f64;
                        *params_dirty = true;
                    }
                    let (w_rect, w_resp) =
                        ui.allocate_exact_size(Vec2::new(58.0_f32, 18.0_f32), Sense::click());
                    if w_resp.double_clicked() {
                        *stereo_width = 1.22;
                        *params_dirty = true;
                    }
                    ui.painter().text(
                        Pos2::new(w_rect.right(), w_rect.center().y),
                        Align2::RIGHT_CENTER,
                        format!("W {:>3.0}%", *stereo_width * 100.0),
                        FontId::monospace(10.0_f32),
                        Color32::from_rgb(148, 163, 184),
                    );
                    w_resp.on_hover_text("Doppio clic per ripristinare ampiezza stereo 122%");
                });
            });
        });
    });
}

/// Disegna una corsia Forma d'Onda ad Alta Definizione (Peak 48% alpha + RMS 92% alpha)
/// e gestisce clic singolo (seek), trascinamento orizzontale (selezione regione) e doppio clic (reset selezione).
#[allow(clippy::too_many_arguments)]
fn draw_waveform_lane(
    ui: &mut Ui,
    wave_w: f32,
    lane_h: f32,
    buffer: &StereoBuffer,
    accent: Color32,
    is_silenced: bool,
    visual_gain: f32,
    start_frame: usize,
    vis_frames: usize,
    total_frames: usize,
    major_tick_xs: &[f32],
    position_frame: &mut usize,
    selection_range: &mut Option<(usize, usize)>,
    tl: &mut TimelineState,
) -> Rect {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(wave_w, lane_h), Sense::click_and_drag());

    // Interazioni mouse sulla corsia: clic seek, Shift+Clic estende, maniglie sinistro/destro regolabili, drag selezione
    let mut dummy_loop_flag = false;
    let had_selection_before = selection_range.is_some();
    handle_range_selection_interaction(
        ui,
        &resp,
        rect,
        start_frame,
        vis_frames,
        total_frames,
        false,
        position_frame,
        selection_range,
        &mut dummy_loop_flag,
        &mut tl.lane_drag,
        egui::CursorIcon::Crosshair,
    );
    if had_selection_before && selection_range.is_none() {
        tl.drag_select_start = None;
    }

    if !ui.is_rect_visible(rect) {
        return rect;
    }

    let painter = ui.painter();
    let radius = CornerRadius::same(8);

    // Sfondo corsia scuro con leggera reazione hover
    let bg_col = if resp.hovered() {
        Color32::from_rgb(15, 21, 33)
    } else {
        Color32::from_rgb(13, 18, 28)
    };
    painter.rect_filled(rect, radius, bg_col);

    // Griglia verticale discreta sincronizzata con le tacche principali del righello
    for &tx in major_tick_xs {
        if tx > rect.left() + 2.0_f32 && tx < rect.right() - 2.0_f32 {
            painter.line_segment(
                [Pos2::new(tx, rect.top() + 1.0_f32), Pos2::new(tx, rect.bottom() - 1.0_f32)],
                Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(148, 163, 184, 15)),
            );
        }
    }

    // Linea centrale 0.0
    let center_y = rect.center().y;
    painter.line_segment(
        [
            Pos2::new(rect.left() + 1.0_f32, center_y),
            Pos2::new(rect.right() - 1.0_f32, center_y),
        ],
        Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(148, 163, 184, 34)),
    );

    // Guadagno visivo percettivo automatico per tracce dal livello contenuto (es. Basso separato)
    let raw_peak = buffer
        .mip_coarse
        .max_val
        .iter()
        .zip(buffer.mip_coarse.min_val.iter())
        .fold(0.0_f32, |acc, (&mx, &mn)| acc.max(mx.abs()).max(mn.abs()));
    let norm_boost = if raw_peak > 0.003_f32 && raw_peak < 0.55_f32 {
        (0.62_f32 / raw_peak).clamp(1.0_f32, 8.0_f32)
    } else {
        1.0_f32
    };

    // Colori forma d'onda: attiva (Peak 48% + RMS brillante 92%) oppure silenziata (grigio-blu 25%)
    let (peak_col, rms_col, eff_gain) = if is_silenced {
        (
            Color32::from_rgba_unmultiplied(100, 116, 139, 36),
            Color32::from_rgba_unmultiplied(118, 136, 162, 64), // 25% alpha
            (visual_gain * norm_boost).clamp(0.35_f32, 6.0_f32),
        )
    } else {
        let rms_bright = brighten(accent, 26);
        (
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 122), // 48% alpha
            Color32::from_rgba_unmultiplied(rms_bright.r(), rms_bright.g(), rms_bright.b(), 235), // 92% alpha
            (visual_gain * norm_boost).clamp(0.05_f32, 10.0_f32),
        )
    };

    // Disegno colonne pixel ad alta definizione tramite mipmap Peak + RMS
    let n_cols = (rect.width().floor() as usize).max(1);
    let half_h = (rect.height() * 0.44_f32).max(8.0_f32);
    let buf_len = buffer.len().max(1);

    for col in 0..n_cols {
        let f0 = (start_frame + ((col as u64 * vis_frames as u64) / n_cols as u64) as usize)
            .min(buf_len - 1);
        let f1 = (start_frame + (((col + 1) as u64 * vis_frames as u64) / n_cols as u64) as usize)
            .max(f0 + 1)
            .min(buf_len);

        let (raw_min, raw_max, raw_rms) = buffer.sample_envelope(f0, f1);
        let mn = (raw_min * eff_gain).clamp(-1.0_f32, 1.0_f32);
        let mx = (raw_max * eff_gain).clamp(-1.0_f32, 1.0_f32);
        let rms = (raw_rms * eff_gain).clamp(0.0_f32, 1.0_f32);

        let x = rect.left() + (col as f32) + 0.5_f32;
        let y_top = center_y - mx * half_h;
        let y_bot = center_y - mn * half_h;

        if (y_bot - y_top) > 0.55_f32 {
            painter.line_segment(
                [Pos2::new(x, y_top), Pos2::new(x, y_bot)],
                Stroke::new(1.15_f32, peak_col),
            );
        }

        let y_rms_top = center_y - rms * half_h;
        let y_rms_bot = center_y + rms * half_h;
        if (y_rms_bot - y_rms_top) > 0.5_f32 {
            painter.line_segment(
                [Pos2::new(x, y_rms_top), Pos2::new(x, y_rms_bot)],
                Stroke::new(1.15_f32, rms_col),
            );
        }
    }

    // Evidenziazione regione selezionata sulla corsia con maniglie laterali stile DAW
    if let Some((a, b)) = *selection_range {
        let s0 = a.min(b);
        let s1 = a.max(b);
        let end_frame = start_frame + vis_frames;
        if s1 > s0 && s1 > start_frame && s0 < end_frame {
            let sx0 = frame_to_x(s0.max(start_frame), rect, start_frame, vis_frames);
            let sx1 = frame_to_x(s1.min(end_frame), rect, start_frame, vis_frames);
            if sx1 > sx0 {
                let sel_rect = Rect::from_min_max(
                    Pos2::new(sx0, rect.top() + 1.0_f32),
                    Pos2::new(sx1, rect.bottom() - 1.0_f32),
                );
                painter.rect_filled(
                    sel_rect,
                    CornerRadius::ZERO,
                    Color32::from_rgba_unmultiplied(56, 189, 248, 26),
                );
                let edge_col = Color32::from_rgba_unmultiplied(125, 211, 252, 200);
                let cap_w = 5.5_f32.min((sx1 - sx0) * 0.45_f32);
                if s0 >= start_frame {
                    painter.line_segment(
                        [Pos2::new(sx0, rect.top() + 1.0_f32), Pos2::new(sx0, rect.bottom() - 1.0_f32)],
                        Stroke::new(1.5_f32, edge_col),
                    );
                    painter.line_segment(
                        [Pos2::new(sx0, rect.top() + 2.0_f32), Pos2::new(sx0 + cap_w, rect.top() + 2.0_f32)],
                        Stroke::new(2.0_f32, edge_col),
                    );
                    painter.line_segment(
                        [Pos2::new(sx0, rect.bottom() - 2.0_f32), Pos2::new(sx0 + cap_w, rect.bottom() - 2.0_f32)],
                        Stroke::new(2.0_f32, edge_col),
                    );
                }
                if s1 <= end_frame {
                    painter.line_segment(
                        [Pos2::new(sx1, rect.top() + 1.0_f32), Pos2::new(sx1, rect.bottom() - 1.0_f32)],
                        Stroke::new(1.5_f32, edge_col),
                    );
                    painter.line_segment(
                        [Pos2::new(sx1 - cap_w, rect.top() + 2.0_f32), Pos2::new(sx1, rect.top() + 2.0_f32)],
                        Stroke::new(2.0_f32, edge_col),
                    );
                    painter.line_segment(
                        [Pos2::new(sx1 - cap_w, rect.bottom() - 2.0_f32), Pos2::new(sx1, rect.bottom() - 2.0_f32)],
                        Stroke::new(2.0_f32, edge_col),
                    );
                }
            }
        }
    }

    // Bordo esterno corsia
    let border_col = if is_silenced {
        Color32::from_rgb(24, 31, 45)
    } else if resp.hovered() {
        Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 95)
    } else {
        Color32::from_rgb(28, 38, 56)
    };
    painter.rect_stroke(
        rect,
        radius,
        Stroke::new(1.0_f32, border_col),
        StrokeKind::Inside,
    );

    if !resp.dragged() && !ui.ctx().is_using_pointer() {
        resp.on_hover_text(
            "Forma d'Onda\n\
             • Clic singolo: salta subito a questo punto del brano\n\
             • Trascina in orizzontale: seleziona un intervallo per il Loop (Tasto L)\n\
             • Doppio clic: rimuovi la selezione\n\
             • Ctrl + Rotellina: Zoom avanti / indietro",
        );
    }

    rect
}

fn draw_playhead_overlay(painter: &egui::Painter, x: f32, top_y: f32, bottom_y: f32) {
    // Alone morbido ciano lungo l'intera altezza della timeline
    painter.line_segment(
        [Pos2::new(x, top_y), Pos2::new(x, bottom_y)],
        Stroke::new(5.5_f32, Color32::from_rgba_unmultiplied(56, 189, 248, 42)),
    );

    // Linea verticale nitida bianca/ciano 2.0_f32
    painter.line_segment(
        [Pos2::new(x, top_y + 7.5_f32), Pos2::new(x, bottom_y)],
        Stroke::new(2.0_f32, Color32::from_rgb(232, 250, 255)),
    );

    // Triangolino superiore sul righello temporale
    let hw = 6.0_f32;
    let hh = 10.5_f32;
    let pts = vec![
        Pos2::new(x - hw, top_y + 1.0_f32),
        Pos2::new(x + hw, top_y + 1.0_f32),
        Pos2::new(x, top_y + hh),
    ];
    painter.add(egui::Shape::convex_polygon(
        pts,
        Color32::from_rgb(56, 189, 248),
        Stroke::new(1.0_f32, Color32::WHITE),
    ));
}

fn draw_thin_scrollbar(ui: &mut Ui, wave_w: f32, vis_frac: f32, scroll_norm: &mut f32) {
    let (rect, resp) =
        ui.allocate_exact_size(Vec2::new(wave_w, 12.0_f32), Sense::click_and_drag());

    let thumb_w = (rect.width() * vis_frac).clamp(28.0_f32, rect.width());
    let travel = (rect.width() - thumb_w).max(1.0_f32);

    if resp.clicked() || resp.dragged() {
        if let Some(pos) = resp.interact_pointer_pos() {
            *scroll_norm = ((pos.x - rect.left() - thumb_w * 0.5_f32) / travel).clamp(0.0_f32, 1.0_f32);
        }
    }

    if ui.is_rect_visible(rect) {
        let track_rect = Rect::from_min_max(
            Pos2::new(rect.left(), rect.center().y - 3.5_f32),
            Pos2::new(rect.right(), rect.center().y + 3.5_f32),
        );
        ui.painter()
            .rect_filled(track_rect, CornerRadius::same(4), Color32::from_rgb(16, 22, 34));

        let thumb_x = rect.left() + (*scroll_norm).clamp(0.0_f32, 1.0_f32) * travel;
        let thumb_rect = Rect::from_min_max(
            Pos2::new(thumb_x, track_rect.top()),
            Pos2::new(thumb_x + thumb_w, track_rect.bottom()),
        );
        let thumb_col = if resp.dragged() || resp.hovered() {
            Color32::from_rgb(56, 189, 248)
        } else {
            Color32::from_rgb(71, 88, 120)
        };
        ui.painter()
            .rect_filled(thumb_rect, CornerRadius::same(4), thumb_col);
    }

    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    if !resp.dragged() {
        resp.on_hover_text(
            "Barra di scorrimento orizzontale del brano (oppure usa Shift + Rotellina del mouse)",
        );
    }
}

/// Pulsante compatto a pillola per Mute ("M"), Solo ("S"), FX o Loop all'interno delle intestazioni traccia.
fn compact_pill_toggle(
    ui: &mut Ui,
    label: &str,
    active: bool,
    accent: Color32,
    size: Vec2,
) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    if ui.is_rect_visible(rect) {
        let hovered = resp.hovered();
        let radius = CornerRadius::same((rect.height() * 0.5_f32) as u8);

        if active {
            ui.painter().rect_filled(
                rect.expand(1.5_f32),
                radius,
                Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 44),
            );
        }

        let bg = if active {
            if hovered {
                brighten(accent, 18)
            } else {
                accent
            }
        } else if hovered {
            Color32::from_rgb(35, 45, 65)
        } else {
            Color32::from_rgb(15, 20, 31)
        };

        let border = if active {
            brighten(accent, 34)
        } else if hovered {
            Color32::from_rgb(76, 95, 130)
        } else {
            Color32::from_rgb(42, 54, 78)
        };

        ui.painter().rect_filled(rect, radius, bg);
        ui.painter().rect_stroke(
            rect,
            radius,
            Stroke::new(1.0_f32, border),
            StrokeKind::Inside,
        );

        let text_col = if active {
            Color32::WHITE
        } else if hovered {
            Color32::from_rgb(226, 232, 240)
        } else {
            Color32::from_rgb(156, 169, 190)
        };

        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(11.0_f32),
            text_col,
        );
    }

    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

/// Controllo Pan bipolare moderno con origine centrale `0.0` (`-1.0` Left .. `+1.0` Right).
/// Doppio clic sul controllo riporta il Pan esattamente al centro `0.0`.
fn sleek_bipolar_pan(
    ui: &mut Ui,
    pan: &mut f32,
    accent: Color32,
    width: f32,
    height: f32,
    tooltip: &str,
) -> bool {
    let (rect, resp) = ui.allocate_exact_size(
        Vec2::new(width.max(40.0_f32), height.max(15.0_f32)),
        Sense::click_and_drag(),
    );
    let mut changed = false;

    if resp.double_clicked() {
        *pan = 0.0_f32;
        changed = true;
    } else if resp.clicked() || resp.dragged() {
        if let Some(pos) = resp.interact_pointer_pos() {
            let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0_f32, 1.0_f32);
            let raw = -1.0_f32 + 2.0_f32 * t;
            *pan = if raw.abs() < 0.035_f32 { 0.0_f32 } else { raw };
            changed = true;
        }
    }

    if ui.is_rect_visible(rect) {
        let track_h = 5.0_f32;
        let track_rect = Rect::from_min_max(
            Pos2::new(rect.left(), rect.center().y - track_h * 0.5_f32),
            Pos2::new(rect.right(), rect.center().y + track_h * 0.5_f32),
        );
        ui.painter()
            .rect_filled(track_rect, CornerRadius::same(3), Color32::from_rgb(26, 33, 48));

        let cx = rect.center().x;
        ui.painter().line_segment(
            [
                Pos2::new(cx, rect.center().y - 5.0_f32),
                Pos2::new(cx, rect.center().y + 5.0_f32),
            ],
            Stroke::new(1.0_f32, Color32::from_rgb(71, 85, 105)),
        );

        let px = cx + pan.clamp(-1.0_f32, 1.0_f32) * (rect.width() * 0.5_f32);
        let fill_min_x = cx.min(px);
        let fill_max_x = cx.max(px);
        if (fill_max_x - fill_min_x) > 0.8_f32 {
            let fill_rect = Rect::from_min_max(
                Pos2::new(fill_min_x, track_rect.top()),
                Pos2::new(fill_max_x, track_rect.bottom()),
            );
            ui.painter()
                .rect_filled(fill_rect, CornerRadius::same(2), accent);
        }

        let thumb_r = if resp.hovered() || resp.dragged() {
            5.6_f32
        } else {
            4.6_f32
        };
        let thumb_pos = Pos2::new(px, rect.center().y);
        ui.painter().circle_filled(
            thumb_pos,
            thumb_r + 2.0_f32,
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 45),
        );
        ui.painter()
            .circle_filled(thumb_pos, thumb_r, Color32::WHITE);
    }

    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    if !resp.dragged() && !tooltip.is_empty() {
        resp.on_hover_text(tooltip);
    }
    changed
}

/// Mini meter verticale LED a segmenti (verde -> ambra -> rosso) pilotato da `live_peak`.
fn draw_vertical_led_meter(painter: &egui::Painter, rect: Rect, live_peak: f32) {
    let n_segs = 12usize;
    let gap = 1.5_f32;
    let total_gaps = gap * ((n_segs - 1) as f32);
    let seg_h = ((rect.height() - total_gaps) / (n_segs as f32)).max(2.0_f32);

    let db = 20.0_f32 * live_peak.max(1e-5_f32).log10();
    let norm = ((db + 42.0_f32) / 42.0_f32).clamp(0.0_f32, 1.0_f32);
    let lit_count = (norm * (n_segs as f32)).round() as usize;

    for i in 0..n_segs {
        let y_bot = rect.bottom() - (i as f32) * (seg_h + gap);
        let y_top = y_bot - seg_h;
        let seg_rect = Rect::from_min_max(Pos2::new(rect.left(), y_top), Pos2::new(rect.right(), y_bot));

        let col = if i < lit_count {
            if i >= 10 {
                Color32::from_rgb(239, 68, 68) // Clip / Hot
            } else if i >= 7 {
                Color32::from_rgb(245, 158, 11) // Headroom ambra
            } else {
                Color32::from_rgb(34, 197, 94) // Segnale nominale verde smeraldo
            }
        } else {
            Color32::from_rgb(13, 18, 28)
        };

        painter.rect_filled(seg_rect, CornerRadius::same(1), col);
    }
}

fn format_pan_label(pan: f32) -> String {
    if pan.abs() < 0.035_f32 {
        "PAN C".to_string()
    } else if pan < 0.0_f32 {
        format!("L {:>2.0}%", (-pan * 100.0_f32).round())
    } else {
        format!("R {:>2.0}%", (pan * 100.0_f32).round())
    }
}

fn draw_empty_timeline_placeholder(ui: &mut Ui, wave_w: f32) {
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(COL_GAP, LANE_GAP);

        ui.horizontal(|ui| {
            let (c_rect, _) =
                ui.allocate_exact_size(Vec2::new(HEADER_WIDTH, RULER_HEIGHT), Sense::hover());
            ui.painter()
                .rect_filled(c_rect, CornerRadius::same(6), Color32::from_rgb(14, 19, 29));
            ui.painter().rect_stroke(
                c_rect,
                CornerRadius::same(6),
                Stroke::new(1.0_f32, Color32::from_rgb(30, 41, 59)),
                StrokeKind::Inside,
            );
            ui.painter().text(
                Pos2::new(c_rect.left() + 10.0_f32, c_rect.center().y),
                Align2::LEFT_CENTER,
                "00:00.0 / 00:00",
                FontId::monospace(10.5_f32),
                Color32::from_rgb(100, 116, 139),
            );

            let (r_rect, _) =
                ui.allocate_exact_size(Vec2::new(wave_w, RULER_HEIGHT), Sense::hover());
            let _ = draw_time_ruler(ui.painter(), r_rect, 0, 44100 * 120, 44100, None, false);
        });

        ui.horizontal(|ui| {
            let empty_h = 136.0_f32;
            let (h_rect, _) =
                ui.allocate_exact_size(Vec2::new(HEADER_WIDTH, empty_h), Sense::hover());
            ui.painter()
                .rect_filled(h_rect, CornerRadius::same(8), Color32::from_rgb(21, 28, 43));
            ui.painter().rect_stroke(
                h_rect,
                CornerRadius::same(8),
                Stroke::new(1.0_f32, Color32::from_rgb(35, 46, 68)),
                StrokeKind::Inside,
            );
            ui.painter().text(
                h_rect.center(),
                Align2::CENTER_CENTER,
                "NESSUN BRANO CARICATO",
                FontId::proportional(11.2_f32),
                Color32::from_rgb(120, 136, 162),
            );

            let (w_rect, _) = ui.allocate_exact_size(Vec2::new(wave_w, empty_h), Sense::hover());
            ui.painter()
                .rect_filled(w_rect, CornerRadius::same(8), Color32::from_rgb(13, 18, 28));
            ui.painter().rect_stroke(
                w_rect,
                CornerRadius::same(8),
                Stroke::new(1.0_f32, Color32::from_rgb(28, 38, 56)),
                StrokeKind::Inside,
            );
            ui.painter().line_segment(
                [
                    Pos2::new(w_rect.left() + 12.0_f32, w_rect.center().y),
                    Pos2::new(w_rect.right() - 12.0_f32, w_rect.center().y),
                ],
                Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(148, 163, 184, 26)),
            );
            ui.painter().text(
                Pos2::new(w_rect.center().x, w_rect.center().y - 9.0_f32),
                Align2::CENTER_CENTER,
                "Trascina qui un file audio (MP3, WAV, FLAC, M4A) o premi 'Apri Audio' in alto",
                FontId::proportional(13.0_f32),
                Color32::from_rgb(203, 213, 225),
            );
            ui.painter().text(
                Pos2::new(w_rect.center().x, w_rect.center().y + 12.0_f32),
                Align2::CENTER_CENTER,
                "Potrai pulire il suono della sala prove con Auto-EQ o dividere Voce, Batteria, Basso e Chitarre con Separa AI",
                FontId::proportional(11.4_f32),
                Color32::from_rgb(112, 130, 158),
            );
        });
    });
}

#[inline]
fn x_to_frame(
    x: f32,
    rect: Rect,
    start_frame: usize,
    vis_frames: usize,
    total_frames: usize,
) -> usize {
    if total_frames == 0 || rect.width() <= 0.0_f32 {
        return 0;
    }
    let t = ((x - rect.left()) / rect.width()).clamp(0.0_f32, 1.0_f32);
    let offset = (t * (vis_frames as f32)).round() as usize;
    (start_frame + offset).min(total_frames.saturating_sub(1))
}

#[inline]
fn frame_to_x(frame: usize, rect: Rect, start_frame: usize, vis_frames: usize) -> f32 {
    if vis_frames == 0 {
        return rect.left();
    }
    let rel = frame.saturating_sub(start_frame) as f32 / (vis_frames as f32);
    (rect.left() + rel * rect.width()).clamp(rect.left(), rect.right())
}
