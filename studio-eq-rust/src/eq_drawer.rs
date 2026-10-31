use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Ui, Vec2,
};

use crate::audio_io::{PlaybackSourceMode, SharedAudioState};
use crate::dsp::{
    stem_eq_band_specs, stem_eq_curve_db, total_curve_db, BiquadCoeffs, DspParams, FilterKind,
    STEM_EQ_MAX_DB, STEM_EQ_MIN_DB,
};
use crate::ui_widgets::{brighten, draw_filled_curve_area, pill_button, studio_knob};

const FREQ_MIN_HZ: f64 = 20.0_f64;
const FREQ_MAX_HZ: f64 = 20_000.0_f64;
const DB_VIEW_RANGE: f64 = 15.0_f64;

const GRID_FREQS: [(f64, &str, bool); 8] = [
    (50.0_f64, "50", false),
    (100.0_f64, "100 Hz", true),
    (200.0_f64, "200", false),
    (500.0_f64, "500", false),
    (1000.0_f64, "1 kHz", true),
    (2000.0_f64, "2k", false),
    (5000.0_f64, "5k", false),
    (10_000.0_f64, "10k", true),
];

const BAND_ROLES_IT: [&str; 8] = [
    "Sub & Rimbombo",
    "Anti-Fango",
    "Pulizia Medio-Bassi",
    "Anti-Scatola",
    "Presenza Voce",
    "Anti-Asprezza",
    "Attacco & Definizione",
    "Aria & Brillantezza",
];

const NODE_PALETTE: [Color32; 8] = [
    Color32::from_rgb(251, 146, 60),  // Band 1: Sub / Low Resonance
    Color32::from_rgb(250, 204, 21),  // Band 2: Anti-Mud
    Color32::from_rgb(74, 222, 128),  // Band 3: Low-Mid Clean
    Color32::from_rgb(45, 212, 191),  // Band 4: Anti-Box
    Color32::from_rgb(34, 211, 238),  // Band 5: Vocal & Guitars Body
    Color32::from_rgb(56, 189, 248),  // Band 6: Harshness Tamer
    Color32::from_rgb(129, 140, 248), // Band 7: Attack & Presence
    Color32::from_rgb(192, 132, 252), // Band 8: Studio Air Shelf
];

const STEM_TAB_LABELS: [&str; 4] = ["Voce", "Batteria", "Basso", "Chitarre"];
const STEM_FULL_TITLES: [&str; 4] = ["VOCE", "BATTERIA", "BASSO", "CHITARRE & ALTRO"];
const STEM_BAND_NODES: [(&str, &str, Color32); 3] = [
    ("B", "Bassi", Color32::from_rgb(245, 158, 11)),
    ("M", "Medi", Color32::from_rgb(34, 211, 238)),
    ("A", "Alti", Color32::from_rgb(192, 132, 252)),
];

fn stem_accent_color(idx: usize) -> Color32 {
    match idx {
        0 => Color32::from_rgb(56, 189, 248),
        1 => Color32::from_rgb(245, 158, 11),
        2 => Color32::from_rgb(168, 85, 247),
        _ => Color32::from_rgb(34, 197, 94),
    }
}

fn stem_knob_subtitles(stem_idx: usize) -> (&'static str, &'static str, &'static str, &'static str) {
    match stem_idx.min(3) {
        0 => ("Calore 135 Hz", "Presenza 1.65 kHz", "Aria 6.5 kHz", "Livello Voce"),
        1 => (
            "Cassa & Sub 95 Hz",
            "Attacco/Box 1.1 kHz",
            "Piatti & Punta 5.2k",
            "Livello Batteria",
        ),
        2 => (
            "Sub 85 Hz",
            "Definizione 780 Hz",
            "Corde 3.2 kHz",
            "Livello Basso",
        ),
        _ => (
            "Corpo 120 Hz",
            "Morso 1.35 kHz",
            "Brillantezza 5.8k",
            "Livello Chitarre",
        ),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqDrawerTarget {
    MasterBus,
    Stem(usize),
}

#[derive(Clone, Debug)]
pub struct EqDrawerState {
    pub active_target: EqDrawerTarget,
    pub dragged_band: Option<usize>,
    pub selected_band: usize,
    pub dragged_stem_band: Option<usize>,
    pub selected_stem_band: usize,
    pub macro_clean: f32,
    pub macro_vocal: f32,
    pub macro_air: f32,
    pub macro_punch: f32,
}

impl Default for EqDrawerState {
    fn default() -> Self {
        Self {
            active_target: EqDrawerTarget::MasterBus,
            dragged_band: None,
            selected_band: 0,
            dragged_stem_band: None,
            selected_stem_band: 0,
            macro_clean: 75.0_f32,
            macro_vocal: 65.0_f32,
            macro_air: 70.0_f32,
            macro_punch: 60.0_f32,
        }
    }
}

#[inline]
fn freq_to_x(freq: f64, rect: Rect) -> f32 {
    let min_l = FREQ_MIN_HZ.log10();
    let max_l = FREQ_MAX_HZ.log10();
    let norm = ((freq.clamp(FREQ_MIN_HZ, FREQ_MAX_HZ).log10() - min_l) / (max_l - min_l)) as f32;
    rect.left() + norm * rect.width()
}

#[inline]
fn x_to_freq(x: f32, rect: Rect) -> f64 {
    let norm = ((x - rect.left()) / rect.width().max(1.0_f32)).clamp(0.0_f32, 1.0_f32) as f64;
    let min_l = FREQ_MIN_HZ.log10();
    let max_l = FREQ_MAX_HZ.log10();
    10.0_f64.powf(min_l + norm * (max_l - min_l))
}

#[inline]
fn db_to_y(db: f64, rect: Rect) -> f32 {
    let mid = rect.center().y;
    let half_span = rect.height() * 0.42_f32;
    let norm = (db.clamp(-DB_VIEW_RANGE, DB_VIEW_RANGE) / DB_VIEW_RANGE) as f32;
    mid - norm * half_span
}

#[inline]
fn y_to_db(y: f32, rect: Rect) -> f64 {
    let mid = rect.center().y;
    let half_span = (rect.height() * 0.42_f32).max(1.0_f32);
    (((mid - y) / half_span) as f64 * DB_VIEW_RANGE).clamp(-DB_VIEW_RANGE, DB_VIEW_RANGE)
}

fn format_freq_hz(freq: f64) -> String {
    if freq >= 10_000.0_f64 {
        format!("{:.1} kHz", freq / 1000.0_f64)
    } else if freq >= 1000.0_f64 {
        format!("{:.2} kHz", freq / 1000.0_f64)
    } else {
        format!("{:.0} Hz", freq)
    }
}

fn format_q(q: f64) -> String {
    if ((q * 10.0_f64) - (q * 10.0_f64).round()).abs() < 0.04_f64 {
        format!("Q {:.1}", q)
    } else {
        format!("Q {:.2}", q)
    }
}

fn detect_active_preset(params: &DspParams) -> Option<usize> {
    if params.bypass || params.bands.len() < 8 {
        return None;
    }
    let all_flat = params.bands.iter().all(|b| b.gain_db.abs() < 0.15_f64)
        && params.comp_drive_db.abs() < 0.25_f64;
    if all_flat {
        return Some(3);
    }
    if (params.bands[0].freq - 68.0_f64).abs() < 4.0_f64 && params.bands[0].gain_db > 1.5_f64 {
        return Some(2);
    }
    if (params.bands[4].gain_db - 6.8_f64).abs() < 0.35_f64
        && (params.bands[1].gain_db - (-7.0_f64)).abs() < 0.35_f64
    {
        return Some(1);
    }
    if (params.bands[1].gain_db - (-6.2_f64)).abs() < 0.35_f64
        && (params.bands[4].gain_db - 5.4_f64).abs() < 0.35_f64
    {
        return Some(0);
    }
    None
}

pub fn draw_eq_and_master_drawer(
    ui: &mut Ui,
    state: &mut SharedAudioState,
    eq_ui: &mut EqDrawerState,
    spectrum_db: &[f32],
) {
    if state.stems.is_none() && matches!(eq_ui.active_target, EqDrawerTarget::Stem(_)) {
        eq_ui.active_target = EqDrawerTarget::MasterBus;
    }

    let total_w = ui.available_width().max(620.0_f32);
    let gap = 12.0_f32;
    let drawer_h = 204.0_f32;
    let usable_w = (total_w - gap).max(600.0_f32);
    // Garantisce almeno 508px alle 4 manopole (4x122 + 3x6 = 506px) così nessuna manopola viene tagliata a 1060px
    let right_w = (usable_w * 0.45_f32)
        .floor()
        .max(508.0_f32)
        .min((usable_w - 340.0_f32).max(280.0_f32));
    let left_w = (usable_w - right_w).max(320.0_f32);

    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = gap;

        // =========================================================================
        // SEZIONE SINISTRA: Grafico Spettro FFT + Curva EQ Master o Curva 3-Bande Stem
        // =========================================================================
        ui.allocate_ui_with_layout(
            Vec2::new(left_w, drawer_h),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                draw_pro_eq_canvas(ui, state, eq_ui, spectrum_db, Vec2::new(left_w, drawer_h));
            },
        );

        // =========================================================================
        // SEZIONE DESTRA: Pillole + 4 Manopole Studio (Master oppure Bassi/Medi/Alti/Vol Stem)
        // =========================================================================
        ui.allocate_ui_with_layout(
            Vec2::new(right_w, drawer_h),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                if state.stems.is_some() {
                    if let EqDrawerTarget::Stem(s_idx) = eq_ui.active_target {
                        draw_stem_three_band_controls(ui, state, eq_ui, s_idx.min(3), right_w);
                        return;
                    }
                }
                draw_smart_master_controls(ui, state, eq_ui, right_w);
            },
        );
    });
}

fn build_canvas_target_tabs(
    painter: &egui::Painter,
    inner_rect: Rect,
) -> Vec<(Rect, EqDrawerTarget, &'static str, Color32)> {
    let items: [(EqDrawerTarget, &'static str, Color32); 5] = [
        (
            EqDrawerTarget::MasterBus,
            "Master",
            Color32::from_rgb(14, 165, 233),
        ),
        (EqDrawerTarget::Stem(0), STEM_TAB_LABELS[0], stem_accent_color(0)),
        (EqDrawerTarget::Stem(1), STEM_TAB_LABELS[1], stem_accent_color(1)),
        (EqDrawerTarget::Stem(2), STEM_TAB_LABELS[2], stem_accent_color(2)),
        (EqDrawerTarget::Stem(3), STEM_TAB_LABELS[3], stem_accent_color(3)),
    ];

    let font = FontId::proportional(9.9_f32);
    let mut x = inner_rect.left() + 8.0_f32;
    let y = inner_rect.top() + 5.0_f32;
    let h = 19.0_f32;
    let mut out = Vec::with_capacity(5);

    for (target, label, col) in items {
        let galley = painter.layout_no_wrap(label.to_string(), font.clone(), Color32::WHITE);
        let w = (galley.size().x + 13.0_f32).ceil();
        let r = Rect::from_min_size(Pos2::new(x, y), Vec2::new(w, h));
        out.push((r, target, label, col));
        x += w + 4.0_f32;
    }
    out
}

fn draw_canvas_target_tabs(
    painter: &egui::Painter,
    tabs: &[(Rect, EqDrawerTarget, &'static str, Color32)],
    active_target: EqDrawerTarget,
    hovered_tab: Option<usize>,
    state: &SharedAudioState,
) -> f32 {
    let font = FontId::proportional(9.9_f32);
    let mut right_end = 0.0_f32;

    for (idx, &(rect, target, label, col)) in tabs.iter().enumerate() {
        right_end = rect.right();
        let is_active = active_target == target;
        let is_hovered = hovered_tab == Some(idx);
        let has_eq_mod = match target {
            EqDrawerTarget::MasterBus => false,
            EqDrawerTarget::Stem(s) => !state.stem_channels[s.min(3)].eq.is_flat(),
        };

        let radius = CornerRadius::same(9);
        if is_active {
            painter.rect_filled(
                rect.expand(1.2_f32),
                radius,
                Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 38),
            );
        }
        let bg = if is_active {
            col
        } else if is_hovered {
            Color32::from_rgb(28, 38, 58)
        } else {
            Color32::from_rgba_unmultiplied(16, 22, 35, 228)
        };
        let border = if is_active {
            brighten(col, 30)
        } else if has_eq_mod {
            Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 165)
        } else if is_hovered {
            Color32::from_rgb(72, 92, 126)
        } else {
            Color32::from_rgb(34, 46, 68)
        };

        painter.rect_filled(rect, radius, bg);
        painter.rect_stroke(rect, radius, Stroke::new(1.0_f32, border), StrokeKind::Inside);

        let txt_col = if is_active {
            Color32::WHITE
        } else if has_eq_mod {
            brighten(col, 24)
        } else if is_hovered {
            Color32::from_rgb(226, 232, 240)
        } else {
            Color32::from_rgb(148, 163, 184)
        };
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            label,
            font.clone(),
            txt_col,
        );
    }

    right_end
}

fn draw_pro_eq_canvas(
    ui: &mut Ui,
    state: &mut SharedAudioState,
    eq_ui: &mut EqDrawerState,
    spectrum_db: &[f32],
    desired_size: Vec2,
) {
    let (rect, response) = ui.allocate_exact_size(desired_size, Sense::click_and_drag());
    let card_radius = CornerRadius::same(12);

    // Card Ossidiana (#0A0E16)
    let outer_painter = ui.painter_at(rect);
    outer_painter.rect_filled(rect, card_radius, Color32::from_rgb(10, 14, 22));

    let inner_rect = rect.shrink(1.0_f32);
    let painter = ui.painter_at(inner_rect);
    let sr = state.device_sr.max(8000) as f64;

    // 1. Griglia sottilissima stile FabFilter Pro-Q 4 (20 Hz .. 20 kHz senza sovrapposizioni)
    let freq_label_col = Color32::from_rgb(104, 122, 150);
    painter.text(
        Pos2::new(inner_rect.left() + 6.0_f32, inner_rect.bottom() - 5.5_f32),
        Align2::LEFT_BOTTOM,
        "20 Hz",
        FontId::proportional(10.0_f32),
        freq_label_col,
    );
    painter.text(
        Pos2::new(inner_rect.right() - 6.0_f32, inner_rect.bottom() - 5.5_f32),
        Align2::RIGHT_BOTTOM,
        "20k",
        FontId::proportional(10.0_f32),
        freq_label_col,
    );

    for &(freq, label, is_decade) in &GRID_FREQS {
        let x = freq_to_x(freq, inner_rect);
        let line_col = if is_decade {
            Color32::from_rgb(24, 33, 50)
        } else {
            Color32::from_rgb(17, 23, 35)
        };
        painter.line_segment(
            [Pos2::new(x, inner_rect.top()), Pos2::new(x, inner_rect.bottom())],
            Stroke::new(1.0_f32, line_col),
        );
        painter.text(
            Pos2::new(x, inner_rect.bottom() - 5.5_f32),
            Align2::CENTER_BOTTOM,
            label,
            FontId::proportional(10.2_f32),
            freq_label_col,
        );
    }

    for &db_mark in &[-12.0_f64, -6.0_f64, 0.0_f64, 6.0_f64, 12.0_f64] {
        let y = db_to_y(db_mark, inner_rect);
        let is_zero = db_mark.abs() < 0.01_f64;
        let line_col = if is_zero {
            Color32::from_rgb(42, 56, 82)
        } else {
            Color32::from_rgb(18, 24, 37)
        };
        painter.line_segment(
            [
                Pos2::new(inner_rect.left(), y),
                Pos2::new(inner_rect.right(), y),
            ],
            Stroke::new(1.0_f32, line_col),
        );
        if db_mark.abs() < 11.5_f64 {
            let lbl = if is_zero {
                "0 dB".to_string()
            } else {
                format!("{:+.0}", db_mark)
            };
            painter.text(
                Pos2::new(inner_rect.right() - 7.0_f32, y - 2.0_f32),
                Align2::RIGHT_BOTTOM,
                lbl,
                FontId::monospace(9.5_f32),
                Color32::from_rgb(72, 88, 114),
            );
        }
    }

    // 2. Spettro FFT riempito morbido (blu/ciano tenue)
    if spectrum_db.len() >= 16 {
        let n_bins = spectrum_db.len();
        let fft_size = (n_bins * 2) as f64;
        let width_px = inner_rect.width().max(2.0_f32) as usize;
        let step = 3usize;
        let mut raw_y: Vec<(f32, f32)> = Vec::with_capacity(width_px / step + 3);

        let sample_fft_y = |x: f32| -> f32 {
            let freq = x_to_freq(x, inner_rect);
            let f_bin = ((freq * fft_size) / sr).clamp(1.0_f64, (n_bins - 1) as f64);
            let b0 = f_bin.floor() as usize;
            let b1 = (b0 + 1).min(n_bins - 1);
            let frac = (f_bin - b0 as f64) as f32;
            let db_val = spectrum_db[b0] * (1.0_f32 - frac) + spectrum_db[b1] * frac;
            let norm = ((db_val + 86.0_f32) / 76.0_f32).clamp(0.0_f32, 1.0_f32);
            inner_rect.bottom() - norm * (inner_rect.height() * 0.86_f32)
        };

        let mut px = 0usize;
        while px < width_px {
            let x = inner_rect.left() + px as f32;
            raw_y.push((x, sample_fft_y(x)));
            px += step;
        }
        let right_x = inner_rect.right();
        raw_y.push((right_x, sample_fft_y(right_x)));

        let mut spec_pts: Vec<Pos2> = Vec::with_capacity(raw_y.len());
        for i in 0..raw_y.len() {
            let y_prev = raw_y[i.saturating_sub(1)].1;
            let y_curr = raw_y[i].1;
            let y_next = raw_y[(i + 1).min(raw_y.len() - 1)].1;
            let smooth_y = 0.25_f32 * y_prev + 0.50_f32 * y_curr + 0.25_f32 * y_next;
            spec_pts.push(Pos2::new(raw_y[i].0, smooth_y));
        }

        draw_filled_curve_area(
            &painter,
            &spec_pts,
            inner_rect.bottom(),
            Color32::from_rgba_unmultiplied(38, 162, 236, 54),
            Color32::from_rgba_unmultiplied(14, 52, 96, 6),
        );
        painter.add(egui::Shape::line(
            spec_pts,
            Stroke::new(1.15_f32, Color32::from_rgba_unmultiplied(56, 189, 248, 96)),
        ));
    }

    // Se sono presenti le 4 tracce isolate, costruiamo la barra selettore rapida (Master | Voce | Batteria | Basso | Chitarre)
    let pointer_pos = response
        .hover_pos()
        .or_else(|| response.interact_pointer_pos());
    let tabs_opt = state
        .stems
        .is_some()
        .then(|| build_canvas_target_tabs(&painter, inner_rect));
    let mut hovered_tab: Option<usize> = None;
    let mut consumed_by_tab = false;

    if let (Some(ref tabs), Some(pos)) = (&tabs_opt, pointer_pos) {
        for (i, &(t_rect, _, _, _)) in tabs.iter().enumerate() {
            if t_rect.contains(pos) {
                hovered_tab = Some(i);
                break;
            }
        }
    }

    if let Some(tab_idx) = hovered_tab {
        if let Some(ref tabs) = tabs_opt {
            let (_, target, label, _) = tabs[tab_idx];
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            if response.secondary_clicked() {
                if let EqDrawerTarget::Stem(s) = target {
                    state.stem_channels[s.min(3)].eq.reset();
                    state.mode = PlaybackSourceMode::FourStemMixer;
                    consumed_by_tab = true;
                }
            } else if response.clicked() || response.drag_started() {
                eq_ui.active_target = target;
                eq_ui.dragged_band = None;
                eq_ui.dragged_stem_band = None;
                if matches!(target, EqDrawerTarget::Stem(_)) {
                    state.mode = PlaybackSourceMode::FourStemMixer;
                }
                consumed_by_tab = true;
            }
            if !response.dragged() {
                let tip = match target {
                    EqDrawerTarget::MasterBus => {
                        "Passa all'Equalizzatore Master a 8 Bande (intero brano)".to_string()
                    }
                    EqDrawerTarget::Stem(_) => format!(
                        "Modifica Bassi / Medi / Alti della traccia isolata: {}\n• Tasto destro sulla scheda per azzerare a 0.0 dB",
                        label.to_uppercase()
                    ),
                };
                response.clone().on_hover_text(tip);
            }
        }
    }

    if state.stems.is_some() {
        if let EqDrawerTarget::Stem(s_idx) = eq_ui.active_target {
            draw_stem_eq_on_canvas(
                ui,
                &painter,
                inner_rect,
                &response,
                state,
                eq_ui,
                s_idx.min(3),
                sr,
                tabs_opt.as_deref(),
                hovered_tab,
                consumed_by_tab,
            );
            outer_painter.rect_stroke(
                rect,
                card_radius,
                Stroke::new(1.0_f32, Color32::from_rgb(30, 41, 62)),
                StrokeKind::Inside,
            );
            return;
        }
    }

    // =========================================================================
    // MODALITÀ MASTER BUS: Interazione e Disegno degli 8 Nodi Parametrici
    // =========================================================================
    if eq_ui.dragged_band.is_some() && !ui.input(|i| i.pointer.primary_down()) {
        eq_ui.dragged_band = None;
    }

    let mut hovered_band: Option<usize> = None;
    if hovered_tab.is_none() {
        if let Some(pos) = pointer_pos {
            let mut best_dist = 17.5_f32;
            for (idx, b) in state.params.bands.iter().enumerate() {
                let np = Pos2::new(freq_to_x(b.freq, inner_rect), db_to_y(b.gain_db, inner_rect));
                let dist = pos.distance(np);
                if dist < best_dist {
                    best_dist = dist;
                    hovered_band = Some(idx);
                }
            }
        }
    }

    if !consumed_by_tab {
        if response.double_clicked() {
            if let Some(idx) = hovered_band.or(eq_ui.dragged_band) {
                if idx < state.params.bands.len() {
                    state.params.bands[idx].gain_db = 0.0_f64;
                    state.params.bands[idx].enabled = true;
                    state.params.bypass = false;
                    state.params_dirty = true;
                    eq_ui.selected_band = idx;
                    eq_ui.dragged_band = None;
                }
            }
        } else {
            if response.clicked() {
                if let Some(idx) = hovered_band {
                    eq_ui.selected_band = idx;
                }
            }
            if response.drag_started() {
                eq_ui.dragged_band = hovered_band;
                if let Some(idx) = hovered_band {
                    eq_ui.selected_band = idx;
                }
            }
            if response.dragged() {
                if let (Some(idx), Some(pos)) = (eq_ui.dragged_band, response.interact_pointer_pos())
                {
                    if idx < state.params.bands.len() {
                        state.params.bands[idx].freq =
                            x_to_freq(pos.x, inner_rect).clamp(25.0_f64, 18_500.0_f64);
                        state.params.bands[idx].gain_db =
                            y_to_db(pos.y, inner_rect).clamp(-15.0_f64, 15.0_f64);
                        state.params.bands[idx].enabled = true;
                        state.params.bypass = false;
                        state.params_dirty = true;
                        eq_ui.selected_band = idx;
                    }
                }
            }
            if response.drag_stopped() {
                eq_ui.dragged_band = None;
            }
        }
    }

    // Regolazione fattore di merito Q con la rotellina del mouse sopra un nodo (0.4 ..= 8.0)
    let active_node = eq_ui.dragged_band.or(hovered_band);
    if response.hovered() && hovered_tab.is_none() {
        if let Some(idx) = active_node {
            let scroll_y = ui.input_mut(|i| {
                let dy = if i.raw_scroll_delta.y.abs() > 0.0_f32 {
                    i.raw_scroll_delta.y
                } else {
                    i.smooth_scroll_delta.y
                };
                if dy.abs() > 0.0_f32 {
                    i.raw_scroll_delta.y = 0.0_f32;
                    i.smooth_scroll_delta.y = 0.0_f32;
                }
                dy
            });
            if scroll_y.abs() > 0.01_f32 && idx < state.params.bands.len() {
                let step_scale =
                    (1.0_f64 + (scroll_y as f64).clamp(-60.0_f64, 60.0_f64) * 0.0045_f64)
                        .clamp(0.75_f64, 1.35_f64);
                state.params.bands[idx].q =
                    (state.params.bands[idx].q * step_scale).clamp(0.4_f64, 8.0_f64);
                state.params.bypass = false;
                state.params_dirty = true;
                eq_ui.selected_band = idx;
            }
        }
    }

    if eq_ui.dragged_band.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if hovered_band.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }

    let zero_y = db_to_y(0.0_f64, inner_rect);

    // Curva individuale soffusa per la banda selezionata/attiva (stile Pro-Q 4)
    let highlight_idx = active_node.unwrap_or(eq_ui.selected_band);
    if !state.params.bypass && highlight_idx < state.params.bands.len() {
        let b = &state.params.bands[highlight_idx];
        if b.enabled && b.gain_db.abs() > 0.08_f64 {
            let node_col = NODE_PALETTE[highlight_idx % NODE_PALETTE.len()];
            let coeffs = BiquadCoeffs::design(b.kind, b.freq, sr, b.gain_db, b.q);
            let width_px = inner_rect.width().max(2.0_f32) as usize;
            let mut band_pts = Vec::with_capacity(width_px / 4 + 3);
            let mut px = 0usize;
            while px < width_px {
                let x = inner_rect.left() + px as f32;
                let db = coeffs.magnitude_db(x_to_freq(x, inner_rect), sr);
                band_pts.push(Pos2::new(x, db_to_y(db, inner_rect)));
                px += 4;
            }
            let rx = inner_rect.right();
            band_pts.push(Pos2::new(
                rx,
                db_to_y(coeffs.magnitude_db(x_to_freq(rx, inner_rect), sr), inner_rect),
            ));

            draw_filled_curve_area(
                &painter,
                &band_pts,
                zero_y,
                Color32::from_rgba_unmultiplied(node_col.r(), node_col.g(), node_col.b(), 30),
                Color32::from_rgba_unmultiplied(node_col.r(), node_col.g(), node_col.b(), 3),
            );
            painter.add(egui::Shape::line(
                band_pts,
                Stroke::new(
                    1.0_f32,
                    Color32::from_rgba_unmultiplied(node_col.r(), node_col.g(), node_col.b(), 95),
                ),
            ));
        }
    }

    // 4. Curva complessiva `total_curve_db` riempita rispetto a 0 dB + linea nitida 2.5_f32
    let width_px = inner_rect.width().max(2.0_f32) as usize;
    let mut curve_pts: Vec<Pos2> = Vec::with_capacity(width_px / 2 + 8);
    let push_with_zero_crossing = |pts: &mut Vec<Pos2>, next: Pos2| {
        if let Some(&prev) = pts.last() {
            let dy0 = prev.y - zero_y;
            let dy1 = next.y - zero_y;
            if dy0 * dy1 < 0.0_f32 && (next.y - prev.y).abs() > 1e-4_f32 {
                let t = (zero_y - prev.y) / (next.y - prev.y);
                let cx = prev.x + t * (next.x - prev.x);
                pts.push(Pos2::new(cx, zero_y));
            }
        }
        pts.push(next);
    };

    let mut px = 0usize;
    while px < width_px {
        let x = inner_rect.left() + px as f32;
        let db = total_curve_db(&state.params, x_to_freq(x, inner_rect), sr);
        push_with_zero_crossing(&mut curve_pts, Pos2::new(x, db_to_y(db, inner_rect)));
        px += 2;
    }
    let rx = inner_rect.right();
    let r_db = total_curve_db(&state.params, x_to_freq(rx, inner_rect), sr);
    push_with_zero_crossing(&mut curve_pts, Pos2::new(rx, db_to_y(r_db, inner_rect)));

    if state.params.bypass {
        draw_filled_curve_area(
            &painter,
            &curve_pts,
            zero_y,
            Color32::from_rgba_unmultiplied(100, 116, 139, 24),
            Color32::from_rgba_unmultiplied(100, 116, 139, 4),
        );
        painter.add(egui::Shape::line(
            curve_pts,
            Stroke::new(2.5_f32, Color32::from_rgb(100, 116, 139)),
        ));
    } else {
        draw_filled_curve_area(
            &painter,
            &curve_pts,
            zero_y,
            Color32::from_rgba_unmultiplied(34, 211, 238, 66),
            Color32::from_rgba_unmultiplied(34, 211, 238, 6),
        );
        painter.add(egui::Shape::line(
            curve_pts.clone(),
            Stroke::new(5.5_f32, Color32::from_rgba_unmultiplied(34, 211, 238, 28)),
        ));
        painter.add(egui::Shape::line(
            curve_pts,
            Stroke::new(2.5_f32, Color32::from_rgb(56, 217, 245)),
        ));
    }

    // 5. Disegno degli 8 Nodi luminosi trascinabili
    for (idx, b) in state.params.bands.iter().enumerate() {
        let np = Pos2::new(freq_to_x(b.freq, inner_rect), db_to_y(b.gain_db, inner_rect));
        let col = NODE_PALETTE[idx % NODE_PALETTE.len()];
        let is_dragged = eq_ui.dragged_band == Some(idx);
        let is_hovered = hovered_band == Some(idx);
        let is_selected = eq_ui.selected_band == idx;

        let halo_r = if is_dragged || is_hovered {
            14.5_f32
        } else if is_selected {
            11.5_f32
        } else {
            9.5_f32
        };
        let halo_alpha = if is_dragged || is_hovered {
            68u8
        } else if is_selected {
            42u8
        } else {
            24u8
        };
        painter.circle_filled(
            np,
            halo_r,
            Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), halo_alpha),
        );

        let node_r = if is_dragged || is_hovered {
            8.2_f32
        } else {
            7.0_f32
        };
        painter.circle_filled(np, node_r, col);

        let ring_stroke = if is_dragged || is_hovered {
            Stroke::new(1.8_f32, Color32::WHITE)
        } else if is_selected {
            Stroke::new(1.3_f32, Color32::from_rgb(241, 245, 249))
        } else {
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 165))
        };
        painter.circle_stroke(np, node_r, ring_stroke);

        painter.text(
            np,
            Align2::CENTER_CENTER,
            format!("{}", idx + 1),
            FontId::proportional(9.5_f32),
            Color32::from_rgb(10, 14, 22),
        );
    }

    // Intestazione superiore: schede selettore (se 4 Stems attivi) oppure testo EQ 8 Bande
    let hdr_left_end = if let Some(ref tabs) = tabs_opt {
        draw_canvas_target_tabs(&painter, tabs, eq_ui.active_target, hovered_tab, state)
    } else {
        let (header_txt, header_col) = if state.params.bypass {
            (
                "EQ 8 BANDE  •  BYPASS ATTIVO (Audio Originale)",
                Color32::from_rgb(245, 158, 11),
            )
        } else {
            (
                "EQ 8 BANDE  •  Trascina: Freq/dB  •  Rotellina: Q  •  2xClic: 0 dB",
                Color32::from_rgb(96, 115, 144),
            )
        };
        let hdr_font = FontId::proportional(9.8_f32);
        let hdr_galley = painter.layout_no_wrap(header_txt.to_string(), hdr_font.clone(), header_col);
        painter.text(
            Pos2::new(inner_rect.left() + 10.0_f32, inner_rect.top() + 7.5_f32),
            Align2::LEFT_TOP,
            header_txt,
            hdr_font,
            header_col,
        );
        inner_rect.left() + 10.0_f32 + hdr_galley.size().x
    };

    let sel_idx = active_node.unwrap_or(eq_ui.selected_band);
    if let Some(sel_b) = state.params.bands.get(sel_idx) {
        let role_it = BAND_ROLES_IT[sel_idx % BAND_ROLES_IT.len()];
        let sel_col = NODE_PALETTE[sel_idx % NODE_PALETTE.len()];
        let sel_str = format!(
            "{} • {}  {}  {:+.1} dB",
            sel_idx + 1,
            role_it,
            format_freq_hz(sel_b.freq),
            sel_b.gain_db
        );
        let sel_font = FontId::proportional(9.8_f32);
        let sel_galley = painter.layout_no_wrap(sel_str.clone(), sel_font.clone(), sel_col);
        if hdr_left_end + sel_galley.size().x + 16.0_f32 <= inner_rect.right() - 8.0_f32 {
            painter.text(
                Pos2::new(inner_rect.right() - 9.0_f32, inner_rect.top() + 7.5_f32),
                Align2::RIGHT_TOP,
                sel_str,
                sel_font,
                sel_col,
            );
        }
    }

    // 6. Badge flottante arrotondato con ruolo musicale in italiano quando un nodo è attivo
    if let Some(idx) = active_node {
        if let Some(b) = state.params.bands.get(idx) {
            let np = Pos2::new(freq_to_x(b.freq, inner_rect), db_to_y(b.gain_db, inner_rect));
            let col = NODE_PALETTE[idx % NODE_PALETTE.len()];
            let role_it = BAND_ROLES_IT[idx % BAND_ROLES_IT.len()];
            let badge_str = format!(
                "{} • {}  |  {}  |  {:+.1} dB  |  {}",
                idx + 1,
                role_it,
                format_freq_hz(b.freq),
                b.gain_db,
                format_q(b.q)
            );
            let font = FontId::monospace(10.8_f32);
            let galley = painter.layout_no_wrap(badge_str.clone(), font.clone(), Color32::WHITE);
            let badge_w = (galley.size().x + 26.0_f32).min(inner_rect.width() - 12.0_f32);
            let badge_h = galley.size().y + 11.0_f32;

            let prefer_above = np.y - 24.0_f32 - badge_h >= inner_rect.top() + 26.0_f32;
            let center_y = if prefer_above {
                np.y - 19.0_f32 - badge_h * 0.5_f32
            } else {
                np.y + 19.0_f32 + badge_h * 0.5_f32
            };
            let min_cx = inner_rect.left() + badge_w * 0.5_f32 + 6.0_f32;
            let max_cx = (inner_rect.right() - badge_w * 0.5_f32 - 6.0_f32).max(min_cx);
            let center_x = np.x.clamp(min_cx, max_cx);

            let badge_rect =
                Rect::from_center_size(Pos2::new(center_x, center_y), Vec2::new(badge_w, badge_h));
            let badge_radius = CornerRadius::same(8);

            painter.rect_filled(
                badge_rect.expand(2.0_f32),
                badge_radius,
                Color32::from_rgba_unmultiplied(0, 0, 0, 95),
            );
            painter.rect_filled(
                badge_rect,
                badge_radius,
                Color32::from_rgba_unmultiplied(15, 22, 35, 244),
            );
            painter.rect_stroke(
                badge_rect,
                badge_radius,
                Stroke::new(1.2_f32, col),
                StrokeKind::Inside,
            );

            let dot_pos = Pos2::new(badge_rect.left() + 10.0_f32, badge_rect.center().y);
            painter.circle_filled(dot_pos, 3.5_f32, col);
            painter.text(
                Pos2::new(badge_rect.left() + 18.0_f32, badge_rect.center().y),
                Align2::LEFT_CENTER,
                badge_str,
                font,
                Color32::from_rgb(241, 245, 249),
            );
        }
    } else if !response.dragged() && hovered_tab.is_none() {
        response.on_hover_text(
            "Equalizzatore Parametrico a 8 Bande (64-bit)\n\
             • Trascina i pallini numerati (1–8) per cambiare Frequenza (Hz) e Guadagno (dB)\n\
             • Usa la rotellina sopra un pallino per stringere o allargare la campana (Q)\n\
             • Doppio clic su un pallino per riportarlo a 0.0 dB",
        );
    }

    outer_painter.rect_stroke(
        rect,
        card_radius,
        Stroke::new(1.0_f32, Color32::from_rgb(30, 41, 62)),
        StrokeKind::Inside,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_stem_eq_on_canvas(
    ui: &mut Ui,
    painter: &egui::Painter,
    inner_rect: Rect,
    response: &egui::Response,
    state: &mut SharedAudioState,
    eq_ui: &mut EqDrawerState,
    stem_idx: usize,
    sr: f64,
    tabs_opt: Option<&[(Rect, EqDrawerTarget, &'static str, Color32)]>,
    hovered_tab: Option<usize>,
    consumed_by_tab: bool,
) {
    let ((low_hz, low_q), (mid_hz, mid_q), (high_hz, high_q)) = stem_eq_band_specs(stem_idx);
    let specs: [(f64, f64, FilterKind); 3] = [
        (low_hz, low_q, FilterKind::LowShelf),
        (mid_hz, mid_q, FilterKind::Peaking),
        (high_hz, high_q, FilterKind::HighShelf),
    ];

    if eq_ui.dragged_stem_band.is_some() && !ui.input(|i| i.pointer.primary_down()) {
        eq_ui.dragged_stem_band = None;
    }

    let get_band_db = |st: &SharedAudioState, b_idx: usize| -> f32 {
        let eq = st.stem_channels[stem_idx].eq;
        match b_idx {
            0 => eq.low_db,
            1 => eq.mid_db,
            _ => eq.high_db,
        }
    };

    let set_band_db = |st: &mut SharedAudioState, b_idx: usize, val: f32| {
        let clamped = val.clamp(STEM_EQ_MIN_DB, STEM_EQ_MAX_DB);
        match b_idx {
            0 => st.stem_channels[stem_idx].eq.low_db = clamped,
            1 => st.stem_channels[stem_idx].eq.mid_db = clamped,
            _ => st.stem_channels[stem_idx].eq.high_db = clamped,
        }
        st.mode = PlaybackSourceMode::FourStemMixer;
    };

    let pointer_pos = response
        .hover_pos()
        .or_else(|| response.interact_pointer_pos());
    let mut hovered_node: Option<usize> = None;
    if hovered_tab.is_none() {
        if let Some(pos) = pointer_pos {
            let mut best_dist = 20.0_f32;
            for (b_idx, &(hz, _, _)) in specs.iter().enumerate() {
                let g_db = get_band_db(state, b_idx) as f64;
                let np = Pos2::new(freq_to_x(hz, inner_rect), db_to_y(g_db, inner_rect));
                let dist = pos.distance(np);
                if dist < best_dist {
                    best_dist = dist;
                    hovered_node = Some(b_idx);
                }
            }
        }
    }

    if !consumed_by_tab {
        if response.double_clicked() || response.secondary_clicked() {
            if let Some(b_idx) = hovered_node.or(eq_ui.dragged_stem_band) {
                set_band_db(state, b_idx, 0.0_f32);
                eq_ui.selected_stem_band = b_idx;
                eq_ui.dragged_stem_band = None;
            }
        } else {
            if response.clicked() {
                if let Some(b_idx) = hovered_node {
                    eq_ui.selected_stem_band = b_idx;
                }
            }
            if response.drag_started() {
                eq_ui.dragged_stem_band = hovered_node;
                if let Some(b_idx) = hovered_node {
                    eq_ui.selected_stem_band = b_idx;
                }
            }
            if response.dragged() {
                if let (Some(b_idx), Some(pos)) =
                    (eq_ui.dragged_stem_band, response.interact_pointer_pos())
                {
                    let new_db =
                        (y_to_db(pos.y, inner_rect) as f32).clamp(STEM_EQ_MIN_DB, STEM_EQ_MAX_DB);
                    set_band_db(state, b_idx, new_db);
                    eq_ui.selected_stem_band = b_idx;
                }
            }
            if response.drag_stopped() {
                eq_ui.dragged_stem_band = None;
            }
        }
    }

    let active_node = eq_ui.dragged_stem_band.or(hovered_node);
    if eq_ui.dragged_stem_band.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
    } else if hovered_node.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }

    let zero_y = db_to_y(0.0_f64, inner_rect);
    let width_px = inner_rect.width().max(2.0_f32) as usize;

    // Traccia di riferimento soffusa della curva Master sullo sfondo
    if !state.params.bypass {
        let mut ref_pts = Vec::with_capacity(width_px / 6 + 3);
        let mut px = 0usize;
        while px < width_px {
            let x = inner_rect.left() + px as f32;
            let db = total_curve_db(&state.params, x_to_freq(x, inner_rect), sr);
            ref_pts.push(Pos2::new(x, db_to_y(db, inner_rect)));
            px += 6;
        }
        let rx = inner_rect.right();
        ref_pts.push(Pos2::new(
            rx,
            db_to_y(total_curve_db(&state.params, x_to_freq(rx, inner_rect), sr), inner_rect),
        ));
        painter.add(egui::Shape::line(
            ref_pts,
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(100, 116, 139, 44)),
        ));
    }

    // Curva individuale soffusa per la banda selezionata (Bassi / Medi / Alti)
    let hi_b = active_node.unwrap_or(eq_ui.selected_stem_band).min(2);
    let hi_gain = get_band_db(state, hi_b) as f64;
    if hi_gain.abs() > 0.05_f64 {
        let (hz, q, kind) = specs[hi_b];
        let (_, _, node_col) = STEM_BAND_NODES[hi_b];
        let coeffs = BiquadCoeffs::design(kind, hz, sr, hi_gain, q);
        let mut band_pts = Vec::with_capacity(width_px / 4 + 3);
        let mut px = 0usize;
        while px < width_px {
            let x = inner_rect.left() + px as f32;
            let db = coeffs.magnitude_db(x_to_freq(x, inner_rect), sr);
            band_pts.push(Pos2::new(x, db_to_y(db, inner_rect)));
            px += 4;
        }
        let rx = inner_rect.right();
        band_pts.push(Pos2::new(
            rx,
            db_to_y(coeffs.magnitude_db(x_to_freq(rx, inner_rect), sr), inner_rect),
        ));
        draw_filled_curve_area(
            painter,
            &band_pts,
            zero_y,
            Color32::from_rgba_unmultiplied(node_col.r(), node_col.g(), node_col.b(), 32),
            Color32::from_rgba_unmultiplied(node_col.r(), node_col.g(), node_col.b(), 4),
        );
        painter.add(egui::Shape::line(
            band_pts,
            Stroke::new(
                1.1_f32,
                Color32::from_rgba_unmultiplied(node_col.r(), node_col.g(), node_col.b(), 105),
            ),
        ));
    }

    // Curva complessiva 3-Bande dello Stem isolato nel suo colore distintivo
    let stem_col = stem_accent_color(stem_idx);
    let eq_now = state.stem_channels[stem_idx].eq;
    let mut curve_pts: Vec<Pos2> = Vec::with_capacity(width_px / 2 + 8);
    let push_with_zero_crossing = |pts: &mut Vec<Pos2>, next: Pos2| {
        if let Some(&prev) = pts.last() {
            let dy0 = prev.y - zero_y;
            let dy1 = next.y - zero_y;
            if dy0 * dy1 < 0.0_f32 && (next.y - prev.y).abs() > 1e-4_f32 {
                let t = (zero_y - prev.y) / (next.y - prev.y);
                let cx = prev.x + t * (next.x - prev.x);
                pts.push(Pos2::new(cx, zero_y));
            }
        }
        pts.push(next);
    };

    let mut px = 0usize;
    while px < width_px {
        let x = inner_rect.left() + px as f32;
        let db = stem_eq_curve_db(stem_idx, &eq_now, x_to_freq(x, inner_rect), sr);
        push_with_zero_crossing(&mut curve_pts, Pos2::new(x, db_to_y(db, inner_rect)));
        px += 2;
    }
    let rx = inner_rect.right();
    let r_db = stem_eq_curve_db(stem_idx, &eq_now, x_to_freq(rx, inner_rect), sr);
    push_with_zero_crossing(&mut curve_pts, Pos2::new(rx, db_to_y(r_db, inner_rect)));

    draw_filled_curve_area(
        painter,
        &curve_pts,
        zero_y,
        Color32::from_rgba_unmultiplied(stem_col.r(), stem_col.g(), stem_col.b(), 68),
        Color32::from_rgba_unmultiplied(stem_col.r(), stem_col.g(), stem_col.b(), 6),
    );
    painter.add(egui::Shape::line(
        curve_pts.clone(),
        Stroke::new(
            5.5_f32,
            Color32::from_rgba_unmultiplied(stem_col.r(), stem_col.g(), stem_col.b(), 32),
        ),
    ));
    painter.add(egui::Shape::line(
        curve_pts,
        Stroke::new(2.6_f32, brighten(stem_col, 18)),
    ));

    // Disegno dei 3 Nodi luminosi trascinabili: B (Bassi), M (Medi), A (Alti)
    for (b_idx, &(hz, _, _)) in specs.iter().enumerate() {
        let g_db = get_band_db(state, b_idx) as f64;
        let np = Pos2::new(freq_to_x(hz, inner_rect), db_to_y(g_db, inner_rect));
        let (glyph, _, col) = STEM_BAND_NODES[b_idx];
        let is_dragged = eq_ui.dragged_stem_band == Some(b_idx);
        let is_hovered = hovered_node == Some(b_idx);
        let is_selected = eq_ui.selected_stem_band == b_idx;

        let halo_r = if is_dragged || is_hovered {
            15.5_f32
        } else if is_selected {
            12.5_f32
        } else {
            10.5_f32
        };
        let halo_alpha = if is_dragged || is_hovered {
            72u8
        } else if is_selected {
            46u8
        } else {
            28u8
        };
        painter.circle_filled(
            np,
            halo_r,
            Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), halo_alpha),
        );

        let node_r = if is_dragged || is_hovered {
            9.0_f32
        } else {
            7.8_f32
        };
        painter.circle_filled(np, node_r, col);
        let ring_stroke = if is_dragged || is_hovered {
            Stroke::new(1.8_f32, Color32::WHITE)
        } else {
            Stroke::new(1.2_f32, Color32::from_rgb(241, 245, 249))
        };
        painter.circle_stroke(np, node_r, ring_stroke);
        painter.text(
            np,
            Align2::CENTER_CENTER,
            glyph,
            FontId::proportional(9.8_f32),
            Color32::from_rgb(10, 14, 22),
        );
    }

    // Schede selettore in alto a sinistra + riepilogo traccia/banda in alto a destra
    let hdr_left_end = if let Some(tabs) = tabs_opt {
        draw_canvas_target_tabs(painter, tabs, eq_ui.active_target, hovered_tab, state)
    } else {
        inner_rect.left() + 10.0_f32
    };

    let sel_b = active_node.unwrap_or(eq_ui.selected_stem_band).min(2);
    let (_, band_it, _) = STEM_BAND_NODES[sel_b];
    let (sel_hz, _, _) = specs[sel_b];
    let sel_db = get_band_db(state, sel_b);
    let right_txt = format!(
        "{} • {} ({}) {:+.1} dB",
        STEM_FULL_TITLES[stem_idx],
        band_it,
        format_freq_hz(sel_hz),
        sel_db
    );
    let right_font = FontId::proportional(9.9_f32);
    let right_col = brighten(stem_col, 22);
    let right_galley = painter.layout_no_wrap(right_txt.clone(), right_font.clone(), right_col);
    if hdr_left_end + right_galley.size().x + 14.0_f32 <= inner_rect.right() - 8.0_f32 {
        painter.text(
            Pos2::new(inner_rect.right() - 9.0_f32, inner_rect.top() + 7.5_f32),
            Align2::RIGHT_TOP,
            right_txt,
            right_font,
            right_col,
        );
    }

    // Badge flottante durante hover/drag di Bassi / Medi / Alti
    if let Some(b_idx) = active_node {
        let (hz, _, _) = specs[b_idx];
        let g_db = get_band_db(state, b_idx) as f64;
        let np = Pos2::new(freq_to_x(hz, inner_rect), db_to_y(g_db, inner_rect));
        let (_, band_name, col) = STEM_BAND_NODES[b_idx];
        let badge_str = format!(
            "{} — {}  |  {}  |  {:+.1} dB",
            STEM_FULL_TITLES[stem_idx],
            band_name,
            format_freq_hz(hz),
            g_db
        );
        let font = FontId::monospace(10.8_f32);
        let galley = painter.layout_no_wrap(badge_str.clone(), font.clone(), Color32::WHITE);
        let badge_w = (galley.size().x + 26.0_f32).min(inner_rect.width() - 12.0_f32);
        let badge_h = galley.size().y + 11.0_f32;
        let prefer_above = np.y - 24.0_f32 - badge_h >= inner_rect.top() + 26.0_f32;
        let center_y = if prefer_above {
            np.y - 19.0_f32 - badge_h * 0.5_f32
        } else {
            np.y + 19.0_f32 + badge_h * 0.5_f32
        };
        let min_cx = inner_rect.left() + badge_w * 0.5_f32 + 6.0_f32;
        let max_cx = (inner_rect.right() - badge_w * 0.5_f32 - 6.0_f32).max(min_cx);
        let center_x = np.x.clamp(min_cx, max_cx);

        let badge_rect =
            Rect::from_center_size(Pos2::new(center_x, center_y), Vec2::new(badge_w, badge_h));
        let badge_radius = CornerRadius::same(8);
        painter.rect_filled(
            badge_rect.expand(2.0_f32),
            badge_radius,
            Color32::from_rgba_unmultiplied(0, 0, 0, 95),
        );
        painter.rect_filled(
            badge_rect,
            badge_radius,
            Color32::from_rgba_unmultiplied(15, 22, 35, 244),
        );
        painter.rect_stroke(
            badge_rect,
            badge_radius,
            Stroke::new(1.2_f32, col),
            StrokeKind::Inside,
        );
        let dot_pos = Pos2::new(badge_rect.left() + 10.0_f32, badge_rect.center().y);
        painter.circle_filled(dot_pos, 3.5_f32, col);
        painter.text(
            Pos2::new(badge_rect.left() + 18.0_f32, badge_rect.center().y),
            Align2::LEFT_CENTER,
            badge_str,
            font,
            Color32::from_rgb(241, 245, 249),
        );
    } else if !response.dragged() && hovered_tab.is_none() {
        response.clone().on_hover_text(format!(
            "Equalizzatore 3-Bande per Traccia Isolata: {}\n\
             • Trascina su/giù i pallini B (Bassi), M (Medi), A (Alti) per regolare -12 .. +12 dB\n\
             • Doppio clic o Tasto destro su un pallino per riportarlo a 0.0 dB\n\
             • Clicca 'Master' in alto a sinistra per tornare all'EQ a 8 Bande del brano",
            STEM_FULL_TITLES[stem_idx]
        ));
    }
}

fn draw_stem_three_band_controls(
    ui: &mut Ui,
    state: &mut SharedAudioState,
    eq_ui: &mut EqDrawerState,
    stem_idx: usize,
    right_w: f32,
) {
    ui.set_width(right_w);
    ui.spacing_mut().item_spacing.y = 0.0_f32;

    let stem_col = stem_accent_color(stem_idx);
    let is_flat = state.stem_channels[stem_idx].eq.is_flat();
    let is_solo = state.stem_channels[stem_idx].solo;

    // Riga superiore: ritorno al Master + preset tonali rapidi per traccia isolata + Flat + Solo
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(6.0_f32, 0.0_f32);
        let pill_min = Vec2::new(0.0_f32, 28.0_f32);

        if pill_button(
            ui,
            "← Master",
            false,
            Color32::from_rgb(14, 165, 233),
            pill_min,
        )
        .on_hover_text("Torna ai controlli Master EQ a 8 Bande dell'intero brano")
        .clicked()
        {
            eq_ui.active_target = EqDrawerTarget::MasterBus;
        }

        if pill_button(
            ui,
            "Spinta & Corpo",
            false,
            Color32::from_rgb(245, 158, 11),
            pill_min,
        )
        .on_hover_text("Dà più corpo e calore ai Bassi (+3.5 dB) riducendo leggermente l'effetto scatola")
        .clicked()
        {
            let (lo, mi, hi) = match stem_idx {
                1 => (4.0_f32, -2.0_f32, 2.0_f32), // Batteria: cassa piena + meno scatola + punta
                2 => (4.2_f32, 1.2_f32, 0.5_f32),  // Basso: sub profondo + definizione
                0 => (2.2_f32, 1.5_f32, 1.5_f32),  // Voce: corpo caldo + presenza
                _ => (3.0_f32, 1.5_f32, 1.0_f32),  // Chitarre: spessore + grinta
            };
            let eq = &mut state.stem_channels[stem_idx].eq;
            eq.low_db = lo;
            eq.mid_db = mi;
            eq.high_db = hi;
            state.mode = PlaybackSourceMode::FourStemMixer;
        }

        if pill_button(
            ui,
            "Chiarezza",
            false,
            Color32::from_rgb(6, 182, 212),
            pill_min,
        )
        .on_hover_text("Toglie l'effetto ovattato scavando le medie (-2.5 dB) e aprendo gli Alti (+3.8 dB)")
        .clicked()
        {
            let (lo, mi, hi) = match stem_idx {
                1 => (2.0_f32, -2.8_f32, 4.0_f32),
                2 => (1.0_f32, 2.5_f32, 3.0_f32),
                0 => (-1.2_f32, 2.2_f32, 3.8_f32),
                _ => (-1.0_f32, 1.8_f32, 3.6_f32),
            };
            let eq = &mut state.stem_channels[stem_idx].eq;
            eq.low_db = lo;
            eq.mid_db = mi;
            eq.high_db = hi;
            state.mode = PlaybackSourceMode::FourStemMixer;
        }

        if pill_button(
            ui,
            "Flat (0 dB)",
            is_flat,
            Color32::from_rgb(100, 116, 139),
            pill_min,
        )
        .on_hover_text("Riporta Bassi, Medi e Alti di questa traccia isolata esattamente a 0.0 dB")
        .clicked()
        {
            state.stem_channels[stem_idx].eq.reset();
            state.mode = PlaybackSourceMode::FourStemMixer;
        }

        if pill_button(
            ui,
            "Solo",
            is_solo,
            Color32::from_rgb(245, 158, 11),
            pill_min,
        )
        .on_hover_text(format!(
            "Ascolta in isolamento solo {} mentre regoli Bassi, Medi e Alti",
            STEM_FULL_TITLES[stem_idx]
        ))
        .clicked()
        {
            let ch = &mut state.stem_channels[stem_idx];
            ch.solo = !ch.solo;
            if ch.solo {
                ch.muted = false;
            }
            state.mode = PlaybackSourceMode::FourStemMixer;
        }
    });

    ui.add_space(5.0_f32);

    // 4 Manopole Hardware da Studio dedicate alla traccia isolata: BASSI | MEDI | ALTI | VOLUME
    ui.spacing_mut().item_spacing.x = 6.0_f32;
    let using_ptr = ui.ctx().is_using_pointer();
    let (sub_low, sub_mid, sub_high, sub_vol) = stem_knob_subtitles(stem_idx);

    ui.columns(4, |cols| {
        for col in cols.iter_mut() {
            col.spacing_mut().item_spacing = Vec2::new(4.0_f32, 1.5_f32);
        }

        let low_lbl = format!("{:+.1} dB", state.stem_channels[stem_idx].eq.low_db);
        let k0 = cols[0].vertical(|ui| {
            studio_knob(
                ui,
                "BASSI",
                sub_low,
                &mut state.stem_channels[stem_idx].eq.low_db,
                STEM_EQ_MIN_DB,
                STEM_EQ_MAX_DB,
                0.0_f32,
                &low_lbl,
                Color32::from_rgb(245, 158, 11),
            )
        });
        if k0.inner {
            eq_ui.selected_stem_band = 0;
            state.mode = PlaybackSourceMode::FourStemMixer;
        }
        if !using_ptr {
            k0.response.on_hover_text(format!(
                "BASSI — {} ({})\nRegola basse frequenze e corpo (-12 .. +12 dB)\n• Doppio clic per 0.0 dB",
                STEM_FULL_TITLES[stem_idx], sub_low
            ));
        }

        let mid_lbl = format!("{:+.1} dB", state.stem_channels[stem_idx].eq.mid_db);
        let k1 = cols[1].vertical(|ui| {
            studio_knob(
                ui,
                "MEDI",
                sub_mid,
                &mut state.stem_channels[stem_idx].eq.mid_db,
                STEM_EQ_MIN_DB,
                STEM_EQ_MAX_DB,
                0.0_f32,
                &mid_lbl,
                Color32::from_rgb(34, 211, 238),
            )
        });
        if k1.inner {
            eq_ui.selected_stem_band = 1;
            state.mode = PlaybackSourceMode::FourStemMixer;
        }
        if !using_ptr {
            k1.response.on_hover_text(format!(
                "MEDI — {} ({})\nRegola presenza sulle medie o scava l'effetto scatola (-12 .. +12 dB)\n• Doppio clic per 0.0 dB",
                STEM_FULL_TITLES[stem_idx], sub_mid
            ));
        }

        let high_lbl = format!("{:+.1} dB", state.stem_channels[stem_idx].eq.high_db);
        let k2 = cols[2].vertical(|ui| {
            studio_knob(
                ui,
                "ALTI",
                sub_high,
                &mut state.stem_channels[stem_idx].eq.high_db,
                STEM_EQ_MIN_DB,
                STEM_EQ_MAX_DB,
                0.0_f32,
                &high_lbl,
                Color32::from_rgb(192, 132, 252),
            )
        });
        if k2.inner {
            eq_ui.selected_stem_band = 2;
            state.mode = PlaybackSourceMode::FourStemMixer;
        }
        if !using_ptr {
            k2.response.on_hover_text(format!(
                "ALTI — {} ({})\nRegola brillantezza, attacco e aria (-12 .. +12 dB)\n• Doppio clic per 0.0 dB",
                STEM_FULL_TITLES[stem_idx], sub_high
            ));
        }

        let vol_lbl = format!("{:+.1} dB", state.stem_channels[stem_idx].gain_db);
        let k3 = cols[3].vertical(|ui| {
            studio_knob(
                ui,
                "VOLUME",
                sub_vol,
                &mut state.stem_channels[stem_idx].gain_db,
                -24.0_f32,
                8.0_f32,
                0.0_f32,
                &vol_lbl,
                stem_col,
            )
        });
        if k3.inner {
            state.mode = PlaybackSourceMode::FourStemMixer;
        }
        if !using_ptr {
            k3.response.on_hover_text(format!(
                "VOLUME — {}\nRegola il volume della traccia isolata (-24 .. +8 dB)\n• Doppio clic per 0.0 dB",
                STEM_FULL_TITLES[stem_idx]
            ));
        }
    });
}

fn draw_smart_master_controls(
    ui: &mut Ui,
    state: &mut SharedAudioState,
    eq_ui: &mut EqDrawerState,
    right_w: f32,
) {
    ui.set_width(right_w);
    ui.spacing_mut().item_spacing.y = 0.0_f32;

    // Riga superiore compatta con 4 pillole Preset + eventuale indicatore Bypass
    let active_preset = detect_active_preset(&state.params);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(6.0_f32, 0.0_f32);
        let pill_min = Vec2::new(0.0_f32, 28.0_f32);

        if pill_button(
            ui,
            "Sala Prove Fix",
            active_preset == Some(0),
            Color32::from_rgb(14, 165, 233),
            pill_min,
        )
        .on_hover_text(
            "Preset consigliato per registrazioni in sala prove:\nriduce il rimbombo dei bassi e l'effetto scatola, portando avanti la voce",
        )
        .clicked()
        {
            state.params = DspParams::preset_sala_prove();
            state.params.bypass = false;
            state.params_dirty = true;
            eq_ui.macro_clean = 75.0_f32;
            eq_ui.macro_vocal = 65.0_f32;
            eq_ui.macro_air = 70.0_f32;
            eq_ui.macro_punch = 60.0_f32;
        }

        if pill_button(
            ui,
            "Voce Nitida",
            active_preset == Some(1),
            Color32::from_rgb(6, 182, 212),
            pill_min,
        )
        .on_hover_text(
            "Mette in primo piano la voce e la chiarezza delle parole scavando le frequenze medie affollate",
        )
        .clicked()
        {
            state.params = DspParams::preset_voce_primo_piano();
            state.params.bypass = false;
            state.params_dirty = true;
            eq_ui.macro_clean = 82.0_f32;
            eq_ui.macro_vocal = 82.0_f32;
            eq_ui.macro_air = 78.0_f32;
            eq_ui.macro_punch = 60.0_f32;
        }

        if pill_button(
            ui,
            "Rock Punch",
            active_preset == Some(2),
            Color32::from_rgb(245, 158, 11),
            pill_min,
        )
        .on_hover_text(
            "Dà spinta e botta a cassa e basso (68 Hz), grinta alle chitarre e coesione sul Master",
        )
        .clicked()
        {
            state.params = DspParams::preset_rock_punch();
            state.params.bypass = false;
            state.params_dirty = true;
            eq_ui.macro_clean = 68.0_f32;
            eq_ui.macro_vocal = 58.0_f32;
            eq_ui.macro_air = 70.0_f32;
            eq_ui.macro_punch = 78.0_f32;
        }

        if pill_button(
            ui,
            "Flat (0 dB)",
            active_preset == Some(3),
            Color32::from_rgb(100, 116, 139),
            pill_min,
        )
        .on_hover_text(
            "Azzera tutte le bande dell'equalizzatore e le 4 manopole (suono neutro a 0 dB)",
        )
        .clicked()
        {
            state.params = DspParams::preset_flat();
            state.params.bypass = false;
            state.params_dirty = true;
            eq_ui.macro_clean = 0.0_f32;
            eq_ui.macro_vocal = 0.0_f32;
            eq_ui.macro_air = 0.0_f32;
            eq_ui.macro_punch = 0.0_f32;
        }

        if state.params.bypass
            && pill_button(
                ui,
                "EQ Off",
                true,
                Color32::from_rgb(245, 158, 11),
                pill_min,
            )
            .on_hover_text(
                "L'Equalizzatore è in Bypass (stai ascoltando l'audio Originale).\nClicca qui o premi B per riattivare l'EQ.",
            )
            .clicked()
        {
            state.params.bypass = false;
            state.params_dirty = true;
        }
    });

    ui.add_space(5.0_f32);

    // 4 Colonne con Manopole Smart Studio (`studio_knob` da `ui_widgets.rs`)
    ui.spacing_mut().item_spacing.x = 6.0_f32;
    let using_ptr = ui.ctx().is_using_pointer();

    ui.columns(4, |cols| {
        for col in cols.iter_mut() {
            col.spacing_mut().item_spacing = Vec2::new(4.0_f32, 1.5_f32);
        }

        // 1. PULIZIA SALA (Anti-Rimbombo) -> scala bands[0..4] + dynamic_mud_amount
        let clean_lbl = format!("{:.0}%", eq_ui.macro_clean);
        let k0 = cols[0].vertical(|ui| {
            studio_knob(
                ui,
                "PULIZIA SALA",
                "Anti-Rimbombo",
                &mut eq_ui.macro_clean,
                0.0_f32,
                100.0_f32,
                75.0_f32,
                &clean_lbl,
                Color32::from_rgb(245, 158, 11),
            )
        });
        if k0.inner {
            let ratio = (eq_ui.macro_clean as f64) / 75.0_f64;
            if state.params.bands.len() >= 4 {
                let base_cuts = [-3.2_f64, -6.2_f64, -3.0_f64, -5.2_f64];
                if state.params.bands[0].freq < 85.0_f64 {
                    state.params.bands[0].freq = 113.0_f64;
                    state.params.bands[0].q = 2.6_f64;
                }
                for (i, &base_db) in base_cuts.iter().enumerate() {
                    state.params.bands[i].gain_db = (base_db * ratio).clamp(-12.0_f64, 0.0_f64);
                    state.params.bands[i].enabled = true;
                }
            }
            state.params.sub_cut_hz = (20.0_f64 + 25.0_f64 * ratio).clamp(20.0_f64, 65.0_f64);
            state.params.dynamic_mud_amount = (1.0_f64 * ratio).clamp(0.0_f64, 1.5_f64);
            state.params.dynamic_mud_enabled = state.params.dynamic_mud_amount > 0.02_f64;
            state.params.bypass = false;
            state.params_dirty = true;
        }
        if !using_ptr {
            k0.response.on_hover_text(
                "Pulizia Sala (Anti-Rimbombo)\n\
                 Asciuga il rimbombo dei muri e il fango sui medio-bassi (110–500 Hz).\n\
                 • Trascina su/giù per regolare (Shift = precisione)\n\
                 • Doppio clic sulla manopola per tornare al 75%",
            );
        }

        // 2. VOCE & MEDIE (Presenza) -> regola boost su 1.42 kHz (bands[4]) e 4.8 kHz (bands[6])
        let vocal_lbl = format!("{:.0}%", eq_ui.macro_vocal);
        let k1 = cols[1].vertical(|ui| {
            studio_knob(
                ui,
                "VOCE & MEDIE",
                "Presenza",
                &mut eq_ui.macro_vocal,
                0.0_f32,
                100.0_f32,
                65.0_f32,
                &vocal_lbl,
                Color32::from_rgb(34, 211, 238),
            )
        });
        if k1.inner {
            let ratio = (eq_ui.macro_vocal as f64) / 65.0_f64;
            if state.params.bands.len() >= 7 {
                state.params.bands[4].gain_db = (5.4_f64 * ratio).clamp(0.0_f64, 10.5_f64);
                state.params.bands[4].enabled = true;
                state.params.bands[6].gain_db = (4.8_f64 * ratio).clamp(0.0_f64, 9.5_f64);
                state.params.bands[6].enabled = true;
            }
            state.params.bypass = false;
            state.params_dirty = true;
        }
        if !using_ptr {
            k1.response.on_hover_text(
                "Voce & Medie (Presenza)\n\
                 Porta avanti la voce, il rullante e l'attacco degli strumenti (1.4 kHz e 4.8 kHz).\n\
                 • Trascina su/giù per regolare (Shift = precisione)\n\
                 • Doppio clic sulla manopola per tornare al 65%",
            );
        }

        // 3. ARIA & ALTI (Brillantezza) -> regola 8.2 kHz shelf (bands[7]) e exciter_drive
        let air_lbl = format!("{:.0}%", eq_ui.macro_air);
        let k2 = cols[2].vertical(|ui| {
            studio_knob(
                ui,
                "ARIA & ALTI",
                "Brillantezza",
                &mut eq_ui.macro_air,
                0.0_f32,
                100.0_f32,
                70.0_f32,
                &air_lbl,
                Color32::from_rgb(192, 132, 252),
            )
        });
        if k2.inner {
            let ratio = (eq_ui.macro_air as f64) / 70.0_f64;
            if state.params.bands.len() >= 8 {
                state.params.bands[7].gain_db = (7.8_f64 * ratio).clamp(0.0_f64, 12.0_f64);
                state.params.bands[7].enabled = true;
            }
            state.params.exciter_drive = (0.22_f64 * ratio).clamp(0.0_f64, 0.50_f64);
            state.params.exciter_enabled = state.params.exciter_drive > 0.005_f64;
            state.params.bypass = false;
            state.params_dirty = true;
        }
        if !using_ptr {
            k2.response.on_hover_text(
                "Aria & Alti (Brillantezza)\n\
                 Apre le alte frequenze (> 8 kHz) per dare respiro ai piatti e limpidezza da studio.\n\
                 • Trascina su/giù per regolare (Shift = precisione)\n\
                 • Doppio clic sulla manopola per tornare al 70%",
            );
        }

        // 4. SPINTA MASTER (Glue & Stereo) -> regola comp_drive_db e stereo_width
        let punch_lbl = format!("{:.0}%", eq_ui.macro_punch);
        let k3 = cols[3].vertical(|ui| {
            studio_knob(
                ui,
                "SPINTA MASTER",
                "Glue & Stereo",
                &mut eq_ui.macro_punch,
                0.0_f32,
                100.0_f32,
                60.0_f32,
                &punch_lbl,
                Color32::from_rgb(52, 211, 153),
            )
        });
        if k3.inner {
            let ratio = (eq_ui.macro_punch as f64) / 60.0_f64;
            state.params.comp_drive_db = (3.8_f64 * ratio).clamp(0.0_f64, 9.0_f64);
            state.params.comp_enabled = state.params.comp_drive_db > 0.05_f64;
            state.params.stereo_width = (1.0_f64 + 0.22_f64 * ratio).clamp(1.0_f64, 1.55_f64);
            state.params.mid_side_enabled = true;
            state.params.bypass = false;
            state.params_dirty = true;
        }
        if !using_ptr {
            k3.response.on_hover_text(format!(
                "Spinta Master (Volume, Coesione & Apertura Stereo)\n\
                 Compatta la band insieme ed espande l'immagine stereo senza mai andare in distorsione.\n\
                 Attuale: Drive {:+.1} dB  •  Ampiezza Stereo {:.0}%\n\
                 • Trascina su/giù per regolare (Shift = precisione)\n\
                 • Doppio clic sulla manopola per tornare al 60%",
                state.params.comp_drive_db,
                state.params.stereo_width * 100.0_f64
            ));
        }
    });
}
