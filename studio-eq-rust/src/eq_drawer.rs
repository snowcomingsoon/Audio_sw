use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Ui, Vec2,
};

use crate::audio_io::SharedAudioState;
use crate::dsp::{total_curve_db, BiquadCoeffs, DspParams};
use crate::ui_widgets::{draw_filled_curve_area, pill_button, studio_knob};

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

#[derive(Clone, Debug)]
pub struct EqDrawerState {
    pub dragged_band: Option<usize>,
    pub selected_band: usize,
    pub macro_clean: f32,
    pub macro_vocal: f32,
    pub macro_air: f32,
    pub macro_punch: f32,
}

impl Default for EqDrawerState {
    fn default() -> Self {
        Self {
            dragged_band: None,
            selected_band: 0,
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
    let total_w = ui.available_width().max(620.0_f32);
    let gap = 12.0_f32;
    let drawer_h = 204.0_f32;
    let usable_w = (total_w - gap).max(600.0_f32);
    // Garantisce almeno 508px alle 4 manopole (4x122 + 3x6 = 506px) così SPINTA MASTER non viene mai tagliata a 1060px
    let right_w = (usable_w * 0.45_f32)
        .floor()
        .max(508.0_f32)
        .min((usable_w - 340.0_f32).max(280.0_f32));
    let left_w = (usable_w - right_w).max(320.0_f32);

    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = gap;

        // =========================================================================
        // SEZIONE SINISTRA: Grafico Spettro FFT + Curva Parametric EQ Pro-Q 4
        // =========================================================================
        ui.allocate_ui_with_layout(
            Vec2::new(left_w, drawer_h),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                draw_pro_eq_canvas(ui, state, eq_ui, spectrum_db, Vec2::new(left_w, drawer_h));
            },
        );

        // =========================================================================
        // SEZIONE DESTRA: Pillole Preset + 4 Manopole Smart Studio
        // =========================================================================
        ui.allocate_ui_with_layout(
            Vec2::new(right_w, drawer_h),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                draw_smart_master_controls(ui, state, eq_ui, right_w);
            },
        );
    });
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
        if db_mark != -12.0_f64 {
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

        // Lieve smoothing spaziale 3-tap per una silhouette FFT setosa stile Pro-Q 4
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

    // 3. Interazione Mouse con gli 8 Nodi (Hover, Click, Drag, Double-Click 0 dB, Wheel Q)
    if eq_ui.dragged_band.is_some() && !ui.input(|i| i.pointer.primary_down()) {
        eq_ui.dragged_band = None;
    }

    let pointer_pos = response
        .hover_pos()
        .or_else(|| response.interact_pointer_pos());
    let mut hovered_band: Option<usize> = None;
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
            if let (Some(idx), Some(pos)) = (eq_ui.dragged_band, response.interact_pointer_pos()) {
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

    // Regolazione fattore di merito Q con la rotellina del mouse sopra un nodo (0.4 ..= 8.0)
    let active_node = eq_ui.dragged_band.or(hovered_band);
    if response.hovered() {
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
        // Alone morbido + curva nitida 2.5_f32
        painter.add(egui::Shape::line(
            curve_pts.clone(),
            Stroke::new(5.5_f32, Color32::from_rgba_unmultiplied(34, 211, 238, 28)),
        ));
        painter.add(egui::Shape::line(
            curve_pts,
            Stroke::new(2.5_f32, Color32::from_rgb(56, 217, 245)),
        ));
    }

    // Intestazione chiara in alto a sinistra + lettura banda selezionata a destra senza sovrapposizioni
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
    let hdr_left_end = inner_rect.left() + 10.0_f32 + hdr_galley.size().x;
    painter.text(
        Pos2::new(inner_rect.left() + 10.0_f32, inner_rect.top() + 7.5_f32),
        Align2::LEFT_TOP,
        header_txt,
        hdr_font,
        header_col,
    );

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
        if hdr_left_end + sel_galley.size().x + 18.0_f32 <= inner_rect.right() - 8.0_f32 {
            painter.text(
                Pos2::new(inner_rect.right() - 9.0_f32, inner_rect.top() + 7.5_f32),
                Align2::RIGHT_TOP,
                sel_str,
                sel_font,
                sel_col,
            );
        }
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

            let prefer_above = np.y - 24.0_f32 - badge_h >= inner_rect.top() + 20.0_f32;
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
    } else if !response.dragged() {
        response.on_hover_text(
            "Equalizzatore Parametrico a 8 Bande (64-bit)\n\
             • Trascina i pallini numerati (1–8) per cambiare Frequenza (Hz) e Guadagno (dB)\n\
             • Usa la rotellina sopra un pallino per stringere o allargare la campana (Q)\n\
             • Doppio clic su un pallino per riportarlo a 0.0 dB",
        );
    }

    // Bordo nitido finale della Card Ossidiana
    outer_painter.rect_stroke(
        rect,
        card_radius,
        Stroke::new(1.0_f32, Color32::from_rgb(30, 41, 62)),
        StrokeKind::Inside,
    );
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
