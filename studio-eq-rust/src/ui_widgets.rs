use std::f32::consts::PI;
use eframe::egui::{
    self, epaint::Mesh, Align2, Color32, CornerRadius, FontId, Pos2, Rect, Response, Sense,
    Stroke, StrokeKind, Ui, Vec2,
};

/// Palette cromatica calibrata Pro Audio 2026 (ispirata a Logic Pro 11, FabFilter Pro-Q 4, Moises.ai).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct StudioPalette;

#[allow(dead_code)]
impl StudioPalette {
    /// `#090C12` — Sfondo principale ultra-profondo ad alto contrasto.
    pub const DEEP_OBSIDIAN: Color32 = Color32::from_rgb(9, 12, 18);
    /// `#111622` — Superficie rialzata per rack, moduli e toolbar.
    pub const ELEVATED_SURFACE: Color32 = Color32::from_rgb(17, 22, 34);
    /// `#161D2C` — Superficie satinata "glass/hardware" per card e controlli.
    pub const CARD_GLASS: Color32 = Color32::from_rgb(22, 29, 44);
    /// `#243046` — Bordo sottile calibrato per separazione strutturale.
    pub const SUBTLE_HAIRLINE: Color32 = Color32::from_rgb(36, 48, 70);
    /// `#0EA5E9` — Accento primario ciano elettrico.
    pub const ELECTRIC_CYAN: Color32 = Color32::from_rgb(14, 165, 233);
    /// `#8B5CF6` — Accento secondario viola da studio (AI / Stem / FX).
    pub const STUDIO_VIOLET: Color32 = Color32::from_rgb(139, 92, 246);
    /// `#F59E0B` — Accento caldo ambra analogico (Warmth / Presence / Solo).
    pub const WARM_AMBER: Color32 = Color32::from_rgb(245, 158, 11);
    /// `#10B981` — Accento smeraldo (Play / Safe / Active DSP).
    pub const EMERALD: Color32 = Color32::from_rgb(16, 185, 129);
    /// `#F8FAFC` — Testo principale nitido ad altissima leggibilità.
    pub const CRISP_TEXT: Color32 = Color32::from_rgb(248, 250, 252);
    /// `#94A3B8` — Etichette secondarie e parametri ausiliari.
    pub const MUTED_LABEL: Color32 = Color32::from_rgb(148, 163, 184);
}

/// Pulsante moderno a pillola in stile Apple / Pro Audio 2026 con micro-ombra morbida,
/// highlight speculare superiore "glass/hardware" e transizione hover fluida.
pub fn pill_button(
    ui: &mut Ui,
    label: &str,
    filled: bool,
    accent: Color32,
    min_size: Vec2,
) -> Response {
    let font = FontId::proportional(13.5_f32);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font.clone(), StudioPalette::CRISP_TEXT);
    let size = Vec2::new(
        (galley.size().x + 28.0_f32).max(min_size.x),
        (galley.size().y + 14.0_f32).max(min_size.y),
    );

    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    if ui.is_rect_visible(rect) {
        let hover_t = ui
            .ctx()
            .animate_bool_responsive(resp.id.with("pill_hover"), resp.hovered());
        let pressed = resp.is_pointer_button_down_on();
        let press_offset = if pressed { 0.6_f32 } else { 0.0_f32 };
        let draw_rect = rect.translate(Vec2::new(0.0_f32, press_offset));

        let radius_u8 = (draw_rect.height() * 0.5_f32)
            .round()
            .clamp(2.0_f32, 120.0_f32) as u8;
        let radius = CornerRadius::same(radius_u8);
        let painter = ui.painter();

        // 1. Micro-ombra sottostante morbida a doppio livello (profondità hardware)
        if !pressed {
            painter.rect_filled(
                draw_rect
                    .translate(Vec2::new(0.0_f32, 2.4_f32))
                    .expand(0.8_f32),
                radius,
                Color32::from_rgba_unmultiplied(0, 0, 0, 38),
            );
            painter.rect_filled(
                draw_rect.translate(Vec2::new(0.0_f32, 1.2_f32)),
                radius,
                Color32::from_rgba_unmultiplied(0, 0, 0, 64),
            );
        }

        // 2. Alone luminoso calibrato (attivo su filled, accennato in hover su glass)
        if filled {
            let glow_alpha = (28.0_f32 + 28.0_f32 * hover_t).round() as u8;
            painter.rect_filled(
                draw_rect.expand(2.2_f32 + 0.8_f32 * hover_t),
                radius,
                with_alpha(accent, glow_alpha),
            );
        } else if hover_t > 0.01_f32 {
            let glow_alpha = (18.0_f32 * hover_t).round() as u8;
            painter.rect_filled(
                draw_rect.expand(1.5_f32),
                radius,
                with_alpha(accent, glow_alpha),
            );
        }

        // 3. Corpo principale con transizione fluida
        let bg_rest = if filled {
            accent
        } else {
            StudioPalette::CARD_GLASS
        };
        let bg_hover = if filled {
            brighten(accent, 20)
        } else {
            Color32::from_rgb(31, 41, 62)
        };
        let mut bg = lerp_color(bg_rest, bg_hover, hover_t);
        if pressed {
            bg = lerp_color(bg, StudioPalette::DEEP_OBSIDIAN, 0.16_f32);
        }
        painter.rect_filled(draw_rect, radius, bg);

        // 4. Bordo perimetrale calibrato
        let border_rest = if filled {
            brighten(accent, 30)
        } else {
            StudioPalette::SUBTLE_HAIRLINE
        };
        let border_hover = if filled {
            brighten(accent, 52)
        } else {
            lerp_color(StudioPalette::SUBTLE_HAIRLINE, accent, 0.58_f32)
        };
        let border_col = lerp_color(border_rest, border_hover, hover_t);
        painter.rect_stroke(
            draw_rect,
            radius,
            Stroke::new(1.0_f32, border_col),
            StrokeKind::Inside,
        );

        // 5. Highlight speculare superiore sottile (1px bianco al 10% alpha sul bordo top)
        let spec_alpha = (26.0_f32 + 10.0_f32 * hover_t).round() as u8; // 26/255 ≈ 10.2%
        let top_band = Rect::from_min_max(
            draw_rect.min,
            Pos2::new(draw_rect.max.x, draw_rect.min.y + 2.5_f32),
        );
        let spec_clip = ui.clip_rect().intersect(top_band);
        if spec_clip.is_positive() {
            painter.with_clip_rect(spec_clip).rect_stroke(
                draw_rect.shrink(0.8_f32),
                radius,
                Stroke::new(
                    1.0_f32,
                    Color32::from_rgba_unmultiplied(255, 255, 255, spec_alpha),
                ),
                StrokeKind::Inside,
            );
        }

        // 6. Tipografia nitida con micro-contrasto
        let text_col = if filled {
            StudioPalette::CRISP_TEXT
        } else {
            lerp_color(
                Color32::from_rgb(208, 218, 232),
                StudioPalette::CRISP_TEXT,
                hover_t,
            )
        };

        if filled {
            painter.text(
                draw_rect.center() + Vec2::new(0.0_f32, 0.8_f32),
                Align2::CENTER_CENTER,
                label,
                font.clone(),
                Color32::from_rgba_unmultiplied(0, 0, 0, 52),
            );
        }

        painter.text(
            draw_rect.center(),
            Align2::CENTER_CENTER,
            label,
            font,
            text_col,
        );
    }

    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

/// Pulsante circolare Master Transport Play / Pausa con anello esterno satinato,
/// ombra morbida, bagliore neon calibrato in riproduzione e icone vettoriali bilanciate.
pub fn transport_play_button(ui: &mut Ui, is_playing: bool, accent: Color32) -> Response {
    let size = Vec2::splat(48.0_f32);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    if ui.is_rect_visible(rect) {
        let hover_t = ui
            .ctx()
            .animate_bool_responsive(resp.id.with("transport_hover"), resp.hovered());
        let play_t = ui
            .ctx()
            .animate_bool_responsive(resp.id.with("transport_playing"), is_playing);
        let pressed = resp.is_pointer_button_down_on();

        let center = rect.center();
        let bezel_r = 22.0_f32;
        let core_r = 18.0_f32 - if pressed { 0.5_f32 } else { 0.0_f32 };
        let painter = ui.painter();

        // 1. Ombra morbida sottostante
        painter.circle_filled(
            center + Vec2::new(0.0_f32, 2.8_f32),
            bezel_r + 1.2_f32,
            Color32::from_rgba_unmultiplied(0, 0, 0, 56),
        );
        painter.circle_filled(
            center + Vec2::new(0.0_f32, 1.4_f32),
            bezel_r,
            Color32::from_rgba_unmultiplied(0, 0, 0, 82),
        );

        // 2. Bagliore neon calibrato (intenso quando is_playing == true, delicato in hover)
        if play_t > 0.01_f32 || hover_t > 0.01_f32 {
            let outer_glow_a = (38.0_f32 * play_t + 16.0_f32 * hover_t)
                .clamp(0.0_f32, 90.0_f32)
                .round() as u8;
            let inner_glow_a = (64.0_f32 * play_t + 24.0_f32 * hover_t)
                .clamp(0.0_f32, 120.0_f32)
                .round() as u8;
            painter.circle_filled(center, bezel_r + 4.2_f32, with_alpha(accent, outer_glow_a));
            painter.circle_filled(center, bezel_r + 1.8_f32, with_alpha(accent, inner_glow_a));
        }

        // 3. Anello esterno satinato (chassis hardware incassato)
        painter.circle_filled(center, bezel_r, StudioPalette::ELEVATED_SURFACE);
        let bezel_border = lerp_color(
            StudioPalette::SUBTLE_HAIRLINE,
            with_alpha(brighten(accent, 24), 210),
            (play_t * 0.72_f32 + hover_t * 0.28_f32).clamp(0.0_f32, 1.0_f32),
        );
        painter.circle_stroke(center, bezel_r, Stroke::new(1.2_f32, bezel_border));

        // Scanalatura scura tra ghiera satinata e nucleo attivo
        painter.circle_stroke(
            center,
            core_r + 1.1_f32,
            Stroke::new(1.2_f32, StudioPalette::DEEP_OBSIDIAN),
        );

        // 4. Nucleo attivo luminoso + bisello speculare superiore
        let core_base = lerp_color(accent, brighten(accent, 14), play_t * 0.45_f32);
        let mut core_fill = lerp_color(core_base, brighten(accent, 28), hover_t);
        if pressed {
            core_fill = lerp_color(core_fill, StudioPalette::DEEP_OBSIDIAN, 0.15_f32);
        }
        painter.circle_filled(center, core_r, core_fill);
        painter.circle_stroke(
            center,
            core_r,
            Stroke::new(1.0_f32, brighten(accent, 38)),
        );

        // Riflesso speculare sulla calotta superiore
        draw_arc(
            painter,
            center,
            core_r - 1.0_f32,
            -0.80_f32 * PI,
            -0.20_f32 * PI,
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 52)),
        );

        // 5. Icone Play / Pausa proporzionate al pixel con micro-ombra
        let icon_shadow = Color32::from_rgba_unmultiplied(0, 0, 0, 58);
        let shadow_off = Vec2::new(0.0_f32, 1.0_f32);

        if is_playing {
            let bar_w = 3.8_f32;
            let bar_h = 13.6_f32;
            let half_gap = 3.8_f32;
            let bar_radius = CornerRadius::same(2);

            let r1 = Rect::from_center_size(
                Pos2::new(center.x - half_gap, center.y),
                Vec2::new(bar_w, bar_h),
            );
            let r2 = Rect::from_center_size(
                Pos2::new(center.x + half_gap, center.y),
                Vec2::new(bar_w, bar_h),
            );

            painter.rect_filled(r1.translate(shadow_off), bar_radius, icon_shadow);
            painter.rect_filled(r2.translate(shadow_off), bar_radius, icon_shadow);
            painter.rect_filled(r1, bar_radius, StudioPalette::CRISP_TEXT);
            painter.rect_filled(r2, bar_radius, StudioPalette::CRISP_TEXT);
        } else {
            // Baricentro ottico centrato (+1.4px a destra del centro geometrico) con vertici morbidi
            let pts_shadow = vec![
                Pos2::new(center.x - 3.6_f32, center.y - 6.2_f32 + 1.0_f32),
                Pos2::new(center.x - 3.6_f32, center.y + 6.2_f32 + 1.0_f32),
                Pos2::new(center.x + 6.8_f32, center.y + 1.0_f32),
            ];
            painter.add(egui::Shape::convex_polygon(
                pts_shadow,
                icon_shadow,
                Stroke::new(1.4_f32, icon_shadow),
            ));

            let pts = vec![
                Pos2::new(center.x - 3.6_f32, center.y - 6.2_f32),
                Pos2::new(center.x - 3.6_f32, center.y + 6.2_f32),
                Pos2::new(center.x + 6.8_f32, center.y),
            ];
            painter.add(egui::Shape::convex_polygon(
                pts,
                StudioPalette::CRISP_TEXT,
                Stroke::new(1.4_f32, StudioPalette::CRISP_TEXT),
            ));
        }
    }

    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

/// Barra di avanzamento / Fader moderno con scanalatura incassata scura,
/// riempimento luminoso con alone morbido e cursore satinato con punto centrale colorato.
pub fn sleek_progress_or_fader(
    ui: &mut Ui,
    value: &mut f32,
    min: f32,
    max: f32,
    accent: Color32,
    desired_width: f32,
    height: f32,
) -> bool {
    let alloc = Vec2::new(desired_width.max(40.0_f32), height.max(15.0_f32));
    let (rect, resp) = ui.allocate_exact_size(alloc, Sense::click_and_drag());
    let mut changed = false;

    if (resp.clicked() || resp.dragged()) && max > min {
        if let Some(pos) = resp.interact_pointer_pos() {
            let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0_f32, 1.0_f32);
            *value = min + t * (max - min);
            changed = true;
        }
    }

    if ui.is_rect_visible(rect) {
        let active_t = ui.ctx().animate_bool_responsive(
            resp.id.with("fader_active"),
            resp.hovered() || resp.dragged(),
        );
        let painter = ui.painter();

        let track_h = 6.0_f32 + 0.8_f32 * active_t;
        let track_radius = CornerRadius::same(3);
        let track_rect = Rect::from_min_max(
            Pos2::new(rect.left(), rect.center().y - track_h * 0.5_f32),
            Pos2::new(rect.right(), rect.center().y + track_h * 0.5_f32),
        );

        // 1. Scanalatura incassata scura con bordo sottile
        painter.rect_filled(track_rect, track_radius, StudioPalette::DEEP_OBSIDIAN);
        let groove_border = lerp_color(
            StudioPalette::SUBTLE_HAIRLINE,
            Color32::from_rgb(52, 68, 98),
            active_t,
        );
        painter.rect_stroke(
            track_rect,
            track_radius,
            Stroke::new(1.0_f32, groove_border),
            StrokeKind::Outside,
        );

        // Ombra interna superiore nella scanalatura per effetto incassato
        painter.line_segment(
            [
                Pos2::new(track_rect.left() + 2.0_f32, track_rect.top() + 0.8_f32),
                Pos2::new(track_rect.right() - 2.0_f32, track_rect.top() + 0.8_f32),
            ],
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(0, 0, 0, 90)),
        );

        let t = if max > min {
            ((*value - min) / (max - min)).clamp(0.0_f32, 1.0_f32)
        } else {
            0.0_f32
        };
        let fill_x = rect.left() + t * rect.width();

        // 2. Riempimento luminoso con alone morbido
        if fill_x > track_rect.left() + 0.5_f32 {
            let fill_rect =
                Rect::from_min_max(track_rect.min, Pos2::new(fill_x, track_rect.max.y));
            let glow_alpha = (28.0_f32 + 24.0_f32 * active_t).round() as u8;
            painter.rect_filled(
                fill_rect.expand(2.2_f32),
                CornerRadius::same(5),
                with_alpha(accent, glow_alpha),
            );

            let bar_col = lerp_color(accent, brighten(accent, 20), active_t * 0.65_f32);
            painter.rect_filled(fill_rect, track_radius, bar_col);

            // Riflesso speculare liquido sul riempimento
            if fill_rect.width() > 5.0_f32 {
                painter.line_segment(
                    [
                        Pos2::new(fill_rect.left() + 2.0_f32, fill_rect.top() + 1.1_f32),
                        Pos2::new(fill_rect.right() - 2.0_f32, fill_rect.top() + 1.1_f32),
                    ],
                    Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 48)),
                );
            }
        }

        // 3. Cursore satinato con ombra, alone e punto centrale colorato
        let base_r = if rect.height() < 18.0_f32 {
            6.0_f32
        } else {
            7.2_f32
        };
        let thumb_r = base_r + 1.1_f32 * active_t;
        let thumb_x = if rect.width() > thumb_r * 2.0_f32 {
            fill_x.clamp(rect.left() + thumb_r, rect.right() - thumb_r)
        } else {
            fill_x
        };
        let thumb_pos = Pos2::new(thumb_x, rect.center().y);

        // Ombra morbida sotto il cursore
        painter.circle_filled(
            thumb_pos + Vec2::new(0.0_f32, 1.8_f32),
            thumb_r + 1.0_f32,
            Color32::from_rgba_unmultiplied(0, 0, 0, 88),
        );

        // Alone cromatico reattivo
        let halo_alpha = (26.0_f32 + 40.0_f32 * active_t).round() as u8;
        painter.circle_filled(
            thumb_pos,
            thumb_r + 3.0_f32 + 1.4_f32 * active_t,
            with_alpha(accent, halo_alpha),
        );

        // Corpo metallico satinato del cursore
        let cap_col = lerp_color(
            Color32::from_rgb(228, 235, 246),
            StudioPalette::CRISP_TEXT,
            active_t,
        );
        painter.circle_filled(thumb_pos, thumb_r, cap_col);
        painter.circle_stroke(
            thumb_pos,
            thumb_r,
            Stroke::new(1.0_f32, StudioPalette::DEEP_OBSIDIAN),
        );
        painter.circle_stroke(
            thumb_pos,
            (thumb_r - 0.9_f32).max(1.0_f32),
            Stroke::new(1.0_f32, Color32::WHITE),
        );

        // Punto centrale colorato
        let dot_r = 2.7_f32 + 0.5_f32 * active_t;
        painter.circle_filled(thumb_pos, dot_r, accent);
    }

    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    changed
}

/// Manopola Rotativa da Studio Pro Audio 2026 (stile FabFilter Pro-Q 4 / Logic Pro 11 / Teenage Engineering).
/// Corona esterna a 21 micro-tacche graduate illuminate, ghiera metallica satinata a doppio anello,
/// tacca indicatrice luminosa allungata e valore tipografico centrale nitido.
/// Trascina su/giù per regolare (Shift per regolazione fine); doppio clic per `default_val`.
pub fn studio_knob(
    ui: &mut Ui,
    title: &str,
    subtitle: &str,
    value: &mut f32,
    min: f32,
    max: f32,
    default_val: f32,
    display_val: &str,
    accent: Color32,
) -> bool {
    let mut changed = false;

    egui::Frame::NONE
        .fill(StudioPalette::ELEVATED_SURFACE)
        .stroke(Stroke::new(1.0_f32, StudioPalette::SUBTLE_HAIRLINE))
        .corner_radius(CornerRadius::same(14))
        .inner_margin(14.0_f32)
        .show(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.label(
                    egui::RichText::new(title)
                        .size(13.0_f32)
                        .strong()
                        .color(StudioPalette::CRISP_TEXT),
                );
                ui.label(
                    egui::RichText::new(subtitle)
                        .size(11.2_f32)
                        .color(StudioPalette::MUTED_LABEL),
                );
                ui.add_space(6.0_f32);

                let knob_size = Vec2::splat(94.0_f32);
                let (rect, resp) = ui.allocate_exact_size(knob_size, Sense::click_and_drag());

                if resp.double_clicked() && max > min {
                    *value = default_val.clamp(min, max);
                    changed = true;
                } else if resp.dragged() && max > min {
                    let delta = -resp.drag_delta().y;
                    let fine = if ui.input(|i| i.modifiers.shift) {
                        0.25_f32
                    } else {
                        1.0_f32
                    };
                    let speed = ((max - min) / 150.0_f32) * fine;
                    *value = (*value + delta * speed).clamp(min, max);
                    changed = true;
                }

                if ui.is_rect_visible(rect) {
                    let active_t = ui.ctx().animate_bool_responsive(
                        resp.id.with("knob_hover"),
                        resp.hovered() || resp.dragged(),
                    );
                    let painter = ui.painter();
                    let c = rect.center();

                    // Escursione 270°: da 135° (basso-sinistra) a 405° (basso-destra)
                    let start_ang = 0.75_f32 * PI;
                    let sweep_ang = 1.50_f32 * PI;
                    let norm = if max > min {
                        ((*value - min) / (max - min)).clamp(0.0_f32, 1.0_f32)
                    } else {
                        0.0_f32
                    };
                    let end_ang = start_ang + norm * sweep_ang;

                    // 1. Corona esterna di 21 micro-tacche radiali graduate (0..=20)
                    let tick_count = 21usize;
                    let tick_r_in = 39.8_f32;
                    for i in 0..tick_count {
                        let t_tick = i as f32 / (tick_count - 1) as f32;
                        let ang = start_ang + t_tick * sweep_ang;
                        let is_major = i % 5 == 0;
                        let tick_r_out = if is_major { 45.2_f32 } else { 43.2_f32 };

                        let p_in = Pos2::new(c.x + tick_r_in * ang.cos(), c.y + tick_r_in * ang.sin());
                        let p_out =
                            Pos2::new(c.x + tick_r_out * ang.cos(), c.y + tick_r_out * ang.sin());

                        let is_lit = norm > 0.004_f32 && t_tick <= norm + 0.002_f32;
                        if is_lit {
                            let glow_a = (44.0_f32 + 32.0_f32 * active_t).round() as u8;
                            painter.line_segment(
                                [p_in, p_out],
                                Stroke::new(2.8_f32, with_alpha(accent, glow_a)),
                            );
                            let lit_col =
                                lerp_color(accent, brighten(accent, 34), active_t * 0.6_f32);
                            let tick_w: f32 = if is_major { 1.65_f32 } else { 1.2_f32 };
                            painter.line_segment([p_in, p_out], Stroke::new(tick_w, lit_col));
                        } else {
                            let unlit_col = if is_major {
                                Color32::from_rgb(66, 82, 112)
                            } else {
                                StudioPalette::SUBTLE_HAIRLINE
                            };
                            let tick_w: f32 = if is_major { 1.35_f32 } else { 1.0_f32 };
                            painter.line_segment([p_in, p_out], Stroke::new(tick_w, unlit_col));
                        }
                    }

                    // 2. Arco LED continuo incassato all'interno della corona graduata
                    let arc_r = 35.6_f32;
                    draw_arc(
                        painter,
                        c,
                        arc_r,
                        start_ang,
                        start_ang + sweep_ang,
                        Stroke::new(3.2_f32, StudioPalette::DEEP_OBSIDIAN),
                    );

                    if norm > 0.005_f32 {
                        let arc_glow_a = (36.0_f32 + 26.0_f32 * active_t).round() as u8;
                        draw_arc(
                            painter,
                            c,
                            arc_r,
                            start_ang,
                            end_ang,
                            Stroke::new(7.2_f32, with_alpha(accent, arc_glow_a)),
                        );
                        let arc_col = lerp_color(accent, brighten(accent, 22), active_t * 0.5_f32);
                        draw_arc(
                            painter,
                            c,
                            arc_r,
                            start_ang,
                            end_ang,
                            Stroke::new(3.2_f32, arc_col),
                        );
                    }

                    // 3. Ombra morbida sotto il corpo della manopola
                    let cap_outer_r = 30.8_f32;
                    let cap_inner_r = 27.6_f32;
                    painter.circle_filled(
                        c + Vec2::new(0.0_f32, 2.6_f32),
                        cap_outer_r + 0.8_f32,
                        Color32::from_rgba_unmultiplied(0, 0, 0, 96),
                    );

                    // 4. Ghiera metallica satinata a doppio anello:
                    //    - Anello esterno anodizzato con bordo scuro profondo
                    painter.circle_filled(c, cap_outer_r, Color32::from_rgb(17, 23, 35));
                    painter.circle_stroke(
                        c,
                        cap_outer_r,
                        Stroke::new(1.5_f32, StudioPalette::DEEP_OBSIDIAN),
                    );

                    //    - Calotta interna satinata con bisello chiaro reattivo all'hover
                    let cap_face = lerp_color(
                        StudioPalette::CARD_GLASS,
                        Color32::from_rgb(29, 39, 58),
                        active_t * 0.75_f32,
                    );
                    painter.circle_filled(c, cap_inner_r, cap_face);

                    let inner_bevel = lerp_color(
                        Color32::from_rgb(56, 72, 102),
                        lerp_color(brighten(accent, 28), Color32::WHITE, 0.25_f32),
                        active_t * 0.85_f32,
                    );
                    painter.circle_stroke(c, cap_inner_r, Stroke::new(1.15_f32, inner_bevel));

                    // Riflesso speculare superiore sulla ghiera satinata
                    draw_arc(
                        painter,
                        c,
                        cap_inner_r - 0.8_f32,
                        -0.80_f32 * PI,
                        -0.20_f32 * PI,
                        Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 34)),
                    );

                    // Lente centrale leggermente ribassata per massima leggibilità del valore
                    let readout_lens_r = 18.2_f32;
                    painter.circle_filled(c, readout_lens_r, Color32::from_rgb(14, 19, 29));
                    painter.circle_stroke(
                        c,
                        readout_lens_r,
                        Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 12)),
                    );

                    // 5. Tacca indicatrice luminosa allungata dal bordo verso il centro
                    let notch_outer = cap_outer_r - 1.4_f32;
                    let notch_inner = readout_lens_r + 1.6_f32;
                    let p_notch_out = Pos2::new(
                        c.x + notch_outer * end_ang.cos(),
                        c.y + notch_outer * end_ang.sin(),
                    );
                    let p_notch_in = Pos2::new(
                        c.x + notch_inner * end_ang.cos(),
                        c.y + notch_inner * end_ang.sin(),
                    );

                    let notch_glow_a = (68.0_f32 + 44.0_f32 * active_t).round() as u8;
                    painter.line_segment(
                        [p_notch_in, p_notch_out],
                        Stroke::new(4.6_f32, with_alpha(accent, notch_glow_a)),
                    );
                    painter.line_segment(
                        [p_notch_in, p_notch_out],
                        Stroke::new(2.2_f32, brighten(accent, 36)),
                    );
                    painter.circle_filled(p_notch_in, 1.3_f32, StudioPalette::CRISP_TEXT);

                    // 6. Valore centrale tipografico nitido (auto-scalato se lungo)
                    let base_font_sz = if display_val.chars().count() > 7 {
                        11.0_f32
                    } else {
                        12.2_f32
                    };
                    painter.text(
                        c,
                        Align2::CENTER_CENTER,
                        display_val,
                        FontId::proportional(base_font_sz),
                        StudioPalette::CRISP_TEXT,
                    );
                }

                if resp.hovered() || resp.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                }
            });
        });

    changed
}

fn draw_arc(
    painter: &egui::Painter,
    center: Pos2,
    radius: f32,
    start_rad: f32,
    end_rad: f32,
    stroke: Stroke,
) {
    let span = (end_rad - start_rad).abs();
    if span < 0.001_f32 {
        return;
    }
    let steps = ((span * 24.0_f32).ceil() as usize).clamp(8, 48);
    let mut pts = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let a = start_rad + t * (end_rad - start_rad);
        pts.push(Pos2::new(
            center.x + radius * a.cos(),
            center.y + radius * a.sin(),
        ));
    }
    painter.add(egui::Shape::line(pts, stroke));
}

/// Riempie in modo traslucido l'area tra una polilinea e una baseline Y usando una Mesh GPU.
pub fn draw_filled_curve_area(
    painter: &egui::Painter,
    points: &[Pos2],
    baseline_y: f32,
    top_color: Color32,
    base_color: Color32,
) {
    if points.len() < 2 {
        return;
    }
    let mut mesh = Mesh::default();
    mesh.vertices.reserve(points.len() * 2);
    mesh.indices.reserve((points.len() - 1) * 6);
    for (i, &p) in points.iter().enumerate() {
        mesh.colored_vertex(p, top_color);
        mesh.colored_vertex(Pos2::new(p.x, baseline_y), base_color);
        if i > 0 {
            let v = (i * 2) as u32;
            mesh.add_triangle(v - 2, v - 1, v);
            mesh.add_triangle(v - 1, v + 1, v);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

pub fn brighten(c: Color32, delta: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(
        c.r().saturating_add(delta),
        c.g().saturating_add(delta),
        c.b().saturating_add(delta),
        c.a(),
    )
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
