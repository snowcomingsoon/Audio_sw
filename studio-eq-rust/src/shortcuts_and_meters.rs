use eframe::egui::{
    self, Align2, Color32, CornerRadius, Event, FontId, Frame, Grid, Id, Key, Margin, Modifiers,
    Order, Pos2, Rect, RichText, ScrollArea, Sense, Stroke, StrokeKind, Ui, Vec2,
};

use crate::dsp::{DspParams, EqBandParam};
use crate::ui_widgets::pill_button;

const MAX_UNDO_SNAPSHOTS: usize = 40;

#[inline]
fn band_approx_eq(a: &EqBandParam, b: &EqBandParam) -> bool {
    a.name == b.name
        && a.role == b.role
        && a.kind == b.kind
        && a.enabled == b.enabled
        && (a.freq - b.freq).abs() < 1e-3_f64
        && (a.gain_db - b.gain_db).abs() < 1e-4_f64
        && (a.q - b.q).abs() < 1e-4_f64
}

#[inline]
fn dsp_params_approx_eq(a: &DspParams, b: &DspParams) -> bool {
    if a.bypass != b.bypass
        || a.dynamic_mud_enabled != b.dynamic_mud_enabled
        || a.exciter_enabled != b.exciter_enabled
        || a.mid_side_enabled != b.mid_side_enabled
        || a.comp_enabled != b.comp_enabled
        || a.bands.len() != b.bands.len()
    {
        return false;
    }

    if (a.sub_cut_hz - b.sub_cut_hz).abs() >= 1e-3_f64
        || (a.dynamic_mud_amount - b.dynamic_mud_amount).abs() >= 1e-4_f64
        || (a.exciter_drive - b.exciter_drive).abs() >= 1e-4_f64
        || (a.mono_bass_hz - b.mono_bass_hz).abs() >= 1e-3_f64
        || (a.stereo_width - b.stereo_width).abs() >= 1e-4_f64
        || (a.comp_drive_db - b.comp_drive_db).abs() >= 1e-4_f64
        || (a.ceiling_db - b.ceiling_db).abs() >= 1e-4_f64
    {
        return false;
    }

    a.bands
        .iter()
        .zip(b.bands.iter())
        .all(|(ba, bb)| band_approx_eq(ba, bb))
}

/// Storico circolare fino a 40 snapshot di `DspParams` per Annulla (`Ctrl+Z`) e Ripristina (`Ctrl+Y`).
#[derive(Clone, Debug)]
pub struct UndoStack {
    snapshots: Vec<DspParams>,
    cursor: usize,
}

impl Default for UndoStack {
    fn default() -> Self {
        Self::new(DspParams::default())
    }
}

#[allow(dead_code)]
impl UndoStack {
    pub fn new(initial: DspParams) -> Self {
        let mut snapshots = Vec::with_capacity(MAX_UNDO_SNAPSHOTS + 1);
        snapshots.push(initial);
        Self {
            snapshots,
            cursor: 0,
        }
    }

    /// Registra un nuovo snapshot solo se i parametri DSP sono effettivamente cambiati rispetto al cursore attuale.
    pub fn push_if_changed(&mut self, current: &DspParams) -> bool {
        if self.snapshots.is_empty() {
            self.snapshots.push(current.clone());
            self.cursor = 0;
            return true;
        }

        let idx = self.cursor.min(self.snapshots.len() - 1);
        if dsp_params_approx_eq(&self.snapshots[idx], current) {
            return false;
        }

        // Tronca eventuali stati futuri se si effettua una nuova modifica dopo un Undo
        self.snapshots.truncate(idx + 1);
        self.snapshots.push(current.clone());

        while self.snapshots.len() > MAX_UNDO_SNAPSHOTS {
            self.snapshots.remove(0);
        }
        self.cursor = self.snapshots.len() - 1;
        true
    }

    pub fn can_undo(&self) -> bool {
        !self.snapshots.is_empty() && self.cursor > 0
    }

    pub fn can_redo(&self) -> bool {
        !self.snapshots.is_empty() && self.cursor + 1 < self.snapshots.len()
    }

    pub fn undo(&mut self) -> Option<DspParams> {
        if self.can_undo() {
            self.cursor -= 1;
            Some(self.snapshots[self.cursor].clone())
        } else {
            None
        }
    }

    pub fn redo(&mut self) -> Option<DspParams> {
        if self.can_redo() {
            self.cursor += 1;
            Some(self.snapshots[self.cursor].clone())
        } else {
            None
        }
    }

    pub fn reset(&mut self, initial: DspParams) {
        self.snapshots.clear();
        self.snapshots.push(initial);
        self.cursor = 0;
    }
}

/// Azioni istantanee prodotte dalle scorciatoie da tastiera stile DAW professionale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShortcutActions {
    pub toggle_play: bool,
    pub rewind: bool,
    pub toggle_loop: bool,
    pub toggle_eq_drawer: bool,
    pub toggle_ab_bypass: bool,
    pub run_auto_eq: bool,
    pub solo_stem: Option<usize>,
    pub undo: bool,
    pub redo: bool,
    pub toggle_help: bool,
    // Nuove scorciatoie Editing Audio Non-Distruttivo
    pub delete_selection: bool,
    pub cut_selection: bool,
    pub copy_selection: bool,
    pub paste_selection: bool,
    pub crop_selection: bool,
    pub duplicate_selection: bool,
    pub clear_selection: bool,
}

/// Consuma la pressione singola (ignorando l'auto-repeat del sistema operativo) di un tasto senza modificatori.
fn consume_unmodified_single_press(i: &mut egui::InputState, target: Key) -> bool {
    let mut triggered = false;
    i.events.retain(|ev| match ev {
        Event::Key {
            key,
            physical_key,
            pressed: true,
            repeat,
            modifiers,
        } if (*key == target || *physical_key == Some(target))
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

/// Consuma la digitazione del carattere `?` (quando non si sta scrivendo in un campo di testo) per aprire/chiudere la guida.
fn consume_question_mark_text(i: &mut egui::InputState) -> bool {
    let mut triggered = false;
    i.events.retain(|ev| match ev {
        Event::Text(txt) if txt == "?" => {
            triggered = true;
            false
        }
        _ => true,
    });
    triggered
}

/// Intercetta le scorciatoie globali della DAW su ogni frame della GUI.
pub fn poll_daw_shortcuts(ctx: &egui::Context) -> ShortcutActions {
    let mut actions = ShortcutActions::default();
    let editing_text = ctx.wants_keyboard_input() || ctx.output(|o| o.ime.is_some());

    ctx.input_mut(|i| {
        if consume_unmodified_single_press(i, Key::F1) {
            actions.toggle_help = true;
        }

        if editing_text {
            return;
        }

        if consume_question_mark_text(i) {
            actions.toggle_help = true;
        }

        // Cattura eventi Cut / Copy / Paste nativi di egui-winit su Windows + combinazioni Ctrl+X/C/V
        let mut saw_cut = false;
        let mut saw_copy = false;
        let mut saw_paste = false;
        i.events.retain(|ev| match ev {
            Event::Cut => {
                saw_cut = true;
                false
            }
            Event::Copy => {
                saw_copy = true;
                false
            }
            Event::Paste(_) => {
                saw_paste = true;
                false
            }
            _ => true,
        });

        if saw_cut || i.consume_key(Modifiers::COMMAND, Key::X) {
            actions.cut_selection = true;
        }
        if saw_copy || i.consume_key(Modifiers::COMMAND, Key::C) {
            actions.copy_selection = true;
        }
        if saw_paste || i.consume_key(Modifiers::COMMAND, Key::V) {
            actions.paste_selection = true;
        }
        if i.consume_key(Modifiers::COMMAND, Key::T) {
            actions.crop_selection = true;
        }
        if i.consume_key(Modifiers::COMMAND, Key::D) {
            actions.duplicate_selection = true;
        }

        // Ctrl+Y oppure Ctrl+Shift+Z -> Ripristina (Redo)
        if i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
            || i.consume_key(Modifiers::COMMAND, Key::Y)
        {
            actions.redo = true;
        } else if i.consume_key(Modifiers::COMMAND, Key::Z) {
            // Ctrl+Z -> Annulla (Undo)
            actions.undo = true;
        }

        if consume_unmodified_single_press(i, Key::Delete)
            || consume_unmodified_single_press(i, Key::Backspace)
        {
            actions.delete_selection = true;
        }
        if consume_unmodified_single_press(i, Key::Escape) {
            actions.clear_selection = true;
        }

        if consume_unmodified_single_press(i, Key::Space) {
            actions.toggle_play = true;
        }
        if consume_unmodified_single_press(i, Key::Home) {
            actions.rewind = true;
        }
        if consume_unmodified_single_press(i, Key::L) {
            actions.toggle_loop = true;
        }
        if consume_unmodified_single_press(i, Key::E) {
            actions.toggle_eq_drawer = true;
        }
        if consume_unmodified_single_press(i, Key::B) {
            actions.toggle_ab_bypass = true;
        }
        if consume_unmodified_single_press(i, Key::A) {
            actions.run_auto_eq = true;
        }

        // Tasti rapidi 1..4 per Solo istantaneo dei 4 Stems AI (Voce, Batteria, Basso, Chitarre)
        if consume_unmodified_single_press(i, Key::Num1) {
            actions.solo_stem = Some(0);
        } else if consume_unmodified_single_press(i, Key::Num2) {
            actions.solo_stem = Some(1);
        } else if consume_unmodified_single_press(i, Key::Num3) {
            actions.solo_stem = Some(2);
        } else if consume_unmodified_single_press(i, Key::Num4) {
            actions.solo_stem = Some(3);
        }
    });

    actions
}

/// Disegna un tasto hardware stilizzato (Keycap Badge) per la tabella scorciatoie.
fn draw_keycap_badge(ui: &mut Ui, key_label: &str, accent: Color32) {
    let font = FontId::monospace(12.0_f32);
    let galley = ui
        .painter()
        .layout_no_wrap(key_label.to_string(), font.clone(), Color32::WHITE);
    let badge_size = Vec2::new(
        (galley.size().x + 18.0_f32).max(52.0_f32),
        (galley.size().y + 8.0_f32).max(24.0_f32),
    );
    let (rect, _) = ui.allocate_exact_size(badge_size, Sense::hover());

    if ui.is_rect_visible(rect) {
        let radius = CornerRadius::same(6);
        // Ombra inferiore sottile effetto keycap fisico
        let shadow_rect = Rect::from_min_max(
            Pos2::new(rect.min.x, rect.min.y + 1.5_f32),
            Pos2::new(rect.max.x, rect.max.y + 1.5_f32),
        );
        ui.painter()
            .rect_filled(shadow_rect, radius, Color32::from_rgb(8, 11, 18));
        ui.painter()
            .rect_filled(rect, radius, Color32::from_rgb(26, 35, 54));
        ui.painter().rect_stroke(
            rect,
            radius,
            Stroke::new(
                1.0_f32,
                Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 135),
            ),
            StrokeKind::Inside,
        );
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            key_label,
            font,
            Color32::from_rgb(241, 245, 249),
        );
    }
}

fn draw_guide_card(
    ui: &mut Ui,
    badge: &str,
    title: &str,
    body: &str,
    accent: Color32,
) {
    Frame::NONE
        .fill(Color32::from_rgb(12, 16, 27))
        .corner_radius(CornerRadius::same(12))
        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(30, 41, 62)))
        .inner_margin(Margin::symmetric(14, 12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                // Pillola accento categoria
                Frame::NONE
                    .fill(Color32::from_rgba_unmultiplied(
                        accent.r(),
                        accent.g(),
                        accent.b(),
                        34,
                    ))
                    .corner_radius(CornerRadius::same(6))
                    .stroke(Stroke::new(
                        1.0_f32,
                        Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 120),
                    ))
                    .inner_margin(Margin::symmetric(8, 2))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(badge)
                                .size(11.0_f32)
                                .strong()
                                .color(accent),
                        );
                    });

                ui.label(
                    RichText::new(title)
                        .size(13.5_f32)
                        .strong()
                        .color(Color32::from_rgb(241, 245, 249)),
                );
            });

            ui.add_space(4.0_f32);
            ui.label(
                RichText::new(body)
                    .size(12.3_f32)
                    .color(Color32::from_rgb(162, 178, 202)),
            );
        });
}

/// Finestra modale centrata con tabella ordinata delle scorciatoie DAW e guida rapida operativa.
pub fn draw_shortcuts_modal(ctx: &egui::Context, open: &mut bool) {
    if !*open {
        return;
    }

    if ctx.input_mut(|i| consume_unmodified_single_press(i, Key::Escape)) {
        *open = false;
        return;
    }

    let screen = ctx.screen_rect();
    let mut request_close = false;

    // Velatura scura morbida sullo sfondo (clic esterno chiude la modale)
    egui::Area::new(Id::new("studio_eq_shortcuts_backdrop"))
        .order(Order::Middle)
        .fixed_pos(screen.min)
        .show(ctx, |ui| {
            let (rect, resp) = ui.allocate_exact_size(screen.size(), Sense::click());
            ui.painter().rect_filled(
                rect,
                CornerRadius::ZERO,
                Color32::from_rgba_unmultiplied(5, 8, 15, 190),
            );
            if resp.clicked() {
                request_close = true;
            }
        });

    let obsidian_bg = Color32::from_rgb(0x11, 0x17, 0x26); // #111726
    let cyan_accent = Color32::from_rgb(56, 189, 248);
    let emerald_accent = Color32::from_rgb(52, 211, 153);
    let violet_accent = Color32::from_rgb(192, 132, 252);
    let amber_accent = Color32::from_rgb(251, 191, 36);

    let modal_frame = Frame::window(&ctx.style())
        .fill(obsidian_bg)
        .corner_radius(CornerRadius::same(16))
        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(46, 62, 92)))
        .inner_margin(Margin::symmetric(24, 20));

    let modal_w = (screen.width() - 48.0_f32).clamp(560.0_f32, 740.0_f32);
    let max_scroll_h = (screen.height() - 175.0_f32).clamp(220.0_f32, 620.0_f32);

    egui::Window::new("Guida Rapida & Scorciatoie Studio")
        .id(Id::new("studio_eq_shortcuts_modal_window"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .order(Order::Foreground)
        .anchor(Align2::CENTER_CENTER, Vec2::new(0.0_f32, 0.0_f32))
        .default_width(modal_w)
        .min_width(modal_w)
        .max_width(modal_w)
        .frame(modal_frame)
        .show(ctx, |ui| {
            // =================================================================
            // HEADER MODALE
            // =================================================================
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        let (dot_rect, _) =
                            ui.allocate_exact_size(Vec2::splat(10.0_f32), Sense::hover());
                        ui.painter()
                            .circle_filled(dot_rect.center(), 4.5_f32, cyan_accent);
                        ui.label(
                            RichText::new("GUIDA RAPIDA & SCORCIATOIE STUDIO")
                                .size(16.5_f32)
                                .strong()
                                .color(Color32::from_rgb(248, 250, 252)),
                        );
                    });
                    ui.label(
                        RichText::new(
                            "Pensato per band e musicisti • Pulisci le registrazioni in sala prove e mixa i 4 strumenti in pochi clic",
                        )
                        .size(12.2_f32)
                        .color(Color32::from_rgb(138, 156, 184)),
                    );
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if pill_button(
                        ui,
                        "Chiudi  Esc",
                        false,
                        cyan_accent,
                        Vec2::new(92.0_f32, 30.0_f32),
                    )
                    .clicked()
                    {
                        request_close = true;
                    }
                });
            });

            ui.add_space(10.0_f32);
            let sep_rect = ui.allocate_space(Vec2::new(ui.available_width(), 1.0_f32)).1;
            ui.painter().line_segment(
                [sep_rect.left_center(), sep_rect.right_center()],
                Stroke::new(1.0_f32, Color32::from_rgb(31, 43, 66)),
            );
            ui.add_space(10.0_f32);

            // =================================================================
            // CORPO SCORREVOLE: SCORCIATOIE TASTIERA + MOUSE + GUIDA OPERATIVA
            // =================================================================
            ScrollArea::vertical()
                .max_height(max_scroll_h)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    ui.label(
                        RichText::new("SCORCIATOIE DA TASTIERA")
                            .size(11.5_f32)
                            .strong()
                            .color(cyan_accent),
                    );
                    ui.add_space(4.0_f32);

                    let shortcuts: [(&str, &str, &str, Color32); 16] = [
                        (
                            "Spazio",
                            "Play / Pausa",
                            "Avvia o ferma all'istante l'ascolto del brano",
                            emerald_accent,
                        ),
                        (
                            "Home",
                            "Torna all'Inizio",
                            "Riporta subito il cursore all'inizio della canzone (00:00.0)",
                            cyan_accent,
                        ),
                        (
                            "Canc / Backspace",
                            "Elimina Selezione",
                            "Taglia via la zona evidenziata con micro-crossfade anti-click da 8 ms",
                            amber_accent,
                        ),
                        (
                            "Ctrl + X",
                            "Taglia negli Appunti",
                            "Taglia la selezione su tutte le tracce e la salva negli appunti Studio",
                            amber_accent,
                        ),
                        (
                            "Ctrl + C / V",
                            "Copia / Incolla",
                            "Copia la selezione e la incolla alla posizione della testina (crossfade 8 ms)",
                            cyan_accent,
                        ),
                        (
                            "Ctrl + T",
                            "Ritaglia Inizio/Fine",
                            "Mantiene SOLO la selezione eliminando chiacchiere iniziali e finali",
                            emerald_accent,
                        ),
                        (
                            "Ctrl + D",
                            "Duplica Selezione",
                            "Duplica subito dopo il tratto selezionato (es. raddoppio ritornello)",
                            violet_accent,
                        ),
                        (
                            "Esc",
                            "Deseleziona Zona",
                            "Rimuove la selezione corrente sulla timeline e disattiva il Loop",
                            cyan_accent,
                        ),
                        (
                            "L",
                            "Loop Ritornello",
                            "Ripete a ciclo continuo il tratto evidenziato sulla timeline",
                            amber_accent,
                        ),
                        (
                            "E",
                            "Pannello EQ & Master",
                            "Mostra o nasconde il pannello inferiore Equalizzatore e Manopole",
                            cyan_accent,
                        ),
                        (
                            "B",
                            "Confronta Prima / Dopo",
                            "Passa al volo tra il tuo Master e la registrazione Originale grezza",
                            amber_accent,
                        ),
                        (
                            "A",
                            "Auto-EQ Sala Prove",
                            "Analizza il brano e corregge in automatico rimbombo e suono chiuso",
                            violet_accent,
                        ),
                        (
                            "1  2  3  4",
                            "Solo Strumenti AI",
                            "Ascolta da solo: 1 Voce • 2 Batteria • 3 Basso • 4 Chitarre & Altro",
                            emerald_accent,
                        ),
                        (
                            "Ctrl + Z",
                            "Annulla (Undo)",
                            "Annulla l'ultima modifica Audio o EQ (storico fino a 15 edit audio / 40 EQ)",
                            cyan_accent,
                        ),
                        (
                            "Ctrl + Y",
                            "Ripristina (Redo)",
                            "Riapplica una modifica Audio o EQ annullata con Ctrl+Z",
                            cyan_accent,
                        ),
                        (
                            "F1  /  ?",
                            "Guida Rapida",
                            "Apre o chiude in qualsiasi momento questa finestra d'aiuto",
                            violet_accent,
                        ),
                    ];

                    Frame::NONE
                        .fill(Color32::from_rgb(12, 16, 27))
                        .corner_radius(CornerRadius::same(12))
                        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(28, 39, 59)))
                        .inner_margin(Margin::symmetric(14, 10))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            Grid::new("studio_eq_shortcuts_grid")
                                .num_columns(3)
                                .spacing(Vec2::new(16.0_f32, 7.0_f32))
                                .with_row_color(|row, _| {
                                    if row > 0 && row % 2 == 0 {
                                        Some(Color32::from_rgba_unmultiplied(24, 34, 54, 90))
                                    } else {
                                        None
                                    }
                                })
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new("TASTO")
                                            .size(10.8_f32)
                                            .strong()
                                            .color(Color32::from_rgb(112, 130, 158)),
                                    );
                                    ui.label(
                                        RichText::new("COMANDO")
                                            .size(10.8_f32)
                                            .strong()
                                            .color(Color32::from_rgb(112, 130, 158)),
                                    );
                                    ui.label(
                                        RichText::new("COSA FA")
                                            .size(10.8_f32)
                                            .strong()
                                            .color(Color32::from_rgb(112, 130, 158)),
                                    );
                                    ui.end_row();

                                    for (key_txt, cmd_txt, desc_txt, acc) in shortcuts {
                                        draw_keycap_badge(ui, key_txt, acc);
                                        ui.label(
                                            RichText::new(cmd_txt)
                                                .size(12.8_f32)
                                                .strong()
                                                .color(Color32::from_rgb(236, 242, 249)),
                                        );
                                        ui.label(
                                            RichText::new(desc_txt)
                                                .size(12.2_f32)
                                                .color(Color32::from_rgb(156, 173, 198)),
                                        );
                                        ui.end_row();
                                    }
                                });
                        });

                    ui.add_space(12.0_f32);
                    ui.label(
                        RichText::new("COMANDI RAPIDI COL MOUSE")
                            .size(11.5_f32)
                            .strong()
                            .color(amber_accent),
                    );
                    ui.add_space(4.0_f32);

                    let mouse_tips: [(&str, &str, &str, Color32); 6] = [
                        (
                            "Doppio Clic",
                            "Azzera Regolazione",
                            "Su Volume, Pan, Manopole o Nodi EQ riporta subito al valore consigliato",
                            emerald_accent,
                        ),
                        (
                            "Trascina Righello",
                            "Seleziona Loop",
                            "Trascina sul righello dei minuti o sull'onda per mettere in Loop una parte",
                            amber_accent,
                        ),
                        (
                            "Ctrl + Rotellina",
                            "Zoom Forme d'Onda",
                            "Ingrandisce o riduce la timeline (oppure usa i pulsanti - / 1.0x / +)",
                            cyan_accent,
                        ),
                        (
                            "Shift + Rotellina",
                            "Scorri Timeline",
                            "Muove la visuale a destra e sinistra quando lo zoom è ingrandito",
                            cyan_accent,
                        ),
                        (
                            "Rotellina su EQ",
                            "Larghezza Campana Q",
                            "Sopra un pallino EQ (1–8) stringe o allarga la fascia di frequenze",
                            violet_accent,
                        ),
                        (
                            "Trascina File",
                            "Carica Audio al Volo",
                            "Trascina un MP3, WAV, FLAC o M4A dentro la finestra per aprirlo subito",
                            emerald_accent,
                        ),
                    ];

                    Frame::NONE
                        .fill(Color32::from_rgb(12, 16, 27))
                        .corner_radius(CornerRadius::same(12))
                        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(28, 39, 59)))
                        .inner_margin(Margin::symmetric(14, 10))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            Grid::new("studio_eq_mouse_grid")
                                .num_columns(3)
                                .spacing(Vec2::new(16.0_f32, 7.0_f32))
                                .with_row_color(|row, _| {
                                    if row > 0 && row % 2 == 0 {
                                        Some(Color32::from_rgba_unmultiplied(24, 34, 54, 90))
                                    } else {
                                        None
                                    }
                                })
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new("GESTO MOUSE")
                                            .size(10.8_f32)
                                            .strong()
                                            .color(Color32::from_rgb(112, 130, 158)),
                                    );
                                    ui.label(
                                        RichText::new("AZIONE")
                                            .size(10.8_f32)
                                            .strong()
                                            .color(Color32::from_rgb(112, 130, 158)),
                                    );
                                    ui.label(
                                        RichText::new("DETTAGLIO PRATICO")
                                            .size(10.8_f32)
                                            .strong()
                                            .color(Color32::from_rgb(112, 130, 158)),
                                    );
                                    ui.end_row();

                                    for (gesture_txt, act_txt, desc_txt, acc) in mouse_tips {
                                        draw_keycap_badge(ui, gesture_txt, acc);
                                        ui.label(
                                            RichText::new(act_txt)
                                                .size(12.6_f32)
                                                .strong()
                                                .color(Color32::from_rgb(236, 242, 249)),
                                        );
                                        ui.label(
                                            RichText::new(desc_txt)
                                                .size(12.1_f32)
                                                .color(Color32::from_rgb(156, 173, 198)),
                                        );
                                        ui.end_row();
                                    }
                                });
                        });

                    ui.add_space(14.0_f32);
                    ui.label(
                        RichText::new("COME OTTENERE UN MIX PULITO IN 3 PASSI")
                            .size(11.5_f32)
                            .strong()
                            .color(emerald_accent),
                    );
                    ui.add_space(4.0_f32);

                    draw_guide_card(
                        ui,
                        "PASSO 1 • SEPARA AI",
                        "Dividi la Registrazione in Voce, Batteria, Basso e Chitarre",
                        "Hai registrato la band con un solo microfono o dal mixer in sala prove? Premi 'Separa AI' per ottenere 4 tracce indipendenti. Puoi alzare la Voce se è coperta dalla batteria, mettere in Muto (M) o Solo (S / tasti 1–4) un singolo strumento per studiare la tua parte, e aprire le chitarre a sinistra/destra col Pan.",
                        emerald_accent,
                    );
                    ui.add_space(8.0_f32);

                    draw_guide_card(
                        ui,
                        "PASSO 2 • PULISCI & SCOLPISCI",
                        "Togli il Rimbombo della Sala con Auto-EQ e le 4 Manopole Smart",
                        "Premi 'Auto-EQ' (o scegli il preset 'Sala Prove Fix') e regola a orecchio le 4 grandi manopole: 'Pulizia Sala' asciuga il fango dei muri sui bassi, 'Voce & Medie' rende nitide le parole, 'Aria & Alti' apre il suono, e 'Spinta Master' compatta la band dandole volume. Premi B per sentire subito la differenza Prima/Dopo!",
                        violet_accent,
                    );
                    ui.add_space(8.0_f32);

                    draw_guide_card(
                        ui,
                        "PASSO 3 • ESPORTA",
                        "Salva il Brano Finito o le 4 Tracce Separate",
                        "Premi 'Esporta' per salvare il mix completo già bilanciato e protetto contro la distorsione: scegli WAV 24-bit per la massima qualità da studio, oppure MP3 320 kbps leggero e pronto da inviare su WhatsApp/Telegram alla band. Con 'Salva 4 Stems' esporti invece le 4 tracce separate in una cartella.",
                        amber_accent,
                    );
                });

            // =================================================================
            // FOOTER
            // =================================================================
            ui.add_space(12.0_f32);
            let foot_sep = ui.allocate_space(Vec2::new(ui.available_width(), 1.0_f32)).1;
            ui.painter().line_segment(
                [foot_sep.left_center(), foot_sep.right_center()],
                Stroke::new(1.0_f32, Color32::from_rgb(31, 43, 66)),
            );
            ui.add_space(10.0_f32);

            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(
                        "Suggerimento: Passa il mouse sopra qualsiasi controllo per leggere cosa fa e come azzerarlo",
                    )
                    .size(11.6_f32)
                    .color(Color32::from_rgb(124, 142, 170)),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if pill_button(
                        ui,
                        "Torna al Mix",
                        true,
                        Color32::from_rgb(14, 165, 233),
                        Vec2::new(132.0_f32, 32.0_f32),
                    )
                    .clicked()
                    {
                        request_close = true;
                    }
                });
            });
        });

    if request_close {
        *open = false;
    }
}
