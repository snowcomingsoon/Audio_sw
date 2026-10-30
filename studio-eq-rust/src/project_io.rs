use std::f32::consts::{FRAC_PI_2, PI};
use std::path::{Path, PathBuf};

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Id, LayerId, Order, Pos2, Rect, Stroke,
    StrokeKind, Vec2,
};

use crate::audio_io::{constant_power_pan_gains, soft_limit_to_pcm24, StemChannel, StereoBuffer};
use crate::dsp::{DspParams, FilterKind};

const STEM_EXPORT_FILENAMES: [&str; 4] = [
    "01_Voce_Mix.wav",
    "02_Batteria_Mix.wav",
    "03_Basso_Mix.wav",
    "04_Chitarre_Altro_Mix.wav",
];

const SUPPORTED_AUDIO_EXTENSIONS: [&str; 6] = ["mp3", "mpeg", "wav", "flac", "ogg", "m4a"];

/// Esporta tutti i 4 Stems separati in formato WAV 24-bit Studio nella cartella indicata,
/// applicando `gain_db` e `pan` impostati dall'utente nel mixer.
pub fn export_all_stems_wav(
    stems: &[StereoBuffer; 4],
    channels: &[StemChannel; 4],
    folder: &Path,
) -> Result<usize, String> {
    if stems.iter().all(|s| s.is_empty()) {
        return Err("Nessuno Stem audio disponibile per l'esportazione.".to_string());
    }

    std::fs::create_dir_all(folder).map_err(|e| {
        format!(
            "Impossibile creare o accedere alla cartella '{}': {e}",
            folder.display()
        )
    })?;

    let mut exported_count = 0usize;

    for idx in 0..4 {
        let stem = &stems[idx];
        let frames = stem.len();
        if frames == 0 {
            continue;
        }

        let ch = &channels[idx];
        let gain_lin = 10.0_f32.powf(ch.gain_db / 20.0_f32);
        let (pan_l, pan_r) = constant_power_pan_gains(ch.pan);
        let mult_l = gain_lin * pan_l;
        let mult_r = gain_lin * pan_r;

        let file_name = STEM_EXPORT_FILENAMES[idx];
        let out_path = folder.join(file_name);

        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: stem.sample_rate.max(8_000),
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(&out_path, spec)
            .map_err(|e| format!("Impossibile creare '{file_name}': {e}"))?;

        for i in 0..frames {
            let out_l = stem.left[i] * mult_l;
            let out_r = stem.right[i] * mult_r;
            writer
                .write_sample(soft_limit_to_pcm24(out_l))
                .map_err(|e| format!("Errore scrittura '{file_name}': {e}"))?;
            writer
                .write_sample(soft_limit_to_pcm24(out_r))
                .map_err(|e| format!("Errore scrittura '{file_name}': {e}"))?;
        }

        writer
            .finalize()
            .map_err(|e| format!("Errore finalizzazione '{file_name}': {e}"))?;

        exported_count += 1;
    }

    Ok(exported_count)
}

fn filter_kind_to_str(kind: FilterKind) -> &'static str {
    match kind {
        FilterKind::Peaking => "Peaking",
        FilterKind::HighPass => "HighPass",
        FilterKind::LowPass => "LowPass",
        FilterKind::HighShelf => "HighShelf",
    }
}

fn parse_filter_kind(raw: &str) -> Option<FilterKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "peaking" | "peak" | "bell" => Some(FilterKind::Peaking),
        "highpass" | "high_pass" | "hp" => Some(FilterKind::HighPass),
        "lowpass" | "low_pass" | "lp" => Some(FilterKind::LowPass),
        "highshelf" | "high_shelf" | "hs" => Some(FilterKind::HighShelf),
        _ => None,
    }
}

fn parse_bool_val(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_finite_f64(raw: &str) -> Option<f64> {
    raw.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Salva i parametri DSP completi (inclusi tutti gli 8 filtri parametrici) in un file `.seqpreset` leggibile (`key=value`).
pub fn save_preset_file(params: &DspParams, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Impossibile creare la cartella per il preset '{}': {e}",
                    parent.display()
                )
            })?;
        }
    }

    let mut content = String::with_capacity(1024);
    content.push_str("# StudioEQ Master Pro — Preset File (.seqpreset)\n");
    content.push_str("format=StudioEQ_SeqPreset_v1\n");
    content.push_str(&format!("bypass={}\n", params.bypass));
    content.push_str(&format!("sub_cut_hz={:.4}\n", params.sub_cut_hz));
    content.push_str(&format!(
        "dynamic_mud_enabled={}\n",
        params.dynamic_mud_enabled
    ));
    content.push_str(&format!(
        "dynamic_mud_amount={:.4}\n",
        params.dynamic_mud_amount
    ));
    content.push_str(&format!("exciter_enabled={}\n", params.exciter_enabled));
    content.push_str(&format!("exciter_drive={:.4}\n", params.exciter_drive));
    content.push_str(&format!("mid_side_enabled={}\n", params.mid_side_enabled));
    content.push_str(&format!("mono_bass_hz={:.4}\n", params.mono_bass_hz));
    content.push_str(&format!("stereo_width={:.4}\n", params.stereo_width));
    content.push_str(&format!("comp_enabled={}\n", params.comp_enabled));
    content.push_str(&format!("comp_drive_db={:.4}\n", params.comp_drive_db));
    content.push_str(&format!("ceiling_db={:.4}\n", params.ceiling_db));
    content.push_str("\n# 8-Band Parametric EQ: freq,gain_db,q,enabled\n");

    let base_template = DspParams::preset_sala_prove();
    for idx in 0..8 {
        let band = params.bands.get(idx).unwrap_or(&base_template.bands[idx]);
        content.push_str(&format!(
            "band_{idx}={:.4},{:.4},{:.4},{}\n",
            band.freq, band.gain_db, band.q, band.enabled
        ));
        content.push_str(&format!(
            "band_{idx}_kind={}\n",
            filter_kind_to_str(band.kind)
        ));
    }

    std::fs::write(path, content)
        .map_err(|e| format!("Impossibile salvare il preset '{}': {e}", path.display()))
}

/// Carica un file `.seqpreset` (`key=value`) restituendo una configurazione `DspParams` validata.
pub fn load_preset_file(path: &Path) -> Result<DspParams, String> {
    let raw_text = std::fs::read_to_string(path)
        .map_err(|e| format!("Impossibile leggere il preset '{}': {e}", path.display()))?;

    let mut params = DspParams::preset_sala_prove();
    let mut recognized_keys = 0usize;
    let mut has_explicit_mud_flag = false;
    let mut has_explicit_exciter_flag = false;

    for raw_line in raw_text.lines() {
        let line = raw_line.trim_start_matches('\u{feff}').trim();
        if line.is_empty()
            || line.starts_with('#')
            || line.starts_with(';')
            || line.starts_with("//")
        {
            continue;
        }

        let Some((k_raw, v_raw)) = line.split_once('=') else {
            continue;
        };
        let key = k_raw.trim().to_ascii_lowercase();
        let val = v_raw.trim();

        match key.as_str() {
            "format" | "version" => {
                recognized_keys += 1;
            }
            "bypass" => {
                if let Some(b) = parse_bool_val(val) {
                    params.bypass = b;
                    recognized_keys += 1;
                }
            }
            "sub_cut_hz" => {
                if let Some(v) = parse_finite_f64(val) {
                    params.sub_cut_hz = v.clamp(15.0_f64, 400.0_f64);
                    recognized_keys += 1;
                }
            }
            "dynamic_mud_enabled" => {
                if let Some(b) = parse_bool_val(val) {
                    params.dynamic_mud_enabled = b;
                    has_explicit_mud_flag = true;
                    recognized_keys += 1;
                }
            }
            "dynamic_mud_amount" => {
                if let Some(v) = parse_finite_f64(val) {
                    params.dynamic_mud_amount = v.clamp(0.0_f64, 1.5_f64);
                    recognized_keys += 1;
                }
            }
            "exciter_enabled" => {
                if let Some(b) = parse_bool_val(val) {
                    params.exciter_enabled = b;
                    has_explicit_exciter_flag = true;
                    recognized_keys += 1;
                }
            }
            "exciter_drive" => {
                if let Some(v) = parse_finite_f64(val) {
                    params.exciter_drive = v.clamp(0.0_f64, 0.6_f64);
                    recognized_keys += 1;
                }
            }
            "mid_side_enabled" => {
                if let Some(b) = parse_bool_val(val) {
                    params.mid_side_enabled = b;
                    recognized_keys += 1;
                }
            }
            "mono_bass_hz" => {
                if let Some(v) = parse_finite_f64(val) {
                    params.mono_bass_hz = v.clamp(20.0_f64, 400.0_f64);
                    recognized_keys += 1;
                }
            }
            "stereo_width" => {
                if let Some(v) = parse_finite_f64(val) {
                    params.stereo_width = v.clamp(0.0_f64, 2.0_f64);
                    recognized_keys += 1;
                }
            }
            "comp_enabled" => {
                if let Some(b) = parse_bool_val(val) {
                    params.comp_enabled = b;
                    recognized_keys += 1;
                }
            }
            "comp_drive_db" => {
                if let Some(v) = parse_finite_f64(val) {
                    params.comp_drive_db = v.clamp(-6.0_f64, 12.0_f64);
                    recognized_keys += 1;
                }
            }
            "ceiling_db" => {
                if let Some(v) = parse_finite_f64(val) {
                    params.ceiling_db = v.clamp(-12.0_f64, 0.0_f64);
                    recognized_keys += 1;
                }
            }
            other if other.starts_with("band_") || other.starts_with("band") => {
                let suffix = other
                    .strip_prefix("band_")
                    .or_else(|| other.strip_prefix("band"))
                    .unwrap_or("");

                if let Ok(band_idx) = suffix.parse::<usize>() {
                    if band_idx < params.bands.len() {
                        let parts: Vec<&str> = val.split(',').map(|s| s.trim()).collect();
                        if parts.len() >= 4 {
                            if let (Some(freq), Some(gain_db), Some(q), Some(enabled)) = (
                                parse_finite_f64(parts[0]),
                                parse_finite_f64(parts[1]),
                                parse_finite_f64(parts[2]),
                                parse_bool_val(parts[3]),
                            ) {
                                let b = &mut params.bands[band_idx];
                                b.freq = freq.clamp(20.0_f64, 20_000.0_f64);
                                b.gain_db = gain_db.clamp(-24.0_f64, 24.0_f64);
                                b.q = q.clamp(0.15_f64, 18.0_f64);
                                b.enabled = enabled;
                                if parts.len() >= 5 {
                                    if let Some(k) = parse_filter_kind(parts[4]) {
                                        b.kind = k;
                                    }
                                }
                                recognized_keys += 1;
                            }
                        }
                    }
                } else if let Some((idx_str, field)) = suffix.split_once('_') {
                    if let Ok(band_idx) = idx_str.parse::<usize>() {
                        if band_idx < params.bands.len() {
                            let b = &mut params.bands[band_idx];
                            match field {
                                "freq" => {
                                    if let Some(v) = parse_finite_f64(val) {
                                        b.freq = v.clamp(20.0_f64, 20_000.0_f64);
                                        recognized_keys += 1;
                                    }
                                }
                                "gain" | "gain_db" => {
                                    if let Some(v) = parse_finite_f64(val) {
                                        b.gain_db = v.clamp(-24.0_f64, 24.0_f64);
                                        recognized_keys += 1;
                                    }
                                }
                                "q" => {
                                    if let Some(v) = parse_finite_f64(val) {
                                        b.q = v.clamp(0.15_f64, 18.0_f64);
                                        recognized_keys += 1;
                                    }
                                }
                                "enabled" => {
                                    if let Some(en) = parse_bool_val(val) {
                                        b.enabled = en;
                                        recognized_keys += 1;
                                    }
                                }
                                "kind" => {
                                    if let Some(k) = parse_filter_kind(val) {
                                        b.kind = k;
                                        recognized_keys += 1;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if recognized_keys == 0 {
        return Err(
            "Il file selezionato non contiene parametri preset StudioEQ validi (.seqpreset)."
                .to_string(),
        );
    }

    if !has_explicit_mud_flag {
        params.dynamic_mud_enabled = params.dynamic_mud_amount > 0.005_f64;
    }
    if !has_explicit_exciter_flag {
        params.exciter_enabled = params.exciter_drive > 0.002_f64;
    }

    Ok(params)
}

fn is_supported_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            SUPPORTED_AUDIO_EXTENSIONS
                .iter()
                .any(|&supported| ext.eq_ignore_ascii_case(supported))
        })
        .unwrap_or(false)
}

/// Rileva i file trascinati con Drag & Drop da Esplora Risorse di Windows sulla finestra dell'app,
/// filtrando le estensioni audio supportate (`mp3`, `mpeg`, `wav`, `flac`, `ogg`, `m4a`).
pub fn extract_dropped_audio_path(ctx: &eframe::egui::Context) -> Option<PathBuf> {
    let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
    for file in dropped_files {
        if let Some(path) = file.path {
            if is_supported_audio_file(&path) {
                return Some(path);
            }
        }
    }
    None
}

fn draw_dashed_segment(
    painter: &egui::Painter,
    from: Pos2,
    to: Pos2,
    dash_len: f32,
    gap_len: f32,
    stroke: Stroke,
) {
    let delta = to - from;
    let total_len = delta.length();
    if total_len < 1.0_f32 {
        return;
    }
    let dir = delta / total_len;
    let step = (dash_len + gap_len).max(2.0_f32);
    let mut dist = 0.0_f32;
    while dist < total_len {
        let end_dist = (dist + dash_len).min(total_len);
        let p0 = from + dir * dist;
        let p1 = from + dir * end_dist;
        painter.line_segment([p0, p1], stroke);
        dist += step;
    }
}

fn draw_corner_arc(
    painter: &egui::Painter,
    center: Pos2,
    radius: f32,
    start_angle: f32,
    end_angle: f32,
    stroke: Stroke,
) {
    let steps = 12usize;
    let mut pts = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let ang = start_angle + t * (end_angle - start_angle);
        pts.push(Pos2::new(
            center.x + radius * ang.cos(),
            center.y + radius * ang.sin(),
        ));
    }
    painter.add(egui::Shape::line(pts, stroke));
}

fn draw_dashed_glow_frame(
    painter: &egui::Painter,
    rect: Rect,
    corner_radius: f32,
    cyan_bright: Color32,
    cyan_core: Color32,
) {
    let r_u8 = corner_radius.clamp(0.0_f32, 48.0_f32) as u8;
    // Alone luminoso continuo Ciano attorno al perimetro
    painter.rect_stroke(
        rect,
        CornerRadius::same(r_u8),
        Stroke::new(
            7.0_f32,
            Color32::from_rgba_unmultiplied(cyan_core.r(), cyan_core.g(), cyan_core.b(), 34),
        ),
        StrokeKind::Inside,
    );
    painter.rect_stroke(
        rect,
        CornerRadius::same(r_u8),
        Stroke::new(
            1.0_f32,
            Color32::from_rgba_unmultiplied(cyan_bright.r(), cyan_bright.g(), cyan_bright.b(), 65),
        ),
        StrokeKind::Inside,
    );

    let r = corner_radius.min(rect.width() * 0.25_f32).min(rect.height() * 0.25_f32);
    let dash_stroke = Stroke::new(2.2_f32, cyan_bright);
    let corner_stroke = Stroke::new(2.6_f32, cyan_bright);
    let dash_len = 14.0_f32;
    let gap_len = 9.0_f32;

    // 4 lati tratteggiati
    draw_dashed_segment(
        painter,
        Pos2::new(rect.left() + r, rect.top()),
        Pos2::new(rect.right() - r, rect.top()),
        dash_len,
        gap_len,
        dash_stroke,
    );
    draw_dashed_segment(
        painter,
        Pos2::new(rect.left() + r, rect.bottom()),
        Pos2::new(rect.right() - r, rect.bottom()),
        dash_len,
        gap_len,
        dash_stroke,
    );
    draw_dashed_segment(
        painter,
        Pos2::new(rect.left(), rect.top() + r),
        Pos2::new(rect.left(), rect.bottom() - r),
        dash_len,
        gap_len,
        dash_stroke,
    );
    draw_dashed_segment(
        painter,
        Pos2::new(rect.right(), rect.top() + r),
        Pos2::new(rect.right(), rect.bottom() - r),
        dash_len,
        gap_len,
        dash_stroke,
    );

    // 4 angoli arrotondati luminosi
    draw_corner_arc(
        painter,
        Pos2::new(rect.left() + r, rect.top() + r),
        r,
        PI,
        PI + FRAC_PI_2,
        corner_stroke,
    );
    draw_corner_arc(
        painter,
        Pos2::new(rect.right() - r, rect.top() + r),
        r,
        -FRAC_PI_2,
        0.0_f32,
        corner_stroke,
    );
    draw_corner_arc(
        painter,
        Pos2::new(rect.right() - r, rect.bottom() - r),
        r,
        0.0_f32,
        FRAC_PI_2,
        corner_stroke,
    );
    draw_corner_arc(
        painter,
        Pos2::new(rect.left() + r, rect.bottom() - r),
        r,
        FRAC_PI_2,
        PI,
        corner_stroke,
    );
}

/// Disegna un overlay moderno semi-trasparente con bordo tratteggiato/luminoso Ciano
/// quando l'utente trascina un file sopra la finestra dell'applicazione.
pub fn draw_drop_zone_overlay(ctx: &eframe::egui::Context) {
    let hovered_files = ctx.input(|i| i.raw.hovered_files.clone());
    if hovered_files.is_empty() {
        return;
    }

    let screen = ctx.screen_rect();
    if screen.width() < 120.0_f32 || screen.height() < 120.0_f32 {
        return;
    }

    ctx.request_repaint();

    let painter = ctx.layer_painter(LayerId::new(
        Order::Foreground,
        Id::new("studio_eq_drop_zone_overlay"),
    ));

    let cyan_core = Color32::from_rgb(14, 165, 233);
    let cyan_bright = Color32::from_rgb(56, 189, 248);

    // Velatura scura semi-trasparente sull'intera finestra
    painter.rect_filled(
        screen,
        CornerRadius::ZERO,
        Color32::from_rgba_unmultiplied(7, 11, 19, 214),
    );

    // Cornice perimetrale tratteggiata e luminosa Ciano
    let outer_frame = screen.shrink(26.0_f32);
    draw_dashed_glow_frame(&painter, outer_frame, 20.0_f32, cyan_bright, cyan_core);

    // Card centrale fluttuante ad alto contrasto
    let card_w = 590.0_f32.min(outer_frame.width() - 32.0_f32).max(240.0_f32);
    let card_h = 216.0_f32.min(outer_frame.height() - 32.0_f32).max(140.0_f32);
    let card_rect = Rect::from_center_size(screen.center(), Vec2::new(card_w, card_h));

    painter.rect_filled(
        card_rect.expand(4.0_f32),
        CornerRadius::same(24),
        Color32::from_rgba_unmultiplied(14, 165, 233, 28),
    );
    painter.rect_filled(
        card_rect,
        CornerRadius::same(20),
        Color32::from_rgba_unmultiplied(13, 19, 31, 242),
    );
    painter.rect_stroke(
        card_rect,
        CornerRadius::same(20),
        Stroke::new(1.5_f32, Color32::from_rgba_unmultiplied(56, 189, 248, 150)),
        StrokeKind::Inside,
    );

    // Badge circolare superiore con icona forma d'onda audio stilizzata
    let badge_center = Pos2::new(card_rect.center().x, card_rect.top() + 54.0_f32);
    painter.circle_filled(
        badge_center,
        31.0_f32,
        Color32::from_rgba_unmultiplied(14, 165, 233, 36),
    );
    painter.circle_stroke(
        badge_center,
        27.0_f32,
        Stroke::new(1.6_f32, cyan_bright),
    );

    let bar_heights = [9.0_f32, 18.0_f32, 26.0_f32, 15.0_f32, 22.0_f32, 10.0_f32];
    let start_x = badge_center.x - 15.0_f32;
    for (idx, &bh) in bar_heights.iter().enumerate() {
        let bx = start_x + idx as f32 * 6.0_f32;
        let half = bh * 0.5_f32;
        painter.line_segment(
            [
                Pos2::new(bx, badge_center.y - half),
                Pos2::new(bx, badge_center.y + half),
            ],
            Stroke::new(2.6_f32, Color32::WHITE),
        );
    }

    // Messaggio principale richiesto
    painter.text(
        Pos2::new(card_rect.center().x, card_rect.center().y + 18.0_f32),
        Align2::CENTER_CENTER,
        "Rilascia qui il file audio per aprirlo nello Studio",
        FontId::proportional(20.0_f32),
        Color32::from_rgb(241, 245, 249),
    );

    // Sottotitolo informativo (mostra il nome file se disponibile o i formati supportati)
    let hovered_info = hovered_files
        .iter()
        .find_map(|hf| hf.path.as_deref())
        .and_then(|p| {
            let fname = p.file_name().and_then(|n| n.to_str())?;
            let ok = is_supported_audio_file(p);
            Some((fname.to_string(), ok))
        });

    let (subtitle_text, subtitle_color) = match hovered_info {
        Some((fname, true)) => (
            format!("File pronto: {fname}  •  Importazione diretta 64-Bit DSP"),
            cyan_bright,
        ),
        Some((fname, false)) => (
            format!("Formato non audio ({fname})  •  Usa MP3, WAV, FLAC, OGG, M4A"),
            Color32::from_rgb(245, 158, 11),
        ),
        None => (
            "Formati supportati: WAV  •  MP3  •  MPEG  •  FLAC  •  OGG  •  M4A".to_string(),
            Color32::from_rgb(148, 163, 184),
        ),
    };

    painter.text(
        Pos2::new(card_rect.center().x, card_rect.bottom() - 34.0_f32),
        Align2::CENTER_CENTER,
        subtitle_text,
        FontId::proportional(13.0_f32),
        subtitle_color,
    );
}
