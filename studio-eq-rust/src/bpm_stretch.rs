#![forbid(unsafe_code)]

//! Rilevamento rapido del BPM (`detect_bpm_fast`) tramite Onset-Flux dual-band + Comb Resonator
//! (ispirato a `fabriziosalmi/open-bpm` e `stratum-dsp`) e Time-Stretching stereo phase-locked WSOLA
//! (`stretch_stereo_wsola`, ispirato a `SoundTouch` `TDStretch::calcCrossCorrNorm` e `Signalsmith-Audio`).

use std::f64::consts::PI;

use crate::audio_io::{SharedAudioState, StereoBuffer};

pub const MIN_DETECT_BPM: f32 = 68.0;
pub const MAX_DETECT_BPM: f32 = 185.0;
pub const MIN_USER_BPM: f32 = 45.0;
pub const MAX_USER_BPM: f32 = 240.0;

const WSOLA_WIN_LEN: usize = 2048;
const WSOLA_SYN_HOP: usize = 512;
const WSOLA_SEARCH_TOL: isize = 480;
const WSOLA_CORR_LEN: usize = 384;

/// Riferimenti non stirati (`Arc` zero-copy) per permettere cambi ripetuti di BPM in `<150 ms`
/// senza mai accumulare artefatti di generazioni successive.
#[derive(Clone, Debug)]
pub struct OriginalTempoRefs {
    pub main_track: Option<StereoBuffer>,
    pub ai_master_track: Option<StereoBuffer>,
    pub stems: Option<[StereoBuffer; 4]>,
    /// BPM a cui si trovano i buffer salvati qui dentro.
    pub base_bpm: f32,
    /// BPM originale rilevato al caricamento iniziale del file.
    pub detected_bpm: f32,
    /// BPM attivo attualmente ascoltato nella timeline.
    pub current_bpm: f32,
}

impl Default for OriginalTempoRefs {
    fn default() -> Self {
        Self {
            main_track: None,
            ai_master_track: None,
            stems: None,
            base_bpm: 120.0,
            detected_bpm: 120.0,
            current_bpm: 120.0,
        }
    }
}

impl OriginalTempoRefs {
    pub fn store_loaded(
        &mut self,
        main_track: Option<StereoBuffer>,
        ai_master_track: Option<StereoBuffer>,
        stems: Option<[StereoBuffer; 4]>,
        detected_bpm: f32,
    ) {
        let clean_bpm = detected_bpm.clamp(MIN_USER_BPM, MAX_USER_BPM);
        self.main_track = main_track;
        self.ai_master_track = ai_master_track;
        self.stems = stems;
        self.base_bpm = clean_bpm;
        self.detected_bpm = clean_bpm;
        self.current_bpm = clean_bpm;
    }

    /// Da invocare dopo ogni modifica di editing audio (Taglia, Ritaglia, Silenzia, Fade, Undo/Redo)
    /// così un successivo cambio BPM mantiene tutte le modifiche eseguite dall'utente!
    pub fn sync_after_audio_edit(&mut self, st: &SharedAudioState) {
        self.main_track = st.main_track.clone();
        self.ai_master_track = st.ai_master_track.clone();
        self.stems = st.stems.clone();
        self.base_bpm = self.current_bpm.clamp(MIN_USER_BPM, MAX_USER_BPM);
    }

    pub fn render_for_target_bpm(
        &self,
        target_bpm: f32,
    ) -> (
        Option<StereoBuffer>,
        Option<StereoBuffer>,
        Option<[StereoBuffer; 4]>,
    ) {
        let clamped_target = target_bpm.clamp(MIN_USER_BPM, MAX_USER_BPM);
        let ratio = (clamped_target as f64) / (self.base_bpm.max(MIN_USER_BPM) as f64);

        let stretched_main = self
            .main_track
            .as_ref()
            .map(|b| stretch_stereo_wsola(b, ratio));
        let stretched_ai = self
            .ai_master_track
            .as_ref()
            .map(|b| stretch_stereo_wsola(b, ratio));
        let stretched_stems = self.stems.as_ref().map(|arr| {
            std::array::from_fn(|i| stretch_stereo_wsola(&arr[i], ratio))
        });

        (stretched_main, stretched_ai, stretched_stems)
    }
}

#[inline]
fn interp_autocorr(ac: &[f64], lag_f: f64) -> f64 {
    if ac.is_empty() || lag_f < 1.0 {
        return 0.0;
    }
    let idx0 = lag_f.floor() as usize;
    if idx0 >= ac.len() {
        return 0.0;
    }
    let idx1 = (idx0 + 1).min(ac.len() - 1);
    let frac = lag_f - (idx0 as f64);
    ac[idx0] * (1.0 - frac) + ac[idx1] * frac
}

/// Rilevamento BPM ultra-veloce (`< 15 ms`) su `StereoBuffer`:
/// 1. Estrae una finestra rappresentativa fino a 45s nella zona centrale più energica del brano.
/// 2. Calcola l'Onset Detection Function (ODF) dual-band (cassa/basso `<165 Hz` + rullante/transienti `>165 Hz`)
///    con compressione logaritmica e sottrazione della soglia mobile locale.
/// 3. Esegue autocorrelazione non distorta + banco di filtri a pettine armonico (`1x, 2x, 3x, 4x`)
///    con curva percettiva gaussiana centrata a 120 BPM nel range `68.0..=185.0 BPM`.
pub fn detect_bpm_fast(buf: &StereoBuffer) -> f32 {
    let total_len = buf.len();
    let sr = buf.sample_rate.max(8_000) as usize;
    if total_len < sr * 2 {
        return 120.0;
    }

    // Analizziamo al massimo 45 secondi nel cuore del brano (evitando intro silenziose)
    let max_window = sr * 45;
    let (seg_start, seg_end) = if total_len > max_window + sr * 12 {
        let start = (total_len / 5).min(sr * 15);
        (start, (start + max_window).min(total_len))
    } else {
        (0, total_len.min(max_window))
    };

    let seg_len = seg_end - seg_start;
    // Frame rate ODF ~200 Hz (risoluzione temporale 5 ms, leggerissimo per l'autocorrelazione)
    let hop = (sr / 200).max(32);
    let n_frames = seg_len / hop;
    if n_frames < 80 {
        return 120.0;
    }

    let fps = (sr as f64) / (hop as f64);

    // Coefficiente filtro 1-polo passa-basso a ~165 Hz (campionato a passo 2 per velocità 2x)
    let eff_sr = (sr as f64) * 0.5;
    let lp_alpha = (1.0 - (-2.0 * PI * 165.0 / eff_sr.max(1000.0)).exp()).clamp(0.005, 0.5) as f32;

    let mut lp_state = 0.0_f32;
    let mut prev_mono = 0.0_f32;
    let mut prev_log_low = 0.0_f32;
    let mut prev_log_high = 0.0_f32;
    let mut raw_flux = Vec::with_capacity(n_frames);
    let mut max_flux = 0.0_f32;

    let left = &buf.left[seg_start..seg_end];
    let right = &buf.right[seg_start..seg_end];

    for f in 0..n_frames {
        let base = f * hop;
        let end = (base + hop).min(seg_len);
        let mut e_low = 0.0_f32;
        let mut e_high = 0.0_f32;
        let mut count = 0usize;

        let mut i = base;
        while i < end {
            let m = 0.5_f32 * (left[i] + right[i]);
            lp_state += lp_alpha * (m - lp_state);
            let hp = (m - lp_state) + 0.5_f32 * (m - prev_mono);
            prev_mono = m;

            e_low += lp_state * lp_state;
            e_high += hp * hp;
            count += 1;
            i += 2;
        }

        let inv = 1.0_f32 / (count.max(1) as f32);
        let log_low = (1.0_f32 + 40.0_f32 * (e_low * inv)).ln();
        let log_high = (1.0_f32 + 30.0_f32 * (e_high * inv)).ln();

        let flux = if f == 0 {
            0.0_f32
        } else {
            (log_low - prev_log_low).max(0.0_f32) * 1.35_f32
                + (log_high - prev_log_high).max(0.0_f32)
        };
        prev_log_low = log_low;
        prev_log_high = log_high;
        max_flux = max_flux.max(flux);
        raw_flux.push(flux);
    }

    if max_flux < 1e-5_f32 {
        return 120.0;
    }

    // Sottrazione soglia mobile locale (±10 frame = ±50 ms) per isolare i transienti ritmici netti
    let mut odf = vec![0.0_f32; n_frames];
    let half_w = 10usize;
    let mut running_sum = 0.0_f32;
    for i in 0..half_w.min(n_frames) {
        running_sum += raw_flux[i];
    }
    for i in 0..n_frames {
        let add_idx = i + half_w;
        if add_idx < n_frames {
            running_sum += raw_flux[add_idx];
        }
        if i > half_w {
            running_sum -= raw_flux[i - half_w - 1];
        }
        let w_start = i.saturating_sub(half_w);
        let w_end = (i + half_w + 1).min(n_frames);
        let local_mean = running_sum / ((w_end - w_start).max(1) as f32);
        odf[i] = (raw_flux[i] - local_mean * 1.02_f32).max(0.0_f32);
    }

    let min_lag = ((60.0 * fps / (MAX_DETECT_BPM as f64)).floor() as usize).max(4);
    let max_base_lag = ((60.0 * fps / (MIN_DETECT_BPM as f64)).ceil() as usize).max(min_lag + 4);
    let max_comb_lag = (max_base_lag * 4).min(n_frames * 2 / 3);

    if max_base_lag >= n_frames / 2 {
        return 120.0;
    }

    // Autocorrelazione non distorta (normalizzata per n - lag)
    let calc_start = (min_lag / 2).max(2);
    let mut ac = vec![0.0_f64; max_comb_lag + 2];
    for lag in calc_start..=max_comb_lag {
        let limit = n_frames - lag;
        let mut sum = 0.0_f32;
        for i in 0..limit {
            sum += odf[i] * odf[i + lag];
        }
        ac[lag] = (sum as f64) / (limit.max(1) as f64);
    }

    // Valutazione pettine armonico continuo con passo 0.1 BPM nel range [68.0 ..= 185.0]
    let mut best_bpm = 120.0_f64;
    let mut best_score = -1.0_f64;

    let steps = (((MAX_DETECT_BPM - MIN_DETECT_BPM) * 10.0).round() as usize) + 1;
    for step_idx in 0..steps {
        let bpm = (MIN_DETECT_BPM as f64) + (step_idx as f64) * 0.1;
        let tau = 60.0 * fps / bpm;

        let r1 = interp_autocorr(&ac, tau);
        let r2 = interp_autocorr(&ac, tau * 2.0);
        let r3 = interp_autocorr(&ac, tau * 3.0);
        let r4 = interp_autocorr(&ac, tau * 4.0);

        let mut comb_energy = 1.00 * r1 + 0.50 * r2 + 0.30 * r3 + 0.20 * r4;
        // Piccolo bonus se i multipli dispari confermano il battito fondamentale (anti half-time)
        let r_half = interp_autocorr(&ac, tau * 0.5);
        if r_half > r1 * 0.92 {
            // Forte energia a metà periodo: probabile candidato half-time di un tempo doppio
            comb_energy *= 0.86;
        }

        // Prior percettivo log-gaussiano centrato su 120 BPM (evita errori d'ottava estremi)
        let oct_dist = (bpm / 120.0_f64).log2();
        let perceptual_prior = (-0.5 * (oct_dist / 0.78).powi(2)).exp();

        let score = comb_energy * perceptual_prior;
        if score > best_score {
            best_score = score;
            best_bpm = bpm;
        }
    }

    if best_score <= 0.0 || !best_bpm.is_finite() {
        return 120.0;
    }

    // Arrotonda al decimo di BPM (es. 120.0, 122.5)
    (((best_bpm * 10.0).round() / 10.0) as f32).clamp(MIN_DETECT_BPM, MAX_DETECT_BPM)
}

#[inline]
fn normalized_cross_corr_stride(
    mono: &[f32],
    sq_prefix: &[f64],
    ref_pos: usize,
    cand_pos: usize,
    corr_len: usize,
    stride: usize,
) -> f32 {
    let energy_ref = (sq_prefix[ref_pos + corr_len] - sq_prefix[ref_pos]).max(0.0);
    let energy_cand = (sq_prefix[cand_pos + corr_len] - sq_prefix[cand_pos]).max(0.0);
    let denom = (energy_ref * energy_cand).sqrt();
    if denom < 1e-9 {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut k = 0usize;
    let ref_slice = &mono[ref_pos..ref_pos + corr_len];
    let cand_slice = &mono[cand_pos..cand_pos + corr_len];
    if stride == 1 {
        for i in 0..corr_len {
            dot += ref_slice[i] * cand_slice[i];
        }
        (dot as f64 / denom) as f32
    } else {
        while k < corr_len {
            dot += ref_slice[k] * cand_slice[k];
            k += stride;
        }
        ((dot as f64 * (stride as f64)) / denom) as f32
    }
}

/// Time-Stretching **Stereo Phase-Locked WSOLA**:
/// - Finestra di Hann periodica da `2048` campioni con hop di sintesi `512` (overlap 75% a somma costante).
/// - Ricerca di correlazione incrociata normalizzata (`±480` campioni, stile `SoundTouch::calcCrossCorrNorm`)
///   calcolata sul canale somma mono `L + R` così Left e Right usano **esattamente lo stesso offset intero**:
///   zero sfasamento stereo, transienti di batteria nitidi e zero variazione di intonazione (pitch).
/// - `tempo_ratio = target_bpm / source_bpm` (`> 1.0` accelera, `< 1.0` rallenta).
pub fn stretch_stereo_wsola(buf: &StereoBuffer, tempo_ratio: f64) -> StereoBuffer {
    let in_len = buf.len();
    if !tempo_ratio.is_finite() || (tempo_ratio - 1.0).abs() < 1e-4 || in_len < WSOLA_WIN_LEN + 64 {
        return buf.clone();
    }

    let ratio = tempo_ratio.clamp(0.35, 3.0);
    let out_len = ((in_len as f64) / ratio).round().max(64.0) as usize;

    // Precalcolo finestra di Hann periodica (2048 campioni)
    let mut hann = vec![0.0_f32; WSOLA_WIN_LEN];
    for k in 0..WSOLA_WIN_LEN {
        let phase = (2.0 * PI * (k as f64)) / (WSOLA_WIN_LEN as f64);
        hann[k] = (0.5 * (1.0 - phase.cos())) as f32;
    }

    // Somma mono L + R e prefix-sum dell'energia quadratica per normalizzazione O(1)
    let mut mono = Vec::with_capacity(in_len);
    let mut sq_prefix = Vec::with_capacity(in_len + 1);
    sq_prefix.push(0.0_f64);
    let mut acc_sq = 0.0_f64;
    for i in 0..in_len {
        let m = buf.left[i] + buf.right[i];
        mono.push(m);
        acc_sq += (m as f64) * (m as f64);
        sq_prefix.push(acc_sq);
    }

    let mut out_l = vec![0.0_f32; out_len + WSOLA_WIN_LEN];
    let mut out_r = vec![0.0_f32; out_len + WSOLA_WIN_LEN];
    let mut norm_acc = vec![0.0_f32; out_len + WSOLA_WIN_LEN];

    let max_grain_start = in_len - WSOLA_WIN_LEN;
    let corr_len = WSOLA_CORR_LEN.min(WSOLA_WIN_LEN / 2);
    let max_corr_start = in_len - corr_len;

    let mut prev_chosen_in: usize = 0;
    let mut out_pos: usize = 0;
    let mut hop_idx: usize = 0;

    while out_pos < out_len {
        let chosen_in = if hop_idx == 0 {
            0usize
        } else {
            let expected_in_f = (out_pos as f64) * ratio;
            let expected_in = (expected_in_f.round() as isize)
                .clamp(0, max_grain_start as isize);

            // Continuazione naturale in fase rispetto al grano precedente
            let natural_ref = (prev_chosen_in + WSOLA_SYN_HOP).min(max_corr_start);

            let min_delta = (-WSOLA_SEARCH_TOL).max(-expected_in);
            let max_delta =
                WSOLA_SEARCH_TOL.min((max_grain_start as isize) - expected_in);

            // Fase 1: scansione coarse (passo 4, correlazione stride 2)
            let mut best_coarse_delta = 0isize;
            let mut best_coarse_score = -2.0_f32;
            let inv_tol = 1.0_f32 / (WSOLA_SEARCH_TOL as f32);

            let mut d = min_delta;
            while d <= max_delta {
                let cand = (expected_in + d) as usize;
                let ncc = normalized_cross_corr_stride(
                    &mono,
                    &sq_prefix,
                    natural_ref,
                    cand,
                    corr_len,
                    2,
                );
                let norm_d = (d as f32) * inv_tol;
                let score = ncc - 0.015_f32 * norm_d * norm_d;
                if score > best_coarse_score {
                    best_coarse_score = score;
                    best_coarse_delta = d;
                }
                d += 4;
            }

            // Fase 2: rifinitura fine al singolo campione attorno al miglior candidato coarse
            let fine_min = (best_coarse_delta - 5).max(min_delta);
            let fine_max = (best_coarse_delta + 5).min(max_delta);
            let mut best_delta = best_coarse_delta;
            let mut best_score = -2.0_f32;

            for fd in fine_min..=fine_max {
                let cand = (expected_in + fd) as usize;
                let ncc = normalized_cross_corr_stride(
                    &mono,
                    &sq_prefix,
                    natural_ref,
                    cand,
                    corr_len,
                    1,
                );
                let norm_d = (fd as f32) * inv_tol;
                let score = ncc - 0.015_f32 * norm_d * norm_d;
                if score > best_score {
                    best_score = score;
                    best_delta = fd;
                }
            }

            (expected_in + best_delta) as usize
        };

        // Overlap-Add identico su Left e Right con l'offset fase-bloccato `chosen_in`
        let in_l_slice = &buf.left[chosen_in..chosen_in + WSOLA_WIN_LEN];
        let in_r_slice = &buf.right[chosen_in..chosen_in + WSOLA_WIN_LEN];
        let out_l_slice = &mut out_l[out_pos..out_pos + WSOLA_WIN_LEN];
        let out_r_slice = &mut out_r[out_pos..out_pos + WSOLA_WIN_LEN];
        let norm_slice = &mut norm_acc[out_pos..out_pos + WSOLA_WIN_LEN];

        for k in 0..WSOLA_WIN_LEN {
            let w = hann[k];
            out_l_slice[k] += in_l_slice[k] * w;
            out_r_slice[k] += in_r_slice[k] * w;
            norm_slice[k] += w;
        }

        prev_chosen_in = chosen_in;
        out_pos += WSOLA_SYN_HOP;
        hop_idx += 1;
    }

    out_l.truncate(out_len);
    out_r.truncate(out_len);

    // Normalizzazione per la somma esatta delle finestre di Hann (elimina ogni modulazione di ampiezza)
    for i in 0..out_len {
        let n = norm_acc[i];
        if n > 1e-5_f32 {
            let inv = 1.0_f32 / n;
            out_l[i] = (out_l[i] * inv).clamp(-1.0_f32, 1.0_f32);
            out_r[i] = (out_r[i] * inv).clamp(-1.0_f32, 1.0_f32);
        }
    }

    StereoBuffer::from_vecs(out_l, out_r, buf.sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_drum_pattern(sr: u32, seconds: f32, target_bpm: f32) -> StereoBuffer {
        let total = (sr as f32 * seconds) as usize;
        let mut l = vec![0.0_f32; total];
        let mut r = vec![0.0_f32; total];
        let beat_samples = (60.0_f32 / target_bpm) * (sr as f32);

        let mut beat_idx = 0usize;
        loop {
            let start = (beat_idx as f32 * beat_samples).round() as usize;
            if start >= total {
                break;
            }
            let is_kick = beat_idx % 2 == 0;
            let hit_len = (0.065 * sr as f32) as usize;
            for k in 0..hit_len {
                if start + k >= total {
                    break;
                }
                let t = k as f32 / sr as f32;
                let env = (-t * 48.0).exp();
                let sig = if is_kick {
                    0.85 * env * (2.0 * std::f32::consts::PI * 62.0 * t).sin()
                } else {
                    // Rullante sintetico medio-alto
                    0.70 * env
                        * ((2.0 * std::f32::consts::PI * 210.0 * t).sin()
                            + 0.6 * (2.0 * std::f32::consts::PI * 930.0 * t).sin())
                };
                l[start + k] += sig;
                r[start + k] += sig * 0.9;
            }
            beat_idx += 1;
        }
        StereoBuffer::from_vecs(l, r, sr)
    }

    #[test]
    fn test_detect_bpm_fast_accuracy_across_tempos() {
        for &expected_bpm in &[92.0_f32, 120.0_f32, 142.0_f32] {
            let buf = synth_drum_pattern(44_100, 12.0, expected_bpm);
            let detected = detect_bpm_fast(&buf);
            assert!(
                (detected - expected_bpm).abs() <= 1.0,
                "Expected ~{expected_bpm} BPM, got {detected} BPM"
            );
        }
    }

    #[test]
    fn test_wsola_stereo_phase_coherence_and_length() {
        let sr = 48_000u32;
        let frames = 48_000 * 2; // 2.0 s
        let mut l = Vec::with_capacity(frames);
        let mut r = Vec::with_capacity(frames);
        for i in 0..frames {
            let t = i as f32 / sr as f32;
            // Segnale stereo in cui R = -0.5 * L esatto: se L e R condividono lo stesso offset,
            // questo rapporto resta invariato al 100% dopo il WSOLA!
            let s = 0.6 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.25 * (2.0 * std::f32::consts::PI * 660.0 * t).sin();
            l.push(s);
            r.push(-0.5 * s);
        }
        let buf = StereoBuffer::from_vecs(l, r, sr);

        // Rallentamento a 0.80x (+25% durata)
        let slower = stretch_stereo_wsola(&buf, 0.80);
        assert_eq!(slower.len(), 120_000);
        for i in 2048..(slower.len() - 2048) {
            assert!(slower.left[i].is_finite() && slower.right[i].is_finite());
            let phase_err = (slower.right[i] - (-0.5 * slower.left[i])).abs();
            assert!(
                phase_err < 1e-5,
                "Sfasamento stereo rilevato al campione {i}: err={phase_err}"
            );
        }

        // Accelerazione a 1.25x (-20% durata)
        let faster = stretch_stereo_wsola(&buf, 1.25);
        assert_eq!(faster.len(), 76_800);
        for i in 2048..(faster.len() - 2048) {
            let phase_err = (faster.right[i] - (-0.5 * faster.left[i])).abs();
            assert!(phase_err < 1e-5);
        }
    }

    #[test]
    fn test_original_tempo_refs_sync_after_audio_edit() {
        let sr = 44_100u32;
        let orig = synth_drum_pattern(sr, 4.0, 120.0); // 176_400 frame
        let mut refs = OriginalTempoRefs::default();
        refs.store_loaded(Some(orig), None, None, 120.0);

        // Simuliamo un ritaglio (crop) dell'utente a metà durata (2.0 s = 88_200 frame)
        let mut st = SharedAudioState::new(sr);
        st.main_track = Some(synth_drum_pattern(sr, 2.0, 120.0));
        refs.sync_after_audio_edit(&st);

        // Cambiando BPM da 120 -> 150 BPM (ratio 1.25x), la durata parte dal buffer editato (88_200 / 1.25 = 70_560)
        let (new_main, new_ai, new_stems) = refs.render_for_target_bpm(150.0);
        assert!(new_ai.is_none() && new_stems.is_none());
        let stretched = new_main.expect("stretched main expected");
        assert_eq!(stretched.len(), 70_560);
    }
}
