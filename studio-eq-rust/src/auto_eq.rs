use std::f64::consts::PI;

use rustfft::{num_complex::Complex, FftPlanner};

use crate::audio_io::StereoBuffer;
use crate::dsp::{BiquadCoeffs, DspParams, EqBandParam, FilterKind};

const N_FFT: usize = 8192;
const MAX_ACTIVE_BLOCKS: usize = 120;
const MAX_CANDIDATE_BLOCKS: usize = 360;
const ACTIVE_RMS_GATE_DB: f64 = -42.0_f64;
const FALLBACK_RMS_GATE_DB: f64 = -58.0_f64;
const MIN_SAFE_GAIN_DB: f64 = -7.5_f64;
const MAX_SAFE_GAIN_DB: f64 = 8.0_f64;
const STUDIO_PINK_SLOPE_DB_OCT: f64 = -3.5_f64;

#[derive(Clone, Copy)]
struct BlockMetrics {
    start: usize,
    len: usize,
    rms_db: f64,
    mid_sq: f64,
    side_sq: f64,
    peak_lin: f64,
}

/// Analizza lo spettro reale del brano tramite Welch PSD (finestra di Hann a 8192 punti)
/// e calcola istantaneamente (< 80 ms) una catena `DspParams` cucita su misura per
/// correggere risonanze modali da sala prove, rimbombo (mud), inscatolamento (boxiness),
/// carenza vocale, acidità microfonica e roll-off sulle alte frequenze.
pub fn analyze_and_compute_auto_eq(buf: &StereoBuffer) -> (DspParams, String) {
    let total_len = buf.len();
    let sr = (buf.sample_rate.max(8000)) as f64;

    if total_len < 1024 {
        return (
            DspParams::preset_sala_prove(),
            "Auto-EQ: traccia troppo breve per l'analisi spettrale, applicato profilo Sala Prove."
                .to_string(),
        );
    }

    // 1. Scansione veloce dei blocchi candidati uniformemente distribuiti lungo il brano
    let candidates = scan_candidate_blocks(buf, total_len);
    let mut active: Vec<BlockMetrics> = candidates
        .iter()
        .copied()
        .filter(|b| b.rms_db > ACTIVE_RMS_GATE_DB)
        .collect();

    if active.len() < 4 {
        active = candidates
            .into_iter()
            .filter(|b| b.rms_db > FALLBACK_RMS_GATE_DB)
            .collect();
    }

    if active.is_empty() {
        return (
            DspParams::preset_sala_prove(),
            "Auto-EQ: segnale silente rilevato, mantenuto profilo Sala Prove standard.".to_string(),
        );
    }

    // Campionamento uniforme fino a MAX_ACTIVE_BLOCKS (~120 blocchi -> < 15 ms di FFT!)
    let selected = subsample_uniform(&active, MAX_ACTIVE_BLOCKS);

    // Statistiche energetiche stereo e dinamiche sui blocchi musicali attivi
    let mut total_mid_sq = 0.0_f64;
    let mut total_side_sq = 0.0_f64;
    let mut total_frames_acc = 0usize;
    let mut block_peaks_db: Vec<f64> = Vec::with_capacity(selected.len());

    for blk in &selected {
        total_mid_sq += blk.mid_sq;
        total_side_sq += blk.side_sq;
        total_frames_acc += blk.len;
        block_peaks_db.push(20.0_f64 * blk.peak_lin.max(1e-9_f64).log10());
    }

    let mean_mid_rms = (total_mid_sq / (total_frames_acc.max(1) as f64)).sqrt();
    let active_rms_db = 20.0_f64 * mean_mid_rms.max(1e-9_f64).log10();
    let side_mid_ratio = (total_side_sq / total_mid_sq.max(1e-15_f64)).sqrt();

    block_peaks_db.sort_by(|a, b| a.total_cmp(b));
    let p90_idx = ((block_peaks_db.len() * 9) / 10).min(block_peaks_db.len() - 1);
    let peak_90_db = block_peaks_db[p90_idx];
    let crest_db = (peak_90_db - active_rms_db).clamp(3.0_f64, 24.0_f64);

    // 2. Stima Welch PSD con finestra di Hann a 8192 punti
    let psd_lin = compute_welch_psd_linear(buf, &selected, sr);
    let num_bins = psd_lin.len(); // N_FFT / 2 + 1

    // Doppio smoothing logaritmico O(N) tramite somme prefisse in potenza:
    // - `fine_db`: ~1/12 d'ottava per individuare la frequenza esatta di picchi modali e risonanze
    // - `broad_db`: ~1/2.5 d'ottava per valutare il bilanciamento tonale macroscopico di studio
    let fine_db = smooth_psd_log_octaves(&psd_lin, sr, 0.042_f64);
    let broad_db = smooth_psd_log_octaves(&psd_lin, sr, 0.20_f64);

    // 3. Allineamento robusto alla curva target da studio (~ -3.5 dB/ottava stile pink-noise morbido)
    let anchor_db = compute_robust_target_anchor(&broad_db, sr);

    let mut fine_err = vec![0.0_f64; num_bins];
    let mut broad_err = vec![0.0_f64; num_bins];
    for k in 1..num_bins {
        let freq = (k as f64) * sr / (N_FFT as f64);
        let target = anchor_db + studio_target_shape_db(freq);
        fine_err[k] = fine_db[k] - target;
        broad_err[k] = broad_db[k] - target;
    }

    // Misura pre-EQ del rimbombo grezzo e della carenza d'aria per calibrare processori dinamici/armonici
    let raw_mud_excess = band_log_mean(&broad_err, sr, 130.0_f64, 340.0_f64);
    let raw_box_excess = band_log_mean(&broad_err, sr, 360.0_f64, 640.0_f64);
    let raw_air_err = estimate_air_error(&broad_err, sr);

    // 4. Calibrazione Sub-Cut High-Pass anti-rumble (dual-stage 24 dB/oct)
    let rumble_db = band_power_db(&psd_lin, sr, 22.0_f64, 48.0_f64);
    let bass_ref_db = band_power_db(&psd_lin, sr, 68.0_f64, 128.0_f64);
    let rumble_diff = rumble_db - bass_ref_db;
    let sub_cut_hz = round_1dec((41.0_f64 + 1.35_f64 * (rumble_diff + 4.5_f64)).clamp(32.0_f64, 56.0_f64));

    let hp_coeffs = BiquadCoeffs::design(FilterKind::HighPass, sub_cut_hz, sr, 0.0_f64, 0.707_f64);
    for k in 1..num_bins {
        let freq = (k as f64) * sr / (N_FFT as f64);
        let hp_resp = 2.0_f64 * hp_coeffs.magnitude_db(freq, sr);
        fine_err[k] += hp_resp;
        broad_err[k] += hp_resp;
    }

    // 5. Fitting spettrale residuo sequenziale delle 8 bande parametriche
    //    Ogni banda aggiorna `fine_err` e `broad_err` sottraendo la propria risposta biquad esatta
    //    così le bande adiacenti non sommano mai tagli o boost ridondanti!

    // --- Banda 1 (indice 1): Mud / Rimbombo Sala (135 - 260 Hz) ---
    let b1_mean = band_log_mean(&broad_err, sr, 135.0_f64, 260.0_f64);
    let (b1_pk_f, b1_pk_err, b1_prom) =
        find_peak_in_range(&fine_err, &broad_err, sr, 138.0_f64, 256.0_f64);
    let (b1_freq, b1_gain, b1_q) = if b1_pk_err > 0.45_f64 || b1_mean > 0.25_f64 {
        let excess = 0.56_f64 * b1_pk_err.max(0.0_f64)
            + 0.44_f64 * b1_mean.max(0.0_f64)
            + 0.24_f64 * b1_prom.max(0.0_f64);
        let gain = (-0.82_f64 * excess).clamp(MIN_SAFE_GAIN_DB, 0.0_f64);
        let q = (1.42_f64 + 0.28_f64 * b1_prom.clamp(0.0_f64, 4.5_f64)).clamp(1.40_f64, 2.75_f64);
        (b1_pk_f, gain, q)
    } else if b1_mean < -1.10_f64 {
        let (dip_f, dip_err) = find_dip_in_range(&broad_err, &fine_err, sr, 145.0_f64, 245.0_f64);
        let deficit = -(0.70_f64 * b1_mean + 0.30_f64 * dip_err);
        let gain = (0.55_f64 * deficit).clamp(0.0_f64, 4.4_f64);
        (dip_f, gain, 1.15_f64)
    } else {
        (b1_pk_f, (-0.45_f64 * b1_mean).clamp(-1.0_f64, 1.0_f64), 1.50_f64)
    };
    let band1 = make_band(
        "Mud Sala",
        "Anti-Mud Sala",
        FilterKind::Peaking,
        b1_freq,
        b1_gain,
        b1_q,
    );
    apply_band_response(&band1, sr, &mut fine_err, &mut broad_err);

    // --- Banda 0 (indice 0): Risonanza Bassi (70 - 135 Hz) ---
    let b0_hi = (band1.freq * 0.84_f64).clamp(106.0_f64, 135.0_f64);
    let b0_mean = band_log_mean(&broad_err, sr, 70.0_f64, b0_hi);
    let (b0_pk_f, b0_pk_err, b0_prom) =
        find_peak_in_range(&fine_err, &broad_err, sr, 70.0_f64, b0_hi);
    let (b0_freq, b0_gain, b0_q) =
        if b0_pk_err > 0.70_f64 || (b0_prom > 1.45_f64 && b0_pk_err > 0.15_f64) {
            let excess = 0.62_f64 * b0_pk_err.max(0.0_f64) + 0.38_f64 * b0_prom.max(0.0_f64);
            let gain = (-0.78_f64 * excess).clamp(MIN_SAFE_GAIN_DB, 0.0_f64);
            let q = (2.05_f64 + 0.34_f64 * b0_prom.clamp(0.0_f64, 4.5_f64)).clamp(1.95_f64, 3.60_f64);
            (b0_pk_f, gain, q)
        } else if b0_mean < -0.95_f64 {
            let (dip_f, dip_err) = find_dip_in_range(&broad_err, &fine_err, sr, 74.0_f64, 118.0_f64);
            let deficit = -(0.65_f64 * b0_mean + 0.35_f64 * dip_err);
            let gain = (0.68_f64 * deficit).clamp(0.0_f64, 6.0_f64);
            (dip_f.clamp(75.0_f64, 115.0_f64), gain, 1.35_f64)
        } else {
            (b0_pk_f, (-0.45_f64 * b0_pk_err).clamp(-1.4_f64, 1.4_f64), 2.20_f64)
        };
    let band0 = make_band(
        "Bassi",
        "Risonanza Basso",
        FilterKind::Peaking,
        b0_freq,
        b0_gain,
        b0_q,
    );
    apply_band_response(&band0, sr, &mut fine_err, &mut broad_err);

    // --- Banda 3 (indice 3): Effetto Scatola / Boxiness (360 - 650 Hz) ---
    let b3_mean = band_log_mean(&broad_err, sr, 360.0_f64, 650.0_f64);
    let (b3_pk_f, b3_pk_err, b3_prom) =
        find_peak_in_range(&fine_err, &broad_err, sr, 366.0_f64, 644.0_f64);
    let (b3_freq, b3_gain, b3_q) = if b3_pk_err > 0.40_f64 || b3_mean > 0.20_f64 {
        let excess = 0.56_f64 * b3_pk_err.max(0.0_f64)
            + 0.44_f64 * b3_mean.max(0.0_f64)
            + 0.22_f64 * b3_prom.max(0.0_f64);
        let gain = (-0.80_f64 * excess).clamp(MIN_SAFE_GAIN_DB, 0.0_f64);
        let q = (1.65_f64 + 0.26_f64 * b3_prom.clamp(0.0_f64, 4.2_f64)).clamp(1.60_f64, 2.85_f64);
        (b3_pk_f, gain, q)
    } else if b3_mean < -1.25_f64 {
        let (dip_f, dip_err) = find_dip_in_range(&broad_err, &fine_err, sr, 380.0_f64, 620.0_f64);
        let deficit = -(0.70_f64 * b3_mean + 0.30_f64 * dip_err);
        let gain = (0.56_f64 * deficit).clamp(0.0_f64, 4.5_f64);
        (dip_f, gain, 1.25_f64)
    } else {
        (b3_pk_f, (-0.45_f64 * b3_mean).clamp(-1.2_f64, 1.2_f64), 1.85_f64)
    };
    let band3 = make_band(
        "Scatola",
        "Anti-Scatola (Box)",
        FilterKind::Peaking,
        b3_freq,
        b3_gain,
        b3_q,
    );
    apply_band_response(&band3, sr, &mut fine_err, &mut broad_err);

    // --- Banda 2 (indice 2): Medio-Bassi (220 - 360 Hz), calcolata sul residuo tra Banda 1 e Banda 3 ---
    let b2_lo = (band1.freq * 1.18_f64).clamp(220.0_f64, 305.0_f64);
    let b2_hi = (band3.freq * 0.85_f64).clamp(b2_lo + 24.0_f64, 360.0_f64);
    let b2_mean = band_log_mean(&broad_err, sr, b2_lo, b2_hi);
    let (b2_pk_f, b2_pk_err, b2_prom) =
        find_peak_in_range(&fine_err, &broad_err, sr, b2_lo, b2_hi);
    let (b2_freq, b2_gain, b2_q) = if b2_pk_err > 0.40_f64 || b2_mean > 0.25_f64 {
        let excess = 0.55_f64 * b2_pk_err.max(0.0_f64)
            + 0.45_f64 * b2_mean.max(0.0_f64)
            + 0.20_f64 * b2_prom.max(0.0_f64);
        let gain = (-0.75_f64 * excess).clamp(MIN_SAFE_GAIN_DB, 0.0_f64);
        let q = (1.38_f64 + 0.24_f64 * b2_prom.clamp(0.0_f64, 3.6_f64)).clamp(1.35_f64, 2.40_f64);
        (b2_pk_f, gain, q)
    } else if b2_mean < -1.25_f64 {
        let (dip_f, dip_err) = find_dip_in_range(&broad_err, &fine_err, sr, b2_lo, b2_hi);
        let deficit = -(0.70_f64 * b2_mean + 0.30_f64 * dip_err);
        let gain = (0.55_f64 * deficit).clamp(0.0_f64, 4.2_f64);
        (dip_f, gain, 1.25_f64)
    } else {
        (
            0.5_f64 * (b2_lo + b2_hi),
            (-0.45_f64 * b2_mean).clamp(-1.2_f64, 1.2_f64),
            1.40_f64,
        )
    };
    let band2 = make_band(
        "Low-Mid",
        "Pulizia Medio-Bassi",
        FilterKind::Peaking,
        b2_freq,
        b2_gain,
        b2_q,
    );
    apply_band_response(&band2, sr, &mut fine_err, &mut broad_err);

    // --- Banda 7 (indice 7): Studio Air High-Shelf (7500 - 15000 Hz) ---
    //     Calcolata prima della Presenza (3.8-6 kHz) così la coda morbida dello shelf viene già compensata!
    let air_err_res = estimate_air_error(&broad_err, sr);
    let upper_mid_bright = band_log_mean(&broad_err, sr, 5800.0_f64, 7800.0_f64);
    let b7_freq = (8350.0_f64 + 190.0_f64 * upper_mid_bright.clamp(-3.5_f64, 4.5_f64))
        .clamp(7650.0_f64, 9600.0_f64);
    let b7_gain = (-0.80_f64 * air_err_res).clamp(MIN_SAFE_GAIN_DB, MAX_SAFE_GAIN_DB);
    let band7 = make_band(
        "Aria",
        "Studio Air Shelf",
        FilterKind::HighShelf,
        b7_freq,
        b7_gain,
        0.707_f64,
    );
    apply_band_response(&band7, sr, &mut fine_err, &mut broad_err);

    // --- Banda 4 (indice 4): Medie Vocali & Chitarre (1000 - 2200 Hz) ---
    let b4_mean = band_log_mean(&broad_err, sr, 1000.0_f64, 2200.0_f64);
    let (b4_dip_f, b4_dip_err) =
        find_dip_in_range(&broad_err, &fine_err, sr, 1060.0_f64, 2050.0_f64);
    let (b4_pk_f, b4_pk_err, b4_prom) =
        find_peak_in_range(&fine_err, &broad_err, sr, 1050.0_f64, 2120.0_f64);
    let (b4_freq, b4_gain, b4_q) =
        if b4_mean < -0.30_f64 || (b4_dip_err < -1.35_f64 && b4_mean < 0.35_f64) {
            let deficit = -(0.62_f64 * b4_mean.min(0.0_f64) + 0.38_f64 * b4_dip_err.min(0.0_f64));
            let gain = (0.84_f64 * deficit).clamp(0.0_f64, MAX_SAFE_GAIN_DB);
            let q = (0.82_f64 + 0.035_f64 * gain).clamp(0.80_f64, 1.15_f64);
            (b4_dip_f.clamp(1080.0_f64, 2020.0_f64), gain, q)
        } else if b4_pk_err > 1.10_f64 || b4_mean > 0.70_f64 {
            let excess = 0.55_f64 * b4_mean.max(0.0_f64) + 0.45_f64 * b4_pk_err.max(0.0_f64);
            let gain = (-0.75_f64 * excess).clamp(MIN_SAFE_GAIN_DB, 0.0_f64);
            let q = (1.25_f64 + 0.22_f64 * b4_prom.clamp(0.0_f64, 3.2_f64)).clamp(1.20_f64, 2.05_f64);
            (b4_pk_f, gain, q)
        } else {
            (
                1420.0_f64,
                (-0.55_f64 * b4_mean).clamp(-1.4_f64, 1.8_f64),
                0.90_f64,
            )
        };
    let band4 = make_band(
        "Voce",
        "Voce & Chitarre",
        FilterKind::Peaking,
        b4_freq,
        b4_gain,
        b4_q,
    );
    apply_band_response(&band4, sr, &mut fine_err, &mut broad_err);

    // --- Banda 6 (indice 6): Attacco & Presenza (3800 - 6000 Hz) ---
    let b6_hi = 6000.0_f64.min(sr * 0.45_f64).max(4200.0_f64);
    let b6_mean = band_log_mean(&broad_err, sr, 3800.0_f64, b6_hi);
    let (b6_dip_f, b6_dip_err) =
        find_dip_in_range(&broad_err, &fine_err, sr, 3920.0_f64, (b6_hi - 150.0_f64).max(4050.0_f64));
    let (b6_pk_f, b6_pk_err, b6_prom) =
        find_peak_in_range(&fine_err, &broad_err, sr, 3850.0_f64, b6_hi);
    let (b6_freq, b6_gain, b6_q) = if b6_mean < -0.30_f64 {
        let deficit = -(0.65_f64 * b6_mean.min(0.0_f64) + 0.35_f64 * b6_dip_err.min(0.0_f64));
        let gain = (0.78_f64 * deficit).clamp(0.0_f64, 6.8_f64);
        (b6_dip_f.clamp(3950.0_f64, 5800.0_f64), gain, 1.00_f64)
    } else if b6_pk_err > 0.95_f64 || b6_mean > 0.55_f64 {
        let excess = 0.55_f64 * b6_pk_err.max(0.0_f64)
            + 0.45_f64 * b6_mean.max(0.0_f64)
            + 0.20_f64 * b6_prom.max(0.0_f64);
        let gain = (-0.75_f64 * excess).clamp(MIN_SAFE_GAIN_DB, 0.0_f64);
        let q = (1.35_f64 + 0.25_f64 * b6_prom.clamp(0.0_f64, 3.5_f64)).clamp(1.30_f64, 2.40_f64);
        (b6_pk_f, gain, q)
    } else {
        (
            4800.0_f64,
            (-0.50_f64 * b6_mean).clamp(-1.4_f64, 1.8_f64),
            1.05_f64,
        )
    };
    let band6 = make_band(
        "Presenza",
        "Attacco & Presenza",
        FilterKind::Peaking,
        b6_freq,
        b6_gain,
        b6_q,
    );
    apply_band_response(&band6, sr, &mut fine_err, &mut broad_err);

    // --- Banda 5 (indice 5): Picco Acido Microfonico (2100 - 3600 Hz) ---
    let b5_lo = (band4.freq * 1.18_f64).clamp(2100.0_f64, 2900.0_f64);
    let b5_hi = (band6.freq * 0.85_f64).clamp(b5_lo + 220.0_f64, 3600.0_f64);
    let b5_mean = band_log_mean(&broad_err, sr, b5_lo, b5_hi);
    let (b5_pk_f, b5_pk_err, b5_prom) =
        find_peak_in_range(&fine_err, &broad_err, sr, b5_lo, b5_hi);
    let (b5_freq, b5_gain, b5_q) = if b5_pk_err > 0.35_f64 || b5_prom > 1.05_f64 {
        let harsh = 0.58_f64 * b5_pk_err.max(0.0_f64) + 0.56_f64 * b5_prom.max(0.0_f64);
        let gain = (-0.76_f64 * harsh).clamp(MIN_SAFE_GAIN_DB, -0.3_f64);
        let q = (2.55_f64 + 0.38_f64 * b5_prom.clamp(0.0_f64, 4.5_f64)).clamp(2.40_f64, 4.40_f64);
        (b5_pk_f, gain, q)
    } else if b5_mean < -1.90_f64 {
        let (dip_f, dip_err) = find_dip_in_range(&broad_err, &fine_err, sr, b5_lo, b5_hi);
        let deficit = -(0.70_f64 * b5_mean + 0.30_f64 * dip_err);
        // Massimo +3.6 dB nella zona 2.1-3.6 kHz per non introdurre mai durezza d'ascolto
        let gain = (0.52_f64 * deficit).clamp(0.0_f64, 3.6_f64);
        (dip_f, gain, 1.25_f64)
    } else {
        let gentle_cut =
            (-0.45_f64 * (b5_pk_err.max(0.0_f64) + 0.45_f64 * b5_prom.max(0.0_f64))).clamp(-1.8_f64, 0.0_f64);
        (b5_pk_f, gentle_cut, 3.20_f64)
    };
    let band5 = make_band(
        "Acidità Mic",
        "Taglio Acidita Mic",
        FilterKind::Peaking,
        b5_freq,
        b5_gain,
        b5_q,
    );

    let mut bands = vec![band0, band1, band2, band3, band4, band5, band6, band7];
    enforce_cumulative_headroom(&mut bands, sr);

    // 6. Calibrazione intelligente dei moduli Master (Dynamic Anti-Mud, Exciter, Stereo M/S, Glue Comp)
    let mud_pressure = (0.68_f64 * raw_mud_excess.max(0.0_f64)
        + 0.32_f64 * raw_box_excess.max(0.0_f64))
    .clamp(0.0_f64, 10.0_f64);
    let dynamic_mud_amount = round_2dec((0.22_f64 + 0.115_f64 * mud_pressure).clamp(0.18_f64, 1.32_f64));
    let dynamic_mud_enabled = dynamic_mud_amount > 0.08_f64;

    let air_deficit = (-raw_air_err).clamp(0.0_f64, 12.0_f64);
    let exciter_drive = round_2dec((0.05_f64 + 0.024_f64 * air_deficit).clamp(0.04_f64, 0.34_f64));
    let exciter_enabled = exciter_drive > 0.02_f64;

    let stereo_width = compute_optimal_stereo_width(side_mid_ratio);
    let mono_bass_hz = (bands[1].freq * 0.80_f64).round().clamp(115.0_f64, 150.0_f64);

    // Stima della variazione energetica complessiva post-EQ per tarare il Bus Glue Compressor (-17 dB thresh)
    let eq_power_shift_db = estimate_eq_loudness_shift_db(&psd_lin, &bands, sub_cut_hz, sr);
    let est_post_eq_rms_db = active_rms_db + eq_power_shift_db;
    let target_bus_rms_db = -14.8_f64;
    let level_need = (target_bus_rms_db - est_post_eq_rms_db).clamp(-2.0_f64, 7.5_f64);
    let crest_bonus = ((crest_db - 11.5_f64) * 0.22_f64).clamp(-0.6_f64, 1.6_f64);
    let comp_drive_db = round_1dec((1.4_f64 + 0.58_f64 * level_need + crest_bonus).clamp(0.8_f64, 6.8_f64));

    let mut params = DspParams::preset_sala_prove();
    params.bypass = false;
    params.sub_cut_hz = sub_cut_hz;
    params.bands = bands;
    params.dynamic_mud_enabled = dynamic_mud_enabled;
    params.dynamic_mud_amount = dynamic_mud_amount;
    params.exciter_enabled = exciter_enabled;
    params.exciter_drive = exciter_drive;
    params.mid_side_enabled = true;
    params.mono_bass_hz = mono_bass_hz;
    params.stereo_width = stereo_width;
    params.comp_enabled = true;
    params.comp_drive_db = comp_drive_db;
    params.ceiling_db = -0.4_f64;

    let report = format_diagnostic_summary(&params);
    (params, report)
}

fn scan_candidate_blocks(buf: &StereoBuffer, total_len: usize) -> Vec<BlockMetrics> {
    if total_len <= N_FFT {
        return vec![measure_block(buf, 0, total_len)];
    }

    let max_start = total_len - N_FFT;
    let natural_hop = N_FFT / 2;
    let natural_count = max_start / natural_hop + 1;
    let count = natural_count.min(MAX_CANDIDATE_BLOCKS);

    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        let start = if count <= 1 {
            0
        } else {
            (idx * max_start) / (count - 1)
        };
        out.push(measure_block(buf, start, N_FFT));
    }
    out
}

fn measure_block(buf: &StereoBuffer, start: usize, len: usize) -> BlockMetrics {
    let end = (start + len).min(buf.len());
    let actual_len = end.saturating_sub(start).max(1);
    let mut mid_sq = 0.0_f64;
    let mut side_sq = 0.0_f64;
    let mut peak_lin = 0.0_f64;

    for i in start..end {
        let l = buf.left[i] as f64;
        let r = buf.right[i] as f64;
        if !l.is_finite() || !r.is_finite() {
            continue;
        }
        let mid = 0.5_f64 * (l + r);
        let side = 0.5_f64 * (l - r);
        mid_sq += mid * mid;
        side_sq += side * side;
        let pk = l.abs().max(r.abs());
        if pk > peak_lin {
            peak_lin = pk;
        }
    }

    let mean_sq = (mid_sq / (actual_len as f64)).max(1e-16_f64);
    let rms_db = 10.0_f64 * mean_sq.log10();

    BlockMetrics {
        start,
        len: actual_len,
        rms_db,
        mid_sq,
        side_sq,
        peak_lin,
    }
}

fn subsample_uniform(blocks: &[BlockMetrics], max_count: usize) -> Vec<BlockMetrics> {
    if blocks.len() <= max_count {
        return blocks.to_vec();
    }
    let mut out = Vec::with_capacity(max_count);
    let last = blocks.len() - 1;
    let denom = (max_count - 1).max(1);
    for i in 0..max_count {
        let idx = (i * last) / denom;
        out.push(blocks[idx]);
    }
    out
}

fn compute_welch_psd_linear(buf: &StereoBuffer, blocks: &[BlockMetrics], sr: f64) -> Vec<f64> {
    let mut hann = vec![0.0_f32; N_FFT];
    let mut win_pwr_sum = 0.0_f64;
    let denom = (N_FFT - 1) as f64;
    for (i, w) in hann.iter_mut().enumerate() {
        let val = 0.5_f64 * (1.0_f64 - (2.0_f64 * PI * (i as f64) / denom).cos());
        *w = val as f32;
        win_pwr_sum += val * val;
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut scratch = vec![Complex { re: 0.0_f32, im: 0.0_f32 }; fft.get_inplace_scratch_len()];
    let mut fft_buf = vec![Complex { re: 0.0_f32, im: 0.0_f32 }; N_FFT];

    let num_bins = N_FFT / 2 + 1;
    let mut psd_acc = vec![0.0_f64; num_bins];

    for blk in blocks {
        let end = (blk.start + blk.len).min(buf.len());
        let copy_len = end.saturating_sub(blk.start).min(N_FFT);

        for k in 0..copy_len {
            let idx = blk.start + k;
            let l = buf.left[idx];
            let r = buf.right[idx];
            let mono = if l.is_finite() && r.is_finite() {
                0.5_f32 * (l + r)
            } else {
                0.0_f32
            };
            fft_buf[k] = Complex {
                re: mono * hann[k],
                im: 0.0_f32,
            };
        }
        for c in &mut fft_buf[copy_len..N_FFT] {
            *c = Complex {
                re: 0.0_f32,
                im: 0.0_f32,
            };
        }

        fft.process_with_scratch(&mut fft_buf, &mut scratch);

        for (k, acc) in psd_acc.iter_mut().enumerate() {
            let c = fft_buf[k];
            let re = c.re as f64;
            let im = c.im as f64;
            *acc += re * re + im * im;
        }
    }

    let norm_base = ((blocks.len().max(1) as f64) * win_pwr_sum * sr).max(1e-12_f64);
    for (k, p) in psd_acc.iter_mut().enumerate() {
        let one_sided = if k == 0 || k == N_FFT / 2 {
            1.0_f64
        } else {
            2.0_f64
        };
        *p = (*p * one_sided / norm_base).max(1e-20_f64);
    }

    psd_acc
}

/// Doppio passaggio di media mobile in scala logaritmica (equivalente a kernel triangolare in potenza)
/// calcolato in tempo O(N) tramite array di somme prefisse.
fn smooth_psd_log_octaves(psd_lin: &[f64], sr: f64, half_octaves: f64) -> Vec<f64> {
    let n = psd_lin.len();
    let pass1 = boxcar_octaves_power(psd_lin, sr, half_octaves * 0.72_f64);
    let pass2 = boxcar_octaves_power(&pass1, sr, half_octaves * 0.72_f64);

    let mut out_db = vec![-120.0_f64; n];
    for k in 0..n {
        out_db[k] = 10.0_f64 * pass2[k].max(1e-20_f64).log10();
    }
    out_db
}

fn boxcar_octaves_power(src: &[f64], sr: f64, half_oct: f64) -> Vec<f64> {
    let n = src.len();
    let mut prefix = vec![0.0_f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + src[i];
    }

    let ratio_lo = 2.0_f64.powf(-half_oct);
    let ratio_hi = 2.0_f64.powf(half_oct);
    let min_musical_bin = ((35.0_f64 * (N_FFT as f64)) / sr).ceil() as usize;

    let mut dst = vec![0.0_f64; n];
    dst[0] = src[0];
    for k in 1..n {
        if k < min_musical_bin {
            dst[k] = src[k];
            continue;
        }
        let lo = ((k as f64) * ratio_lo).floor().max(1.0_f64) as usize;
        let hi = (((k as f64) * ratio_hi).ceil() as usize).min(n - 1).max(lo);
        let cnt = (hi - lo + 1) as f64;
        dst[k] = ((prefix[hi + 1] - prefix[lo]) / cnt).max(1e-20_f64);
    }
    dst
}

/// Curva target da studio bilanciata: inclinazione morbida stile pink-noise (-3.5 dB/ottava)
/// centrata a 0.0 dB su 1 kHz, con plateau naturale sui bassi sotto 85 Hz.
#[inline]
fn studio_target_shape_db(freq: f64) -> f64 {
    let f = freq.clamp(20.0_f64, 20_000.0_f64);
    let f_tilt = f.max(85.0_f64);
    let base_tilt = STUDIO_PINK_SLOPE_DB_OCT * (f_tilt / 1000.0_f64).log2();
    if f < 55.0_f64 {
        base_tilt - 5.5_f64 * (55.0_f64 / f).log2()
    } else {
        base_tilt
    }
}

/// Ancora verticalmente la curva target allo spettro del brano usando una media troncata (trimmed mean)
/// su 12 sotto-bande a mezza ottava (85 Hz .. 5600 Hz), immune sia ai picchi di mud che ai buchi vocali.
fn compute_robust_target_anchor(broad_db: &[f64], sr: f64) -> f64 {
    const SUB_BANDS: [(f64, f64); 12] = [
        (85.0_f64, 120.0_f64),
        (120.0_f64, 170.0_f64),
        (170.0_f64, 240.0_f64),
        (240.0_f64, 340.0_f64),
        (340.0_f64, 480.0_f64),
        (480.0_f64, 680.0_f64),
        (680.0_f64, 960.0_f64),
        (960.0_f64, 1360.0_f64),
        (1360.0_f64, 1920.0_f64),
        (1920.0_f64, 2720.0_f64),
        (2720.0_f64, 3840.0_f64),
        (3840.0_f64, 5500.0_f64),
    ];

    let mut offsets: Vec<f64> = Vec::with_capacity(SUB_BANDS.len());
    for &(f_lo, f_hi) in &SUB_BANDS {
        if f_lo >= sr * 0.44_f64 {
            continue;
        }
        let hi_clamped = f_hi.min(sr * 0.46_f64);
        let center = (f_lo * hi_clamped).sqrt();
        let meas = band_log_mean(broad_db, sr, f_lo, hi_clamped);
        offsets.push(meas - studio_target_shape_db(center));
    }

    if offsets.is_empty() {
        return -45.0_f64;
    }

    offsets.sort_by(|a, b| a.total_cmp(b));
    let trim = offsets.len() / 4;
    let slice = &offsets[trim..offsets.len().saturating_sub(trim).max(trim + 1)];
    let sum: f64 = slice.iter().sum();
    sum / (slice.len() as f64)
}

fn band_power_db(psd_lin: &[f64], sr: f64, f_min: f64, f_max: f64) -> f64 {
    let (k0, k1) = freq_range_to_bins(sr, psd_lin.len(), f_min, f_max);
    let mut sum = 0.0_f64;
    let mut cnt = 0usize;
    for &p in &psd_lin[k0..=k1] {
        sum += p;
        cnt += 1;
    }
    let avg = (sum / (cnt.max(1) as f64)).max(1e-20_f64);
    10.0_f64 * avg.log10()
}

/// Media pesata per ottava (`1 / k`) di uno spettro in dB nell'intervallo `[f_min, f_max]`.
fn band_log_mean(spec_db: &[f64], sr: f64, f_min: f64, f_max: f64) -> f64 {
    let (k0, k1) = freq_range_to_bins(sr, spec_db.len(), f_min, f_max);
    let mut w_sum = 0.0_f64;
    let mut acc = 0.0_f64;
    for (offset, &val) in spec_db[k0..=k1].iter().enumerate() {
        let k = (k0 + offset).max(1) as f64;
        let w = 1.0_f64 / k;
        acc += val * w;
        w_sum += w;
    }
    if w_sum > 0.0_f64 {
        acc / w_sum
    } else {
        0.0_f64
    }
}

/// Stima robusta del roll-off Aria (7500 - 15000 Hz), protetta dai tagli brickwall dei codec MP3 sopra 13-15 kHz.
fn estimate_air_error(broad_err: &[f64], sr: f64) -> f64 {
    let nyq_safe = (sr * 0.45_f64).max(8200.0_f64);
    let hi_1 = 11_200.0_f64.min(nyq_safe);
    let hi_2 = 15_000.0_f64.min(nyq_safe);

    let low_air = band_log_mean(broad_err, sr, 7500.0_f64.min(hi_1 - 300.0_f64), hi_1);
    if hi_2 <= hi_1 + 400.0_f64 {
        return low_air;
    }
    let upper_air_raw = band_log_mean(broad_err, sr, hi_1, hi_2);
    // Evita che un filtro passa-basso MP3 a 14-15 kHz falsi la stima spingendo lo shelf all'estremo
    let upper_air_clamped = upper_air_raw.max(low_air - 7.5_f64);
    0.70_f64 * low_air + 0.30_f64 * upper_air_clamped
}

/// Individua la frequenza esatta (con interpolazione parabolica sub-bin), l'eccesso in dB
/// e la prominenza risonante locale del picco dominante in `[f_min, f_max]`.
fn find_peak_in_range(
    fine_err: &[f64],
    broad_err: &[f64],
    sr: f64,
    f_min: f64,
    f_max: f64,
) -> (f64, f64, f64) {
    let lo_f = f_min.min(f_max);
    let hi_f = f_max.max(f_min);
    let n = fine_err.len();
    let (k0, k1) = freq_range_to_bins(sr, n, lo_f, hi_f);

    let mut best_k = k0;
    let mut best_score = f64::NEG_INFINITY;
    let span = (k1.saturating_sub(k0)).max(1) as f64;

    for k in k0..=k1 {
        let prom = (fine_err[k] - broad_err[k]).max(0.0_f64);
        // Leggera preferenza verso i massimi interni rispetto ai bordi estremi della finestra
        let edge_norm = ((k - k0) as f64 / span) * 2.0_f64 - 1.0_f64;
        let edge_guard = 0.18_f64 * edge_norm * edge_norm;
        let score = fine_err[k] + 0.35_f64 * prom - edge_guard;
        if score > best_score {
            best_score = score;
            best_k = k;
        }
    }

    let interp_k = parabolic_bin_interp(fine_err, best_k, true);
    let freq = (interp_k * sr / (N_FFT as f64)).clamp(lo_f, hi_f);

    // Prominenza rispetto ai fianchi a +/- ~0.45 ottave
    let flank_lo = ((best_k as f64) * 0.73_f64).round().clamp(1.0_f64, (n - 1) as f64) as usize;
    let flank_hi = ((best_k as f64) * 1.37_f64).round().clamp(1.0_f64, (n - 1) as f64) as usize;
    let surround = 0.5_f64 * (broad_err[flank_lo] + broad_err[flank_hi]);
    let prominence = (fine_err[best_k] - surround).max(0.0_f64);

    (freq, fine_err[best_k], prominence)
}

/// Individua la frequenza centrale della carenza spettrale (scoop) in `[f_min, f_max]`.
fn find_dip_in_range(
    broad_err: &[f64],
    fine_err: &[f64],
    sr: f64,
    f_min: f64,
    f_max: f64,
) -> (f64, f64) {
    let lo_f = f_min.min(f_max);
    let hi_f = f_max.max(f_min);
    let n = broad_err.len();
    let (k0, k1) = freq_range_to_bins(sr, n, lo_f, hi_f);

    let mut best_k = (k0 + k1) / 2;
    let mut min_val = f64::INFINITY;
    let span = (k1.saturating_sub(k0)).max(1) as f64;

    for k in k0..=k1 {
        let blended = 0.68_f64 * broad_err[k] + 0.32_f64 * fine_err[k];
        let edge_norm = ((k - k0) as f64 / span) * 2.0_f64 - 1.0_f64;
        let penalized = blended + 0.22_f64 * edge_norm * edge_norm;
        if penalized < min_val {
            min_val = penalized;
            best_k = k;
        }
    }

    let interp_k = parabolic_bin_interp(broad_err, best_k, false);
    let freq = (interp_k * sr / (N_FFT as f64)).clamp(lo_f, hi_f);
    let dip_err = 0.68_f64 * broad_err[best_k] + 0.32_f64 * fine_err[best_k];
    (freq, dip_err)
}

fn parabolic_bin_interp(arr: &[f64], k: usize, is_max: bool) -> f64 {
    if k == 0 || k + 1 >= arr.len() {
        return k as f64;
    }
    let y0 = arr[k - 1];
    let y1 = arr[k];
    let y2 = arr[k + 1];
    let denom = y0 - 2.0_f64 * y1 + y2;
    if denom.abs() < 1e-9_f64 {
        return k as f64;
    }
    // Verifica concavità coerente con massimo o minimo locale
    if (is_max && denom >= 0.0_f64) || (!is_max && denom <= 0.0_f64) {
        return k as f64;
    }
    let delta = (0.5_f64 * (y0 - y2) / denom).clamp(-0.5_f64, 0.5_f64);
    (k as f64) + delta
}

fn freq_range_to_bins(sr: f64, num_bins: usize, f_min: f64, f_max: f64) -> (usize, usize) {
    let lo_f = f_min.min(f_max);
    let hi_f = f_max.max(f_min);
    let max_bin = num_bins.saturating_sub(1).max(1);
    let k0 = ((lo_f * (N_FFT as f64) / sr).floor() as usize).clamp(1, max_bin);
    let k1 = ((hi_f * (N_FFT as f64) / sr).ceil() as usize).clamp(k0, max_bin);
    (k0, k1)
}

fn make_band(
    name: &'static str,
    role: &'static str,
    kind: FilterKind,
    freq: f64,
    gain_db: f64,
    q: f64,
) -> EqBandParam {
    let clean_freq = if freq < 1000.0_f64 {
        freq.round()
    } else {
        (freq / 5.0_f64).round() * 5.0_f64
    };
    let clean_gain = round_1dec(gain_db.clamp(MIN_SAFE_GAIN_DB, MAX_SAFE_GAIN_DB));
    let clean_q = round_2dec(q.clamp(0.50_f64, 8.0_f64));

    EqBandParam {
        name,
        role,
        kind,
        freq: clean_freq,
        gain_db: clean_gain,
        q: clean_q,
        enabled: true,
    }
}

fn apply_band_response(
    band: &EqBandParam,
    sr: f64,
    fine_err: &mut [f64],
    broad_err: &mut [f64],
) {
    if !band.enabled || band.gain_db.abs() < 0.05_f64 {
        return;
    }
    let coeffs = BiquadCoeffs::design(band.kind, band.freq, sr, band.gain_db, band.q);
    for k in 1..fine_err.len() {
        let freq = (k as f64) * sr / (N_FFT as f64);
        let h_db = coeffs.magnitude_db(freq, sr);
        fine_err[k] += h_db;
        broad_err[k] += h_db;
    }
}

/// Garantisce che ogni singola banda resti entro `[-7.5, +8.0]` dB e che anche la somma
/// delle code dei filtri adiacenti mantenga una risposta cumulativa musicale e naturale.
fn enforce_cumulative_headroom(bands: &mut [EqBandParam], sr: f64) {
    for b in bands.iter_mut() {
        b.gain_db = round_1dec(b.gain_db.clamp(MIN_SAFE_GAIN_DB, MAX_SAFE_GAIN_DB));
    }

    let coeffs_vec: Vec<BiquadCoeffs> = bands
        .iter()
        .map(|b| BiquadCoeffs::design(b.kind, b.freq, sr, b.gain_db, b.q))
        .collect();

    let mut max_boost = 0.0_f64;
    let mut max_cut = 0.0_f64;
    let nyq_safe = (sr * 0.45_f64).max(4000.0_f64);

    for step in 0..96 {
        let t = (step as f64) / 95.0_f64;
        let freq = 55.0_f64 * (nyq_safe / 55.0_f64).powf(t);
        let sum_db: f64 = coeffs_vec.iter().map(|c| c.magnitude_db(freq, sr)).sum();
        if sum_db > max_boost {
            max_boost = sum_db;
        }
        if sum_db < max_cut {
            max_cut = sum_db;
        }
    }

    if max_boost > 8.6_f64 {
        let scale = 8.4_f64 / max_boost;
        for b in bands.iter_mut() {
            if b.gain_db > 0.0_f64 {
                b.gain_db = round_1dec((b.gain_db * scale).clamp(0.0_f64, MAX_SAFE_GAIN_DB));
            }
        }
    }

    if max_cut < -8.4_f64 {
        let scale = -8.2_f64 / max_cut;
        for b in bands.iter_mut() {
            if b.gain_db < 0.0_f64 {
                b.gain_db = round_1dec((b.gain_db * scale).clamp(MIN_SAFE_GAIN_DB, 0.0_f64));
            }
        }
    }
}

fn compute_optimal_stereo_width(side_mid_ratio: f64) -> f64 {
    let raw = if side_mid_ratio < 0.04_f64 {
        // Sorgente quasi mono: evita di amplificare rumore di fondo incoerente sul canale Side
        1.05_f64
    } else if side_mid_ratio < 0.35_f64 {
        // Tipica ripresa stereo stretta da registratore ambientale in sala prove (es. Zoom XY)
        let norm = (side_mid_ratio - 0.04_f64) / 0.31_f64;
        1.32_f64 - 0.12_f64 * norm
    } else if side_mid_ratio < 0.62_f64 {
        let norm = (side_mid_ratio - 0.35_f64) / 0.27_f64;
        1.20_f64 - 0.14_f64 * norm
    } else {
        1.04_f64
    };
    round_2dec(raw.clamp(1.00_f64, 1.36_f64))
}

fn estimate_eq_loudness_shift_db(
    psd_lin: &[f64],
    bands: &[EqBandParam],
    sub_cut_hz: f64,
    sr: f64,
) -> f64 {
    let hp = BiquadCoeffs::design(FilterKind::HighPass, sub_cut_hz, sr, 0.0_f64, 0.707_f64);
    let band_coeffs: Vec<BiquadCoeffs> = bands
        .iter()
        .map(|b| BiquadCoeffs::design(b.kind, b.freq, sr, b.gain_db, b.q))
        .collect();

    let mut pre_pwr = 0.0_f64;
    let mut post_pwr = 0.0_f64;
    let (k0, k1) = freq_range_to_bins(sr, psd_lin.len(), 35.0_f64, (sr * 0.45_f64).max(4000.0_f64));

    for (offset, &p_in) in psd_lin[k0..=k1].iter().enumerate() {
        let k = k0 + offset;
        let freq = (k as f64) * sr / (N_FFT as f64);
        let mut g_db = 2.0_f64 * hp.magnitude_db(freq, sr);
        for c in &band_coeffs {
            g_db += c.magnitude_db(freq, sr);
        }
        let lin_gain_sq = 10.0_f64.powf(g_db / 10.0_f64);
        pre_pwr += p_in;
        post_pwr += p_in * lin_gain_sq;
    }

    if pre_pwr <= 1e-20_f64 || post_pwr <= 1e-20_f64 {
        return 0.0_f64;
    }
    (10.0_f64 * (post_pwr / pre_pwr).log10()).clamp(-8.0_f64, 8.0_f64)
}

fn format_hz_compact(freq: f64) -> String {
    if freq >= 1000.0_f64 {
        format!("{:.2} kHz", freq / 1000.0_f64)
    } else {
        format!("{:.0} Hz", freq)
    }
}

fn format_diagnostic_summary(params: &DspParams) -> String {
    // Raccoglie i principali tagli anti-risonanza (Bassi, Mud, Medio-Bassi, Scatola, Acidità Mic)
    let mut cuts: Vec<(f64, f64)> = Vec::new();
    for &idx in &[0usize, 1, 2, 3, 5] {
        if let Some(b) = params.bands.get(idx) {
            if b.gain_db <= -1.2_f64 {
                cuts.push((b.freq, b.gain_db));
            }
        }
    }
    cuts.sort_by(|a, b| a.1.total_cmp(&b.1)); // I tagli più profondi per primi

    let mut clauses: Vec<String> = Vec::new();

    if cuts.len() >= 2 {
        clauses.push(format!(
            "tagli risonanze a {} ({:+.1} dB) e {} ({:+.1} dB)",
            format_hz_compact(cuts[0].0),
            cuts[0].1,
            format_hz_compact(cuts[1].0),
            cuts[1].1
        ));
    } else if cuts.len() == 1 {
        clauses.push(format!(
            "taglio risonanza a {} ({:+.1} dB)",
            format_hz_compact(cuts[0].0),
            cuts[0].1
        ));
    }

    if let Some(b0) = params.bands.first() {
        if b0.gain_db >= 1.4_f64 {
            clauses.push(format!(
                "corpo Bassi {} {:+.1} dB",
                format_hz_compact(b0.freq),
                b0.gain_db
            ));
        }
    }

    if let Some(voce) = params.bands.get(4) {
        if voce.gain_db >= 1.0_f64 {
            clauses.push(format!("recupero Voce {:+.1} dB", voce.gain_db));
        } else if voce.gain_db <= -1.4_f64 {
            clauses.push(format!(
                "controllo Medie {} {:+.1} dB",
                format_hz_compact(voce.freq),
                voce.gain_db
            ));
        }
    }

    if let Some(pres) = params.bands.get(6) {
        if pres.gain_db.abs() >= 2.2_f64 && clauses.len() < 3 {
            clauses.push(format!("Presenza {:+.1} dB", pres.gain_db));
        }
    }

    if let Some(aria) = params.bands.get(7) {
        if aria.gain_db.abs() >= 1.0_f64 {
            clauses.push(format!("Aria {:+.1} dB", aria.gain_db));
        }
    }

    if clauses.is_empty() {
        let voce_g = params.bands.get(4).map(|b| b.gain_db).unwrap_or(0.0_f64);
        let aria_g = params.bands.get(7).map(|b| b.gain_db).unwrap_or(0.0_f64);
        format!(
            "Auto-EQ completato: spettro già bilanciato, rifinitura fine (Voce {:+.1} dB, Aria {:+.1} dB)",
            voce_g, aria_g
        )
    } else {
        format!("Auto-EQ completato: {}", clauses.join(", "))
    }
}

#[inline]
fn round_1dec(v: f64) -> f64 {
    (v * 10.0_f64).round() / 10.0_f64
}

#[inline]
fn round_2dec(v: f64) -> f64 {
    (v * 100.0_f64).round() / 100.0_f64
}
