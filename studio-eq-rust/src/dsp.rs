use std::f64::consts::PI;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilterKind {
    Peaking,
    HighPass,
    LowPass,
    HighShelf,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct EqBandParam {
    pub name: &'static str,
    pub role: &'static str,
    pub kind: FilterKind,
    pub freq: f64,
    pub gain_db: f64,
    pub q: f64,
    pub enabled: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct DspParams {
    pub bypass: bool,
    pub sub_cut_hz: f64,
    pub bands: Vec<EqBandParam>,
    pub dynamic_mud_enabled: bool,
    pub dynamic_mud_amount: f64, // 0.0 ..= 1.5
    pub exciter_enabled: bool,
    pub exciter_drive: f64,      // 0.0 ..= 0.6
    pub mid_side_enabled: bool,
    pub mono_bass_hz: f64,
    pub stereo_width: f64,       // 0.0 ..= 2.0 (1.0 = normal, 1.22 = studio wide)
    pub comp_enabled: bool,
    pub comp_drive_db: f64,      // -6.0 ..= 12.0
    pub ceiling_db: f64,         // -0.4 dBTP
}

impl Default for DspParams {
    fn default() -> Self {
        Self::preset_sala_prove()
    }
}

#[allow(dead_code)]
impl DspParams {
    pub fn preset_sala_prove() -> Self {
        Self {
            bypass: false,
            sub_cut_hz: 45.0,
            bands: vec![
                EqBandParam {
                    name: "113 Hz",
                    role: "Risonanza Basso",
                    kind: FilterKind::Peaking,
                    freq: 113.0,
                    gain_db: -3.2,
                    q: 2.6,
                    enabled: true,
                },
                EqBandParam {
                    name: "164 Hz",
                    role: "Anti-Mud Sala",
                    kind: FilterKind::Peaking,
                    freq: 164.0,
                    gain_db: -6.2,
                    q: 1.5,
                    enabled: true,
                },
                EqBandParam {
                    name: "265 Hz",
                    role: "Pulizia Medio-Bassi",
                    kind: FilterKind::Peaking,
                    freq: 265.0,
                    gain_db: -3.0,
                    q: 1.4,
                    enabled: true,
                },
                EqBandParam {
                    name: "495 Hz",
                    role: "Anti-Scatola (Box)",
                    kind: FilterKind::Peaking,
                    freq: 495.0,
                    gain_db: -5.2,
                    q: 2.0,
                    enabled: true,
                },
                EqBandParam {
                    name: "1.42 kHz",
                    role: "Voce & Chitarre",
                    kind: FilterKind::Peaking,
                    freq: 1420.0,
                    gain_db: 5.4,
                    q: 0.85,
                    enabled: true,
                },
                EqBandParam {
                    name: "2.34 kHz",
                    role: "Taglio Acidita Mic",
                    kind: FilterKind::Peaking,
                    freq: 2336.0,
                    gain_db: -2.4,
                    q: 3.4,
                    enabled: true,
                },
                EqBandParam {
                    name: "4.8 kHz",
                    role: "Attacco & Presenza",
                    kind: FilterKind::Peaking,
                    freq: 4800.0,
                    gain_db: 4.8,
                    q: 1.0,
                    enabled: true,
                },
                EqBandParam {
                    name: "8.2 kHz",
                    role: "Studio Air Shelf",
                    kind: FilterKind::HighShelf,
                    freq: 8200.0,
                    gain_db: 7.8,
                    q: 0.707,
                    enabled: true,
                },
            ],
            dynamic_mud_enabled: true,
            dynamic_mud_amount: 1.0,
            exciter_enabled: true,
            exciter_drive: 0.22,
            mid_side_enabled: true,
            mono_bass_hz: 135.0,
            stereo_width: 1.22,
            comp_enabled: true,
            comp_drive_db: 3.8,
            ceiling_db: -0.4,
        }
    }

    pub fn preset_voce_primo_piano() -> Self {
        let mut p = Self::preset_sala_prove();
        p.bands[1].gain_db = -7.0;
        p.bands[3].gain_db = -5.8;
        p.bands[4].gain_db = 6.8;
        p.bands[6].gain_db = 5.6;
        p.bands[7].gain_db = 8.4;
        p.exciter_drive = 0.28;
        p
    }

    pub fn preset_rock_punch() -> Self {
        let mut p = Self::preset_sala_prove();
        p.sub_cut_hz = 38.0;
        p.bands[0].freq = 68.0;
        p.bands[0].gain_db = 2.8;
        p.bands[0].q = 1.4;
        p.bands[1].gain_db = -6.5;
        p.bands[4].gain_db = 4.6;
        p.bands[6].gain_db = 5.4;
        p.stereo_width = 1.32;
        p.comp_drive_db = 4.8;
        p
    }

    pub fn preset_flat() -> Self {
        let mut p = Self::preset_sala_prove();
        p.sub_cut_hz = 20.0;
        for b in &mut p.bands {
            b.gain_db = 0.0;
        }
        p.dynamic_mud_enabled = false;
        p.exciter_enabled = false;
        p.stereo_width = 1.0;
        p.comp_drive_db = 0.0;
        p
    }

    /// Preset delicatissimo da Master Bus pensato per quando sono attivi i 4 Stems AI separati
    /// (già equalizzati singolarmente da Pedalboard).
    pub fn preset_stem_polish() -> Self {
        let mut p = Self::preset_sala_prove();
        p.sub_cut_hz = 30.0;
        for b in &mut p.bands {
            b.gain_db = 0.0;
        }
        p.dynamic_mud_enabled = false;
        p.exciter_enabled = false;
        p.mid_side_enabled = true;
        p.stereo_width = 1.05;
        p.comp_enabled = true;
        p.comp_drive_db = 1.2;
        p.ceiling_db = -0.4;
        p
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Default for BiquadCoeffs {
    fn default() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }
}

#[allow(dead_code)]
impl BiquadCoeffs {
    pub fn design(kind: FilterKind, freq: f64, sr: f64, gain_db: f64, q: f64) -> Self {
        let f_clamped = freq.clamp(15.0, (sr * 0.48).max(20.0));
        let q_clamped = q.clamp(0.15, 18.0);
        let a_amp = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * f_clamped / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q_clamped);

        let (b0, b1, b2, a0, a1, a2) = match kind {
            FilterKind::Peaking => (
                1.0 + alpha * a_amp,
                -2.0 * cos_w0,
                1.0 - alpha * a_amp,
                1.0 + alpha / a_amp,
                -2.0 * cos_w0,
                1.0 - alpha / a_amp,
            ),
            FilterKind::HighPass => (
                (1.0 + cos_w0) * 0.5,
                -(1.0 + cos_w0),
                (1.0 + cos_w0) * 0.5,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            FilterKind::LowPass => (
                (1.0 - cos_w0) * 0.5,
                1.0 - cos_w0,
                (1.0 - cos_w0) * 0.5,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            FilterKind::HighShelf => {
                let sq_a = a_amp.sqrt();
                (
                    a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + 2.0 * sq_a * alpha),
                    -2.0 * a_amp * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0),
                    a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - 2.0 * sq_a * alpha),
                    (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + 2.0 * sq_a * alpha,
                    2.0 * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0),
                    (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - 2.0 * sq_a * alpha,
                )
            }
        };

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Risposta in ampiezza esatta H(e^jw) in dB per il grafico della curva EQ.
    pub fn magnitude_db(&self, freq: f64, sr: f64) -> f64 {
        let w = 2.0 * PI * freq / sr;
        let cos1 = w.cos();
        let sin1 = w.sin();
        let cos2 = (2.0 * w).cos();
        let sin2 = (2.0 * w).sin();

        let num_re = self.b0 + self.b1 * cos1 + self.b2 * cos2;
        let num_im = -self.b1 * sin1 - self.b2 * sin2;
        let den_re = 1.0 + self.a1 * cos1 + self.a2 * cos2;
        let den_im = -self.a1 * sin1 - self.a2 * sin2;

        let num_mag2 = num_re * num_re + num_im * num_im;
        let den_mag2 = (den_re * den_re + den_im * den_im).max(1e-18);
        10.0 * (num_mag2 / den_mag2).max(1e-18).log10()
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Default)]
pub struct StereoBiquad {
    pub coeffs: BiquadCoeffs,
    z1_l: f64,
    z2_l: f64,
    z1_r: f64,
    z2_r: f64,
}

#[allow(dead_code)]
impl StereoBiquad {
    #[inline]
    pub fn process(&mut self, l: f64, r: f64) -> (f64, f64) {
        let c = &self.coeffs;
        let out_l = c.b0 * l + self.z1_l;
        self.z1_l = c.b1 * l - c.a1 * out_l + self.z2_l + 1e-24;
        self.z2_l = c.b2 * l - c.a2 * out_l;

        let out_r = c.b0 * r + self.z1_r;
        self.z1_r = c.b1 * r - c.a1 * out_r + self.z2_r + 1e-24;
        self.z2_r = c.b2 * r - c.a2 * out_r;
        (out_l, out_r)
    }

    pub fn reset(&mut self) {
        self.z1_l = 0.0;
        self.z2_l = 0.0;
        self.z1_r = 0.0;
        self.z2_r = 0.0;
    }
}

#[allow(dead_code)]
pub fn total_curve_db(params: &DspParams, freq: f64, sr: f64) -> f64 {
    if params.bypass {
        return 0.0;
    }
    let hp = BiquadCoeffs::design(FilterKind::HighPass, params.sub_cut_hz, sr, 0.0, 0.707);
    let mut sum_db = hp.magnitude_db(freq, sr) * 2.0;
    for b in &params.bands {
        if b.enabled {
            let c = BiquadCoeffs::design(b.kind, b.freq, sr, b.gain_db, b.q);
            sum_db += c.magnitude_db(freq, sr);
        }
    }
    sum_db
}

#[inline]
fn soft_knee_over(env_db: f64, thresh_db: f64, knee_db: f64) -> f64 {
    let over = env_db - thresh_db;
    let half_knee = 0.5 * knee_db;
    if over <= -half_knee {
        0.0
    } else if over >= half_knee {
        over
    } else {
        let t = over + half_knee;
        (t * t) / (2.0 * knee_db)
    }
}

#[allow(dead_code)]
pub struct StudioDspChain {
    sr: f64,
    hp1: StereoBiquad,
    hp2: StereoBiquad,
    bands: Vec<StereoBiquad>,
    // Dynamic Mud (110-360 Hz)
    mud_bp_lp: StereoBiquad,
    mud_bp_hp: StereoBiquad,
    mud_env: f64,
    mud_att_coeff: f64,
    mud_rel_coeff: f64,
    // Harmonic Air Exciter (>4500 Hz)
    exc_hp1: StereoBiquad,
    exc_hp2: StereoBiquad,
    // Mid/Side Side High-Pass (<135 Hz Mono)
    side_hp1: StereoBiquad,
    side_hp2: StereoBiquad,
    // Bus Glue Compressor
    comp_env: f64,
    comp_att_coeff: f64,
    comp_rel_coeff: f64,
    // Stereo-Linked Envelope Ceiling Limiter
    lim_gain: f64,
    lim_att_coeff: f64,
    lim_rel_coeff: f64,
    pub last_gain_reduction_db: f32,
}

#[allow(dead_code)]
impl StudioDspChain {
    pub fn new(sr: f64, params: &DspParams) -> Self {
        let sr_safe = sr.max(8000.0);
        let mut s = Self {
            sr: sr_safe,
            hp1: StereoBiquad::default(),
            hp2: StereoBiquad::default(),
            bands: vec![StereoBiquad::default(); params.bands.len()],
            mud_bp_lp: StereoBiquad::default(),
            mud_bp_hp: StereoBiquad::default(),
            mud_env: 0.0,
            mud_att_coeff: (-1.0 / (0.018 * sr_safe)).exp(),
            mud_rel_coeff: (-1.0 / (0.150 * sr_safe)).exp(),
            exc_hp1: StereoBiquad::default(),
            exc_hp2: StereoBiquad::default(),
            side_hp1: StereoBiquad::default(),
            side_hp2: StereoBiquad::default(),
            comp_env: 0.0,
            comp_att_coeff: (-1.0 / (0.025 * sr_safe)).exp(),
            comp_rel_coeff: (-1.0 / (0.180 * sr_safe)).exp(),
            lim_gain: 1.0,
            lim_att_coeff: (-1.0 / (0.0015 * sr_safe)).exp(),
            lim_rel_coeff: (-1.0 / (0.085 * sr_safe)).exp(),
            last_gain_reduction_db: 0.0,
        };
        s.update_coefficients(params);
        s
    }

    pub fn reset_state(&mut self) {
        self.hp1.reset();
        self.hp2.reset();
        for b in &mut self.bands {
            b.reset();
        }
        self.mud_bp_lp.reset();
        self.mud_bp_hp.reset();
        self.mud_env = 0.0;
        self.exc_hp1.reset();
        self.exc_hp2.reset();
        self.side_hp1.reset();
        self.side_hp2.reset();
        self.comp_env = 0.0;
        self.lim_gain = 1.0;
        self.last_gain_reduction_db = 0.0;
    }

    pub fn update_coefficients(&mut self, params: &DspParams) {
        let hp_c = BiquadCoeffs::design(FilterKind::HighPass, params.sub_cut_hz, self.sr, 0.0, 0.707);
        self.hp1.coeffs = hp_c;
        self.hp2.coeffs = hp_c;

        if self.bands.len() != params.bands.len() {
            self.bands.resize(params.bands.len(), StereoBiquad::default());
        }
        for (st, p) in self.bands.iter_mut().zip(params.bands.iter()) {
            let g = if p.enabled { p.gain_db } else { 0.0 };
            st.coeffs = BiquadCoeffs::design(p.kind, p.freq, self.sr, g, p.q);
        }

        self.mud_bp_hp.coeffs = BiquadCoeffs::design(FilterKind::HighPass, 110.0, self.sr, 0.0, 0.707);
        self.mud_bp_lp.coeffs = BiquadCoeffs::design(FilterKind::LowPass, 360.0, self.sr, 0.0, 0.707);

        let exc_c = BiquadCoeffs::design(FilterKind::HighPass, 4500.0, self.sr, 0.0, 0.707);
        self.exc_hp1.coeffs = exc_c;
        self.exc_hp2.coeffs = exc_c;

        let side_c = BiquadCoeffs::design(FilterKind::HighPass, params.mono_bass_hz, self.sr, 0.0, 0.707);
        self.side_hp1.coeffs = side_c;
        self.side_hp2.coeffs = side_c;
    }

    #[inline]
    pub fn process_frame(&mut self, in_l: f32, in_r: f32, params: &DspParams) -> (f32, f32) {
        if params.bypass {
            self.last_gain_reduction_db = 0.0;
            return (in_l, in_r);
        }

        // Offset anti-denormali impercettibile (-480 dBFS) per stabilita CPU real-time nei filtri IIR
        let mut l = (in_l as f64) + 1e-24;
        let mut r = (in_r as f64) + 1e-24;

        // 1. Dynamic Anti-Mud (110-360 Hz) con Soft-Knee musicale (6 dB)
        if params.dynamic_mud_enabled && params.dynamic_mud_amount > 0.01 {
            let (m1_l, m1_r) = self.mud_bp_hp.process(l, r);
            let (mud_l, mud_r) = self.mud_bp_lp.process(m1_l, m1_r);
            let mud_pwr = 0.5 * (mud_l * mud_l + mud_r * mud_r);
            let coeff = if mud_pwr > self.mud_env {
                self.mud_att_coeff
            } else {
                self.mud_rel_coeff
            };
            self.mud_env = coeff * self.mud_env + (1.0 - coeff) * mud_pwr + 1e-24;

            let env_db = 10.0 * (self.mud_env + 1e-12).log10();
            let over_eff = soft_knee_over(env_db, -18.5, 6.0);
            if over_eff > 0.0 {
                let gr_db = (-over_eff * 0.66 * params.dynamic_mud_amount).max(-5.5);
                let sub_factor = 1.0 - 10.0_f64.powf(gr_db / 20.0);
                l -= mud_l * sub_factor;
                r -= mud_r * sub_factor;
            }
        }

        // 2. Dual-Stage Sub Rumble High-Pass + 8-Band 64-bit Parametric EQ
        (l, r) = self.hp1.process(l, r);
        (l, r) = self.hp2.process(l, r);
        for b in &mut self.bands {
            (l, r) = b.process(l, r);
        }

        // 3. Harmonic Air Exciter (>4.5 kHz) con saturazione asimmetrica morbida stile valvolare
        if params.exciter_enabled && params.exciter_drive > 0.005 {
            let (h1_l, h1_r) = self.exc_hp1.process(l, r);
            let (hi_l, hi_r) = self.exc_hp2.process(h1_l, h1_r);
            let drive = 1.8 + params.exciter_drive * 2.2;
            let soft_sat = |x: f64| -> f64 {
                let x_d = x * drive;
                let even = 0.08 * x_d * x_d / (1.0 + x_d * x_d);
                ((x_d + even).tanh() - even.tanh()) / drive
            };
            l += soft_sat(hi_l) * params.exciter_drive;
            r += soft_sat(hi_r) * params.exciter_drive;
        }

        // 4. Mid/Side Bass Mono-Maker + Stereo Widener
        if params.mid_side_enabled {
            let mid = 0.5 * (l + r);
            let side = 0.5 * (l - r);
            let (s1, _) = self.side_hp1.process(side, 0.0);
            let (side_clean, _) = self.side_hp2.process(s1, 0.0);
            let side_wide = side_clean * params.stereo_width;
            l = mid + side_wide;
            r = mid - side_wide;
        }

        // 5. SSL-Style Bus Glue Compressor con Soft-Knee analogico (6 dB)
        if params.comp_enabled {
            let drive_lin = 10.0_f64.powf(params.comp_drive_db / 20.0);
            l *= drive_lin;
            r *= drive_lin;

            let pwr = 0.5 * (l * l + r * r);
            let c = if pwr > self.comp_env {
                self.comp_att_coeff
            } else {
                self.comp_rel_coeff
            };
            self.comp_env = c * self.comp_env + (1.0 - c) * pwr + 1e-24;

            let env_db = 10.0 * (self.comp_env + 1e-12).log10();
            let over_eff = soft_knee_over(env_db, -17.0, 6.0);
            let mut gr_db = 0.0;
            if over_eff > 0.0 {
                gr_db = -over_eff * (1.0 - 1.0 / 2.4);
                let g = 10.0_f64.powf(gr_db / 20.0);
                l *= g;
                r *= g;
            }
            self.last_gain_reduction_db = gr_db as f32;
        } else {
            self.last_gain_reduction_db = 0.0;
        }

        // 6. Stereo-Linked Envelope Limiter + Soft-Knee Safety Ceiling
        let ceiling = 10.0_f64.powf(params.ceiling_db / 20.0);
        let lim_thresh = ceiling * 0.92;
        let peak = l.abs().max(r.abs());
        let target_gain = if peak > lim_thresh {
            (lim_thresh / peak).max(0.05)
        } else {
            1.0
        };
        let lim_c = if target_gain < self.lim_gain {
            self.lim_att_coeff
        } else {
            self.lim_rel_coeff
        };
        self.lim_gain = (lim_c * self.lim_gain + (1.0 - lim_c) * target_gain + 1e-24).min(1.0);
        l *= self.lim_gain;
        r *= self.lim_gain;

        let knee = ceiling * 0.90;
        let span = (ceiling - knee).max(1e-6);
        let limit_sample = |x: f64| -> f32 {
            if !x.is_finite() {
                return 0.0;
            }
            let ax = x.abs();
            let limited = if ax > knee {
                knee + span * ((ax - knee) / span).tanh()
            } else {
                ax
            };
            (x.signum() * limited.min(ceiling)) as f32
        };

        (limit_sample(l), limit_sample(r))
    }
}

#[cfg(test)]
mod argon_verification {
    use super::*;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
        }
    }

    #[test]
    fn verify_10000_deterministic_dsp_vectors() {
        let mut rng = SplitMix64(0xA460_4400_2026_CAFE);
        let mut params = DspParams::preset_sala_prove();
        let mut chain = StudioDspChain::new(44100.0, &params);
        let ceiling_lin = 10.0_f32.powf((params.ceiling_db as f32) / 20.0) + 1e-4;

        for vec_idx in 0..10_000 {
            if vec_idx % 250 == 0 {
                for b in &mut params.bands {
                    b.freq = 25.0 + rng.next_f64() * 17500.0;
                    b.gain_db = -12.0 + rng.next_f64() * 24.0;
                    b.q = 0.25 + rng.next_f64() * 10.0;
                }
                chain.update_coefficients(&params);
            }
            let in_l = ((rng.next_f64() * 2.4) - 1.2) as f32;
            let in_r = ((rng.next_f64() * 2.4) - 1.2) as f32;
            let (out_l, out_r) = chain.process_frame(in_l, in_r, &params);
            assert!(out_l.is_finite() && out_r.is_finite(), "Non-finite sample at {vec_idx}");
            assert!(
                out_l.abs() <= ceiling_lin && out_r.abs() <= ceiling_lin,
                "Ceiling exceeded at {vec_idx}: {out_l}, {out_r}"
            );
        }
    }
}
