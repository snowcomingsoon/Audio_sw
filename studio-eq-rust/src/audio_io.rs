#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::dsp::{
    BiquadCoeffs, DspParams, FilterKind, StemEqParams, StemThreeBandChain, StudioDspChain,
};

/// Cache multi-risoluzione (Peak + RMS) per disegnare forme d'onda fluide a 60 FPS stile DAW moderna.
#[derive(Clone, Debug)]
pub struct WaveformMipmap {
    pub block_size: usize,
    pub min_val: Vec<f32>,
    pub max_val: Vec<f32>,
    pub rms_val: Vec<f32>,
}

impl WaveformMipmap {
    pub fn build(left: &[f32], right: &[f32], block_size: usize) -> Self {
        let len = left.len().min(right.len());
        let step = block_size.max(1);
        let n_blocks = (len + step - 1) / step;
        let mut min_val = Vec::with_capacity(n_blocks);
        let mut max_val = Vec::with_capacity(n_blocks);
        let mut rms_val = Vec::with_capacity(n_blocks);

        for b in 0..n_blocks {
            let start = b * step;
            let end = (start + step).min(len);
            let mut mn = 0.0_f32;
            let mut mx = 0.0_f32;
            let mut sq = 0.0_f32;
            let cnt = (end - start).max(1) as f32;
            for i in start..end {
                let m = 0.5 * (left[i] + right[i]);
                if m < mn {
                    mn = m;
                }
                if m > mx {
                    mx = m;
                }
                sq += m * m;
            }
            min_val.push(mn);
            max_val.push(mx);
            rms_val.push((sq / cnt).sqrt());
        }

        Self {
            block_size: step,
            min_val,
            max_val,
            rms_val,
        }
    }
}

/// Buffer audio stereo zero-copy basato su `Arc`: `StereoBuffer::clone()` è O(1) istantaneo!
#[derive(Clone, Debug)]
pub struct StereoBuffer {
    pub left: Arc<Vec<f32>>,
    pub right: Arc<Vec<f32>>,
    pub sample_rate: u32,
    pub mip_fine: Arc<WaveformMipmap>,   // 128 samples/block (zoom ravvicinato)
    pub mip_coarse: Arc<WaveformMipmap>, // 2048 samples/block (vista intero brano)
}

#[inline]
fn hermite4_interp(src: &[f32], src_pos: f64) -> f32 {
    if src.is_empty() {
        return 0.0;
    }
    let max_idx = src.len() - 1;
    let idx0 = (src_pos.floor() as isize).clamp(0, max_idx as isize) as usize;
    let frac = (src_pos - idx0 as f64).clamp(0.0, 1.0) as f32;

    let im1 = idx0.saturating_sub(1);
    let i1 = (idx0 + 1).min(max_idx);
    let i2 = (idx0 + 2).min(max_idx);

    let xm1 = src[im1];
    let x0 = src[idx0];
    let x1 = src[i1];
    let x2 = src[i2];

    let c0 = x0;
    let c1 = 0.5_f32 * (x1 - xm1);
    let c2 = xm1 - 2.5_f32 * x0 + 2.0_f32 * x1 - 0.5_f32 * x2;
    let c3 = 0.5_f32 * (x2 - xm1) + 1.5_f32 * (x0 - x1);

    ((c3 * frac + c2) * frac + c1) * frac + c0
}

fn prefilter_zero_phase_lowpass(samples: &[f32], cutoff_hz: f64, sr_hz: f64) -> Vec<f32> {
    if samples.len() < 4 {
        return samples.to_vec();
    }
    let coeffs = BiquadCoeffs::design(
        FilterKind::LowPass,
        cutoff_hz,
        sr_hz,
        0.0,
        std::f64::consts::FRAC_1_SQRT_2,
    );

    let mut out: Vec<f32> = samples.to_vec();
    // Doppio stadio forward-backward (48 dB/oct Linkwitz-Riley a fase lineare, zero ritardo di gruppo)
    for _stage in 0..2 {
        let mut z1 = 0.0_f64;
        let mut z2 = 0.0_f64;
        for s in out.iter_mut() {
            let x = *s as f64;
            let y = coeffs.b0 * x + z1;
            z1 = coeffs.b1 * x - coeffs.a1 * y + z2;
            z2 = coeffs.b2 * x - coeffs.a2 * y;
            *s = y as f32;
        }

        z1 = 0.0_f64;
        z2 = 0.0_f64;
        for s in out.iter_mut().rev() {
            let x = *s as f64;
            let y = coeffs.b0 * x + z1;
            z1 = coeffs.b1 * x - coeffs.a1 * y + z2;
            z2 = coeffs.b2 * x - coeffs.a2 * y;
            *s = y as f32;
        }
    }

    out
}

impl StereoBuffer {
    pub fn from_vecs(left: Vec<f32>, right: Vec<f32>, sample_rate: u32) -> Self {
        let mip_fine = Arc::new(WaveformMipmap::build(&left, &right, 128));
        let mip_coarse = Arc::new(WaveformMipmap::build(&left, &right, 2048));
        Self {
            left: Arc::new(left),
            right: Arc::new(right),
            sample_rate,
            mip_fine,
            mip_coarse,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.left.len().min(self.right.len())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ricampionamento studio con interpolazione cubica Hermite a 4 punti
    /// e pre-filtro biquad anti-aliasing a fase zero in caso di downsampling (`ratio > 1.01`).
    pub fn resample_to(&self, target_sr: u32) -> Self {
        let safe_target_sr = target_sr.max(8_000);
        if self.sample_rate == safe_target_sr || self.len() < 2 {
            return self.clone();
        }
        let ratio = self.sample_rate as f64 / safe_target_sr as f64;
        let out_len = ((self.len() as f64) / ratio).round().max(1.0) as usize;

        let filtered_pair = if ratio > 1.01 {
            let cutoff = 0.45 * (safe_target_sr as f64);
            let sr = self.sample_rate as f64;
            let fl = prefilter_zero_phase_lowpass(&self.left[..self.len()], cutoff, sr);
            let fr = prefilter_zero_phase_lowpass(&self.right[..self.len()], cutoff, sr);
            Some((fl, fr))
        } else {
            None
        };

        let (src_l, src_r): (&[f32], &[f32]) = match filtered_pair {
            Some((ref fl, ref fr)) => (fl.as_slice(), fr.as_slice()),
            None => (&self.left[..self.len()], &self.right[..self.len()]),
        };

        let mut out_l = Vec::with_capacity(out_len);
        let mut out_r = Vec::with_capacity(out_len);

        for i in 0..out_len {
            let src_pos = i as f64 * ratio;
            out_l.push(hermite4_interp(src_l, src_pos));
            out_r.push(hermite4_interp(src_r, src_pos));
        }

        Self::from_vecs(out_l, out_r, safe_target_sr)
    }

    /// Campiona istantaneamente (min, max, rms) nell'intervallo [f0..f1] per una colonna pixel.
    pub fn sample_envelope(&self, f0: usize, f1: usize) -> (f32, f32, f32) {
        let span = f1.saturating_sub(f0).max(1);
        let mip = if span >= 2048 {
            &*self.mip_coarse
        } else {
            &*self.mip_fine
        };
        if mip.max_val.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let b0 = (f0 / mip.block_size).min(mip.max_val.len() - 1);
        let b1 = ((f1 + mip.block_size - 1) / mip.block_size)
            .max(b0 + 1)
            .min(mip.max_val.len());

        let mut mn = 0.0_f32;
        let mut mx = 0.0_f32;
        let mut rms_acc = 0.0_f32;
        for b in b0..b1 {
            mn = mn.min(mip.min_val[b]);
            mx = mx.max(mip.max_val[b]);
            rms_acc += mip.rms_val[b];
        }
        let cnt = (b1 - b0).max(1) as f32;
        (mn, mx, rms_acc / cnt)
    }
}

/// Legge di pan a potenza costante normalizzata: `pan == 0.0` produce guadagno unitario `(1.0, 1.0)`
/// identico tra riproduzione Live WASAPI, Export Mix Master ed Export 4 Stems WAV!
#[inline]
pub fn constant_power_pan_gains(pan: f32) -> (f32, f32) {
    let angle = (pan.clamp(-1.0_f32, 1.0_f32) + 1.0_f32) * std::f32::consts::FRAC_PI_4;
    (
        angle.cos() * std::f32::consts::SQRT_2,
        angle.sin() * std::f32::consts::SQRT_2,
    )
}

/// Soft-limiter analogico trasparente a ginocchio morbido prima della conversione PCM 24-bit
/// per evitare qualsiasi clipping digitale secco durante l'esportazione.
#[inline]
pub fn soft_limit_to_pcm24(sample: f32) -> i32 {
    const SCALE_24BIT: f32 = 8_388_607.0_f32;
    const KNEE: f32 = 0.94_f32;
    const CEIL: f32 = 0.9999_f32;

    if !sample.is_finite() {
        return 0;
    }

    let ax = sample.abs();
    let shaped = if ax > KNEE {
        let span = CEIL - KNEE;
        KNEE + span * ((ax - KNEE) / span).tanh()
    } else {
        ax
    };
    let clamped = (sample.signum() * shaped).clamp(-1.0_f32, 1.0_f32);
    (clamped * SCALE_24BIT).round() as i32
}

fn collect_ffmpeg_bins_recursive(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 4 || !dir.is_dir() {
        return;
    }
    let ffmpeg_exe = if cfg!(target_os = "windows") {
        dir.join("ffmpeg.exe")
    } else {
        dir.join("ffmpeg")
    };
    if ffmpeg_exe.is_file() {
        out.push(dir.to_path_buf());
        return;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                collect_ffmpeg_bins_recursive(&p, depth + 1, out);
            }
        }
    }
}

/// Raccoglie le directory FFmpeg installate tramite WinGet (`Links` e `Packages\*FFmpeg*\**\bin`)
/// così FFmpeg funziona immediatamente anche senza riavviare Windows Explorer.
pub fn discover_winget_ffmpeg_dirs() -> Vec<PathBuf> {
    let mut extra_dirs = Vec::new();

    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        let base = PathBuf::from(local_app_data).join("Microsoft").join("WinGet");
        let links = base.join("Links");
        if links.is_dir() {
            extra_dirs.push(links);
        }

        let packages = base.join("Packages");
        if let Ok(entries) = std::fs::read_dir(packages) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    let name_upper = entry.file_name().to_string_lossy().to_uppercase();
                    if name_upper.contains("FFMPEG") {
                        collect_ffmpeg_bins_recursive(&p, 0, &mut extra_dirs);
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    for fallback in [r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin"] {
        let p = Path::new(fallback);
        if p.is_dir() {
            extra_dirs.push(p.to_path_buf());
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let sys_ffmpeg = PathBuf::from("/Library/Application Support/StudioEQ_Master_Pro/ffmpeg/bin");
        if sys_ffmpeg.is_dir() {
            extra_dirs.push(sys_ffmpeg);
        }
        if let Some(home) = std::env::var_os("HOME") {
            let home_support = PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("StudioEQ_Master_Pro");
            for sub in [home_support.join("ffmpeg").join("bin"), home_support.join("bin")] {
                if sub.is_dir() {
                    extra_dirs.push(sub);
                }
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(contents) = exe.parent().and_then(|p| p.parent()) {
                let bundled_mac_ffmpeg = contents.join("Resources").join("ffmpeg").join("bin");
                if bundled_mac_ffmpeg.is_dir() {
                    extra_dirs.push(bundled_mac_ffmpeg);
                }
            }
        }
        for mac_dir in ["/opt/homebrew/bin", "/usr/local/bin"] {
            let p = Path::new(mac_dir);
            if p.is_dir() {
                extra_dirs.push(p.to_path_buf());
            }
        }
    }

    extra_dirs
}

/// Costruisce la variabile `PATH` arricchita con i percorsi WinGet FFmpeg per i processi figli.
pub fn build_augmented_path_env() -> Option<OsString> {
    let mut paths: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|val| std::env::split_paths(&val).collect())
        .unwrap_or_default();

    for extra in discover_winget_ffmpeg_dirs() {
        let already = paths.iter().any(|existing| {
            existing
                .to_string_lossy()
                .eq_ignore_ascii_case(&extra.to_string_lossy())
        });
        if !already {
            paths.push(extra);
        }
    }

    std::env::join_paths(paths).ok()
}

/// Risolve il percorso dell'eseguibile `ffmpeg` (cercando anche nelle cartelle WinGet).
pub fn resolve_ffmpeg_executable() -> PathBuf {
    let bin_name = if cfg!(target_os = "windows") {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };

    if let Some(path_os) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_os) {
            let candidate = dir.join(bin_name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }

    for dir in discover_winget_ffmpeg_dirs() {
        let candidate = dir.join(bin_name);
        if candidate.is_file() {
            return candidate;
        }
    }

    PathBuf::from("ffmpeg")
}

pub fn decode_audio_file(path: &Path) -> Result<StereoBuffer, String> {
    let probe_once = |use_ext: bool| -> Result<symphonia::core::probe::ProbeResult, String> {
        let file = File::open(path).map_err(|e| format!("Impossibile aprire il file: {e}"))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if use_ext {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let mapped = if ext.eq_ignore_ascii_case("mpeg") {
                    "mp3"
                } else {
                    ext
                };
                hint.with_extension(mapped);
            }
        }
        symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| format!("Formato audio non riconosciuto: {e}"))
    };

    let probed = probe_once(true).or_else(|_| probe_once(false))?;
    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| "Nessuna traccia audio trovata nel file".to_string())?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    let mut sample_rate = codec_params.sample_rate.unwrap_or(44100);

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Impossibile inizializzare il decoder: {e}"))?;

    let mut left = Vec::new();
    let mut right = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                sample_rate = spec.rate;
                let channels = spec.channels.count().max(1);
                let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(decoded);
                let samples = sample_buf.samples();

                if channels == 1 {
                    for &s in samples {
                        left.push(s);
                        right.push(s);
                    }
                } else {
                    for frame in samples.chunks_exact(channels) {
                        left.push(frame[0]);
                        right.push(frame[1]);
                    }
                }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(_) => break,
        }
    }

    if left.is_empty() {
        return Err("Il file audio decodificato risulta vuoto.".to_string());
    }

    Ok(StereoBuffer::from_vecs(left, right, sample_rate))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlaybackSourceMode {
    OriginalWithLiveDsp,
    PreRenderedAiMaster,
    FourStemMixer,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct StemChannel {
    pub name: &'static str,
    pub subtitle: &'static str,
    pub gain_db: f32,
    pub pan: f32, // -1.0 (Left) .. 0.0 (Center) .. +1.0 (Right)
    pub eq: StemEqParams,
    pub muted: bool,
    pub solo: bool,
    pub live_peak: f32,
}

pub struct SharedAudioState {
    pub playing: bool,
    pub position_frame: usize,
    pub device_sr: u32,
    pub mode: PlaybackSourceMode,

    // Selezione / Loop sulla timeline (stile Audacity moderno)
    pub loop_enabled: bool,
    pub selection_range: Option<(usize, usize)>,

    pub main_track: Option<StereoBuffer>,
    pub ai_master_track: Option<StereoBuffer>,
    pub stems: Option<[StereoBuffer; 4]>,
    pub stem_channels: [StemChannel; 4],

    pub params: DspParams,
    pub params_dirty: bool,

    // Telemetria real-time verso la GUI (60 FPS)
    pub fft_ring: VecDeque<f32>,
    pub peak_l: f32,
    pub peak_r: f32,
    pub rms_db: f32,
    pub gr_db: f32,
}

impl SharedAudioState {
    pub fn new(device_sr: u32) -> Self {
        Self {
            playing: false,
            position_frame: 0,
            device_sr,
            mode: PlaybackSourceMode::OriginalWithLiveDsp,
            loop_enabled: false,
            selection_range: None,
            main_track: None,
            ai_master_track: None,
            stems: None,
            stem_channels: [
                StemChannel {
                    name: "Voce",
                    subtitle: "De-Noise + Vocal Strip",
                    gain_db: 0.0,
                    pan: 0.0,
                    eq: StemEqParams::default(),
                    muted: false,
                    solo: false,
                    live_peak: 0.0,
                },
                StemChannel {
                    name: "Batteria",
                    subtitle: "Punch 62Hz + Anti-Box",
                    gain_db: 0.0,
                    pan: 0.0,
                    eq: StemEqParams::default(),
                    muted: false,
                    solo: false,
                    live_peak: 0.0,
                },
                StemChannel {
                    name: "Basso",
                    subtitle: "Mono Sub + Anti-Mud",
                    gain_db: 0.0,
                    pan: 0.0,
                    eq: StemEqParams::default(),
                    muted: false,
                    solo: false,
                    live_peak: 0.0,
                },
                StemChannel {
                    name: "Chitarre & Altro",
                    subtitle: "Body 1.4kHz + Stereo Wide",
                    gain_db: 0.0,
                    pan: 0.0,
                    eq: StemEqParams::default(),
                    muted: false,
                    solo: false,
                    live_peak: 0.0,
                },
            ],
            params: DspParams::preset_sala_prove(),
            params_dirty: true,
            fft_ring: VecDeque::with_capacity(8192),
            peak_l: 0.0,
            peak_r: 0.0,
            rms_db: -60.0,
            gr_db: 0.0,
        }
    }

    pub fn active_length(&self) -> usize {
        match self.mode {
            PlaybackSourceMode::PreRenderedAiMaster => self
                .ai_master_track
                .as_ref()
                .or(self.main_track.as_ref())
                .map(|b| b.len())
                .unwrap_or(0),
            PlaybackSourceMode::FourStemMixer => self
                .stems
                .as_ref()
                .map(|s| s[0].len())
                .or_else(|| self.main_track.as_ref().map(|b| b.len()))
                .unwrap_or(0),
            PlaybackSourceMode::OriginalWithLiveDsp => {
                self.main_track.as_ref().map(|b| b.len()).unwrap_or(0)
            }
        }
    }
}

pub struct RealtimeAudioEngine {
    _stream: cpal::Stream,
    pub state: Arc<Mutex<SharedAudioState>>,
}

impl RealtimeAudioEngine {
    pub fn start() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "Nessuna scheda audio di output rilevata su Windows".to_string())?;
        let config = device
            .default_output_config()
            .map_err(|e| format!("Errore configurazione output audio: {e}"))?;

        let device_sr = config.sample_rate().0;
        let channels = config.channels() as usize;
        let state = Arc::new(Mutex::new(SharedAudioState::new(device_sr)));
        let state_cb = Arc::clone(&state);

        let stream_config: cpal::StreamConfig = config.into();
        let mut dsp = StudioDspChain::new(device_sr as f64, &DspParams::preset_sala_prove());
        let mut last_stem_eq = [StemEqParams::default(); 4];
        let mut stem_eq_chains: [StemThreeBandChain; 4] = std::array::from_fn(|i| {
            StemThreeBandChain::new(device_sr as f64, i, &StemEqParams::default())
        });

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut st = match state_cb.lock() {
                        Ok(g) => g,
                        Err(_) => {
                            data.fill(0.0);
                            return;
                        }
                    };

                    if st.params_dirty {
                        dsp.update_coefficients(&st.params);
                        st.params_dirty = false;
                    }
                    for i in 0..4 {
                        if st.stem_channels[i].eq != last_stem_eq[i] {
                            last_stem_eq[i] = st.stem_channels[i].eq;
                            stem_eq_chains[i].update_coefficients(i, &last_stem_eq[i]);
                        }
                    }

                    if !st.playing || st.active_length() == 0 {
                        data.fill(0.0);
                        st.peak_l *= 0.84;
                        st.peak_r *= 0.84;
                        for ch in &mut st.stem_channels {
                            ch.live_peak *= 0.84;
                        }
                        st.gr_db = 0.0;
                        return;
                    }

                    let total_len = st.active_length();
                    let loop_bounds = if st.loop_enabled {
                        st.selection_range.filter(|&(a, b)| b > a + 512)
                    } else {
                        None
                    };

                    let any_solo = st.stem_channels.iter().any(|c| c.solo);
                    let stem_cfg: [(f32, f32, f32); 4] = std::array::from_fn(|i| {
                        let ch = &st.stem_channels[i];
                        let audible = if any_solo { ch.solo } else { !ch.muted };
                        let g = if audible {
                            10.0_f32.powf(ch.gain_db / 20.0)
                        } else {
                            0.0
                        };
                        let (pl, pr) = constant_power_pan_gains(ch.pan);
                        (g, pl, pr)
                    });

                    let mut max_l = 0.0_f32;
                    let mut max_r = 0.0_f32;
                    let mut stem_peaks = [0.0_f32; 4];
                    let mut sum_sq = 0.0_f32;
                    let mut n_frames = 0usize;

                    for frame in data.chunks_exact_mut(channels) {
                        if let Some((l_start, l_end)) = loop_bounds {
                            if st.position_frame >= l_end.min(total_len) {
                                st.position_frame = l_start.min(total_len.saturating_sub(1));
                            }
                        }

                        let pos = st.position_frame;
                        if pos >= total_len {
                            st.position_frame = 0;
                            st.playing = false;
                            frame.fill(0.0);
                            continue;
                        }

                        let (out_l, out_r) = match st.mode {
                            PlaybackSourceMode::PreRenderedAiMaster => {
                                if let Some(ref ai) = st.ai_master_track {
                                    let idx = pos.min(ai.len() - 1);
                                    dsp.process_frame(ai.left[idx], ai.right[idx], &st.params)
                                } else if let Some(ref m) = st.main_track {
                                    let idx = pos.min(m.len() - 1);
                                    dsp.process_frame(m.left[idx], m.right[idx], &st.params)
                                } else {
                                    (0.0, 0.0)
                                }
                            }
                            PlaybackSourceMode::FourStemMixer => {
                                if let Some(ref stems) = st.stems {
                                    let mut sl = 0.0_f32;
                                    let mut sr_val = 0.0_f32;
                                    for i in 0..4 {
                                        if stems[i].is_empty() {
                                            continue;
                                        }
                                        let (g, pl, pr) = stem_cfg[i];
                                        if g <= 0.0_f32 {
                                            continue;
                                        }
                                        let idx = pos.min(stems[i].len() - 1);
                                        let (eq_l, eq_r) = stem_eq_chains[i]
                                            .process_frame(stems[i].left[idx], stems[i].right[idx]);
                                        let l_s = eq_l * g * pl;
                                        let r_s = eq_r * g * pr;
                                        stem_peaks[i] =
                                            stem_peaks[i].max(l_s.abs().max(r_s.abs()));
                                        sl += l_s;
                                        sr_val += r_s;
                                    }
                                    // Passa anche attraverso il Master EQ live (parità 100% con export_session)
                                    dsp.process_frame(sl * 0.82, sr_val * 0.82, &st.params)
                                } else {
                                    (0.0, 0.0)
                                }
                            }
                            PlaybackSourceMode::OriginalWithLiveDsp => {
                                if let Some(ref m) = st.main_track {
                                    let idx = pos.min(m.len() - 1);
                                    dsp.process_frame(m.left[idx], m.right[idx], &st.params)
                                } else {
                                    (0.0, 0.0)
                                }
                            }
                        };

                        st.position_frame += 1;

                        if channels >= 2 {
                            frame[0] = out_l;
                            frame[1] = out_r;
                            for extra in &mut frame[2..] {
                                *extra = 0.0;
                            }
                        } else if channels == 1 {
                            frame[0] = 0.5 * (out_l + out_r);
                        }

                        max_l = max_l.max(out_l.abs());
                        max_r = max_r.max(out_r.abs());
                        let mono = 0.5 * (out_l + out_r);
                        sum_sq += mono * mono;
                        n_frames += 1;

                        if st.fft_ring.len() >= 4096 {
                            st.fft_ring.pop_front();
                        }
                        st.fft_ring.push_back(mono);
                    }

                    st.peak_l = st.peak_l * 0.72 + max_l * 0.28;
                    st.peak_r = st.peak_r * 0.72 + max_r * 0.28;
                    for i in 0..4 {
                        st.stem_channels[i].live_peak =
                            st.stem_channels[i].live_peak * 0.70 + stem_peaks[i] * 0.30;
                    }
                    if n_frames > 0 {
                        let rms = (sum_sq / n_frames as f32).sqrt();
                        st.rms_db = 20.0 * (rms + 1e-6).log10();
                    }
                    st.gr_db = dsp.last_gain_reduction_db;
                },
                |err| eprintln!("Errore stream audio: {err}"),
                None,
            )
            .map_err(|e| {
                #[cfg(target_os = "macos")]
                let backend = "CoreAudio";
                #[cfg(not(target_os = "macos"))]
                let backend = "WASAPI";
                format!("Impossibile avviare lo stream audio {backend}: {e}")
            })?;

        stream
            .play()
            .map_err(|e| format!("Impossibile riprodurre lo stream audio: {e}"))?;

        Ok(Self {
            _stream: stream,
            state,
        })
    }
}

/// Esporta la sessione attiva (sia Mix Multi-Traccia 4 Stems che Traccia Singola + Master EQ) in WAV 24-bit o MP3 320k.
pub fn export_session(
    mode: PlaybackSourceMode,
    main_track: Option<&StereoBuffer>,
    ai_master: Option<&StereoBuffer>,
    stems: Option<&[StereoBuffer; 4]>,
    stem_channels: &[StemChannel; 4],
    params: &DspParams,
    out_path: &Path,
) -> Result<(), String> {
    let sr = main_track
        .map(|b| b.sample_rate)
        .or_else(|| stems.map(|s| s[0].sample_rate))
        .or_else(|| ai_master.map(|b| b.sample_rate))
        .unwrap_or(44100);

    let len = match mode {
        PlaybackSourceMode::FourStemMixer => stems.map(|s| s[0].len()).unwrap_or(0),
        PlaybackSourceMode::PreRenderedAiMaster => {
            ai_master.or(main_track).map(|b| b.len()).unwrap_or(0)
        }
        PlaybackSourceMode::OriginalWithLiveDsp => main_track.map(|b| b.len()).unwrap_or(0),
    };

    if len == 0 {
        return Err("Nessun audio disponibile per l'esportazione.".to_string());
    }

    let mut dsp = StudioDspChain::new(sr as f64, params);
    let mut stem_eq_chains: [StemThreeBandChain; 4] =
        std::array::from_fn(|i| StemThreeBandChain::new(sr as f64, i, &stem_channels[i].eq));
    let any_solo = stem_channels.iter().any(|c| c.solo);
    let stem_cfg: [(f32, f32, f32); 4] = std::array::from_fn(|i| {
        let ch = &stem_channels[i];
        let audible = if any_solo { ch.solo } else { !ch.muted };
        let g = if audible {
            10.0_f32.powf(ch.gain_db / 20.0)
        } else {
            0.0
        };
        let (pl, pr) = constant_power_pan_gains(ch.pan);
        (g, pl, pr)
    });

    let is_mp3 = out_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.eq_ignore_ascii_case("mp3"))
        .unwrap_or(false);

    let wav_target = if is_mp3 {
        out_path.with_extension("temp_render.wav")
    } else {
        out_path.to_path_buf()
    };

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: sr,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&wav_target, spec)
        .map_err(|e| format!("Impossibile creare file WAV: {e}"))?;

    for i in 0..len {
        let (in_l, in_r) = match mode {
            PlaybackSourceMode::FourStemMixer => {
                if let Some(st_arr) = stems {
                    let mut sl = 0.0_f32;
                    let mut sr_v = 0.0_f32;
                    for k in 0..4 {
                        if st_arr[k].is_empty() {
                            continue;
                        }
                        let (g, pl, pr) = stem_cfg[k];
                        if g <= 0.0_f32 {
                            continue;
                        }
                        let idx = i.min(st_arr[k].len() - 1);
                        let (eq_l, eq_r) = stem_eq_chains[k]
                            .process_frame(st_arr[k].left[idx], st_arr[k].right[idx]);
                        sl += eq_l * g * pl;
                        sr_v += eq_r * g * pr;
                    }
                    (sl * 0.82, sr_v * 0.82)
                } else {
                    (0.0, 0.0)
                }
            }
            PlaybackSourceMode::PreRenderedAiMaster => {
                if let Some(ai) = ai_master.or(main_track) {
                    let idx = i.min(ai.len() - 1);
                    (ai.left[idx], ai.right[idx])
                } else {
                    (0.0, 0.0)
                }
            }
            PlaybackSourceMode::OriginalWithLiveDsp => {
                if let Some(m) = main_track {
                    let idx = i.min(m.len() - 1);
                    (m.left[idx], m.right[idx])
                } else {
                    (0.0, 0.0)
                }
            }
        };

        let (l, r) = dsp.process_frame(in_l, in_r, params);
        let il = soft_limit_to_pcm24(l);
        let ir = soft_limit_to_pcm24(r);
        writer.write_sample(il).map_err(|e| e.to_string())?;
        writer.write_sample(ir).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;

    if is_mp3 {
        let ffmpeg_bin = resolve_ffmpeg_executable();
        let mut cmd = Command::new(ffmpeg_bin);
        if let Some(aug_path) = build_augmented_path_env() {
            cmd.env("PATH", aug_path);
        }
        cmd.args([
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            wav_target.to_str().unwrap_or_default(),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "320k",
            out_path.to_str().unwrap_or_default(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

        #[cfg(target_os = "windows")]
        {
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }

        let status = cmd
            .status()
            .map_err(|e| format!("FFmpeg richiesto per esportazione MP3: {e}"))?;
        let _ = std::fs::remove_file(&wav_target);
        if !status.success() {
            return Err("Conversione FFmpeg in MP3 fallita.".to_string());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_stereo_buffer_clone_and_pan_parity() {
        let buf = StereoBuffer::from_vecs(vec![0.25; 4096], vec![-0.25; 4096], 48_000);
        let cloned = buf.clone();
        assert!(Arc::ptr_eq(&buf.left, &cloned.left));
        assert!(Arc::ptr_eq(&buf.right, &cloned.right));
        assert!(Arc::ptr_eq(&buf.mip_fine, &cloned.mip_fine));

        let (pl, pr) = constant_power_pan_gains(0.0);
        assert!((pl - 1.0).abs() < 1e-5, "Center pan L gain was {pl}");
        assert!((pr - 1.0).abs() < 1e-5, "Center pan R gain was {pr}");
    }

    #[test]
    fn test_hermite_resample_up_and_down() {
        let sr_in = 48_000u32;
        let len_in = 4_800usize; // 100 ms
        let mut l = Vec::with_capacity(len_in);
        let mut r = Vec::with_capacity(len_in);
        for i in 0..len_in {
            let t = i as f32 / sr_in as f32;
            let s = 0.7 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            l.push(s);
            r.push(-s);
        }
        let buf = StereoBuffer::from_vecs(l, r, sr_in);

        let down = buf.resample_to(44_100);
        assert_eq!(down.sample_rate, 44_100);
        assert_eq!(down.len(), 4_410);
        assert!(down.left.iter().all(|x| x.is_finite() && x.abs() <= 0.75));

        let up = down.resample_to(48_000);
        assert_eq!(up.sample_rate, 48_000);
        assert_eq!(up.len(), 4_800);
        assert!(up.right.iter().all(|x| x.is_finite() && x.abs() <= 0.75));
    }
}
