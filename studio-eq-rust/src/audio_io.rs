use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;
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

use crate::dsp::{DspParams, StudioDspChain};

/// Cache multi-risoluzione (Peak + RMS) per disegnare forme d'onda fluide a 60 FPS stile DAW moderna.
#[derive(Clone)]
pub struct WaveformMipmap {
    pub block_size: usize,
    pub min_val: Vec<f32>,
    pub max_val: Vec<f32>,
    pub rms_val: Vec<f32>,
}

impl WaveformMipmap {
    pub fn build(left: &[f32], right: &[f32], block_size: usize) -> Self {
        let len = left.len().min(right.len());
        let n_blocks = (len + block_size - 1) / block_size.max(1);
        let mut min_val = Vec::with_capacity(n_blocks);
        let mut max_val = Vec::with_capacity(n_blocks);
        let mut rms_val = Vec::with_capacity(n_blocks);

        for b in 0..n_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(len);
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
            block_size,
            min_val,
            max_val,
            rms_val,
        }
    }
}

#[derive(Clone)]
pub struct StereoBuffer {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: u32,
    pub mip_fine: WaveformMipmap,   // 128 samples/block (zoom ravvicinato)
    pub mip_coarse: WaveformMipmap, // 2048 samples/block (vista intero brano)
}

impl StereoBuffer {
    pub fn from_vecs(left: Vec<f32>, right: Vec<f32>, sample_rate: u32) -> Self {
        let mip_fine = WaveformMipmap::build(&left, &right, 128);
        let mip_coarse = WaveformMipmap::build(&left, &right, 2048);
        Self {
            left,
            right,
            sample_rate,
            mip_fine,
            mip_coarse,
        }
    }

    pub fn len(&self) -> usize {
        self.left.len().min(self.right.len())
    }

    pub fn resample_to(&self, target_sr: u32) -> Self {
        if self.sample_rate == target_sr || self.len() < 2 {
            return self.clone();
        }
        let ratio = self.sample_rate as f64 / target_sr as f64;
        let out_len = ((self.len() as f64) / ratio) as usize;
        let mut out_l = Vec::with_capacity(out_len);
        let mut out_r = Vec::with_capacity(out_len);
        let max_idx = self.len() - 1;

        for i in 0..out_len {
            let src_pos = i as f64 * ratio;
            let idx0 = (src_pos.floor() as usize).min(max_idx);
            let idx1 = (idx0 + 1).min(max_idx);
            let frac = (src_pos - idx0 as f64) as f32;
            out_l.push(self.left[idx0] * (1.0 - frac) + self.left[idx1] * frac);
            out_r.push(self.right[idx0] * (1.0 - frac) + self.right[idx1] * frac);
        }

        Self::from_vecs(out_l, out_r, target_sr)
    }

    /// Campiona istantaneamente (min, max, rms) nell'intervallo [f0..f1] per una colonna pixel.
    pub fn sample_envelope(&self, f0: usize, f1: usize) -> (f32, f32, f32) {
        let span = f1.saturating_sub(f0).max(1);
        let mip = if span >= 2048 {
            &self.mip_coarse
        } else {
            &self.mip_fine
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

pub fn decode_audio_file(path: &Path) -> Result<StereoBuffer, String> {
    let probe_once = |use_ext: bool| -> Result<symphonia::core::probe::ProbeResult, String> {
        let file = File::open(path).map_err(|e| format!("Impossibile aprire il file: {e}"))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if use_ext {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let mapped = if ext.eq_ignore_ascii_case("mpeg") { "mp3" } else { ext };
                hint.with_extension(mapped);
            }
        }
        symphonia::default::get_probe()
            .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
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
                    muted: false,
                    solo: false,
                    live_peak: 0.0,
                },
                StemChannel {
                    name: "Batteria",
                    subtitle: "Punch 62Hz + Anti-Box",
                    gain_db: 0.0,
                    pan: 0.0,
                    muted: false,
                    solo: false,
                    live_peak: 0.0,
                },
                StemChannel {
                    name: "Basso",
                    subtitle: "Mono Sub + Anti-Mud",
                    gain_db: 0.0,
                    pan: 0.0,
                    muted: false,
                    solo: false,
                    live_peak: 0.0,
                },
                StemChannel {
                    name: "Chitarre & Altro",
                    subtitle: "Body 1.4kHz + Stereo Wide",
                    gain_db: 0.0,
                    pan: 0.0,
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

#[inline]
fn pan_gains(pan: f32) -> (f32, f32) {
    // Constant-power pan law
    let angle = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    (angle.cos(), angle.sin())
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
                        let (pl, pr) = pan_gains(ch.pan);
                        (g, pl * std::f32::consts::SQRT_2 * 0.7071, pr * std::f32::consts::SQRT_2 * 0.7071)
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
                                        let idx = pos.min(stems[i].len() - 1);
                                        let (g, pl, pr) = stem_cfg[i];
                                        let l_s = stems[i].left[idx] * g * pl;
                                        let r_s = stems[i].right[idx] * g * pr;
                                        stem_peaks[i] = stem_peaks[i].max(l_s.abs().max(r_s.abs()));
                                        sl += l_s;
                                        sr_val += r_s;
                                    }
                                    // Passa anche attraverso il Master EQ live (in Flat di default sui 4 stems, o personalizzabile!)
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
            .map_err(|e| format!("Impossibile avviare lo stream audio WASAPI: {e}"))?;

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
        .unwrap_or(44100);

    let len = match mode {
        PlaybackSourceMode::FourStemMixer => stems.map(|s| s[0].len()).unwrap_or(0),
        PlaybackSourceMode::PreRenderedAiMaster => ai_master
            .or(main_track)
            .map(|b| b.len())
            .unwrap_or(0),
        PlaybackSourceMode::OriginalWithLiveDsp => main_track.map(|b| b.len()).unwrap_or(0),
    };

    if len == 0 {
        return Err("Nessun audio disponibile per l'esportazione.".to_string());
    }

    let mut dsp = StudioDspChain::new(sr as f64, params);
    let any_solo = stem_channels.iter().any(|c| c.solo);
    let stem_cfg: [(f32, f32, f32); 4] = std::array::from_fn(|i| {
        let ch = &stem_channels[i];
        let audible = if any_solo { ch.solo } else { !ch.muted };
        let g = if audible {
            10.0_f32.powf(ch.gain_db / 20.0)
        } else {
            0.0
        };
        let (pl, pr) = pan_gains(ch.pan);
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

    let scale_24 = 8_388_607.0_f32;
    for i in 0..len {
        let (in_l, in_r) = match mode {
            PlaybackSourceMode::FourStemMixer => {
                if let Some(st_arr) = stems {
                    let mut sl = 0.0_f32;
                    let mut sr_v = 0.0_f32;
                    for k in 0..4 {
                        let idx = i.min(st_arr[k].len() - 1);
                        let (g, pl, pr) = stem_cfg[k];
                        sl += st_arr[k].left[idx] * g * pl;
                        sr_v += st_arr[k].right[idx] * g * pr;
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
        let il = (l.clamp(-1.0, 1.0) * scale_24) as i32;
        let ir = (r.clamp(-1.0, 1.0) * scale_24) as i32;
        writer.write_sample(il).map_err(|e| e.to_string())?;
        writer.write_sample(ir).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;

    if is_mp3 {
        let mut cmd = Command::new("ffmpeg");
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
