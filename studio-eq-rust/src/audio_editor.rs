#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::f32::consts::{FRAC_PI_2, PI};

use crate::audio_io::{PlaybackSourceMode, SharedAudioState, StereoBuffer};

pub const MAX_AUDIO_SNAPSHOTS: usize = 15;

/// Snapshot leggero O(1) grazie ad `Arc` dentro `StereoBuffer`:
/// `(main_track, ai_master_track, stems, selection_range, position_frame)`
pub type AudioSnapshotTuple = (
    Option<StereoBuffer>,
    Option<StereoBuffer>,
    Option<[StereoBuffer; 4]>,
    Option<(usize, usize)>,
    usize,
);

#[derive(Clone, Debug)]
pub struct AudioEditHistory {
    undo_stack: VecDeque<AudioSnapshotTuple>,
    redo_stack: VecDeque<AudioSnapshotTuple>,
}

impl Default for AudioEditHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioEditHistory {
    pub fn new() -> Self {
        Self {
            undo_stack: VecDeque::with_capacity(MAX_AUDIO_SNAPSHOTS),
            redo_stack: VecDeque::with_capacity(MAX_AUDIO_SNAPSHOTS),
        }
    }

    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    #[inline]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    #[inline]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    #[allow(dead_code)]
    #[inline]
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    #[allow(dead_code)]
    #[inline]
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    fn capture(st: &SharedAudioState) -> AudioSnapshotTuple {
        (
            st.main_track.clone(),
            st.ai_master_track.clone(),
            st.stems.clone(),
            st.selection_range,
            st.position_frame,
        )
    }

    fn restore(st: &mut SharedAudioState, snap: AudioSnapshotTuple) {
        let (main_track, ai_master_track, stems, selection_range, pos) = snap;
        st.main_track = main_track;
        st.ai_master_track = ai_master_track;
        st.stems = stems;
        st.selection_range = selection_range;
        let max_len = st.active_length();
        st.position_frame = if max_len > 0 {
            pos.min(max_len - 1)
        } else {
            0
        };
        if st.selection_range.is_none() {
            st.loop_enabled = false;
        }
    }

    pub fn push_before_edit(&mut self, st: &SharedAudioState) {
        if self.undo_stack.len() >= MAX_AUDIO_SNAPSHOTS {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(Self::capture(st));
        self.redo_stack.clear();
    }

    pub fn undo(&mut self, st: &mut SharedAudioState) -> Result<String, String> {
        let Some(prev) = self.undo_stack.pop_back() else {
            return Err("Nessuna modifica audio da annullare.".to_string());
        };
        if self.redo_stack.len() >= MAX_AUDIO_SNAPSHOTS {
            self.redo_stack.pop_front();
        }
        self.redo_stack.push_back(Self::capture(st));
        Self::restore(st, prev);
        Ok(format!(
            "Annullata modifica audio ({} rimaste)",
            self.undo_stack.len()
        ))
    }

    pub fn redo(&mut self, st: &mut SharedAudioState) -> Result<String, String> {
        let Some(next) = self.redo_stack.pop_back() else {
            return Err("Nessuna modifica audio da ripristinare.".to_string());
        };
        if self.undo_stack.len() >= MAX_AUDIO_SNAPSHOTS {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(Self::capture(st));
        Self::restore(st, next);
        Ok(format!(
            "Ripristinata modifica audio ({} disponibili)",
            self.redo_stack.len()
        ))
    }
}

/// Appunti audio interni con supporto sia per traccia singola sia per tutti i 4 Stems sincronizzati.
#[derive(Clone, Debug, Default)]
pub struct AudioClipboard {
    pub main_clip: Option<StereoBuffer>,
    pub ai_master_clip: Option<StereoBuffer>,
    pub stems_clip: Option<[StereoBuffer; 4]>,
    pub sample_rate: u32,
}

impl AudioClipboard {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn has_content(&self) -> bool {
        self.clip_frames() > 0
    }

    pub fn clip_frames(&self) -> usize {
        self.main_clip
            .as_ref()
            .map(|b| b.len())
            .or_else(|| self.stems_clip.as_ref().map(|s| s[0].len()))
            .or_else(|| self.ai_master_clip.as_ref().map(|b| b.len()))
            .unwrap_or(0)
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GainEditOp {
    /// Variazione di guadagno in dB (es. `+3.0` o `-3.0`).
    BoostDb(f32),
    /// Normalizzazione del picco al livello indicato in dBFS (es. `-1.0`).
    NormalizePeakDb(f32),
}

#[allow(dead_code)]
impl GainEditOp {
    pub const PLUS_3DB: Self = Self::BoostDb(3.0);
    pub const MINUS_3DB: Self = Self::BoostDb(-3.0);
    pub const NORMALIZE_MINUS_1DB: Self = Self::NormalizePeakDb(-1.0);
}

#[inline]
fn s_curve_01(t: f32) -> f32 {
    0.5_f32 * (1.0_f32 - (PI * t.clamp(0.0_f32, 1.0_f32)).cos())
}

#[inline]
fn equal_power_s_weights(t: f32) -> (f32, f32) {
    let s = s_curve_01(t);
    let angle = FRAC_PI_2 * s;
    (angle.cos(), angle.sin())
}

#[inline]
fn ms_to_frames(ms: f64, sr: u32) -> usize {
    (((ms * 0.001) * (sr.max(8_000) as f64)).round() as usize).max(4)
}

#[inline]
fn map_frame_pos(pos: usize, ref_len: usize, buf_len: usize) -> usize {
    if buf_len == 0 || ref_len == 0 {
        return 0;
    }
    if buf_len == ref_len {
        return pos.min(buf_len);
    }
    (((pos as u128) * (buf_len as u128)) / (ref_len as u128)).min(buf_len as u128) as usize
}

#[inline]
fn map_range(sel_start: usize, sel_end: usize, ref_len: usize, buf_len: usize) -> (usize, usize) {
    let a = map_frame_pos(sel_start, ref_len, buf_len);
    let b = map_frame_pos(sel_end, ref_len, buf_len);
    (a.min(b), a.max(b).min(buf_len))
}

#[inline]
fn soft_ceiling_sample(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    let ax = x.abs();
    if ax <= 0.96_f32 {
        x
    } else {
        let span = 0.0399_f32;
        let shaped = 0.96_f32 + span * ((ax - 0.96_f32) / span).tanh();
        x.signum() * shaped.min(0.9999_f32)
    }
}

fn validated_selection(st: &SharedAudioState) -> Result<((usize, usize), usize), String> {
    let ref_len = st.active_length();
    if ref_len == 0 {
        return Err("Nessuna traccia audio caricata su cui eseguire l'editing.".to_string());
    }
    let Some((raw_a, raw_b)) = st.selection_range else {
        return Err(
            "Seleziona prima un intervallo trascinando il mouse sulle tracce audio.".to_string(),
        );
    };
    let s0 = raw_a.min(raw_b).min(ref_len);
    let s1 = raw_a.max(raw_b).min(ref_len);
    if s1 <= s0 + 16 {
        return Err("La zona selezionata è troppo breve per questa operazione.".to_string());
    }
    Ok(((s0, s1), ref_len))
}

fn slice_stereo_buffer(buf: &StereoBuffer, s0: usize, s1: usize) -> StereoBuffer {
    let len = buf.len();
    let a = s0.min(len);
    let b = s1.min(len).max(a);
    let l = buf.left[a..b].to_vec();
    let r = buf.right[a..b].to_vec();
    StereoBuffer::from_vecs(l, r, buf.sample_rate)
}

/// Taglia via `[s0..s1]` da un `StereoBuffer` applicando un micro-crossfade equal-power S-curve da 8 ms
/// centrato sulla giunzione (oppure micro-fade anti-click da 5 ms se il taglio tocca l'inizio o la fine).
fn splice_out_buffer(buf: &StereoBuffer, s0: usize, s1: usize) -> StereoBuffer {
    let len = buf.len();
    let a = s0.min(len);
    let b = s1.min(len).max(a);
    if a == b || len == 0 {
        return buf.clone();
    }

    let cut_len = b - a;
    if cut_len >= len {
        return StereoBuffer::from_vecs(vec![0.0; 64], vec![0.0; 64], buf.sample_rate);
    }

    let out_len = len - cut_len;
    let edge_fade = ms_to_frames(5.0, buf.sample_rate).min(out_len / 2);

    // Caso 1: taglio dall'inizio [0..b] -> teniamo [b..len] con 5 ms fade-in anti-click
    if a == 0 {
        let mut out_l = buf.left[b..len].to_vec();
        let mut out_r = buf.right[b..len].to_vec();
        for k in 0..edge_fade {
            let g = s_curve_01((k as f32) / (edge_fade.max(1) as f32));
            out_l[k] *= g;
            out_r[k] *= g;
        }
        return StereoBuffer::from_vecs(out_l, out_r, buf.sample_rate);
    }

    // Caso 2: taglio fino alla fine [a..len] -> teniamo [0..a] con 5 ms fade-out anti-click
    if b >= len {
        let mut out_l = buf.left[0..a].to_vec();
        let mut out_r = buf.right[0..a].to_vec();
        let start_f = out_len.saturating_sub(edge_fade);
        for k in 0..(out_len - start_f) {
            let g = 1.0_f32 - s_curve_01(((k + 1) as f32) / (edge_fade.max(1) as f32));
            out_l[start_f + k] *= g;
            out_r[start_f + k] *= g;
        }
        return StereoBuffer::from_vecs(out_l, out_r, buf.sample_rate);
    }

    // Caso 3: taglio interno [a..b] -> micro-crossfade equal-power S-curve da 8 ms attraverso la giunzione
    let xfade_target = ms_to_frames(8.0, buf.sample_rate);
    let half_max = (xfade_target / 2)
        .min(a)
        .min(len - b)
        .min(cut_len / 2);

    let mut out_l = Vec::with_capacity(out_len);
    let mut out_r = Vec::with_capacity(out_len);

    if half_max < 2 {
        out_l.extend_from_slice(&buf.left[0..a]);
        out_l.extend_from_slice(&buf.left[b..len]);
        out_r.extend_from_slice(&buf.right[0..a]);
        out_r.extend_from_slice(&buf.right[b..len]);
        return StereoBuffer::from_vecs(out_l, out_r, buf.sample_rate);
    }

    let ov = half_max * 2;
    let pre_end = a - half_max;
    let post_start = b + half_max;

    out_l.extend_from_slice(&buf.left[0..pre_end]);
    out_r.extend_from_slice(&buf.right[0..pre_end]);

    for k in 0..ov {
        let t = ((k + 1) as f32) / ((ov + 1) as f32);
        let (w_a, w_b) = equal_power_s_weights(t);
        let idx_a = pre_end + k; // [a - half_max .. a + half_max]
        let idx_b = (b - half_max) + k; // [b - half_max .. b + half_max]
        out_l.push(soft_ceiling_sample(
            buf.left[idx_a] * w_a + buf.left[idx_b] * w_b,
        ));
        out_r.push(soft_ceiling_sample(
            buf.right[idx_a] * w_a + buf.right[idx_b] * w_b,
        ));
    }

    out_l.extend_from_slice(&buf.left[post_start..len]);
    out_r.extend_from_slice(&buf.right[post_start..len]);

    debug_assert_eq!(out_l.len(), out_len);
    StereoBuffer::from_vecs(out_l, out_r, buf.sample_rate)
}

/// Mantiene SOLO `[s0..s1]` con micro-fade anti-click da 5 ms su entrambi i bordi.
fn crop_buffer(buf: &StereoBuffer, s0: usize, s1: usize) -> StereoBuffer {
    let len = buf.len();
    let a = s0.min(len);
    let b = s1.min(len).max(a);
    let mut out_l = buf.left[a..b].to_vec();
    let mut out_r = buf.right[a..b].to_vec();
    let out_len = out_l.len();

    let fade = ms_to_frames(5.0, buf.sample_rate).min(out_len / 2);
    for k in 0..fade {
        let g_in = s_curve_01((k as f32) / (fade.max(1) as f32));
        out_l[k] *= g_in;
        out_r[k] *= g_in;

        let end_idx = out_len - 1 - k;
        out_l[end_idx] *= g_in;
        out_r[end_idx] *= g_in;
    }

    StereoBuffer::from_vecs(out_l, out_r, buf.sample_rate)
}

/// Inserisce `clip` alla posizione `ins_pos` con micro-crossfade equal-power da 8 ms su entrambi i bordi.
fn insert_into_buffer(buf: &StereoBuffer, clip: &StereoBuffer, ins_pos: usize) -> StereoBuffer {
    let resampled_clip = if clip.sample_rate != buf.sample_rate {
        clip.resample_to(buf.sample_rate)
    } else {
        clip.clone()
    };

    let len = buf.len();
    let c_len = resampled_clip.len();
    if c_len == 0 {
        return buf.clone();
    }
    if len == 0 {
        return resampled_clip;
    }

    let pos = ins_pos.min(len);
    let out_len = len + c_len;
    let xfade = ms_to_frames(8.0, buf.sample_rate).min(c_len / 2);
    let edge_fade = ms_to_frames(5.0, buf.sample_rate).min(c_len / 2);

    let mut out_l = Vec::with_capacity(out_len);
    let mut out_r = Vec::with_capacity(out_len);

    // 1. Parte precedente del brano [0..pos]
    out_l.extend_from_slice(&buf.left[0..pos]);
    out_r.extend_from_slice(&buf.right[0..pos]);

    // 2. Clip inserita [0..c_len], con crossfade sinistro e destro usando la continuazione naturale di `buf`
    let ov_left = if pos > 0 { xfade.min(len - pos) } else { 0 };
    let ov_right = if pos < len { xfade.min(pos) } else { 0 };

    for k in 0..c_len {
        let mut sl = resampled_clip.left[k];
        let mut sr = resampled_clip.right[k];

        if k < ov_left {
            let t = ((k + 1) as f32) / ((ov_left + 1) as f32);
            let (w_prev, w_clip) = equal_power_s_weights(t);
            sl = soft_ceiling_sample(buf.left[pos + k] * w_prev + sl * w_clip);
            sr = soft_ceiling_sample(buf.right[pos + k] * w_prev + sr * w_clip);
        } else if pos == 0 && k < edge_fade {
            let g = s_curve_01((k as f32) / (edge_fade.max(1) as f32));
            sl *= g;
            sr *= g;
        }

        let from_end = c_len - 1 - k;
        if from_end < ov_right && k >= ov_left {
            // `m` va da 0 .. ov_right
            let m = ov_right - 1 - from_end;
            let t = ((m + 1) as f32) / ((ov_right + 1) as f32);
            let (w_clip, w_next) = equal_power_s_weights(t);
            let b_idx = (pos - ov_right) + m;
            sl = soft_ceiling_sample(sl * w_clip + buf.left[b_idx] * w_next);
            sr = soft_ceiling_sample(sr * w_clip + buf.right[b_idx] * w_next);
        } else if pos == len && from_end < edge_fade {
            let g = s_curve_01((from_end as f32) / (edge_fade.max(1) as f32));
            sl *= g;
            sr *= g;
        }

        out_l.push(sl);
        out_r.push(sr);
    }

    // 3. Parte successiva del brano [pos..len]
    out_l.extend_from_slice(&buf.left[pos..len]);
    out_r.extend_from_slice(&buf.right[pos..len]);

    debug_assert_eq!(out_l.len(), out_len);
    StereoBuffer::from_vecs(out_l, out_r, buf.sample_rate)
}

/// Silenzia morbido `[s0..s1]` con rampe cosinusoidali da 10 ms ai bordi.
fn silence_buffer_region(buf: &StereoBuffer, s0: usize, s1: usize) -> StereoBuffer {
    let len = buf.len();
    let a = s0.min(len);
    let b = s1.min(len).max(a);
    if a == b || len == 0 {
        return buf.clone();
    }

    let mut out_l = (*buf.left).clone();
    let mut out_r = (*buf.right).clone();

    let span = b - a;
    let ramp = ms_to_frames(10.0, buf.sample_rate).min(span / 2);

    for idx in a..b {
        let g = if a > 0 && idx < a + ramp {
            let t = ((idx - a + 1) as f32) / ((ramp + 1) as f32);
            0.5_f32 * (1.0_f32 + (PI * t).cos())
        } else if b < len && idx >= b - ramp {
            let t = ((idx - (b - ramp) + 1) as f32) / ((ramp + 1) as f32);
            0.5_f32 * (1.0_f32 - (PI * t).cos())
        } else {
            0.0_f32
        };
        out_l[idx] *= g;
        out_r[idx] *= g;
    }

    StereoBuffer::from_vecs(out_l, out_r, buf.sample_rate)
}

fn fade_in_buffer_region(buf: &StereoBuffer, s0: usize, s1: usize) -> StereoBuffer {
    let len = buf.len();
    let a = s0.min(len);
    let b = s1.min(len).max(a);
    if b <= a + 2 || len == 0 {
        return buf.clone();
    }

    let mut out_l = (*buf.left).clone();
    let mut out_r = (*buf.right).clone();

    // Se la zona inizia dentro il brano (`a > 0`), ammorbidiamo i 5 ms precedenti verso 0 per evitare scalini
    if a > 0 {
        let pre = ms_to_frames(5.0, buf.sample_rate).min(a);
        let start_pre = a - pre;
        for k in 0..pre {
            let t = ((k + 1) as f32) / ((pre + 1) as f32);
            let g = 1.0_f32 - s_curve_01(t);
            out_l[start_pre + k] *= g;
            out_r[start_pre + k] *= g;
        }
    }

    let span = (b - a).max(1);
    for k in 0..span {
        let t = (k as f32) / ((span - 1).max(1) as f32);
        let g = s_curve_01(t);
        out_l[a + k] *= g;
        out_r[a + k] *= g;
    }

    StereoBuffer::from_vecs(out_l, out_r, buf.sample_rate)
}

fn fade_out_buffer_region(buf: &StereoBuffer, s0: usize, s1: usize) -> StereoBuffer {
    let len = buf.len();
    let a = s0.min(len);
    let b = s1.min(len).max(a);
    if b <= a + 2 || len == 0 {
        return buf.clone();
    }

    let mut out_l = (*buf.left).clone();
    let mut out_r = (*buf.right).clone();

    let span = (b - a).max(1);
    for k in 0..span {
        let t = (k as f32) / ((span - 1).max(1) as f32);
        let g = 1.0_f32 - s_curve_01(t);
        out_l[a + k] *= g;
        out_r[a + k] *= g;
    }

    // Se la zona finisce prima della fine del brano (`b < len`), rampa morbida da 5 ms per riprendere senza click
    if b < len {
        let post = ms_to_frames(5.0, buf.sample_rate).min(len - b);
        for k in 0..post {
            let t = ((k + 1) as f32) / ((post + 1) as f32);
            let g = s_curve_01(t);
            out_l[b + k] *= g;
            out_r[b + k] *= g;
        }
    }

    StereoBuffer::from_vecs(out_l, out_r, buf.sample_rate)
}

fn apply_gain_buffer_region(
    buf: &StereoBuffer,
    s0: usize,
    s1: usize,
    gain_lin: f32,
) -> StereoBuffer {
    let len = buf.len();
    let a = s0.min(len);
    let b = s1.min(len).max(a);
    if b <= a || len == 0 || (gain_lin - 1.0_f32).abs() < 1e-5 {
        return buf.clone();
    }

    let mut out_l = (*buf.left).clone();
    let mut out_r = (*buf.right).clone();
    let span = b - a;
    let ramp = ms_to_frames(8.0, buf.sample_rate).min(span / 2);

    for idx in a..b {
        let blend = if a > 0 && idx < a + ramp {
            s_curve_01(((idx - a + 1) as f32) / ((ramp + 1) as f32))
        } else if b < len && idx >= b - ramp {
            s_curve_01(((b - idx) as f32) / ((ramp + 1) as f32))
        } else {
            1.0_f32
        };
        let g = 1.0_f32 + (gain_lin - 1.0_f32) * blend;
        out_l[idx] = soft_ceiling_sample(out_l[idx] * g);
        out_r[idx] = soft_ceiling_sample(out_r[idx] * g);
    }

    StereoBuffer::from_vecs(out_l, out_r, buf.sample_rate)
}

fn peak_in_region(buf: &StereoBuffer, s0: usize, s1: usize) -> f32 {
    let len = buf.len();
    let a = s0.min(len);
    let b = s1.min(len).max(a);
    let mut pk = 0.0_f32;
    for i in a..b {
        pk = pk.max(buf.left[i].abs()).max(buf.right[i].abs());
    }
    pk
}

/// Copia la selezione corrente negli appunti interni (`AudioClipboard`) su tutte le tracce attive.
pub fn copy_selection(
    st: &SharedAudioState,
    clipboard: &mut AudioClipboard,
) -> Result<String, String> {
    let ((s0, s1), ref_len) = validated_selection(st)?;
    let sr = st.device_sr.max(8_000);

    clipboard.main_clip = st.main_track.as_ref().map(|b| {
        let (a, e) = map_range(s0, s1, ref_len, b.len());
        slice_stereo_buffer(b, a, e)
    });
    clipboard.ai_master_clip = st.ai_master_track.as_ref().map(|b| {
        let (a, e) = map_range(s0, s1, ref_len, b.len());
        slice_stereo_buffer(b, a, e)
    });
    clipboard.stems_clip = st.stems.as_ref().map(|arr| {
        std::array::from_fn(|i| {
            let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
            slice_stereo_buffer(&arr[i], a, e)
        })
    });
    clipboard.sample_rate = sr;

    let dur_s = ((s1 - s0) as f32) / (sr as f32);
    Ok(format!("Copiati {dur_s:.2}s negli appunti Studio"))
}

/// Elimina la selezione `[sel_start..sel_end]` sincronizzando `main_track`, `ai_master_track` e tutti i 4 `stems`
/// con micro-crossfade equal-power S-curve da 8 ms sulla giunzione.
pub fn delete_selection(
    st: &mut SharedAudioState,
    history: &mut AudioEditHistory,
) -> Result<String, String> {
    let ((s0, s1), ref_len) = validated_selection(st)?;
    let min_remaining = ms_to_frames(50.0, st.device_sr).max(64);
    if ref_len.saturating_sub(s1 - s0) < min_remaining {
        return Err(
            "Impossibile eliminare l'intero brano: lascia almeno una parte audio attiva."
                .to_string(),
        );
    }

    history.push_before_edit(st);

    if let Some(ref b) = st.main_track {
        let (a, e) = map_range(s0, s1, ref_len, b.len());
        st.main_track = Some(splice_out_buffer(b, a, e));
    }
    if let Some(ref b) = st.ai_master_track {
        let (a, e) = map_range(s0, s1, ref_len, b.len());
        st.ai_master_track = Some(splice_out_buffer(b, a, e));
    }
    if let Some(ref arr) = st.stems {
        let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
            let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
            splice_out_buffer(&arr[i], a, e)
        });
        st.stems = Some(new_stems);
    }

    let new_len = st.active_length();
    st.position_frame = s0.min(new_len.saturating_sub(1));
    st.selection_range = None;
    st.loop_enabled = false;

    let dur_s = ((s1 - s0) as f32) / (st.device_sr.max(1) as f32);
    Ok(format!("Eliminati {dur_s:.2}s con micro-crossfade 8 ms"))
}

/// Taglia (`Ctrl+X`): copia la selezione negli appunti e la rimuove con crossfade da 8 ms.
pub fn cut_selection(
    st: &mut SharedAudioState,
    history: &mut AudioEditHistory,
    clipboard: &mut AudioClipboard,
) -> Result<String, String> {
    let _ = copy_selection(st, clipboard)?;
    let ((s0, s1), _) = validated_selection(st)?;
    let _ = delete_selection(st, history)?;
    let dur_s = ((s1 - s0) as f32) / (st.device_sr.max(1) as f32);
    Ok(format!("Tagliati {dur_s:.2}s negli appunti (Ctrl+V per incollare)"))
}

/// Ritaglia Inizio/Fine (`Ctrl+T`): mantiene SOLO la zona selezionata `[sel_start..sel_end]`
/// applicando micro-fade anti-click da 5 ms su inizio e fine.
pub fn crop_to_selection(
    st: &mut SharedAudioState,
    history: &mut AudioEditHistory,
) -> Result<String, String> {
    let ((s0, s1), ref_len) = validated_selection(st)?;
    if s0 == 0 && s1 >= ref_len {
        return Err("L'intero brano è già selezionato: restringi la selezione per ritagliare.".to_string());
    }

    history.push_before_edit(st);

    if let Some(ref b) = st.main_track {
        let (a, e) = map_range(s0, s1, ref_len, b.len());
        st.main_track = Some(crop_buffer(b, a, e));
    }
    if let Some(ref b) = st.ai_master_track {
        let (a, e) = map_range(s0, s1, ref_len, b.len());
        st.ai_master_track = Some(crop_buffer(b, a, e));
    }
    if let Some(ref arr) = st.stems {
        let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
            let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
            crop_buffer(&arr[i], a, e)
        });
        st.stems = Some(new_stems);
    }

    st.position_frame = 0;
    st.selection_range = None;
    st.loop_enabled = false;

    let dur_s = ((s1 - s0) as f32) / (st.device_sr.max(1) as f32);
    Ok(format!("Brano ritagliato alla selezione ({dur_s:.2}s)"))
}

fn build_fallback_single_clip(clipboard: &AudioClipboard, sr: u32) -> Option<StereoBuffer> {
    if let Some(ref m) = clipboard.main_clip {
        return Some(m.resample_to(sr));
    }
    if let Some(ref ai) = clipboard.ai_master_clip {
        return Some(ai.resample_to(sr));
    }
    if let Some(ref stems) = clipboard.stems_clip {
        let len = stems[0].len();
        let mut l = vec![0.0_f32; len];
        let mut r = vec![0.0_f32; len];
        for s in stems {
            let n = len.min(s.len());
            for i in 0..n {
                l[i] += s.left[i] * 0.82_f32;
                r[i] += s.right[i] * 0.82_f32;
            }
        }
        return Some(StereoBuffer::from_vecs(l, r, stems[0].sample_rate).resample_to(sr));
    }
    None
}

/// Incolla il contenuto degli appunti alla posizione corrente della testina (`position_frame`)
/// con crossfade da 8 ms su entrambi i lati della giunzione.
pub fn paste_at_playhead(
    st: &mut SharedAudioState,
    history: &mut AudioEditHistory,
    clipboard: &AudioClipboard,
) -> Result<String, String> {
    if !clipboard.has_content() {
        return Err("Appunti vuoti: usa prima Copia (Ctrl+C) o Taglia (Ctrl+X).".to_string());
    }
    let ref_len = st.active_length();
    if ref_len == 0 {
        return Err("Nessuna traccia attiva in cui incollare gli appunti.".to_string());
    }

    let ins_pos = st.position_frame.min(ref_len);
    let fallback_single = build_fallback_single_clip(clipboard, st.device_sr.max(8_000));

    history.push_before_edit(st);

    if let Some(ref b) = st.main_track {
        if let Some(clip) = clipboard.main_clip.as_ref().or(fallback_single.as_ref()) {
            let p = map_frame_pos(ins_pos, ref_len, b.len());
            st.main_track = Some(insert_into_buffer(b, clip, p));
        }
    }
    if let Some(ref b) = st.ai_master_track {
        if let Some(clip) = clipboard
            .ai_master_clip
            .as_ref()
            .or(clipboard.main_clip.as_ref())
            .or(fallback_single.as_ref())
        {
            let p = map_frame_pos(ins_pos, ref_len, b.len());
            st.ai_master_track = Some(insert_into_buffer(b, clip, p));
        }
    }
    if let Some(ref arr) = st.stems {
        if let Some(ref stem_clips) = clipboard.stems_clip {
            let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
                let p = map_frame_pos(ins_pos, ref_len, arr[i].len());
                insert_into_buffer(&arr[i], &stem_clips[i], p)
            });
            st.stems = Some(new_stems);
        } else if let Some(ref single) = fallback_single {
            // Mantiene l'allineamento temporale perfetto tra main_track e 4 stems
            let scaled_l: Vec<f32> = single.left.iter().map(|&x| x * 0.28_f32).collect();
            let scaled_r: Vec<f32> = single.right.iter().map(|&x| x * 0.28_f32).collect();
            let stem_fill = StereoBuffer::from_vecs(scaled_l, scaled_r, single.sample_rate);
            let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
                let p = map_frame_pos(ins_pos, ref_len, arr[i].len());
                insert_into_buffer(&arr[i], &stem_fill, p)
            });
            st.stems = Some(new_stems);
        }
    }

    let pasted_len = st.active_length().saturating_sub(ref_len);
    let new_end = (ins_pos + pasted_len).min(st.active_length());
    if new_end > ins_pos {
        st.selection_range = Some((ins_pos, new_end));
    }
    st.position_frame = ins_pos;

    let dur_s = (pasted_len as f32) / (st.device_sr.max(1) as f32);
    Ok(format!("Incollati {dur_s:.2}s alla testina con crossfade 8 ms"))
}

/// Duplica la selezione corrente (`Ctrl+D`) subito dopo `sel_end` con crossfade da 8 ms.
pub fn duplicate_selection(
    st: &mut SharedAudioState,
    history: &mut AudioEditHistory,
) -> Result<String, String> {
    let ((_, s1), _) = validated_selection(st)?;
    let mut temp_clip = AudioClipboard::new();
    let _ = copy_selection(st, &mut temp_clip)?;

    let prev_pos = st.position_frame;
    // Registra lo stato con la posizione originale prima di spostare il cursore a `s1`
    history.push_before_edit(st);

    let ref_len = st.active_length();
    let ins_pos = s1.min(ref_len);
    let fallback_single = build_fallback_single_clip(&temp_clip, st.device_sr.max(8_000));

    if let Some(ref b) = st.main_track {
        if let Some(clip) = temp_clip.main_clip.as_ref().or(fallback_single.as_ref()) {
            let p = map_frame_pos(ins_pos, ref_len, b.len());
            st.main_track = Some(insert_into_buffer(b, clip, p));
        }
    }
    if let Some(ref b) = st.ai_master_track {
        if let Some(clip) = temp_clip
            .ai_master_clip
            .as_ref()
            .or(temp_clip.main_clip.as_ref())
            .or(fallback_single.as_ref())
        {
            let p = map_frame_pos(ins_pos, ref_len, b.len());
            st.ai_master_track = Some(insert_into_buffer(b, clip, p));
        }
    }
    if let Some(ref arr) = st.stems {
        if let Some(ref stem_clips) = temp_clip.stems_clip {
            let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
                let p = map_frame_pos(ins_pos, ref_len, arr[i].len());
                insert_into_buffer(&arr[i], &stem_clips[i], p)
            });
            st.stems = Some(new_stems);
        }
    }

    let dup_len = st.active_length().saturating_sub(ref_len);
    if dup_len > 0 {
        st.selection_range = Some((ins_pos, (ins_pos + dup_len).min(st.active_length())));
        st.position_frame = ins_pos;
    } else {
        st.position_frame = prev_pos;
    }

    let dur_s = (dup_len as f32) / (st.device_sr.max(1) as f32);
    Ok(format!("Duplicata selezione ({dur_s:.2}s) con crossfade 8 ms"))
}

/// Silenzia la selezione `[sel_start..sel_end]` con rampe cosinusoidali morbide da 10 ms.
/// Funzione Smart Sala Prove: se si è in `FourStemMixer` e almeno uno Stem è in `Solo`,
/// silenzia SOLO gli Stem in `Solo` (es. per pulire i rientri microfonici della Voce durante un assolo!).
pub fn silence_selection(
    st: &mut SharedAudioState,
    history: &mut AudioEditHistory,
) -> Result<String, String> {
    let ((s0, s1), ref_len) = validated_selection(st)?;
    history.push_before_edit(st);

    let dur_s = ((s1 - s0) as f32) / (st.device_sr.max(1) as f32);
    let solo_mask: [bool; 4] = std::array::from_fn(|i| st.stem_channels[i].solo);
    let solo_only = st.mode == PlaybackSourceMode::FourStemMixer
        && st.stems.is_some()
        && solo_mask.iter().any(|&s| s);

    if solo_only {
        if let Some(ref arr) = st.stems {
            let mut names = Vec::new();
            let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
                if solo_mask[i] {
                    names.push(st.stem_channels[i].name);
                    let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
                    silence_buffer_region(&arr[i], a, e)
                } else {
                    arr[i].clone()
                }
            });
            st.stems = Some(new_stems);
            return Ok(format!(
                "Zona silenziata ({dur_s:.2}s) solo su Stem in Solo: {}",
                names.join(", ")
            ));
        }
    }

    if let Some(ref b) = st.main_track {
        let (a, e) = map_range(s0, s1, ref_len, b.len());
        st.main_track = Some(silence_buffer_buffer_safe(b, a, e));
    }
    if let Some(ref b) = st.ai_master_track {
        let (a, e) = map_range(s0, s1, ref_len, b.len());
        st.ai_master_track = Some(silence_buffer_buffer_safe(b, a, e));
    }
    if let Some(ref arr) = st.stems {
        let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
            let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
            silence_buffer_buffer_safe(&arr[i], a, e)
        });
        st.stems = Some(new_stems);
    }

    Ok(format!("Zona silenziata su tutte le tracce ({dur_s:.2}s)"))
}

#[inline]
fn silence_buffer_buffer_safe(buf: &StereoBuffer, a: usize, e: usize) -> StereoBuffer {
    silence_buffer_region(buf, a, e)
}

/// Applica un Fade In musicale S-curve (`0.5 * (1 - cos(pi * t))`) sulla selezione attiva,
/// oppure sui primi `2.0 s` del brano se non c'è alcuna selezione.
pub fn apply_fade_in(
    st: &mut SharedAudioState,
    history: &mut AudioEditHistory,
) -> Result<String, String> {
    let ref_len = st.active_length();
    if ref_len == 0 {
        return Err("Nessuna traccia audio caricata per applicare il Fade In.".to_string());
    }

    let sr = st.device_sr.max(8_000);
    let (s0, s1) = match st.selection_range {
        Some((a, b)) if a.max(b) > a.min(b) + 16 => (a.min(b).min(ref_len), a.max(b).min(ref_len)),
        _ => (0, ((2.0_f64 * sr as f64).round() as usize).min(ref_len)),
    };
    if s1 <= s0 + 4 {
        return Err("Durata troppo breve per applicare il Fade In.".to_string());
    }

    history.push_before_edit(st);

    let solo_mask: [bool; 4] = std::array::from_fn(|i| st.stem_channels[i].solo);
    let solo_only = st.mode == PlaybackSourceMode::FourStemMixer
        && st.stems.is_some()
        && solo_mask.iter().any(|&s| s);

    if solo_only {
        if let Some(ref arr) = st.stems {
            let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
                if solo_mask[i] {
                    let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
                    fade_in_buffer_region(&arr[i], a, e)
                } else {
                    arr[i].clone()
                }
            });
            st.stems = Some(new_stems);
        }
    } else {
        if let Some(ref b) = st.main_track {
            let (a, e) = map_range(s0, s1, ref_len, b.len());
            st.main_track = Some(fade_in_buffer_region(b, a, e));
        }
        if let Some(ref b) = st.ai_master_track {
            let (a, e) = map_range(s0, s1, ref_len, b.len());
            st.ai_master_track = Some(fade_in_buffer_region(b, a, e));
        }
        if let Some(ref arr) = st.stems {
            let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
                let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
                fade_in_buffer_region(&arr[i], a, e)
            });
            st.stems = Some(new_stems);
        }
    }

    let dur_s = ((s1 - s0) as f32) / (sr as f32);
    Ok(format!("Applicato Studio Fade In S-curve ({dur_s:.2}s)"))
}

/// Applica un Fade Out musicale S-curve (`0.5 * (1 + cos(pi * t))`) sulla selezione attiva,
/// oppure sugli ultimi `3.0 s` del brano se non c'è alcuna selezione.
pub fn apply_fade_out(
    st: &mut SharedAudioState,
    history: &mut AudioEditHistory,
) -> Result<String, String> {
    let ref_len = st.active_length();
    if ref_len == 0 {
        return Err("Nessuna traccia audio caricata per applicare il Fade Out.".to_string());
    }

    let sr = st.device_sr.max(8_000);
    let (s0, s1) = match st.selection_range {
        Some((a, b)) if a.max(b) > a.min(b) + 16 => (a.min(b).min(ref_len), a.max(b).min(ref_len)),
        _ => {
            let span = ((3.0_f64 * sr as f64).round() as usize).min(ref_len);
            (ref_len.saturating_sub(span), ref_len)
        }
    };
    if s1 <= s0 + 4 {
        return Err("Durata troppo breve per applicare il Fade Out.".to_string());
    }

    history.push_before_edit(st);

    let solo_mask: [bool; 4] = std::array::from_fn(|i| st.stem_channels[i].solo);
    let solo_only = st.mode == PlaybackSourceMode::FourStemMixer
        && st.stems.is_some()
        && solo_mask.iter().any(|&s| s);

    if solo_only {
        if let Some(ref arr) = st.stems {
            let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
                if solo_mask[i] {
                    let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
                    fade_out_buffer_region(&arr[i], a, e)
                } else {
                    arr[i].clone()
                }
            });
            st.stems = Some(new_stems);
        }
    } else {
        if let Some(ref b) = st.main_track {
            let (a, e) = map_range(s0, s1, ref_len, b.len());
            st.main_track = Some(fade_out_buffer_region(b, a, e));
        }
        if let Some(ref b) = st.ai_master_track {
            let (a, e) = map_range(s0, s1, ref_len, b.len());
            st.ai_master_track = Some(fade_out_buffer_region(b, a, e));
        }
        if let Some(ref arr) = st.stems {
            let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
                let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
                fade_out_buffer_region(&arr[i], a, e)
            });
            st.stems = Some(new_stems);
        }
    }

    let dur_s = ((s1 - s0) as f32) / (sr as f32);
    Ok(format!("Applicato Studio Fade Out S-curve ({dur_s:.2}s)"))
}

/// Regola il guadagno (`+3 dB` / `-3 dB`) oppure normalizza il picco (es. `-1.0 dBFS`)
/// sulla selezione attiva o sull'intero brano se non vi è alcuna selezione.
pub fn boost_or_normalize_selection(
    st: &mut SharedAudioState,
    history: &mut AudioEditHistory,
    op: GainEditOp,
) -> Result<String, String> {
    let ref_len = st.active_length();
    if ref_len == 0 {
        return Err("Nessuna traccia audio caricata.".to_string());
    }

    let (s0, s1) = match st.selection_range {
        Some((a, b)) if a.max(b) > a.min(b) + 16 => (a.min(b).min(ref_len), a.max(b).min(ref_len)),
        _ => (0, ref_len),
    };

    let solo_mask: [bool; 4] = std::array::from_fn(|i| st.stem_channels[i].solo);
    let solo_only = st.mode == PlaybackSourceMode::FourStemMixer
        && st.stems.is_some()
        && solo_mask.iter().any(|&s| s);

    let gain_lin = match op {
        GainEditOp::BoostDb(db) => 10.0_f32.powf(db.clamp(-24.0, 24.0) / 20.0),
        GainEditOp::NormalizePeakDb(target_db) => {
            let target_lin = 10.0_f32.powf(target_db.clamp(-18.0, -0.1) / 20.0);
            let mut pk = 0.0_f32;
            if solo_only {
                if let Some(ref arr) = st.stems {
                    for i in 0..4 {
                        if solo_mask[i] {
                            let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
                            pk = pk.max(peak_in_region(&arr[i], a, e));
                        }
                    }
                }
            } else {
                match st.mode {
                    PlaybackSourceMode::PreRenderedAiMaster => {
                        if let Some(ref b) = st.ai_master_track.as_ref().or(st.main_track.as_ref())
                        {
                            let (a, e) = map_range(s0, s1, ref_len, b.len());
                            pk = peak_in_region(b, a, e);
                        }
                    }
                    PlaybackSourceMode::FourStemMixer => {
                        if let Some(ref arr) = st.stems {
                            for stem in arr {
                                let (a, e) = map_range(s0, s1, ref_len, stem.len());
                                pk = pk.max(peak_in_region(stem, a, e));
                            }
                        }
                    }
                    PlaybackSourceMode::OriginalWithLiveDsp => {
                        if let Some(ref b) = st.main_track {
                            let (a, e) = map_range(s0, s1, ref_len, b.len());
                            pk = peak_in_region(b, a, e);
                        }
                    }
                }
            }
            if pk < 1e-5_f32 {
                return Err("La zona selezionata è silenziosa: impossibile normalizzare.".to_string());
            }
            (target_lin / pk).clamp(0.0625_f32, 15.8489_f32)
        }
    };

    history.push_before_edit(st);

    if solo_only {
        if let Some(ref arr) = st.stems {
            let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
                if solo_mask[i] {
                    let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
                    apply_gain_buffer_region(&arr[i], a, e, gain_lin)
                } else {
                    arr[i].clone()
                }
            });
            st.stems = Some(new_stems);
        }
    } else {
        if let Some(ref b) = st.main_track {
            let (a, e) = map_range(s0, s1, ref_len, b.len());
            st.main_track = Some(apply_gain_buffer_region(b, a, e, gain_lin));
        }
        if let Some(ref b) = st.ai_master_track {
            let (a, e) = map_range(s0, s1, ref_len, b.len());
            st.ai_master_track = Some(apply_gain_buffer_region(b, a, e, gain_lin));
        }
        if let Some(ref arr) = st.stems {
            let new_stems: [StereoBuffer; 4] = std::array::from_fn(|i| {
                let (a, e) = map_range(s0, s1, ref_len, arr[i].len());
                apply_gain_buffer_region(&arr[i], a, e, gain_lin)
            });
            st.stems = Some(new_stems);
        }
    }

    let applied_db = 20.0_f32 * gain_lin.log10();
    match op {
        GainEditOp::BoostDb(db) => Ok(format!("Guadagno zona regolato ({db:+.1} dB)")),
        GainEditOp::NormalizePeakDb(t_db) => Ok(format!(
            "Normalizzato a {t_db:.1} dBFS ({applied_db:+.1} dB)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine_buffer(sr: u32, frames: usize, amp: f32, freq: f32) -> StereoBuffer {
        let mut l = Vec::with_capacity(frames);
        let mut r = Vec::with_capacity(frames);
        for i in 0..frames {
            let t = (i as f32) / (sr as f32);
            let s = amp * (2.0 * PI * freq * t).sin();
            l.push(s);
            r.push(s * 0.8);
        }
        StereoBuffer::from_vecs(l, r, sr)
    }

    fn make_4stem_state(sr: u32, frames: usize) -> SharedAudioState {
        let mut st = SharedAudioState::new(sr);
        st.main_track = Some(make_sine_buffer(sr, frames, 0.5, 220.0));
        st.ai_master_track = Some(make_sine_buffer(sr, frames, 0.55, 220.0));
        st.stems = Some([
            make_sine_buffer(sr, frames, 0.45, 440.0), // Voce
            make_sine_buffer(sr, frames, 0.50, 110.0), // Batteria
            make_sine_buffer(sr, frames, 0.48, 82.4),  // Basso
            make_sine_buffer(sr, frames, 0.40, 330.0), // Chitarre
        ]);
        st.mode = PlaybackSourceMode::FourStemMixer;
        st
    }

    #[test]
    fn test_cut_paste_and_history_cap() {
        let sr = 48_000;
        let frames = 48_000; // 1.0 s
        let mut st = make_4stem_state(sr, frames);
        let mut hist = AudioEditHistory::new();
        let mut clip = AudioClipboard::new();

        st.selection_range = Some((12_000, 24_000)); // 0.25 s
        cut_selection(&mut st, &mut hist, &mut clip).expect("cut failed");

        assert!(clip.has_content());
        assert_eq!(clip.clip_frames(), 12_000);
        assert_eq!(st.active_length(), 36_000);
        assert_eq!(st.main_track.as_ref().unwrap().len(), 36_000);
        assert_eq!(st.stems.as_ref().unwrap()[3].len(), 36_000);

        // Nessun click sulla giunzione a 12_000: la differenza tra campioni adiacenti resta limitata
        let vox = &st.stems.as_ref().unwrap()[0];
        for i in 11_700..12_300 {
            let step = (vox.left[i + 1] - vox.left[i]).abs();
            assert!(step < 0.15, "Discontinuità eccessiva su {i}: {step}");
        }

        // Incolla a frame 18_000 -> torna a 48_000 frame esatti
        st.position_frame = 18_000;
        paste_at_playhead(&mut st, &mut hist, &clip).expect("paste failed");
        assert_eq!(st.active_length(), 48_000);

        // Undo x2 -> ripristina stato iniziale
        hist.undo(&mut st).expect("undo paste");
        assert_eq!(st.active_length(), 36_000);
        hist.undo(&mut st).expect("undo cut");
        assert_eq!(st.active_length(), 48_000);
        assert_eq!(st.selection_range, Some((12_000, 24_000)));

        // Verifica limite massimo di 15 snapshot
        for _ in 0..22 {
            hist.push_before_edit(&st);
        }
        assert_eq!(hist.undo_count(), MAX_AUDIO_SNAPSHOTS);
    }

    #[test]
    fn test_crop_and_duplicate() {
        let sr = 44_100;
        let mut st = make_4stem_state(sr, 44_100);
        let mut hist = AudioEditHistory::new();

        st.selection_range = Some((10_000, 30_000));
        crop_to_selection(&mut st, &mut hist).expect("crop failed");
        assert_eq!(st.active_length(), 20_000);
        assert_eq!(st.position_frame, 0);
        // Bordi con micro-fade anti-click: il primo e l'ultimo campione partono da 0
        let m = st.main_track.as_ref().unwrap();
        assert!(m.left[0].abs() < 1e-5);
        assert!(m.left[m.len() - 1].abs() < 0.05);

        st.selection_range = Some((5_000, 10_000));
        duplicate_selection(&mut st, &mut hist).expect("duplicate failed");
        assert_eq!(st.active_length(), 25_000);
        assert_eq!(st.selection_range, Some((10_000, 15_000)));
    }

    #[test]
    fn test_smart_solo_silence_leaves_other_stems_intact() {
        let sr = 48_000;
        let mut st = make_4stem_state(sr, 48_000);
        let mut hist = AudioEditHistory::new();

        // Attiviamo Solo SOLO sulla Voce (canale 0)
        st.stem_channels[0].solo = true;
        st.selection_range = Some((10_000, 20_000));

        silence_selection(&mut st, &mut hist).expect("silence solo failed");

        let stems = st.stems.as_ref().unwrap();
        // Al centro della zona silenziata (frame 15_000): Voce è 0.0, Batteria/Basso/Chitarre sono intatti!
        assert!(stems[0].left[15_000].abs() < 1e-6);
        assert!(
            stems[1]
                .left[14_950..15_050]
                .iter()
                .any(|x| x.abs() > 0.2),
            "Batteria non doveva essere silenziata!"
        );
        assert!(
            stems[2]
                .left[14_950..15_050]
                .iter()
                .any(|x| x.abs() > 0.2),
            "Basso non doveva essere silenziato!"
        );
    }

    #[test]
    fn test_fades_and_normalize() {
        let sr = 44_100;
        let mut st = SharedAudioState::new(sr);
        st.main_track = Some(make_sine_buffer(sr, 44_100 * 4, 0.25, 440.0)); // 4 secondi, picco 0.25 (-12 dBFS)
        let mut hist = AudioEditHistory::new();

        // Senza selezione: Fade In sui primi 2.0s e Fade Out sugli ultimi 3.0s
        apply_fade_in(&mut st, &mut hist).expect("fade in failed");
        assert!(st.main_track.as_ref().unwrap().left[0].abs() < 1e-6);

        apply_fade_out(&mut st, &mut hist).expect("fade out failed");
        let len = st.active_length();
        assert!(st.main_track.as_ref().unwrap().left[len - 1].abs() < 1e-6);

        // Normalizzazione a -1.0 dBFS (~0.89125)
        boost_or_normalize_selection(&mut st, &mut hist, GainEditOp::NORMALIZE_MINUS_1DB)
            .expect("normalize failed");
        let pk = peak_in_region(st.main_track.as_ref().unwrap(), 0, len);
        assert!(
            (pk - 0.89125).abs() < 0.03,
            "Picco normalizzato inatteso: {pk}"
        );
    }

    #[test]
    fn test_boost_minus_plus_3db_redo_and_solo_normalize() {
        let sr = 48_000;
        let mut st = make_4stem_state(sr, 48_000);
        let mut hist = AudioEditHistory::new();

        // Riduciamo la Voce (canale 0) a picco basso (0.15) e mettiamola in Solo
        if let Some(ref mut stems) = st.stems {
            stems[0] = make_sine_buffer(sr, 48_000, 0.15, 440.0);
        }
        st.stem_channels[0].solo = true;
        st.selection_range = Some((8_000, 40_000));

        // Boost +3 dB seguito da -3 dB e verifica Undo / Redo completi
        boost_or_normalize_selection(&mut st, &mut hist, GainEditOp::PLUS_3DB)
            .expect("plus 3dB failed");
        let pk_boosted = peak_in_region(&st.stems.as_ref().unwrap()[0], 12_000, 36_000);
        assert!((pk_boosted - 0.15 * 1.4125375).abs() < 0.01);

        boost_or_normalize_selection(&mut st, &mut hist, GainEditOp::MINUS_3DB)
            .expect("minus 3dB failed");
        assert_eq!(hist.undo_count(), 2);
        assert_eq!(hist.redo_count(), 0);

        hist.undo(&mut st).expect("undo minus 3dB");
        assert_eq!(hist.redo_count(), 1);
        hist.redo(&mut st).expect("redo minus 3dB");
        assert_eq!(hist.redo_count(), 0);

        // Normalizzazione Smart Solo: porta la Voce a ~ -1.0 dBFS lasciando invariato il Basso
        let bass_before = peak_in_region(&st.stems.as_ref().unwrap()[2], 12_000, 36_000);
        boost_or_normalize_selection(&mut st, &mut hist, GainEditOp::NORMALIZE_MINUS_1DB)
            .expect("solo normalize failed");
        let vox_norm = peak_in_region(&st.stems.as_ref().unwrap()[0], 12_000, 36_000);
        let bass_after = peak_in_region(&st.stems.as_ref().unwrap()[2], 12_000, 36_000);
        assert!((vox_norm - 0.89125).abs() < 0.03);
        assert!((bass_after - bass_before).abs() < 1e-6);
    }
}
