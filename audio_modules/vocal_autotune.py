"""
StudioEQ Master Pro — Musical Key Detection & Formant-Preserving Vocal Auto-Tune.

Self-contained NumPy + SciPy DSP implementation (with optional librosa chroma CQT
refinement when installed, never failing when absent). Integrates open-source reference
DSP techniques from:
- `jurihock/stftpitchshift` (cepstral low-quefrency liftering to decouple stationary
  vocal tract formant envelopes from pitch-shifted fine harmonic excitation + per-frame
  RMS energy conservation).
- `JanWilczek/python-auto-tune` (diatonic scale pitch quantization with hysteresis and
  natural/modern retune smoothing).
- Laroche & Dolson Identity Phase Locking + Linkwitz-Riley 4th-order zero-phase
  crossover at 4200 Hz isolating unvoiced sibilance ('s', 't', 'f') and breath/air.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

try:
    import librosa as _librosa  # type: ignore

    _HAS_LIBROSA = True
except Exception:
    _librosa = None
    _HAS_LIBROSA = False


# Krumhansl-Kessler (1982 / 1990) Major and Natural Minor key profiles (C=0 .. B=11)
_KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

_MAJOR_INTERVALS = (0, 2, 4, 5, 7, 9, 11)
_MINOR_INTERVALS = (0, 2, 3, 5, 7, 8, 10)

_KEY_LABELS_MAJOR: dict[int, str] = {
    0: "Do maggiore (C maj)",
    1: "Reb maggiore (Db maj)",
    2: "Re maggiore (D maj)",
    3: "Mib maggiore (Eb maj)",
    4: "Mi maggiore (E maj)",
    5: "Fa maggiore (F maj)",
    6: "Fa# maggiore (F# maj)",
    7: "Sol maggiore (G maj)",
    8: "Lab maggiore (Ab maj)",
    9: "La maggiore (A maj)",
    10: "Sib maggiore (Bb maj)",
    11: "Si maggiore (B maj)",
}

_KEY_LABELS_MINOR: dict[int, str] = {
    0: "Do minore (C min)",
    1: "Do# minore (C# min)",
    2: "Re minore (D min)",
    3: "Mib minore (Eb min)",
    4: "Mi minore (E min)",
    5: "Fa minore (F min)",
    6: "Fa# minore (F# min)",
    7: "Sol minore (G min)",
    8: "Sol# minore (G# min)",
    9: "La minore (A min)",
    10: "Sib minore (Bb min)",
    11: "Si minore (B min)",
}


def _ensure_stereo_2xN(audio: np.ndarray) -> np.ndarray:
    """Normalize input array to float32 stereo shape (2, N)."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return np.stack([arr, arr], axis=0)
    if arr.ndim == 2:
        if arr.shape[0] == 2:
            return arr
        if arr.shape[1] == 2 and arr.shape[0] != 2:
            return np.ascontiguousarray(arr.T)
        if arr.shape[0] == 1:
            return np.vstack([arr, arr])
        return arr[:2]
    raise ValueError(f"Expected 1D or 2D audio array, got shape {arr.shape}")


def _princarg(phase: np.ndarray) -> np.ndarray:
    """Wrap phase radians into (-pi, pi]."""
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def detect_musical_key(
    other_2xN: np.ndarray,
    bass_2xN: np.ndarray | None = None,
    sr: int = 44100,
) -> dict:
    """
    Detect musical key from harmonic stems (`0.65 * other + 0.35 * bass`) across 55–2000 Hz
    by correlating high-resolution log-chroma energy against the 24 Krumhansl-Kessler
    Major and Natural Minor key profiles.

    Returns:
        {"root_pc": int, "scale": "major"|"minor", "label": str, "allowed_pcs": list[int]}
    """
    other = _ensure_stereo_2xN(other_2xN)
    other_mono = 0.5 * (other[0].astype(np.float64) + other[1].astype(np.float64))

    if bass_2xN is not None:
        bass = _ensure_stereo_2xN(bass_2xN)
        bass_mono = 0.5 * (bass[0].astype(np.float64) + bass[1].astype(np.float64))
        min_len = min(other_mono.shape[0], bass_mono.shape[0])
        harm = 0.65 * other_mono[:min_len] + 0.35 * bass_mono[:min_len]
    else:
        harm = other_mono

    if harm.shape[0] < 2048 or float(np.max(np.abs(harm))) < 1e-7:
        return {
            "root_pc": 0,
            "scale": "major",
            "label": _KEY_LABELS_MAJOR[0],
            "allowed_pcs": list(_MAJOR_INTERVALS),
        }

    # High-resolution STFT (~2.69 Hz bin spacing at 44.1 kHz via 16384-pt zero-padded FFT)
    nperseg = min(8192, harm.shape[0])
    nfft = max(16384, 1 << int(np.ceil(np.log2(nperseg))))
    noverlap = int(nperseg * 0.75)

    freqs, _, zxx = signal.stft(
        harm,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        padded=True,
    )
    band_mask = (freqs >= 55.0) & (freqs <= 2000.0)
    freqs_sub = freqs[band_mask]
    mag_sub = np.abs(zxx[band_mask, :])

    if freqs_sub.size == 0 or mag_sub.size == 0:
        return {
            "root_pc": 0,
            "scale": "major",
            "label": _KEY_LABELS_MAJOR[0],
            "allowed_pcs": list(_MAJOR_INTERVALS),
        }

    # Suppress broadband non-tonal floor along frequency axis to isolate chord partials
    smooth_floor = ndimage.uniform_filter1d(mag_sub, size=15, axis=0, mode="nearest")
    mag_harm = np.maximum(0.0, mag_sub - 0.55 * smooth_floor)

    med_val = float(np.median(mag_harm)) + 1e-9
    mag_log = np.log1p(25.0 * (mag_harm / med_val))

    # Gate out near-silent frames so intros/outros do not dilute key correlation
    frame_energy = np.sum(mag_log, axis=0)
    if frame_energy.size > 4:
        e_thresh = 0.20 * float(np.percentile(frame_energy, 85))
        keep_frames = frame_energy >= e_thresh
        if np.any(keep_frames):
            mag_log = mag_log[:, keep_frames]

    # Gaussian pitch-class filterbank (C=0 .. B=11; MIDI 60 = C4 -> 60 % 12 == 0)
    midi_bins = 69.0 + 12.0 * np.log2(freqs_sub / 440.0)
    oct_weight = np.exp(-0.5 * ((np.log2(freqs_sub / 260.0)) / 1.85) ** 2)

    chroma_fb = np.zeros((12, freqs_sub.size), dtype=np.float64)
    for pc in range(12):
        semitone_diff = ((midi_bins - float(pc) + 6.0) % 12.0) - 6.0
        chroma_fb[pc, :] = np.exp(-0.5 * (semitone_diff / 0.28) ** 2) * oct_weight

    # Normalize filterbank per pitch class so uneven FFT bin density across octaves is balanced
    row_sums = np.sum(chroma_fb, axis=1, keepdims=True) + 1e-12
    chroma_fb /= row_sums

    chroma_stft = np.sum(chroma_fb @ (mag_log ** 2), axis=1)
    chroma_vec = chroma_stft / (np.linalg.norm(chroma_stft) + 1e-12)

    # Optional librosa CQT chroma blend if installed
    if _HAS_LIBROSA and harm.shape[0] >= 8192:
        try:
            cqt_ch = _librosa.feature.chroma_cqt(
                y=harm.astype(np.float32),
                sr=sr,
                fmin=55.0,
                n_octaves=5,
            )
            cqt_vec = np.mean(cqt_ch.astype(np.float64), axis=1)
            cqt_norm = np.linalg.norm(cqt_vec)
            if cqt_norm > 1e-9:
                chroma_vec = 0.55 * chroma_vec + 0.45 * (cqt_vec / cqt_norm)
        except Exception:
            pass

    c_centered = chroma_vec - np.mean(chroma_vec)
    c_denom = float(np.linalg.norm(c_centered)) + 1e-12

    best_corr = -2.0
    best_pc = 0
    best_scale = "major"

    maj_centered = _KK_MAJOR - np.mean(_KK_MAJOR)
    maj_norm = maj_centered / (float(np.linalg.norm(maj_centered)) + 1e-12)

    min_centered = _KK_MINOR - np.mean(_KK_MINOR)
    min_norm = min_centered / (float(np.linalg.norm(min_centered)) + 1e-12)

    for pc in range(12):
        corr_maj = float(np.dot(c_centered / c_denom, np.roll(maj_norm, pc)))
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_pc = pc
            best_scale = "major"

        corr_min = float(np.dot(c_centered / c_denom, np.roll(min_norm, pc)))
        if corr_min > best_corr:
            best_corr = corr_min
            best_pc = pc
            best_scale = "minor"

    if best_scale == "major":
        intervals = _MAJOR_INTERVALS
        label = _KEY_LABELS_MAJOR[best_pc]
    else:
        intervals = _MINOR_INTERVALS
        label = _KEY_LABELS_MINOR[best_pc]

    allowed_pcs = sorted((best_pc + iv) % 12 for iv in intervals)
    return {
        "root_pc": int(best_pc),
        "scale": best_scale,
        "label": label,
        "allowed_pcs": allowed_pcs,
    }


def _track_voiced_f0_yin(
    mono_body: np.ndarray,
    sr: int,
    hop: int,
    f_min: float = 80.0,
    f_max: float = 900.0,
    periodicity_thresh: float = 0.72,
    rms_gate_db: float = -46.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batched FFT-accelerated YIN + Normalized Cross-Correlation (NCCF) pitch tracker.

    Returns:
        f0_hz: (n_frames,) float64 (0.0 on unvoiced frames)
        voiced: (n_frames,) bool
        rms_db: (n_frames,) float64
    """
    n_samples = mono_body.shape[0]
    nyq = 0.5 * sr
    # Remove sub-70 Hz rumble before pitch tracking
    sos_hp = signal.butter(2, min(70.0 / nyq, 0.45), btype="high", output="sos")
    sig = signal.sosfiltfilt(sos_hp, mono_body)

    tau_min = max(2, int(np.floor(sr / f_max)))
    tau_max = max(tau_min + 4, int(np.ceil(sr / f_min)))
    win_w = max(1024, 2 * tau_max + 16)
    full_len = win_w + tau_max + 2
    n_fft_yin = 1 << int(np.ceil(np.log2(win_w + full_len)))

    n_frames = max(1, 1 + (n_samples - 1) // hop)
    pad_left = win_w // 2
    pad_right = full_len + hop
    sig_pad = np.pad(sig, (pad_left, pad_right), mode="constant")

    f0_hz = np.zeros(n_frames, dtype=np.float64)
    voiced = np.zeros(n_frames, dtype=bool)
    rms_db = np.full(n_frames, -120.0, dtype=np.float64)

    batch_size = 512
    lags_1_to_max = np.arange(1, tau_max + 2, dtype=np.float64)[None, :]

    for b_start in range(0, n_frames, batch_size):
        b_end = min(n_frames, b_start + batch_size)
        b_len = b_end - b_start

        # Vectorized frame gather via stride indices
        starts = np.arange(b_start, b_end) * hop
        idx_mat = starts[:, None] + np.arange(full_len)[None, :]
        x_full = sig_pad[idx_mat]  # (b_len, full_len)
        x_win = x_full[:, :win_w]  # (b_len, win_w)

        # Frame RMS in dBFS
        e0 = np.sum(x_win * x_win, axis=1)  # (b_len,)
        batch_rms_db = 20.0 * np.log10(np.sqrt(e0 / float(win_w)) + 1e-12)
        rms_db[b_start:b_end] = batch_rms_db

        active_rows = np.flatnonzero(batch_rms_db > rms_gate_db)
        if active_rows.size == 0:
            continue

        xf_act = x_full[active_rows]
        xw_act = x_win[active_rows]
        e0_act = e0[active_rows]

        # Exact linear cross-correlation via zero-padded FFT
        spec_w = np.fft.rfft(xw_act, n=n_fft_yin, axis=1)
        spec_f = np.fft.rfft(xf_act, n=n_fft_yin, axis=1)
        corr = np.fft.irfft(np.conj(spec_w) * spec_f, n=n_fft_yin, axis=1)[:, : tau_max + 2]

        # Cumulative energy E(tau) = sum_{j=0}^{W-1} x[j + tau]^2
        csum = np.concatenate(
            [np.zeros((active_rows.size, 1), dtype=np.float64), np.cumsum(xf_act * xf_act, axis=1)],
            axis=1,
        )
        e_tau = csum[:, win_w : win_w + tau_max + 2] - csum[:, 0 : tau_max + 2]

        # YIN squared difference and Cumulative Mean Normalized Difference Function
        diff = np.maximum(0.0, e0_act[:, None] + e_tau - 2.0 * corr)
        cmndf = np.ones_like(diff)
        cum_mean = np.cumsum(diff[:, 1:], axis=1) / lags_1_to_max
        cmndf[:, 1:] = diff[:, 1:] / (cum_mean + 1e-12)

        # Normalized periodicity (McLeod / NCCF harmonicity in [-1, 1])
        nccf = corr / np.sqrt(e0_act[:, None] * e_tau + 1e-15)

        yin_dip_thresh = 0.22
        for r_idx, frame_idx in enumerate(active_rows):
            c_row = cmndf[r_idx]
            d_row = diff[r_idx]
            n_row = nccf[r_idx]

            # Local minima in [tau_min, tau_max]
            seg_c = c_row[tau_min : tau_max + 1]
            prev_c = c_row[tau_min - 1 : tau_max]
            next_c = c_row[tau_min + 1 : tau_max + 2]
            local_mins = np.flatnonzero((seg_c <= prev_c) & (seg_c <= next_c))

            below = local_mins[seg_c[local_mins] < yin_dip_thresh]
            if below.size > 0:
                tau_star = int(tau_min + below[0])
            elif local_mins.size > 0:
                best_loc = local_mins[int(np.argmin(seg_c[local_mins]))]
                tau_star = int(tau_min + best_loc)
            else:
                tau_star = int(tau_min + np.argmin(seg_c))

            # Check strict periodicity / harmonicity around tau_star
            lo_p = max(1, tau_star - 1)
            hi_p = min(tau_max + 1, tau_star + 2)
            periodicity = float(np.max(n_row[lo_p:hi_p]))
            if periodicity < periodicity_thresh:
                continue

            # Parabolic interpolation on YIN difference function
            if 1 <= tau_star <= tau_max:
                y0 = float(d_row[tau_star - 1])
                y1 = float(d_row[tau_star])
                y2 = float(d_row[tau_star + 1])
                denom = y0 - 2.0 * y1 + y2
                delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
            else:
                delta = 0.0

            tau_refined = float(tau_star) + delta
            if tau_refined <= 0.0:
                continue
            freq = float(sr) / tau_refined
            if f_min <= freq <= f_max:
                out_i = b_start + int(frame_idx)
                f0_hz[out_i] = freq
                voiced[out_i] = True

    # Prune isolated 1-frame voiced spikes (< 20 ms) and median-despike voiced runs
    if n_frames >= 3:
        for i in range(1, n_frames - 1):
            if voiced[i] and not voiced[i - 1] and not voiced[i + 1]:
                voiced[i] = False
                f0_hz[i] = 0.0

        # 3-frame median filter inside contiguous voiced segments to remove boundary glitches
        idx = 0
        while idx < n_frames:
            if not voiced[idx]:
                idx += 1
                continue
            end = idx + 1
            while end < n_frames and voiced[end]:
                end += 1
            if end - idx >= 3:
                f0_hz[idx:end] = signal.medfilt(f0_hz[idx:end], kernel_size=3)
            idx = end

    return f0_hz, voiced, rms_db


def _build_retune_shift_curve(
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    hop_samples: int,
    sr: int,
    mode: str,
    allowed_pcs: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute smoothed pitch-shift curve in semitones (clamped to [-2.0, +2.0])
    across voiced frames.

    Returns:
        shift_st: (n_frames,) smoothed shift in semitones on voiced frames
        raw_shift_st: (n_frames,) unsmoothed target shift in semitones
    """
    n_frames = f0_hz.shape[0]
    allowed_set = set(int(pc) % 12 for pc in allowed_pcs) if allowed_pcs else set(range(12))
    if not allowed_set:
        allowed_set = set(range(12))

    if mode == "modern":
        strength = 0.94
        smooth_ms = 12.0
    else:
        strength = 0.72
        smooth_ms = 45.0

    raw_shift_st = np.zeros(n_frames, dtype=np.float64)
    prev_target_midi: int | None = None

    for i in range(n_frames):
        if not voiced[i] or f0_hz[i] <= 0.0:
            prev_target_midi = None
            continue

        midi_cur = 69.0 + 12.0 * float(np.log2(f0_hz[i] / 440.0))
        lo_m = int(np.floor(midi_cur)) - 2
        hi_m = int(np.ceil(midi_cur)) + 2

        best_m = int(round(midi_cur))
        best_cost = 1e9
        for m_cand in range(lo_m, hi_m + 1):
            if (m_cand % 12) not in allowed_set:
                continue
            dist = abs(float(m_cand) - midi_cur)
            # Subtle 10-cent hysteresis toward active note prevents boundary chatter
            cost = dist - (0.10 if (prev_target_midi is not None and m_cand == prev_target_midi and dist <= 0.65) else 0.0)
            if cost < best_cost:
                best_cost = cost
                best_m = m_cand

        prev_target_midi = best_m
        delta_st = float(np.clip(float(best_m) - midi_cur, -2.0, 2.0))
        raw_shift_st[i] = float(np.clip(strength * delta_st, -2.0, 2.0))

    # Smooth inside contiguous voiced runs using edge-replicated Gaussian smoothing
    # (~45 ms for natural preserving vibrato; ~12 ms for modern tight snap)
    hop_ms = 1000.0 * float(hop_samples) / float(sr)
    sigma_frames = max(0.35, (smooth_ms / max(hop_ms, 1.0)) * 0.55)

    shift_st = np.zeros_like(raw_shift_st)
    idx = 0
    while idx < n_frames:
        if not voiced[idx]:
            idx += 1
            continue
        end = idx + 1
        while end < n_frames and voiced[end]:
            end += 1
        seg = raw_shift_st[idx:end]
        if seg.size >= 2:
            smoothed = ndimage.gaussian_filter1d(seg, sigma=sigma_frames, mode="nearest")
        else:
            smoothed = seg.copy()
        shift_st[idx:end] = np.clip(smoothed, -2.0, 2.0)
        idx = end

    return shift_st, raw_shift_st


def _cepstral_formant_envelope(mag_km: np.ndarray, n_fft: int, sr: int) -> np.ndarray:
    """
    Estimate smooth vocal spectral formant envelope across all frames simultaneously
    via vectorized real-cepstrum low-quefrency liftering (< 1.5 ms).
    """
    log_mag = np.log(mag_km + 1e-7)
    cep = np.fft.irfft(log_mag, n=n_fft, axis=0)

    q_cut = max(8, int(round(sr * 0.0015)))  # ~1.5 ms quefrency cutoff (~66 samples at 44.1 kHz)
    lifter = np.zeros(n_fft, dtype=np.float64)
    lifter[0] = 1.0
    taper = 0.5 * (1.0 + np.cos(np.pi * np.arange(1, q_cut, dtype=np.float64) / float(q_cut)))
    lifter[1:q_cut] = taper
    lifter[n_fft - q_cut + 1 : n_fft] = taper[::-1]

    log_env = np.fft.rfft(cep * lifter[:, None], n=n_fft, axis=0).real
    return np.exp(log_env)


def _pitch_shift_segment_formant_locked(
    seg_2xM: np.ndarray,
    shift_samples_st: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop: int = 256,
) -> np.ndarray:
    """
    Formant-preserving, stereo phase-locked short-window STFT pitch shifter for a
    `< 4200 Hz` vocal segment.

    Uses zero-phase (`fftshift`) Hann windows, cepstral formant liftering so vowel
    formants stay fixed while fine harmonics shift, Identity Phase Locking around
    spectral peaks, and exact L/R inter-channel phase preservation.
    """
    seg_len = seg_2xM.shape[1]
    pad = n_fft
    seg_pad = np.pad(seg_2xM, ((0, 0), (pad, pad)), mode="constant")
    shift_pad = np.pad(shift_samples_st, (pad, pad), mode="edge")

    total_len = seg_pad.shape[1]
    n_frames = max(1, 1 + (total_len - n_fft) // hop)
    win = signal.windows.hann(n_fft, sym=False).astype(np.float64)

    # Frame extraction + zero-phase fftshift for coherent main-lobe phase
    starts = np.arange(n_frames) * hop
    idx_mat = starts[:, None] + np.arange(n_fft)[None, :]
    frames_l = np.fft.fftshift(seg_pad[0][idx_mat] * win[None, :], axes=1)
    frames_r = np.fft.fftshift(seg_pad[1][idx_mat] * win[None, :], axes=1)

    spec_l = np.fft.rfft(frames_l, n=n_fft, axis=1).T  # (K, n_frames)
    spec_r = np.fft.rfft(frames_r, n=n_fft, axis=1).T  # (K, n_frames)
    n_bins = spec_l.shape[0]

    # Active bins up to 6800 Hz (signal is already 24 dB/oct lowpassed at 4200 Hz)
    k_act = min(n_bins, int(np.ceil(6800.0 * n_fft / float(sr))) + 2)
    bins_act = np.arange(k_act, dtype=np.float64)
    bins_idx = np.arange(k_act, dtype=np.int64)

    mag_l = np.abs(spec_l)
    mag_r = np.abs(spec_r)
    env_l = _cepstral_formant_envelope(mag_l, n_fft=n_fft, sr=sr)
    env_r = _cepstral_formant_envelope(mag_r, n_fft=n_fft, sr=sr)

    fine_l = mag_l[:k_act, :] / np.maximum(env_l[:k_act, :], 1e-8)
    fine_r = mag_r[:k_act, :] / np.maximum(env_r[:k_act, :], 1e-8)

    # Sample shift curve at each STFT frame center
    centers = np.clip(starts + (n_fft // 2), 0, shift_pad.shape[0] - 1)
    frame_shift_st = shift_pad[centers]

    out_l = np.zeros_like(spec_l)
    out_r = np.zeros_like(spec_r)

    omega_expect = 2.0 * np.pi * bins_act * float(hop) / float(n_fft)
    phi_syn_prev = np.angle(spec_l[:k_act, 0]).copy()

    for m in range(n_frames):
        st_m = float(frame_shift_st[m])
        if m == 0:
            dphi_in = omega_expect
        else:
            # Stereo Hermitian phase advance (immune to L/R out-of-phase cancellation)
            cross = (
                spec_l[:k_act, m] * np.conj(spec_l[:k_act, m - 1])
                + spec_r[:k_act, m] * np.conj(spec_r[:k_act, m - 1])
            )
            dphi_in = omega_expect + _princarg(np.angle(cross) - omega_expect)

        if abs(st_m) < 1e-4:
            # Unshifted frame: maintain continuous phase while preserving exact magnitudes & IPD
            phi_cur = _princarg(phi_syn_prev + dphi_in)
            rot = np.exp(1j * (phi_cur - np.angle(spec_l[:k_act, m] + 1e-18)))
            out_l[:k_act, m] = spec_l[:k_act, m] * rot
            out_r[:k_act, m] = spec_r[:k_act, m] * rot
            phi_syn_prev = phi_cur
            continue

        alpha = float(2.0 ** (st_m / 12.0))
        k_src = bins_act / alpha

        # Shift fine harmonic excitation while preserving stationary formant envelopes
        # (jurihock/stftpitchshift cepstral liftering + per-frame energy conservation)
        fl_shift = np.interp(k_src, bins_act, fine_l[:, m], left=0.0, right=0.0)
        fr_shift = np.interp(k_src, bins_act, fine_r[:, m], left=0.0, right=0.0)
        ml_out = fl_shift * env_l[:k_act, m]
        mr_out = fr_shift * env_r[:k_act, m]

        e_in_l = float(np.linalg.norm(mag_l[:k_act, m]))
        e_out_l = float(np.linalg.norm(ml_out))
        if e_out_l > 1e-9 and e_in_l > 1e-9:
            ml_out *= np.clip(e_in_l / e_out_l, 0.5, 2.0)

        e_in_r = float(np.linalg.norm(mag_r[:k_act, m]))
        e_out_r = float(np.linalg.norm(mr_out))
        if e_out_r > 1e-9 and e_in_r > 1e-9:
            mr_out *= np.clip(e_in_r / e_out_r, 0.5, 2.0)

        # Preserve stereo L/R inter-channel phase difference from source bins
        xl_m = spec_l[:k_act, m]
        xr_m = spec_r[:k_act, m]
        ipd = (xr_m * np.conj(xl_m)) / (mag_r[:k_act, m] * mag_l[:k_act, m] + 1e-18)
        ipd_re = np.interp(k_src, bins_act, ipd.real, left=1.0, right=1.0)
        ipd_im = np.interp(k_src, bins_act, ipd.imag, left=0.0, right=0.0)
        ipd_norm = np.hypot(ipd_re, ipd_im)
        ipd_shift = np.where(ipd_norm > 1e-6, (ipd_re + 1j * ipd_im) / (ipd_norm + 1e-12), 1.0 + 0.0j)

        # Shifted instantaneous frequency + Laroche-Dolson Identity Phase Locking
        omega_out = alpha * np.interp(k_src, bins_act, dphi_in, left=0.0, right=0.0)
        mag_ref = np.hypot(ml_out, mr_out)

        peaks = np.flatnonzero((mag_ref[1:-1] >= mag_ref[:-2]) & (mag_ref[1:-1] > mag_ref[2:])) + 1
        if peaks.size == 0:
            phi_cur = _princarg(phi_syn_prev + omega_out)
        else:
            mids = (peaks[:-1] + peaks[1:] + 1) >> 1
            owner = peaks[np.searchsorted(mids, bins_idx, side="right")]

            phi_peaks = _princarg(phi_syn_prev[peaks] + omega_out[peaks])

            # Preserve fine main-lobe phase offset around each peak in zero-phase coordinates
            dom_spec = np.where(mag_l[:k_act, m] >= mag_r[:k_act, m], xl_m, xr_m)
            in_unit = dom_spec / (np.maximum(mag_l[:k_act, m], mag_r[:k_act, m]) + 1e-18)
            u_src = np.interp(k_src, bins_act, in_unit.real, left=1.0, right=1.0) + 1j * np.interp(
                k_src, bins_act, in_unit.imag, left=0.0, right=0.0
            )
            p_src_idx = peaks.astype(np.float64) / alpha
            up_src = np.interp(p_src_idx, bins_act, in_unit.real, left=1.0, right=1.0) + 1j * np.interp(
                p_src_idx, bins_act, in_unit.imag, left=0.0, right=0.0
            )
            owner_pos = np.searchsorted(mids, bins_idx, side="right")
            lobe_offset = np.angle(u_src * np.conj(up_src[owner_pos]) + 1e-18)
            phi_cur = _princarg(phi_peaks[owner_pos] + lobe_offset)
            # Ensure exact phase continuity at peak bins
            phi_cur[peaks] = phi_peaks

        phasor_l = np.exp(1j * phi_cur)
        out_l[:k_act, m] = ml_out * phasor_l
        out_r[:k_act, m] = mr_out * phasor_l * ipd_shift
        phi_syn_prev = phi_cur

    # Inverse zero-phase STFT + COLA overlap-add
    frames_out_l = np.fft.ifftshift(np.fft.irfft(out_l.T, n=n_fft, axis=1), axes=1) * win[None, :]
    frames_out_r = np.fft.ifftshift(np.fft.irfft(out_r.T, n=n_fft, axis=1), axes=1) * win[None, :]

    y_pad = np.zeros((2, total_len), dtype=np.float64)
    w_sum = np.zeros(total_len, dtype=np.float64)
    win_sq = win * win

    for m in range(n_frames):
        pos = m * hop
        y_pad[0, pos : pos + n_fft] += frames_out_l[m]
        y_pad[1, pos : pos + n_fft] += frames_out_r[m]
        w_sum[pos : pos + n_fft] += win_sq

    valid = w_sum > 1e-8
    y_pad[:, valid] /= w_sum[None, valid]
    return y_pad[:, pad : pad + seg_len]


def apply_vocal_autotune(
    vocals_2xN: np.ndarray,
    sr: int = 44100,
    mode: str = "natural",
    allowed_pcs: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Apply formant-preserving, sibilance-protected stereo vocal pitch correction.

    Args:
        vocals_2xN: Stereo vocal stem shape (2, N).
        sr: Sample rate in Hz (default 44100).
        mode: "off" | "natural" | "modern".
        allowed_pcs: List of allowed pitch classes (0=C .. 11=B); chromatic if None.

    Returns:
        (tuned_vocals_2xN_float32, stats_dict)
    """
    mode_norm = (mode or "off").strip().lower()
    vocals = _ensure_stereo_2xN(vocals_2xN)

    if mode_norm == "off":
        return vocals, {"autotune_mode": "off", "corrected_frames_pct": 0.0}

    if mode_norm not in ("natural", "modern"):
        mode_norm = "natural"

    n_samples = vocals.shape[1]
    if n_samples < 4096 or float(np.max(np.abs(vocals))) < 1e-6:
        return vocals, {
            "autotune_mode": mode_norm,
            "corrected_frames_pct": 0.0,
            "mean_cents_correction": 0.0,
        }

    # 1. Sibilance & Breath Protection Crossover at 4200 Hz (Zero-phase LR8 48 dB/oct complementary split)
    nyq = 0.5 * float(sr)
    crossover_hz = min(4200.0, 0.88 * nyq)
    pre_vocoder_hz = min(6200.0, 0.92 * nyq)
    sos_lp = signal.butter(4, crossover_hz / nyq, btype="low", output="sos")
    sos_pre_lp = signal.butter(6, pre_vocoder_hz / nyq, btype="low", output="sos")

    vocals_f64 = vocals.astype(np.float64)
    body_2xN = signal.sosfiltfilt(sos_lp, vocals_f64, axis=-1)
    vocoder_src_2xN = signal.sosfiltfilt(sos_pre_lp, vocals_f64, axis=-1)

    # 2. Voiced F0 Pitch Tracking (80–900 Hz, 10 ms hop, periodicity >= 0.72, RMS > -46 dBFS)
    hop_f0 = max(1, int(round(0.010 * sr)))
    mono_body = 0.5 * (body_2xN[0] + body_2xN[1])
    f0_hz, voiced, rms_db = _track_voiced_f0_yin(
        mono_body=mono_body,
        sr=sr,
        hop=hop_f0,
        f_min=80.0,
        f_max=900.0,
        periodicity_thresh=0.72,
        rms_gate_db=-46.0,
    )

    if not np.any(voiced):
        return vocals, {
            "autotune_mode": mode_norm,
            "corrected_frames_pct": 0.0,
            "mean_cents_correction": 0.0,
        }

    # 3. Scale Target & Smooth Retune Curve
    pcs = allowed_pcs if allowed_pcs is not None else list(range(12))
    shift_st, _ = _build_retune_shift_curve(
        f0_hz=f0_hz,
        voiced=voiced,
        hop_samples=hop_f0,
        sr=sr,
        mode=mode_norm,
        allowed_pcs=pcs,
    )

    applied_cents = shift_st * 100.0
    active_vocal_frames = int(np.sum(rms_db > -46.0))
    corrected_frames = int(np.sum(voiced & (np.abs(applied_cents) >= 4.0)))
    pct = (100.0 * float(corrected_frames) / float(active_vocal_frames)) if active_vocal_frames > 0 else 0.0
    mean_cents = float(np.mean(np.abs(applied_cents[voiced]))) if np.any(voiced) else 0.0

    if corrected_frames == 0:
        return vocals, {
            "autotune_mode": mode_norm,
            "corrected_frames_pct": 0.0,
            "mean_cents_correction": round(mean_cents, 1),
        }

    # Interpolate sample-rate pitch shift curve & smooth wet crossfade mask
    frame_centers = np.arange(f0_hz.shape[0], dtype=np.float64) * float(hop_f0)
    sample_idx = np.arange(n_samples, dtype=np.float64)

    # Hold shift steady briefly across voiced edges so release tails don't droop before crossfade finishes
    shift_held = shift_st.copy()
    n_frames = shift_st.shape[0]
    hold_frames = 3
    for i in range(1, n_frames):
        if not voiced[i] and voiced[i - 1]:
            for h in range(hold_frames):
                if i + h < n_frames and not voiced[i + h]:
                    shift_held[i + h] = shift_st[i - 1] * (1.0 - float(h + 1) / float(hold_frames + 1))
    for i in range(n_frames - 2, -1, -1):
        if not voiced[i] and voiced[i + 1]:
            for h in range(hold_frames):
                if i - h >= 0 and not voiced[i - h]:
                    shift_held[i - h] = shift_st[i + 1] * (1.0 - float(h + 1) / float(hold_frames + 1))

    shift_samples = np.interp(sample_idx, frame_centers, shift_held, left=0.0, right=0.0)
    wet_frames = ndimage.gaussian_filter1d(voiced.astype(np.float64), sigma=1.2, mode="constant", cval=0.0)
    wet_frames = np.clip(wet_frames * 1.15, 0.0, 1.0)
    wet_samples = np.interp(sample_idx, frame_centers, wet_frames, left=0.0, right=0.0)

    # 4. Process merged voiced islands with Formant-Preserving Stereo Phase-Locked STFT Shifter
    wet_delta = np.zeros_like(vocals_f64)
    collar = int(round(0.045 * sr))
    merge_gap = int(round(0.080 * sr))

    voiced_samples_idx = np.flatnonzero(wet_samples > 0.01)
    if voiced_samples_idx.size > 0:
        # Group into contiguous islands separated by >= merge_gap
        gaps = np.flatnonzero(np.diff(voiced_samples_idx) > merge_gap)
        island_starts = np.r_[voiced_samples_idx[0], voiced_samples_idx[gaps + 1]]
        island_ends = np.r_[voiced_samples_idx[gaps], voiced_samples_idx[-1]]

        for s_raw, e_raw in zip(island_starts, island_ends):
            s0 = max(0, int(s_raw) - collar)
            e0 = min(n_samples, int(e_raw) + collar)
            if e0 - s0 < 512:
                continue

            seg_shift = shift_samples[s0:e0].copy()
            # Ensure zero-shift phase anchor at outer island collar boundaries
            anchor = min(256, seg_shift.size // 4)
            if anchor > 1:
                ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(anchor, dtype=np.float64) / float(anchor)))
                seg_shift[:anchor] *= ramp
                seg_shift[-anchor:] *= ramp[::-1]

            seg_wet_wide = _pitch_shift_segment_formant_locked(
                seg_2xM=vocoder_src_2xN[:, s0:e0],
                shift_samples_st=seg_shift,
                sr=sr,
                n_fft=2048,
                hop=256,
            )
            w = wet_samples[s0:e0][None, :]
            wet_delta[:, s0:e0] = w * (seg_wet_wide - vocoder_src_2xN[:, s0:e0])

    # Apply zero-phase LR8 4200 Hz lowpass AFTER windowing wet_delta so > 4.2 kHz sibilance & air
    # pass through with exact 48 dB/oct complementary protection and zero modulation skirts
    lowpassed_delta = signal.sosfiltfilt(sos_lp, wet_delta, axis=-1)
    tuned = np.nan_to_num(vocals_f64 + lowpassed_delta, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return tuned, {
        "autotune_mode": mode_norm,
        "corrected_frames_pct": round(pct, 1),
        "mean_cents_correction": round(mean_cents, 1),
    }
