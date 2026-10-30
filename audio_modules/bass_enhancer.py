#!/usr/bin/env python3
"""
Studio Bass Re-Synthesis, Cabinet IR Modeling & Dynamic Kick Sidechain Enhancer v2.

Risolve il problema del basso registrato in sala prove / smartphone che perde le sub-basse
o si impasta nel rimbombo generale:
  1. De-Mudding dinamico adattivo (150-295 Hz, calibrato sul livello reale dello stem) e
     allineamento rigoroso in fase Mono sulle basse.
  2. Tracciamento F0 fondamentale nota-per-nota (34-195 Hz) tramite differenza quadratica
     cumulativa normalizzata (YIN/CMND) + verifica picchi locali per ricostruire la VERA
     fondamentale Sub anche quando il microfono ha attenuato la prima armonica ("missing fundamental").
  3. Risintesi continua in fase della fondamentale Sub + armoniche valvolari + IR Cabinet 8x10.
  4. Sidechain dinamico veloce e privo di artefatti Cassa -> Sub Basso (<115 Hz).
"""

from __future__ import annotations

import math
import numpy as np
from scipy.signal import butter, fftconvolve, find_peaks, lfilter, sosfilt, sosfiltfilt


def _design_ampeg_8x10_cabinet_ir(sr: int = 44100) -> np.ndarray:
    """
    Genera una Impulse Response (IR) minima-fase/early-reflection modellata su cassa
    chiusa 8x10 da studio (punch fermo 62-110 Hz, scavo 240 Hz, presenza corde 850-1350 Hz,
    roll-off naturale sopra 4.2 kHz per eliminare qualsiasi asprezza digitale).
    """
    n = int(0.032 * sr)
    imp = np.zeros(n, dtype=np.float64)
    imp[0] = 1.0
    d1 = min(n - 1, max(1, int(0.0009 * sr)))
    d2 = min(n - 1, max(2, int(0.0023 * sr)))
    imp[d1] += 0.16
    imp[d2] -= 0.09

    sos_hp = butter(2, 35.0 / (0.5 * sr), btype="highpass", output="sos")
    sos_lp = butter(4, min(0.92, 4200.0 / (0.5 * sr)), btype="lowpass", output="sos")
    ir = sosfilt(sos_lp, sosfilt(sos_hp, imp))
    win = np.exp(-np.linspace(0.0, 6.5, n))
    ir *= win
    norm = np.sqrt(np.sum(ir ** 2)) + 1e-12
    return (ir / norm).astype(np.float64)


def _track_bass_f0_and_confidence(mono_low: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Tracciamento F0 fondamentale (34 - 185 Hz) tramite autocorrelazione FFT + YIN/CMND
    su veri massimi locali (`find_peaks`), immune all'aggancio spurio sulla 2a/3a armonica
    tipico delle registrazioni smartphone/sala prove con fondamentale attenuata.
    """
    hop = max(64, int(0.014 * sr))
    win_len = max(2048, int(0.056 * sr))  # >= 2 periodi completi fino a 36 Hz!
    n_samples = len(mono_low)
    if n_samples < win_len:
        return np.full(max(1, n_samples), 65.4, dtype=np.float64), np.zeros(max(1, n_samples), dtype=np.float64)

    lag_min = max(4, int(round(sr / 185.0)))
    lag_max = min(win_len // 2 - 2, int(round(sr / 34.0)))
    n_fft = 1 << int(np.ceil(np.log2(win_len * 2)))
    window = np.hanning(win_len)
    w_spec = np.fft.rfft(window, n=n_fft)
    win_acf = np.fft.irfft(w_spec * np.conj(w_spec), n=n_fft)[: lag_max + 5]
    win_acf = np.maximum(win_acf / (win_acf[0] + 1e-15), 0.15)

    starts = np.arange(0, n_samples - win_len + 1, hop)
    n_frames = len(starts)
    f0_frames = np.zeros(n_frames, dtype=np.float64)
    conf_frames = np.zeros(n_frames, dtype=np.float64)

    # Calibrazione adattiva della soglia energetica sul livello reale dello stem
    frame_energies = np.array(
        [float(np.dot(mono_low[st : st + win_len], mono_low[st : st + win_len])) for st in starts],
        dtype=np.float64,
    )
    active_e = frame_energies[frame_energies > 1e-14]
    energy_floor = (float(np.percentile(active_e, 20)) * 0.15) if active_e.size > 8 else 1e-12
    energy_floor = max(energy_floor, 1e-13)

    last_f0 = 65.4  # C2 / E-minor bass register anchor

    for idx, st in enumerate(starts):
        if frame_energies[idx] < energy_floor:
            f0_frames[idx] = last_f0
            conf_frames[idx] = 0.0
            continue

        seg = mono_low[st : st + win_len] * window
        r0 = float(np.dot(seg, seg))
        if r0 <= 1e-15:
            f0_frames[idx] = last_f0
            conf_frames[idx] = 0.0
            continue

        spec = np.fft.rfft(seg, n=n_fft)
        acf = np.fft.irfft(spec * np.conj(spec), n=n_fft)[: lag_max + 5]
        # Normalizzazione McLeod/NSDF priva di bias della finestra di Hanning sui lag lunghi (basse frequenze)
        acf_norm = (acf / (acf[0] + 1e-15)) / win_acf

        region = acf_norm[lag_min : lag_max + 1]
        if region.size < 6:
            f0_frames[idx] = last_f0
            conf_frames[idx] = 0.0
            continue

        # Cerca ESCLUSIVAMENTE veri massimi locali (evita la discesa dal lag 0!)
        peaks, _ = find_peaks(region, distance=max(3, lag_min // 3), prominence=0.05)
        if peaks.size == 0:
            f0_frames[idx] = last_f0
            conf_frames[idx] = 0.0
            continue

        pk_vals = region[peaks]
        max_pk_val = float(np.max(pk_vals))
        if max_pk_val < 0.36:
            f0_frames[idx] = last_f0
            conf_frames[idx] = 0.0
            continue

        viable = peaks[pk_vals >= max(0.42, 0.88 * max_pk_val)]
        rel_pk = int(viable[0]) if viable.size > 0 else int(peaks[int(np.argmax(pk_vals))])
        pk_val = float(region[rel_pk])
        abs_lag = lag_min + rel_pk

        # Disambiguazione fisica Missing-Fundamental (2a / 3a armonica -> vera fondamentale Sub):
        # Su una nota con fondamentale attenuata (es. 65.4 Hz registrata da smartphone), a T/2 o T/3
        # la 1a armonica si sottrae (-a1^2), mentre al vero periodo T TUTTE le armoniche sommano (+a1^2),
        # per cui `acf_norm[mult * abs_lag] > pk_val + 0.015` (mentre su una vera nota acuta è <= pk_val).
        for mult in (3, 2):
            mult_lag = mult * abs_lag
            if mult_lag <= lag_max - 2:
                sub_win = acf_norm[mult_lag - 3 : mult_lag + 4]
                if sub_win.size > 0 and float(np.max(sub_win)) > pk_val + 0.015:
                    abs_lag = (mult_lag - 3) + int(np.argmax(sub_win))
                    pk_val = float(acf_norm[abs_lag])
                    break

        best_lag = float(abs_lag)
        if 1 <= abs_lag < len(acf_norm) - 1:
            y0 = float(acf_norm[abs_lag - 1])
            y1 = float(acf_norm[abs_lag])
            y2 = float(acf_norm[abs_lag + 1])
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-9:
                best_lag += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))

        cand_f0 = float(np.clip(sr / max(best_lag, 1.0), 34.0, 185.0))
        last_f0 = cand_f0
        f0_frames[idx] = cand_f0
        conf_frames[idx] = float(np.clip((pk_val - 0.34) / 0.50, 0.0, 1.0))

    # Despiking mediano a 5 punti contro salti d'ottava isolati
    if n_frames >= 5:
        f0_med = f0_frames.copy()
        for i in range(2, n_frames - 2):
            f0_med[i] = float(np.median(f0_frames[i - 2 : i + 3]))
        f0_frames = f0_med

    centers = (starts + win_len // 2).astype(np.float64)
    t_idx = np.arange(n_samples, dtype=np.float64)
    f0_sample = np.interp(t_idx, centers, f0_frames)
    conf_sample = np.interp(t_idx, centers, conf_frames)
    return f0_sample, conf_sample


def _fast_kick_sidechain_envelope(drums_mono: np.ndarray, sr: int, target_len: int) -> np.ndarray:
    """
    Inviluppo sidechain veloce C-level (`scipy.signal.lfilter`) sulla banda sub-kick 42-102 Hz:
    attacco rapido (~3.5 ms) + rilascio musicale (~62 ms), zero loop Python campione-per-campione.
    """
    nyq = 0.5 * float(sr)
    sos_kick = butter(4, [42.0 / nyq, 102.0 / nyq], btype="bandpass", output="sos")
    sos_sn = butter(3, [185.0 / nyq, 360.0 / nyq], btype="bandpass", output="sos")
    kick_sub = sosfiltfilt(sos_kick, drums_mono)
    sn_body = sosfiltfilt(sos_sn, drums_mono)

    # Sopprimi il piccolo bleed della cordiera/rullante quando domina 185-360 Hz (es. intro solo rullante!)
    k_abs = np.abs(kick_sub)
    s_abs = np.abs(sn_body)
    alpha_fast = math.exp(-1.0 / max(1.0, 0.004 * sr))
    k_fast = lfilter([1.0 - alpha_fast], [1.0, -alpha_fast], k_abs)
    s_fast = lfilter([1.0 - alpha_fast], [1.0, -alpha_fast], s_abs)

    # Gate anti-rullante: ducka il basso SOLO sui veri colpi di cassa!
    is_true_kick = np.clip((k_fast - 0.08 * s_fast) / (k_fast + 1e-12), 0.0, 1.0)
    gated_k = k_fast * is_true_kick

    alpha_rel = math.exp(-1.0 / max(1.0, 0.058 * sr))
    k_rel = lfilter([1.0 - alpha_rel], [1.0, -alpha_rel], gated_k)
    env = np.maximum(gated_k, 0.88 * k_rel)

    out = np.zeros(target_len, dtype=np.float64)
    c_len = min(target_len, env.size)
    out[:c_len] = env[:c_len]
    return out


def enhance_and_resynthesize_bass(
    bass_2xN: np.ndarray,
    drums_2xN: np.ndarray | None = None,
    sr: int = 44100,
    bass_mode: str = "natural",
) -> tuple[np.ndarray, dict]:
    """
    Ricampiona/risintetizza armonicamente lo stem di basso isolato eliminando il pastume
    e sincronizzandolo in sidechain selettivo con la grancassa.
    """
    mode = (bass_mode or "natural").strip().lower()
    if mode == "off" or bass_2xN.size == 0:
        return bass_2xN.astype(np.float32), {
            "bass_mode": "off",
            "voiced_ratio": 0.0,
            "median_f0_hz": 0.0,
        }

    orig_ndim = bass_2xN.ndim
    transposed = False
    if orig_ndim == 1:
        work = np.vstack([bass_2xN, bass_2xN]).astype(np.float64)
    elif orig_ndim == 2 and bass_2xN.shape[0] != 2 and bass_2xN.shape[1] == 2:
        work = bass_2xN.T.astype(np.float64)
        transposed = True
    else:
        work = bass_2xN.astype(np.float64)

    work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    mono = 0.5 * (work[0] + work[1])
    n_samples = len(mono)
    if n_samples < 2048 or float(np.max(np.abs(mono))) < 1e-7:
        out_e = work[0] if orig_ndim == 1 else (work.T if transposed else work)
        return out_e.astype(np.float32), {
            "bass_mode": mode,
            "voiced_ratio": 0.0,
            "median_f0_hz": 0.0,
        }

    # 1. De-Mudding dinamico adattivo sulla fascia fangosa 155-295 Hz
    nyq = 0.5 * float(sr)
    sos_mud = butter(2, [155.0 / nyq, 295.0 / nyq], btype="bandpass", output="sos")
    mud_band = sosfiltfilt(sos_mud, mono)
    alpha_mud = math.exp(-1.0 / max(1.0, 0.026 * sr))
    mud_env = np.sqrt(lfilter([1.0 - alpha_mud], [1.0, -alpha_mud], mud_band ** 2) + 1e-18)

    active_mud = mud_env[mud_env > float(np.max(mud_env)) * 0.05]
    mud_thr_db = 20.0 * np.log10(float(np.percentile(active_mud, 65)) + 1e-15) if active_mud.size > 32 else -36.0
    mud_db = 20.0 * np.log10(mud_env + 1e-15)
    over_mud = np.maximum(0.0, mud_db - mud_thr_db)
    mud_gr_db = np.maximum(-over_mud * 0.65, -6.0 if mode == "reamp" else -4.5)
    mud_sub = 1.0 - (10.0 ** (mud_gr_db / 20.0))
    clean_mono = mono - mud_band * mud_sub

    # 2. Tracciamento F0 sulla banda fondamentale 32-240 Hz e Risintesi Studio DI
    sos_f0 = butter(3, [32.0 / nyq, 240.0 / nyq], btype="bandpass", output="sos")
    low_for_f0 = sosfiltfilt(sos_f0, clean_mono)
    f0_curve, conf_curve = _track_bass_f0_and_confidence(low_for_f0, sr)

    alpha_amp = math.exp(-1.0 / max(1.0, 0.012 * sr))
    amp_env = np.sqrt(lfilter([1.0 - alpha_amp], [1.0, -alpha_amp], low_for_f0 ** 2) + 1e-18) * math.sqrt(2.0)

    # Sintesi in fase continua: fondamentale Sub piena + 2a/3a/4a armonica stile valvolare Ampeg
    phase = np.cumsum(2.0 * np.pi * f0_curve / float(sr))
    synth_wave = (
        1.00 * np.sin(phase)
        + 0.44 * np.sin(2.0 * phase + 0.12)
        + 0.22 * np.sin(3.0 * phase)
        + 0.09 * np.sin(4.0 * phase)
    ) / 1.30
    synth_bass = synth_wave * amp_env * (0.20 + 0.80 * conf_curve)

    # 3. Articolazione corde / plettro (580-1450 Hz) + Convoluzione Cabinet 8x10
    sos_growl = butter(2, [580.0 / nyq, min(0.90, 1450.0 / nyq)], btype="bandpass", output="sos")
    growl_band = sosfiltfilt(sos_growl, clean_mono)
    g_pk = float(np.percentile(np.abs(growl_band), 99.0)) + 1e-9
    drive = 2.6 if mode == "reamp" else 1.8
    growl_sat = (np.tanh((growl_band / g_pk) * drive) / math.tanh(drive)) * g_pk

    synth_mix = 0.52 if mode == "reamp" else 0.36
    growl_mix = 0.38 if mode == "reamp" else 0.22
    combined = (1.0 - 0.28 * synth_mix) * clean_mono + synth_mix * synth_bass + growl_mix * growl_sat

    cab_ir = _design_ampeg_8x10_cabinet_ir(sr)
    cab_shaped = fftconvolve(combined, cab_ir, mode="full")[:n_samples]
    blended = (0.62 * combined + 0.38 * cab_shaped) if mode == "reamp" else (0.76 * combined + 0.24 * cab_shaped)

    # 4. Sidechain Dinamico Selettivo Cassa -> Sub Basso (<115 Hz)
    duck_active_pct = 0.0
    if drums_2xN is not None and drums_2xN.size > 0:
        if drums_2xN.ndim == 1:
            d_mono = drums_2xN.astype(np.float64)
        elif drums_2xN.shape[0] == 2:
            d_mono = 0.5 * (drums_2xN[0].astype(np.float64) + drums_2xN[1].astype(np.float64))
        else:
            d_mono = 0.5 * (drums_2xN[:, 0].astype(np.float64) + drums_2xN[:, 1].astype(np.float64))

        d_len = min(n_samples, d_mono.size)
        if d_len > 256:
            k_env = _fast_kick_sidechain_envelope(d_mono[:d_len], sr=sr, target_len=n_samples)
            active_k = k_env[k_env > 1e-6]
            if active_k.size > 32:
                pk_kick = float(np.percentile(active_k, 98.5)) + 1e-12
                norm_k = np.clip((k_env / pk_kick - 0.22) / 0.78, 0.0, 1.0)
                max_duck_db = -4.8 if mode == "reamp" else -3.6
                duck_gain = 10.0 ** ((max_duck_db * norm_k) / 20.0)
                duck_active_pct = float(np.mean(norm_k > 0.15) * 100.0)

                sos_sub_split = butter(3, 115.0 / nyq, btype="lowpass", output="sos")
                bass_sub_band = sosfiltfilt(sos_sub_split, blended)
                bass_upper_band = blended - bass_sub_band
                blended = bass_sub_band * duck_gain + bass_upper_band

    # Calibrazione energetica + Soft-Knee Ceiling (-0.49 dBFS, zero clipping)
    in_rms = float(np.sqrt(np.mean(mono ** 2) + 1e-15))
    out_rms = float(np.sqrt(np.mean(blended ** 2) + 1e-15))
    if out_rms > 1e-8 and in_rms > 1e-8:
        target_rms = in_rms * (1.14 if mode == "reamp" else 1.06)
        blended *= float(np.clip(target_rms / out_rms, 0.5, 1.8))

    ceiling = 0.945
    knee = 0.78 * ceiling
    abs_b = np.abs(blended)
    span = ceiling - knee
    above = np.maximum(0.0, abs_b - knee)
    limited = np.where(abs_b > knee, knee + span * np.tanh(above / span), abs_b)
    blended = np.sign(blended) * np.minimum(limited, ceiling)

    if orig_ndim == 1:
        out_arr = blended.astype(np.float32)
    elif transposed:
        out_arr = np.column_stack([blended, blended]).astype(np.float32)
    else:
        out_arr = np.stack([blended, blended], axis=0).astype(np.float32)

    voiced_mask = conf_curve > 0.25
    voiced_ratio = float(np.mean(voiced_mask))
    median_f0 = float(np.median(f0_curve[voiced_mask])) if np.any(voiced_mask) else float(np.median(f0_curve))

    return out_arr, {
        "bass_mode": mode,
        "voiced_ratio": round(voiced_ratio, 3),
        "median_f0_hz": round(median_f0, 1),
        "sidechain_active_pct": round(duck_active_pct, 2),
    }
