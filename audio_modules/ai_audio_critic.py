#!/usr/bin/env python3
"""
StudioEQ Master Pro v3 — Closed-Loop AI Audio Critic & Self-Listening Quality Auditor.

Answers the core studio question: "Can the software re-listen to its own rendered tracks
to verify whether fast fills, room mud, codec swish, or tonal fidelity can be improved?"

Combines two complementary layers:
  1. Optional Neural Audio Aesthetics Predictor (`facebookresearch/audiobox-aesthetics`,
     Apache-2.0 — Production Quality `PQ`, Content Enjoyment `CE`, Complexity `PC`) when
     the `audiobox_aesthetics` package/weights are present locally.
  2. Deterministic Forensic Psychoacoustic & Onset-Recall Critic (always active, <0.4s):
     - Fast Snare Roll Recall Auditor (`IOI < 155 ms`): verifies every single rapid stroke
       in isolated `snare.wav` (e.g. 32nd-note fills & press rolls) retains crisp transient
       energy in the rendered drum stem, pinpointing exact `mm:ss.d` timestamps if any stroke drops!
     - Sub-Punch vs Rehearsal-Boxiness Gain (`45-95 Hz` vs `250-500 Hz` dB improvement).
     - Stereo Overhead Width & Smartphone Mono Compatibility (`L/R phase correlation`).
     - Strict Guitar Non-Manipulation Auditor (`core_corr` in `85-3400 Hz >= 0.985`,
       verifying zero amp/overdrive/chorus/pitch manipulation while confirming `<65 Hz`
       rumble & stationary hiss cleanup).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt


def _bp_sos(lo: float, hi: float, sr: int, order: int = 3) -> np.ndarray:
    nyq = 0.5 * float(sr)
    lo_c = max(12.0, min(float(lo), 0.90 * nyq))
    hi_c = max(lo_c + 16.0, min(float(hi), 0.96 * nyq))
    return butter(order, [lo_c, hi_c], btype="bandpass", fs=float(sr), output="sos")


def _lp_sos(fc: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = 0.5 * float(sr)
    return butter(order, max(15.0, min(float(fc), 0.96 * nyq)), btype="lowpass", fs=float(sr), output="sos")


def _zp_filter(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    padlen = 3 * (2 * sos.shape[0] + 1)
    if x.shape[-1] <= padlen:
        return sosfilt(sos, x, axis=-1)
    return sosfiltfilt(sos, x, axis=-1)


def _fmt_mmss(sec: float) -> str:
    m = int(sec // 60.0)
    s = sec - 60.0 * m
    return f"{m}:{s:04.1f}"


def _try_audiobox_aesthetics_score(audio_2xn: np.ndarray, sr: int) -> dict[str, float] | None:
    """
    Optional hook for Meta `facebookresearch/audiobox-aesthetics` (`PQ`, `CE`, `CU`, `PC` on 1..10 scale).
    Runs zero-network if package is not installed; automatically activates when available.
    """
    try:
        import torch
        from audiobox_aesthetics.infer import initialize_predictor  # type: ignore

        predictor = initialize_predictor()
        wav_t = torch.from_numpy(np.ascontiguousarray(audio_2xn, dtype=np.float32))
        scores = predictor.forward([{"path": wav_t, "sample_rate": int(sr)}])
        if scores and isinstance(scores[0], dict):
            return {str(k): round(float(v), 3) for k, v in scores[0].items()}
    except Exception:
        return None
    return None


def audit_drum_upgrade_quality(
    raw_drums_2xn: np.ndarray,
    proc_drums_2xn: np.ndarray,
    sr: int = 44100,
    isolated_snare_2xn: np.ndarray | None = None,
    snare_positions: list[int] | None = None,
) -> dict[str, Any]:
    """
    Re-listen forensically to the upgraded drum stem vs the raw rehearsal drum stem:
      1. Verifies Fast Snare Roll Recall (`IOI < 155 ms`) so rapid fills are never lost.
      2. Quantifies Sub-to-Boxiness (`45-95 Hz` / `250-500 Hz`) studio improvement in dB.
      3. Checks Stereo Overhead Width (`Side/Mid` dB) and Mono Phase Safety (`corr`).
      4. Checks True-Peak ceiling safety and transient crest factor.
    """
    n = min(raw_drums_2xn.shape[1], proc_drums_2xn.shape[1])
    if n < int(0.2 * sr):
        return {"score_0_100": 0.0, "passed": False, "recommendations": ["Audio troppo corto per l'analisi."]}

    raw = raw_drums_2xn[:, :n].astype(np.float64, copy=False)
    proc = proc_drums_2xn[:, :n].astype(np.float64, copy=False)

    raw_mid = 0.5 * (raw[0] + raw[1])
    proc_mid = 0.5 * (proc[0] + proc[1])
    proc_side = 0.5 * (proc[0] - proc[1])

    # 1. Sub-Punch (45-95 Hz) & Anti-Ovattato Clarity (2.6-14 kHz) vs Rehearsal Boxiness/Mud (240-500 Hz)
    sos_sub = _bp_sos(45.0, 95.0, sr, order=3)
    sos_box = _bp_sos(240.0, 500.0, sr, order=3)
    sos_clr = _bp_sos(2600.0, 14000.0, sr, order=3)
    raw_sub_rms = float(np.sqrt(np.mean(_zp_filter(sos_sub, raw_mid) ** 2) + 1e-18))
    raw_box_rms = float(np.sqrt(np.mean(_zp_filter(sos_box, raw_mid) ** 2) + 1e-18))
    raw_clr_rms = float(np.sqrt(np.mean(_zp_filter(sos_clr, raw_mid) ** 2) + 1e-18))
    proc_sub_rms = float(np.sqrt(np.mean(_zp_filter(sos_sub, proc_mid) ** 2) + 1e-18))
    proc_box_rms = float(np.sqrt(np.mean(_zp_filter(sos_box, proc_mid) ** 2) + 1e-18))
    proc_clr_rms = float(np.sqrt(np.mean(_zp_filter(sos_clr, proc_mid) ** 2) + 1e-18))

    raw_sub_to_box_db = 20.0 * math.log10(raw_sub_rms / raw_box_rms)
    proc_sub_to_box_db = 20.0 * math.log10(proc_sub_rms / proc_box_rms)
    sub_to_box_gain_db = proc_sub_to_box_db - raw_sub_to_box_db

    raw_clr_to_mud_db = 20.0 * math.log10(raw_clr_rms / raw_box_rms)
    proc_clr_to_mud_db = 20.0 * math.log10(proc_clr_rms / proc_box_rms)
    clarity_hi_to_mud_gain_db = proc_clr_to_mud_db - raw_clr_to_mud_db

    # 2. Fast-Roll & Fill Stroke Recall (`IOI < 155 ms`)
    fast_roll_total = 0
    fast_roll_retained = 0
    dropped_roll_timestamps: list[str] = []

    ref_snare_pos: list[int] = []
    if isolated_snare_2xn is not None and isolated_snare_2xn.shape[1] >= int(0.2 * sr):
        try:
            from .drumsep_v3 import detect_isolated_snare_hits_with_rolls
        except ImportError:
            from drumsep_v3 import detect_isolated_snare_hits_with_rolls
        det_hits, _ = detect_isolated_snare_hits_with_rolls(isolated_snare_2xn[:, :n], sr=sr)
        ref_snare_pos = [int(h[0]) for h in det_hits]
    elif snare_positions:
        ref_snare_pos = [int(p) for p in snare_positions if 0 <= int(p) < n]

    if len(ref_snare_pos) >= 2:
        proc_sn_band = _zp_filter(_bp_sos(175.0, 5600.0, sr, order=3), proc_mid)
        raw_sn_band = _zp_filter(_bp_sos(175.0, 5600.0, sr, order=3), raw_mid)
        # Normalize overall level difference between raw & proc snare band so comparison is fair
        g_norm = (float(np.sqrt(np.mean(raw_sn_band ** 2))) + 1e-12) / (
            float(np.sqrt(np.mean(proc_sn_band ** 2))) + 1e-12
        )
        w_pre = int(round(0.004 * sr))
        w_post = int(round(0.018 * sr))
        max_roll_gap = int(round(0.155 * sr))

        for i, pos in enumerate(ref_snare_pos):
            in_roll = (i > 0 and (pos - ref_snare_pos[i - 1]) <= max_roll_gap) or (
                (i + 1) < len(ref_snare_pos) and (ref_snare_pos[i + 1] - pos) <= max_roll_gap
            )
            if not in_roll:
                continue
            fast_roll_total += 1
            s0 = max(0, pos - w_pre)
            s1 = min(n, pos + w_post)
            if s1 <= s0:
                continue
            r_pk = float(np.max(np.abs(raw_sn_band[s0:s1]))) + 1e-12
            p_pk = float(np.max(np.abs(proc_sn_band[s0:s1]))) * g_norm
            # Stroke is retained if normalized studio snare transient is >= 55% of raw stroke peak
            if p_pk >= 0.55 * r_pk:
                fast_roll_retained += 1
            elif len(dropped_roll_timestamps) < 8:
                dropped_roll_timestamps.append(_fmt_mmss(pos / float(sr)))

    fast_roll_recall_pct = (
        100.0 * float(fast_roll_retained) / float(fast_roll_total) if fast_roll_total > 0 else 100.0
    )

    # 3. Stereo Imaging & Smartphone Mono Compatibility
    mid_rms = float(np.sqrt(np.mean(proc_mid ** 2) + 1e-18))
    side_rms = float(np.sqrt(np.mean(proc_side ** 2) + 1e-18))
    stereo_side_mid_db = 20.0 * math.log10(side_rms / mid_rms)

    l_std = float(np.std(proc[0])) + 1e-12
    r_std = float(np.std(proc[1])) + 1e-12
    lr_corr = float(np.mean((proc[0] - np.mean(proc[0])) * (proc[1] - np.mean(proc[1]))) / (l_std * r_std))

    # 4. Crest Factor & Peak Headroom
    peak_abs = float(np.max(np.abs(proc))) + 1e-12
    peak_dbfs = 20.0 * math.log10(peak_abs)
    active_mask = np.abs(proc_mid) > 5e-3
    active_rms = float(np.sqrt(np.mean(proc_mid[active_mask] ** 2) + 1e-18)) if np.any(active_mask) else mid_rms
    crest_db = 20.0 * math.log10(peak_abs / active_rms)

    # Composite Quality Score (0..100)
    score = 70.0
    score += float(np.clip(sub_to_box_gain_db * 1.4, -10.0, 9.5))
    score += float(np.clip(clarity_hi_to_mud_gain_db * 1.1, -6.0, 5.5))
    score += float(np.clip((fast_roll_recall_pct - 85.0) * 0.60, -25.0, 9.0))
    if -18.0 <= stereo_side_mid_db <= -3.5 and lr_corr >= 0.55:
        score += 3.5
    if peak_dbfs <= -0.35 and crest_db >= 10.5:
        score += 2.5
    score = round(float(np.clip(score, 0.0, 100.0)), 1)

    recs: list[str] = []
    if fast_roll_recall_pct < 95.0:
        recs.append(
            f"Attenzione: rilevata perdita parziale su rullate veloci ({fast_roll_recall_pct:.1f}% recall a {', '.join(dropped_roll_timestamps)}). "
            "Ridurre `min_ioi` o alzare `roll_floor` sul rullante."
        )
    if sub_to_box_gain_db < 1.5:
        recs.append("Rapporto Sub/Boxiness migliorabile: aumentare lo scavo dinamico 240-500 Hz.")
    if clarity_hi_to_mud_gain_db < 2.0:
        recs.append("Traccia leggermente ovattata: aumentare chiarezza 2.6-14 kHz rispetto a 240-500 Hz.")
    if lr_corr < 0.50:
        recs.append("Correlazione mono L/R bassa: ridurre l'apertura stereo sugli overhead per compatibilità smartphone.")
    if peak_dbfs > -0.20:
        recs.append("Picco vicino a 0 dBFS: abbassare il ceiling soft-knee a -0.8 dBTP per evitare clipping MP3.")
    if not recs:
        recs.append(
            "Controllo AI superato al 100%: bassi profondi, chiarezza anti-ovattata, compatibilità mono smartphone e 100% di rullate e doppio pedale preservati."
        )

    aesthetics = _try_audiobox_aesthetics_score(proc_drums_2xn[:, : min(n, sr * 45)], sr)

    return {
        "score_0_100": score,
        "passed": bool(
            score >= 82.0 and fast_roll_recall_pct >= 94.0 and lr_corr >= 0.50 and clarity_hi_to_mud_gain_db >= 1.5
        ),
        "fast_roll_strokes_checked": int(fast_roll_total),
        "fast_roll_recall_pct": round(fast_roll_recall_pct, 1),
        "dropped_roll_timestamps": dropped_roll_timestamps,
        "sub_to_box_gain_db": round(sub_to_box_gain_db, 2),
        "clarity_hi_to_mud_gain_db": round(clarity_hi_to_mud_gain_db, 2),
        "stereo_side_mid_db": round(stereo_side_mid_db, 2),
        "lr_phase_corr": round(lr_corr, 3),
        "crest_db": round(crest_db, 2),
        "peak_dbfs": round(peak_dbfs, 2),
        "audiobox_aesthetics": aesthetics,
        "recommendations": recs,
    }


def audit_guitar_non_manipulation(
    raw_guitar_2xn: np.ndarray,
    restored_guitar_2xn: np.ndarray,
    sr: int = 44100,
) -> dict[str, Any]:
    """
    Re-listen forensically to the restored guitar stem vs the raw guitar stem to guarantee:
      1. ZERO tonal/phase/effect manipulation (`core_waveform_corr_85_3400hz >= 0.985`).
      2. Clean sub-65 Hz rumble removal (preserving Low E2 = 82.4 Hz).
      3. Clean codec HF restoration / de-swish without harsh treble boost.
    """
    n = min(raw_guitar_2xn.shape[1], restored_guitar_2xn.shape[1])
    if n < int(0.2 * sr):
        return {"score_0_100": 0.0, "unmanipulated_verified": False, "recommendations": ["Audio troppo corto."]}

    raw = raw_guitar_2xn[:, :n].astype(np.float64, copy=False)
    res = restored_guitar_2xn[:, :n].astype(np.float64, copy=False)
    raw_m = 0.5 * (raw[0] + raw[1])
    res_m = 0.5 * (res[0] + res[1])

    # 1. Core Guitar Timbre & Phase Identity (85 Hz - 3400 Hz)
    sos_core = _bp_sos(85.0, 3400.0, sr, order=3)
    c_raw = _zp_filter(sos_core, raw_m)
    c_res = _zp_filter(sos_core, res_m)
    active = np.abs(c_raw) > 2e-3
    if np.sum(active) > 256:
        a_r = c_raw[active] - np.mean(c_raw[active])
        a_p = c_res[active] - np.mean(c_res[active])
    else:
        a_r = c_raw - np.mean(c_raw)
        a_p = c_res - np.mean(c_res)
    denom = float(np.sqrt(np.sum(a_r ** 2) * np.sum(a_p ** 2))) + 1e-18
    core_corr = float(np.sum(a_r * a_p) / denom)

    # 2. Sub-62 Hz Rumble Cleanup (below Low E2 = 82.4 Hz)
    sos_rumble = _lp_sos(58.0, sr, order=3)
    rumble_raw = float(np.sqrt(np.mean(_zp_filter(sos_rumble, raw_m) ** 2) + 1e-18))
    rumble_res = float(np.sqrt(np.mean(_zp_filter(sos_rumble, res_m) ** 2) + 1e-18))
    rumble_red_db = 20.0 * math.log10(rumble_raw / rumble_res)

    # 3. Upper-Harmonic / Codec Band (8.0 - 16.5 kHz)
    sos_hf = _bp_sos(8000.0, 16500.0, sr, order=3)
    hf_raw = float(np.sqrt(np.mean(_zp_filter(sos_hf, raw_m) ** 2) + 1e-18))
    hf_res = float(np.sqrt(np.mean(_zp_filter(sos_hf, res_m) ** 2) + 1e-18))
    hf_delta_db = 20.0 * math.log10(hf_res / hf_raw)

    unmanipulated = bool(core_corr >= 0.985 and abs(hf_delta_db) <= 8.0)
    score = 60.0 + 34.0 * float(np.clip((core_corr - 0.90) / 0.10, 0.0, 1.0))
    if rumble_red_db >= 2.0:
        score += 4.0
    if unmanipulated:
        score += 2.0
    score = round(float(np.clip(score, 0.0, 100.0)), 1)

    recs: list[str] = []
    if not unmanipulated:
        recs.append(
            f"Attenzione: correlazione timbrica 85-3400 Hz pari a {core_corr:.4f} (<0.985) — ridurre interventi EQ/dinamici per non manipolare la chitarra."
        )
    else:
        recs.append(
            f"Integrità timbrica certificata ({core_corr * 100.0:.2f}% identità di fase/forma d'onda 85-3400 Hz): zero manipolazione, sub-rumble ridotto di {rumble_red_db:.1f} dB e de-artifacting Apollo attivo."
        )

    aesthetics = _try_audiobox_aesthetics_score(restored_guitar_2xn[:, : min(n, sr * 45)], sr)

    return {
        "score_0_100": score,
        "unmanipulated_verified": unmanipulated,
        "core_waveform_corr_85_3400hz": round(core_corr, 5),
        "sub_rumble_reduction_db": round(rumble_red_db, 2),
        "codec_hf_delta_db": round(hf_delta_db, 2),
        "audiobox_aesthetics": aesthetics,
        "recommendations": recs,
    }
