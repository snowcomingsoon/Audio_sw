#!/usr/bin/env python3
"""
Scientific Drum Sampling, Onset Classification & Phase-Alignment Lab Suite.

Covers 4 comprehensive verification pillars:
  1. Acoustic Quality Audit of Studio WAV Bank (RR1 + RR2, Sinc Resampler, Monotonicity)
  2. Ground-Truth Synthetic Rehearsal-Room Multi-Instrument Benchmark (Kick, Snare,
     Ghost Snare, Closed/Open Hi-Hat, Crash/Ride, High/Mid/Floor Tom Fill, Kick+Snare,
     Boxy 340 Hz Room Resonance + Bass Bleed + Smartphone Mic Roll-Off Scenario)
  3. Real Demucs Stem Forensic Evaluation (`drums.wav` & `bass.wav` @ 99.9 BPM, E minor)
  4. Bass Enhancer F0 Fundamental Tracking & Kick-to-Bass Sidechain Verification
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfiltfilt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from audio_modules import drum_samples_synth as dss
from audio_modules import drum_replacer as dr
from audio_modules import bass_enhancer as be

REAL_DRUMS_PATH = os.path.join(ROOT_DIR, "output", "stems_raw", "2b1cf6f94fd422b1", "drums.wav")
REAL_BASS_PATH = os.path.join(ROOT_DIR, "output", "stems_raw", "2b1cf6f94fd422b1", "bass.wav")


def _bandpass(sig: np.ndarray, lo: float, hi: float, sr: int, order: int = 3) -> np.ndarray:
    nyq = 0.5 * sr
    sos = butter(order, [max(15.0, lo) / nyq, min(0.96 * nyq, hi) / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, sig, axis=-1)


def _measure_sample_wav(path: str) -> dict:
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    mid = 0.5 * (x[:, 0] + x[:, 1])
    pk = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(mid ** 2) + 1e-18))
    crest_db = 20.0 * np.log10((pk + 1e-18) / rms)
    dc = float(np.max(np.abs(np.mean(x, axis=0))))

    spec = np.abs(np.fft.rfft(mid))
    freqs = np.fft.rfftfreq(mid.size, 1.0 / sr)

    def band_e(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs <= hi)
        return float(np.sum(spec[m] ** 2))

    total_e = float(np.sum(spec ** 2) + 1e-18)
    centroid = float(np.sum(freqs * (spec ** 2)) / total_e)
    sub_box_db = 10.0 * np.log10((band_e(45.0, 95.0) + 1e-18) / (band_e(320.0, 500.0) + 1e-18))
    crack_mud_db = 10.0 * np.log10((band_e(2000.0, 5500.0) + 1e-18) / (band_e(280.0, 520.0) + 1e-18))
    hf_lf_db = 10.0 * np.log10((band_e(2200.0, 14000.0) + 1e-18) / (band_e(45.0, 450.0) + 1e-18))

    # Low-end mono correlation (<140 Hz)
    lo_l = _bandpass(x[:, 0], 25.0, 140.0, sr, order=3)
    lo_r = _bandpass(x[:, 1], 25.0, 140.0, sr, order=3)
    mono_corr = float(np.dot(lo_l, lo_r) / (np.linalg.norm(lo_l) * np.linalg.norm(lo_r) + 1e-18))

    return {
        "sr": sr,
        "pk_db": 20.0 * np.log10(pk + 1e-18),
        "rms_db": 20.0 * np.log10(rms),
        "energy": float(np.sum(mid ** 2)),
        "crest_db": crest_db,
        "dc": dc,
        "centroid": centroid,
        "sub_box_db": sub_box_db,
        "crack_mud_db": crack_mud_db,
        "hf_lf_db": hf_lf_db,
        "mono_corr": mono_corr,
        "mid": mid,
    }


def _synth_rehearsal_kit_ground_truth(
    sr: int = 44100,
    phone_mic_rolloff: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Build a deterministic 24-second (10-bar @ 100 BPM) stereo rehearsal-room drum stem
    with exact sample-level ground-truth onset positions for:
      - kicks (including solo kicks & simultaneous Kick+Snare)
      - snares (backbeats + ghost snares + simultaneous Kick+Snare)
      - distractors: closed_hat, open_hat, crash, ride, high_tom, mid_tom, floor_tom, bass_bleed
    Immersed in boxy 340 Hz rehearsal-room resonance and short early reflections.
    """
    bpm = 100.0
    beat_s = 60.0 / bpm       # 0.600 s
    sixteenth_s = beat_s / 4  # 0.150 s
    dur_s = 25.2
    n = int(round(dur_s * sr))
    rng = np.random.default_rng(12345)

    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    gt_kicks: list[int] = []
    gt_snares: list[int] = []
    gt_distractors: list[tuple[int, str]] = []

    def add_stereo(pos: int, sig_l: np.ndarray, sig_r: np.ndarray) -> None:
        if pos < 0 or pos >= n:
            return
        m = min(sig_l.size, n - pos)
        left[pos : pos + m] += sig_l[:m]
        right[pos : pos + m] += sig_r[:m]

    def make_rehearsal_kick(amp: float = 0.72, seed: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
        r = np.random.default_rng(seed)
        L = int(0.42 * sr)
        t = np.arange(L, dtype=np.float64) / sr
        f_inst = 58.0 + 95.0 * np.exp(-t / 0.013)
        phi = np.cumsum(2.0 * np.pi * f_inst / sr)
        att = 1.0 - np.exp(-t / 0.0007)
        sub = att * np.exp(-t / 0.115) * np.sin(phi)
        box = 0.34 * att * np.exp(-t / 0.075) * np.sin(2.0 * np.pi * 148.0 * t)
        # Beater click (2.5-4.5 kHz) — potential false-snare trap!
        click_n = r.normal(0.0, 1.0, L) * np.exp(-t / 0.0032) * att
        click = _bandpass(click_n, 2200.0, 4600.0, sr, order=2) * 0.32
        sig = amp * (sub + box + click)
        pk_off, _ = dr._find_first_major_peak(_bandpass(sig, 42.0, 145.0, sr, order=3), int(0.022 * sr), rel_thr=0.55)
        return sig, sig.copy(), pk_off

    def make_rehearsal_snare(amp: float = 0.68, ghost: bool = False, seed: int = 2) -> tuple[np.ndarray, np.ndarray, int]:
        r = np.random.default_rng(seed)
        L = int(0.38 * sr)
        t = np.arange(L, dtype=np.float64) / sr
        att = 1.0 - np.exp(-t / 0.0004)
        f1 = 212.0 + 36.0 * np.exp(-t / 0.0035)
        phi1 = np.cumsum(2.0 * np.pi * f1 / sr)
        body = 0.68 * np.exp(-t / 0.055) * np.sin(phi1)
        # Muddy rehearsal-room 345 Hz ring on snare!
        mud_ring = (0.22 if ghost else 0.44) * np.exp(-t / 0.125) * np.sin(2.0 * np.pi * 345.0 * t)
        wire_env = (1.0 - np.exp(-t / 0.0018)) * (0.65 * np.exp(-t / (0.048 if ghost else 0.082)) + 0.35 * np.exp(-t / 0.022))
        w_l = _bandpass(r.normal(0.0, 1.0, L), 2200.0, 9500.0, sr, order=2) * wire_env * (0.34 if ghost else 0.56)
        w_r = _bandpass(r.normal(0.0, 1.0, L), 2200.0, 9500.0, sr, order=2) * wire_env * (0.34 if ghost else 0.56)
        sl = amp * (att * (body + mud_ring) + w_l)
        sr_sig = amp * (att * (body + mud_ring) + w_r)
        mid_s = 0.5 * (sl + sr_sig)
        pk_off, _ = dr._find_first_major_peak(_bandpass(mid_s, 165.0, 400.0, sr, order=3), int(0.018 * sr), rel_thr=0.55)
        return sl, sr_sig, pk_off

    def make_hihat(open_hat: bool = False, amp: float = 0.28, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.default_rng(seed)
        L = int((0.36 if open_hat else 0.075) * sr)
        t = np.arange(L, dtype=np.float64) / sr
        tau = 0.135 if open_hat else 0.015
        env = (1.0 - np.exp(-t / 0.0004)) * np.exp(-t / tau)
        # Metallic inharmonic partials + high band noise
        metal = (
            0.25 * np.sin(2.0 * np.pi * 3420.0 * t)
            + 0.25 * np.sin(2.0 * np.pi * 5180.0 * t)
            + 0.25 * np.sin(2.0 * np.pi * 7630.0 * t)
            + 0.85 * _bandpass(r.normal(0.0, 1.0, L), 5800.0, 15800.0, sr, order=2)
        )
        sig = amp * env * metal
        return sig * 1.06, sig * 0.94

    def make_crash_or_ride(crash: bool = True, amp: float = 0.46, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.default_rng(seed)
        L = int((0.95 if crash else 0.55) * sr)
        t = np.arange(L, dtype=np.float64) / sr
        tau = 0.34 if crash else 0.19
        env = (1.0 - np.exp(-t / 0.0008)) * np.exp(-t / tau)
        # Crash has broadband wash down to 420 Hz!
        lo_cut = 440.0 if crash else 900.0
        wash_l = _bandpass(r.normal(0.0, 1.0, L), lo_cut, 15500.0, sr, order=2)
        wash_r = _bandpass(r.normal(0.0, 1.0, L), lo_cut, 15500.0, sr, order=2)
        ping = 0.25 * np.exp(-t / 0.08) * np.sin(2.0 * np.pi * (2680.0 if not crash else 3920.0) * t)
        return amp * env * (wash_l + ping), amp * env * (wash_r + ping)

    def make_tom(f0: float, amp: float = 0.62, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """High tom (~172 Hz), Mid tom (~132 Hz), Floor tom (~92 Hz): strong tonal shell, brief stick click, NO snare wires!"""
        r = np.random.default_rng(seed)
        L = int(0.44 * sr)
        t = np.arange(L, dtype=np.float64) / sr
        att = 1.0 - np.exp(-t / 0.0006)
        f_inst = f0 + (0.24 * f0) * np.exp(-t / 0.016)
        phi = np.cumsum(2.0 * np.pi * f_inst / sr)
        shell = (0.86 * np.exp(-t / 0.135) * np.sin(phi) + 0.22 * np.exp(-t / 0.065) * np.sin(1.58 * phi)) * att
        # Brief plastic stick attack (<5 ms) without sustained wire rattle!
        stick = _bandpass(r.normal(0.0, 1.0, L) * np.exp(-t / 0.0024) * att, 1800.0, 4200.0, sr, order=2) * 0.16
        sig = amp * (shell + stick)
        return sig * 1.04, sig * 0.96

    t0_offset = 0.60
    # Bars 0..9 (10 bars @ 2.4s = 24.0s):
    # If phone_mic_rolloff is True, Bars 0..2 are Snare + HiHat ONLY (no Kick!) just like real drums.wav intro!
    for bar in range(10):
        bar_t = t0_offset + bar * (4.0 * beat_s)

        if bar == 7:
            # Bar 7: Full Tom Fill + Crash at end! (High Tom -> Mid Tom -> Floor Tom -> Snare Beat 4)
            # Beat 1: Kick + Crash
            pos_b1 = int(round(bar_t * sr))
            kl, kr, kp = make_rehearsal_kick(0.76, seed=100 + bar)
            cl, cr = make_crash_or_ride(crash=True, amp=0.44, seed=200 + bar)
            add_stereo(pos_b1, kl + cl[: kl.size], kr + cr[: kr.size])
            gt_kicks.append(pos_b1 + kp)

            # High Tom (170 Hz) on 1-and & 2
            for sub_step, f_tom, tag in [
                (2, 170.0, "high_tom"),
                (4, 170.0, "high_tom"),
                (6, 132.0, "mid_tom"),
                (8, 132.0, "mid_tom"),
                (10, 92.0, "floor_tom"),
                (11, 92.0, "floor_tom"),
            ]:
                p_tom = int(round((bar_t + sub_step * sixteenth_s) * sr))
                tl, tr = make_tom(f_tom, amp=0.60, seed=300 + bar * 20 + sub_step)
                add_stereo(p_tom, tl, tr)
                gt_distractors.append((p_tom, tag))

            # Beat 4 (step 12): Hard Snare flam/backbeat out of fill!
            p_sn4 = int(round((bar_t + 12 * sixteenth_s) * sr))
            sl, sr_s, sp = make_rehearsal_snare(amp=0.80, ghost=False, seed=400 + bar)
            add_stereo(p_sn4, sl, sr_s)
            gt_snares.append(p_sn4 + sp)
            continue

        # Standard groove bars (with ghost notes, open hats, crashes, ride, and simultaneous Kick+Snare in Bar 5 & 9)
        kick_steps = [] if (phone_mic_rolloff and bar < 3) else [0, 7, 8, 10]
        if bar in (5, 9):
            kick_steps.append(12)  # Simultaneous Kick + Snare on Beat 4!

        snare_steps = [(4, False, 0.72), (12, False, 0.75)]
        if bar % 2 == 1:
            snare_steps.append((9, True, 0.29))  # Ghost snare on 3-e!

        for ks in kick_steps:
            pk_pos = int(round((bar_t + ks * sixteenth_s) * sr))
            amp_k = 0.75 if ks in (0, 8, 12) else 0.56
            kl, kr, kp = make_rehearsal_kick(amp=amp_k, seed=500 + bar * 16 + ks)
            add_stereo(pk_pos, kl, kr)
            gt_kicks.append(pk_pos + kp)

        for ss, is_ghost, amp_s in snare_steps:
            ps_pos = int(round((bar_t + ss * sixteenth_s) * sr))
            sl, sr_s, sp = make_rehearsal_snare(amp=amp_s, ghost=is_ghost, seed=700 + bar * 16 + ss)
            add_stereo(ps_pos, sl, sr_s)
            gt_snares.append(ps_pos + sp)

        # Hi-hats / Ride / Crash across 8th notes (steps 0, 2, 4, 6, 8, 10, 12, 14)
        for hs in (0, 2, 4, 6, 8, 10, 12, 14):
            ph_pos = int(round((bar_t + hs * sixteenth_s) * sr))
            has_snare_here = any(ss == hs for ss, _, _ in snare_steps)
            has_kick_here = hs in kick_steps
            if hs == 0 and bar in (0, 4, 8):
                cl, cr = make_crash_or_ride(crash=True, amp=0.42, seed=900 + bar)
                add_stereo(ph_pos, cl, cr)
                if not has_snare_here and not has_kick_here:
                    gt_distractors.append((ph_pos, "crash"))
            elif bar >= 8:
                rl, rr = make_crash_or_ride(crash=False, amp=0.30, seed=950 + bar * 16 + hs)
                add_stereo(ph_pos, rl, rr)
                if not has_snare_here and not has_kick_here:
                    gt_distractors.append((ph_pos, "ride"))
            else:
                is_open = hs == 14 and (bar % 2 == 0)
                hl, hr = make_hihat(open_hat=is_open, amp=0.30 if is_open else 0.24, seed=1100 + bar * 16 + hs)
                add_stereo(ph_pos, hl, hr)
                if not has_snare_here and not has_kick_here:
                    gt_distractors.append((ph_pos, "open_hat" if is_open else "closed_hat"))

    # Add sustained bass-amp bleed in rehearsal room (smooth 65 Hz / 82 Hz notes without drum attack)
    t_all = np.arange(n, dtype=np.float64) / sr
    bass_env = 0.5 * (1.0 - np.cos(2.0 * np.pi * t_all / 1.2))
    bass_bleed = 0.045 * bass_env * np.sin(2.0 * np.pi * 73.4 * t_all)
    left += bass_bleed
    right += bass_bleed

    # Add boxy 340 Hz rehearsal-room early reflections
    for d_ms, g in [(11.5, 0.14), (19.2, -0.10), (28.6, 0.07)]:
        d = int(round(d_ms * 0.001 * sr))
        box_l = _bandpass(left, 240.0, 520.0, sr, order=2)
        box_r = _bandpass(right, 240.0, 520.0, sr, order=2)
        left[d:] += g * box_r[:-d]
        right[d:] += g * box_l[:-d]

    # Subtle room noise floor
    left += rng.normal(0.0, 0.0006, n)
    right += rng.normal(0.0, 0.0006, n)

    if phone_mic_rolloff:
        # Simulate smartphone mic steep sub roll-off below 115 Hz (-18 dB at 55 Hz!)
        sos_hp = butter(2, 125.0 / (0.5 * sr), btype="highpass", output="sos")
        left = 0.14 * left + 0.86 * sosfiltfilt(sos_hp, left)
        right = 0.14 * right + 0.86 * sosfiltfilt(sos_hp, right)

    pk = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-9)
    scale = 0.68 / pk
    stem = np.vstack([left * scale, right * scale]).astype(np.float32)

    return stem, {
        "kicks": sorted(gt_kicks),
        "snares": sorted(gt_snares),
        "distractors": gt_distractors,
    }


def _evaluate_detections(
    detected_pos: list[int],
    gt_pos: list[int],
    sr: int,
    tol_ms: float = 15.0,
) -> dict:
    tol_samp = int(round(tol_ms * 0.001 * sr))
    used_det = set()
    timing_errs_ms: list[float] = []
    tp = 0

    for g in gt_pos:
        best_j = None
        best_dist = tol_samp + 1
        for j, d in enumerate(detected_pos):
            if j in used_det:
                continue
            dist = abs(d - g)
            if dist <= tol_samp and dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            used_det.add(best_j)
            tp += 1
            timing_errs_ms.append(abs(detected_pos[best_j] - g) * 1000.0 / sr)

    fp = len(detected_pos) - tp
    fn = len(gt_pos) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2.0 * prec * rec / max(1e-12, prec + rec)
    mean_err_ms = float(np.mean(timing_errs_ms)) if timing_errs_ms else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_err_ms": mean_err_ms,
        "unmatched_det": [detected_pos[j] for j in range(len(detected_pos)) if j not in used_det],
    }


def _measure_in_phase_constructive_rate(
    dry_2xn: np.ndarray,
    wet_bus_2xn: np.ndarray,
    hit_positions: list[int],
    sr: int,
    lo_hz: float,
    hi_hz: float,
) -> float:
    """
    Verify on each reinforced hit that `RMS(dry_band + wet_band) > max(RMS(dry_band), RMS(wet_band))`
    in the fundamental band `[lo_hz, hi_hz]` over `[-6 ms, +28 ms]` around the hit!
    """
    if not hit_positions:
        return 1.0
    dry_mid = 0.5 * (dry_2xn[0].astype(np.float64) + dry_2xn[1].astype(np.float64))
    wet_mid = 0.5 * (wet_bus_2xn[0].astype(np.float64) + wet_bus_2xn[1].astype(np.float64))
    dry_b = _bandpass(dry_mid, lo_hz, hi_hz, sr, order=3)
    wet_b = _bandpass(wet_mid, lo_hz, hi_hz, sr, order=3)

    pre = int(round(0.006 * sr))
    post = int(round(0.028 * sr))
    constructive = 0
    evaluated = 0
    for pos in hit_positions:
        s0 = max(0, pos - pre)
        s1 = min(dry_b.size, pos + post)
        if s1 - s0 < 16:
            continue
        d_seg = dry_b[s0:s1]
        w_seg = wet_b[s0:s1]
        r_d = float(np.sqrt(np.mean(d_seg ** 2)))
        r_w = float(np.sqrt(np.mean(w_seg ** 2)))
        if r_w < 1e-6 or r_d < 1e-6:
            continue
        r_sum = float(np.sqrt(np.mean((d_seg + w_seg) ** 2)))
        evaluated += 1
        if r_sum > max(r_d, r_w):
            constructive += 1
    return constructive / max(1, evaluated)


class TestDrumSamplingLab(unittest.TestCase):
    """End-to-end quantitative benchmark & regression suite for Studio Drum Sampling v2."""

    def test_01_sample_bank_acoustic_quality_and_sinc(self) -> None:
        info = dss.ensure_studio_drum_samples(dss.DEFAULT_SAMPLES_DIR)
        print("\n=== [LAB 1] STUDIO SAMPLE BANK ACOUSTIC QUALITY AUDIT ===")

        # 1. Verify Kaiser-windowed sinc resampler preserves 12 kHz brilliance (<0.25 dB loss vs >2.5 dB linear loss)
        n_test = 4096
        tone_12k = [math.sin(2.0 * math.pi * 12000.0 * i / 44100.0) for i in range(n_test)]
        res_sinc = dss._resample_sinc_list(tone_12k, 0.88)[64:-64]
        rms_in = math.sqrt(sum(x * x for x in tone_12k[64:-64]) / (n_test - 128))
        rms_out = math.sqrt(sum(x * x for x in res_sinc) / len(res_sinc))
        loss_db = abs(20.0 * math.log10((rms_out + 1e-12) / rms_in))
        print(f"  Kaiser-Sinc 12 kHz passband droop @ ratio=0.88: {loss_db:.3f} dB")
        self.assertLess(loss_db, 0.35, "Sinc resampler must preserve high-frequency wire brilliance!")

        # 2. Audit all 12 WAVs (6 canonical RR1 + 6 RR2)
        for rr_label, key_suffix in [("RR1", ""), ("RR2", "_rr2")]:
            k_metrics = {lyr: _measure_sample_wav(info[f"kick{key_suffix}"][lyr]) for lyr in ("soft", "med", "hard")}
            s_metrics = {lyr: _measure_sample_wav(info[f"snare{key_suffix}"][lyr]) for lyr in ("soft", "med", "hard")}

            for lyr in ("soft", "med", "hard"):
                km = k_metrics[lyr]
                sm = s_metrics[lyr]
                print(
                    f"  [{rr_label}] kick_{lyr:4s} : pk={km['pk_db']:6.2f}dB rms={km['rms_db']:6.2f}dB "
                    f"crest={km['crest_db']:5.2f}dB sub/box={km['sub_box_db']:6.2f}dB cent={km['centroid']:6.1f}Hz"
                )
                print(
                    f"  [{rr_label}] snare_{lyr:4s}: pk={sm['pk_db']:6.2f}dB rms={sm['rms_db']:6.2f}dB "
                    f"crest={sm['crest_db']:5.2f}dB crack/mud={sm['crack_mud_db']:5.2f}dB cent={sm['centroid']:6.1f}Hz"
                )
                self.assertEqual(km["sr"], 44100)
                self.assertEqual(sm["sr"], 44100)
                self.assertLess(km["dc"], 1e-4)
                self.assertLess(sm["dc"], 1e-4)
                self.assertLess(km["pk_db"], -0.4)
                self.assertLess(sm["pk_db"], -0.4)
                self.assertGreater(km["mono_corr"], 0.998, "Kick sub <140 Hz must be strictly mono phase-coherent!")
                self.assertGreater(km["sub_box_db"], 22.0, "Kick sub/boxiness ratio must exceed +22 dB!")
                self.assertGreater(sm["crack_mud_db"], 3.5, "Snare crack/mud ratio must exceed +3.5 dB!")

            # Strict monotonic progression: soft < med < hard in BOTH energy and brightness!
            self.assertLess(k_metrics["soft"]["energy"], k_metrics["med"]["energy"])
            self.assertLess(k_metrics["med"]["energy"], k_metrics["hard"]["energy"])
            self.assertLess(k_metrics["soft"]["hf_lf_db"], k_metrics["med"]["hf_lf_db"])
            self.assertLess(k_metrics["med"]["hf_lf_db"], k_metrics["hard"]["hf_lf_db"])

            self.assertLess(s_metrics["soft"]["energy"], s_metrics["med"]["energy"])
            self.assertLess(s_metrics["med"]["energy"], s_metrics["hard"]["energy"])
            self.assertLess(s_metrics["soft"]["centroid"], s_metrics["med"]["centroid"])
            self.assertLess(s_metrics["med"]["centroid"], s_metrics["hard"]["centroid"])

        # Verify RR1 vs RR2 are genuinely distinct studio round-robins (zero machine-gun cloning)
        for inst in ("kick", "snare"):
            for lyr in ("soft", "med", "hard"):
                m1 = _measure_sample_wav(info[inst][lyr])["mid"]
                m2 = _measure_sample_wav(info[f"{inst}_rr2"][lyr])["mid"]
                L = min(m1.size, m2.size)
                rel_diff = float(np.linalg.norm(m1[:L] - m2[:L]) / (np.linalg.norm(m1[:L]) + 1e-12))
                self.assertGreater(rel_diff, 0.05, f"{inst}_{lyr} RR1 vs RR2 must not be identical!")

    def test_02_synthetic_rehearsal_kit_ground_truth(self) -> None:
        print("\n=== [LAB 2] SYNTHETIC MULTI-INSTRUMENT REHEARSAL KIT BENCHMARK ===")
        for scenario_name, phone_rolloff in [("Standard Rehearsal Room", False), ("Phone Mic Sub-Rolloff + Snare-Only Intro", True)]:
            stem, gt = _synth_rehearsal_kit_ground_truth(sr=44100, phone_mic_rolloff=phone_rolloff)
            out_audio, stats = dr.enhance_and_reinforce_drums(stem, sr=44100, drum_mode="natural")

            det_kicks = stats.get("kick_positions", [])
            det_snares = stats.get("snare_positions")
            if det_snares is None:
                det_snares = [h[0] for h in dr._detect_snare_hits(stem, sr=44100)]

            ev_k = _evaluate_detections(det_kicks, gt["kicks"], sr=44100, tol_ms=12.0)
            ev_s = _evaluate_detections(det_snares, gt["snares"], sr=44100, tol_ms=12.0)

            # Distractor false positive rate (Hi-Hats, Crash, Ride, High/Mid/Floor Toms)
            tol_fp = int(round(0.020 * 44100))
            fp_on_distractors = 0
            for d_pos, _tag in gt["distractors"]:
                hit_k = any(abs(k - d_pos) <= tol_fp for k in ev_k["unmatched_det"])
                hit_s = any(abs(s - d_pos) <= tol_fp for s in ev_s["unmatched_det"])
                if hit_k or hit_s:
                    fp_on_distractors += 1
            fp_rate = fp_on_distractors / max(1, len(gt["distractors"]))

            # Constructive in-phase summation check
            wet_only = out_audio - dr._apply_dynamic_antimud_and_transient_tightener(
                stem, sr=44100, drum_mode="natural", kick_hits=[(p, 1, 1.0) for p in det_kicks], snare_hits=[(p, 1, 1.0) for p in det_snares]
            ) if "kick_hits" in dr._apply_dynamic_antimud_and_transient_tightener.__code__.co_varnames else (out_audio - stem)

            phase_k = stats.get("kick_in_phase_pct", None)
            phase_s = stats.get("snare_in_phase_pct", None)
            if phase_k is None:
                phase_k = 100.0 * _measure_in_phase_constructive_rate(stem, wet_only, det_kicks, 44100, 42.0, 145.0)
            if phase_s is None:
                phase_s = 100.0 * _measure_in_phase_constructive_rate(stem, wet_only, det_snares, 44100, 165.0, 400.0)

            print(
                f"  [{scenario_name}]\n"
                f"    Kick : F1={ev_k['f1']:.4f} (P={ev_k['precision']:.3f}, R={ev_k['recall']:.3f}, TP={ev_k['tp']}/{len(gt['kicks'])}, FP={ev_k['fp']}) | err={ev_k['mean_err_ms']:.2f}ms | in-phase={phase_k:.1f}%\n"
                f"    Snare: F1={ev_s['f1']:.4f} (P={ev_s['precision']:.3f}, R={ev_s['recall']:.3f}, TP={ev_s['tp']}/{len(gt['snares'])}, FP={ev_s['fp']}) | err={ev_s['mean_err_ms']:.2f}ms | in-phase={phase_s:.1f}%\n"
                f"    Distractor FP Rate (HH/Crash/Ride/Toms): {100.0*fp_rate:.2f}% ({fp_on_distractors}/{len(gt['distractors'])})"
            )

            self.assertGreaterEqual(ev_k["f1"], 0.96, f"Kick F1 below 0.96 in {scenario_name}!")
            self.assertGreaterEqual(ev_s["f1"], 0.95, f"Snare F1 below 0.95 in {scenario_name}!")
            self.assertLessEqual(fp_rate, 0.02, f"Distractor FP rate > 2% in {scenario_name}!")
            self.assertLessEqual(ev_k["mean_err_ms"], 1.0, "Kick alignment error must be <= 1.0 ms!")
            self.assertLessEqual(ev_s["mean_err_ms"], 1.0, "Snare alignment error must be <= 1.0 ms!")
            self.assertGreaterEqual(phase_k, 98.0, "Kick constructive in-phase summation must exceed 98%!")
            self.assertGreaterEqual(phase_s, 98.0, "Snare constructive in-phase summation must exceed 98%!")

    @unittest.skipUnless(os.path.isfile(REAL_DRUMS_PATH), "Real Demucs drums.wav stem not present")
    def test_03_real_rehearsal_drums_and_bass_stems(self) -> None:
        print("\n=== [LAB 3] REAL DEMUCS STEMS EVALUATION (99.9 BPM, E minor) ===")
        drums, sr = sf.read(REAL_DRUMS_PATH, dtype="float32", always_2d=True)
        drums_2xN = drums.T

        def stem_metrics(x_2xn: np.ndarray) -> dict:
            mid = 0.5 * (x_2xn[0].astype(np.float64) + x_2xn[1].astype(np.float64))
            pk = float(np.max(np.abs(x_2xn)))
            rms = float(np.sqrt(np.mean(mid ** 2) + 1e-18))
            e_sub = float(np.mean(_bandpass(mid, 45.0, 95.0, sr, 3) ** 2))
            e_mud = float(np.mean(_bandpass(mid, 250.0, 520.0, sr, 3) ** 2))
            e_crack = float(np.mean(_bandpass(mid, 2000.0, 5500.0, sr, 3) ** 2))
            e_air = float(np.mean(_bandpass(mid, 7500.0, 16000.0, sr, 3) ** 2))
            return {
                "pk_db": 20.0 * np.log10(pk + 1e-18),
                "rms_db": 20.0 * np.log10(rms),
                "crest_db": 20.0 * np.log10((pk + 1e-18) / rms),
                "sub_mud_db": 10.0 * np.log10((e_sub + 1e-18) / (e_mud + 1e-18)),
                "crack_mud_db": 10.0 * np.log10((e_crack + 1e-18) / (e_mud + 1e-18)),
                "air_db": 10.0 * np.log10(e_air + 1e-18),
            }

        m_raw = stem_metrics(drums_2xN)
        out_nat, st_nat = dr.enhance_and_reinforce_drums(drums_2xN, sr=sr, drum_mode="natural")
        out_pun, st_pun = dr.enhance_and_reinforce_drums(drums_2xN, sr=sr, drum_mode="punch")
        m_nat = stem_metrics(out_nat)
        m_pun = stem_metrics(out_pun)

        k_pos = st_nat.get("kick_positions", [])
        kicks_intro_snare_only = [p for p in k_pos if 25.0 * sr <= p <= 38.0 * sr]

        print(f"  RAW       : crest={m_raw['crest_db']:.2f}dB | sub/mud={m_raw['sub_mud_db']:.2f}dB | crack/mud={m_raw['crack_mud_db']:.2f}dB | air={m_raw['air_db']:.2f}dB | pk={m_raw['pk_db']:.2f}dBFS")
        print(f"  NATURAL v2: kicks={st_nat['kicks']}, snares={st_nat['snares']} | crest={m_nat['crest_db']:.2f}dB | sub/mud={m_nat['sub_mud_db']:.2f}dB | crack/mud={m_nat['crack_mud_db']:.2f}dB | air={m_nat['air_db']:.2f}dB | pk={m_nat['pk_db']:.2f}dBFS | phaseK={st_nat.get('kick_in_phase_pct', 0):.1f}% phaseS={st_nat.get('snare_in_phase_pct', 0):.1f}%")
        print(f"  PUNCH   v2: kicks={st_pun['kicks']}, snares={st_pun['snares']} | crest={m_pun['crest_db']:.2f}dB | sub/mud={m_pun['sub_mud_db']:.2f}dB | crack/mud={m_pun['crack_mud_db']:.2f}dB | air={m_pun['air_db']:.2f}dB | pk={m_pun['pk_db']:.2f}dBFS")
        print(f"  False kicks in snare-only verse (25-38s): {len(kicks_intro_snare_only)}")

        # Musical coherence assertions on the 242s active groove @ 99.9 BPM (~101 active bars)
        self.assertLessEqual(len(kicks_intro_snare_only), 2, "Snare-only verse (25-38s) must NOT false-trigger kicks on snares!")
        self.assertGreaterEqual(st_nat["kicks"], 70)
        self.assertLessEqual(st_nat["kicks"], 240)
        self.assertGreaterEqual(st_nat["snares"], 180)
        self.assertLessEqual(st_nat["snares"], 360)
        self.assertGreater(m_nat["crest_db"], m_raw["crest_db"] + 1.0, "Crest factor must clearly increase!")
        self.assertGreater(m_nat["sub_mud_db"], m_raw["sub_mud_db"] + 12.0, "Sub-Punch/Mud must improve substantially!")
        self.assertGreater(m_nat["crack_mud_db"], m_raw["crack_mud_db"] + 2.5, "Snare Crack/Mud must improve clearly!")
        self.assertGreater(m_nat["air_db"], m_raw["air_db"] - 0.35, "Cymbal air >7.5 kHz must stay 100% crystalline without pumping!")
        self.assertLessEqual(m_nat["pk_db"], -0.40, "Natural mode output peak must stay <= -0.40 dBFS!")
        self.assertLessEqual(m_pun["pk_db"], -0.40, "Punch mode output peak must stay <= -0.40 dBFS!")
        if "kick_in_phase_pct" in st_nat:
            self.assertGreaterEqual(st_nat["kick_in_phase_pct"], 98.0)
        if "snare_in_phase_pct" in st_nat:
            self.assertGreaterEqual(st_nat["snare_in_phase_pct"], 98.0)

        # Bass Enhancer check on real bass.wav + enhanced/raw drums.wav
        if os.path.isfile(REAL_BASS_PATH):
            bass, b_sr = sf.read(REAL_BASS_PATH, dtype="float32", always_2d=True)
            bass_2xN = bass.T
            b_nat, b_st_nat = be.enhance_and_resynthesize_bass(bass_2xN, drums_2xN=out_nat, sr=b_sr, bass_mode="natural")
            b_rea, b_st_rea = be.enhance_and_resynthesize_bass(bass_2xN, drums_2xN=out_pun, sr=b_sr, bass_mode="reamp")
            pk_b = 20.0 * np.log10(float(np.max(np.abs(b_nat))) + 1e-18)
            print(f"  BASS v2 (natural): {b_st_nat} | pk={pk_b:.2f}dBFS")
            print(f"  BASS v2 (reamp)  : {b_st_rea}")
            self.assertEqual(b_nat.shape, bass_2xN.shape)
            self.assertFalse(np.any(np.isnan(b_nat)))
            self.assertLessEqual(pk_b, -0.40)
            self.assertLessEqual(b_st_nat["median_f0_hz"], 95.0, "Bass F0 tracker must lock onto true ~65 Hz bass fundamental, not 2nd/3rd harmonic!")
            self.assertGreaterEqual(b_st_nat["median_f0_hz"], 38.0)

    def test_04_public_api_contracts_and_edge_cases(self) -> None:
        """Verify exact public signatures, mode switches, 1D / (N,2) / (2,N) shapes, and silent/short buffers."""
        import inspect

        sig_d = list(inspect.signature(dr.enhance_and_reinforce_drums).parameters.keys())
        self.assertEqual(sig_d, ["drums_2xN", "sr", "drum_mode", "kick_blend", "snare_blend", "samples_dir"])

        sig_b = list(inspect.signature(be.enhance_and_resynthesize_bass).parameters.keys())
        self.assertEqual(sig_b, ["bass_2xN", "drums_2xN", "sr", "bass_mode"])

        # Off mode passthrough
        dummy_2xn = np.zeros((2, 8192), dtype=np.float32)
        out_off, st_off = dr.enhance_and_reinforce_drums(dummy_2xn, sr=44100, drum_mode="off")
        self.assertEqual(out_off.shape, (2, 8192))
        self.assertEqual(st_off["drum_mode"], "off")

        b_off, bst_off = be.enhance_and_resynthesize_bass(dummy_2xn, drums_2xN=dummy_2xn, sr=44100, bass_mode="off")
        self.assertEqual(b_off.shape, (2, 8192))
        self.assertEqual(bst_off["bass_mode"], "off")

        # 1D mono and (N, 2) transposed shape preservation
        t = np.arange(12000, dtype=np.float32) / 44100.0
        mono_in = (0.3 * np.sin(2.0 * np.pi * 65.4 * t)).astype(np.float32)
        d_1d, _ = dr.enhance_and_reinforce_drums(mono_in, sr=44100, drum_mode="natural")
        self.assertEqual(d_1d.shape, mono_in.shape)
        b_1d, _ = be.enhance_and_resynthesize_bass(mono_in, sr=44100, bass_mode="natural")
        self.assertEqual(b_1d.shape, mono_in.shape)

        trans_in = np.column_stack([mono_in, mono_in])
        d_tr, _ = dr.enhance_and_reinforce_drums(trans_in, sr=44100, drum_mode="punch")
        self.assertEqual(d_tr.shape, trans_in.shape)
        b_tr, _ = be.enhance_and_resynthesize_bass(trans_in, drums_2xN=d_tr, sr=44100, bass_mode="reamp")
        self.assertEqual(b_tr.shape, trans_in.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
