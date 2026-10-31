#!/usr/bin/env python3
"""
Studio Acoustic Drum Sample Hybridizer & Synthesizer v2 (Pure Python Stdlib).

Generates 24-bit 44.1 kHz stereo studio one-shot acoustic drum WAV samples
with both multi-velocity (`soft`, `med`, `hard`) and true Round-Robin (`rr1`, `rr2`):
  - Canonical RR1:
      kick_studio_soft.wav, kick_studio_med.wav, kick_studio_hard.wav
      snare_studio_soft.wav, snare_studio_med.wav, snare_studio_hard.wav
  - Alternate RR2:
      kick_studio_soft_rr2.wav, kick_studio_med_rr2.wav, kick_studio_hard_rr2.wav
      snare_studio_soft_rr2.wav, snare_studio_med_rr2.wav, snare_studio_hard_rr2.wav

Hybrid Studio Architecture:
  1. Real Open-Source CC0-1.0 Studio Multi-Mic Layer (when regenerating with
     ffmpeg + network available):
       - `sfzinstruments/virtuosity_drums` (CC0-1.0): close kick/snare mics +
         stereo studio overheads + studio room + hickory rimshot crack, using
         distinct real studio Round-Robin takes (`rr1` vs `rr2`).
       - `sfzinstruments/karoryfer.frankensnare` (CC0-1.0): 14"x6.5" studio maple
         dedicated bottom snare-wire microphone (`btm`) `rr1` & `rr2`.
     Phase-aligns transient peaks & polarity, resamples via pure-Python polyphase
     Kaiser-windowed sinc interpolation (preserving 8-16 kHz wire shimmer), and
     carves 320-500 Hz boxiness.
  2. Physical Studio Membrane & Snare-Wire Synthesis Layer (100% pure Python stdlib,
     also serving as full offline fallback):
       - Kick: 22"x18" maple membrane sweep (155 Hz -> 52 Hz), 82 Hz warm body,
         velocity-scaled felt/wood beater transient (2.4-5.2 kHz), mono sub (<150 Hz).
       - Snare: tuned 188 Hz + 252 Hz drumhead modes (zero 280-520 Hz boxiness),
         crisp hickory stick impact (2.2-5.5 kHz), stereo decorrelated 20-strand
         snare-wire rattle (2.4-14.5 kHz), and subtle 85 ms studio plate ambience.

Uses ONLY Python standard library modules (glob, math, os, random, shutil,
struct, subprocess, tempfile, urllib.request, wave) with zero external packages.
"""

from __future__ import annotations

import glob
import math
import os
import random
import shutil
import struct
import subprocess
import tempfile
import urllib.request
import wave

SAMPLE_RATE = 44100
TARGET_PEAK_DBFS = -1.0
TARGET_PEAK_LINEAR = 10.0 ** (TARGET_PEAK_DBFS / 20.0)
LAYER_TARGET_PEAK_DBFS = {
    "soft": -3.60,
    "med": -1.85,
    "hard": -0.85,
}
MAX_INT24 = 8388607  # (1 << 23) - 1

DEFAULT_SAMPLES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "drum_samples")
)

STUDIO_SAMPLE_FILES = {
    "kick": {
        "soft": "kick_studio_soft.wav",
        "med": "kick_studio_med.wav",
        "hard": "kick_studio_hard.wav",
    },
    "snare": {
        "soft": "snare_studio_soft.wav",
        "med": "snare_studio_med.wav",
        "hard": "snare_studio_hard.wav",
    },
}

STUDIO_SAMPLE_FILES_RR2 = {
    "kick": {
        "soft": "kick_studio_soft_rr2.wav",
        "med": "kick_studio_med_rr2.wav",
        "hard": "kick_studio_hard_rr2.wav",
    },
    "snare": {
        "soft": "snare_studio_soft_rr2.wav",
        "med": "snare_studio_med_rr2.wav",
        "hard": "snare_studio_hard_rr2.wav",
    },
}

STUDIO_SNARE_ROLL_RR = {
    3: {
        "soft": "snare_studio_soft_rr3.wav",
        "med": "snare_studio_med_rr3.wav",
        "hard": "snare_studio_hard_rr3.wav",
    },
    4: {
        "soft": "snare_studio_soft_rr4.wav",
        "med": "snare_studio_med_rr4.wav",
        "hard": "snare_studio_hard_rr4.wav",
    },
}

STUDIO_TOM_FILES = {
    1: {
        "high": "tom_studio_high.wav",
        "mid": "tom_studio_mid.wav",
        "floor": "tom_studio_floor.wav",
    },
    2: {
        "high": "tom_studio_high_rr2.wav",
        "mid": "tom_studio_mid_rr2.wav",
        "floor": "tom_studio_floor_rr2.wav",
    },
}

# CC0 1.0 Public Domain multi-mic studio drum sessions on GitHub
VIRTUOSITY_CC0_BASE = "https://raw.githubusercontent.com/sfzinstruments/virtuosity_drums/master/Samples"
FRANKENSNARE_CC0_BASE = "https://raw.githubusercontent.com/sfzinstruments/karoryfer.frankensnare/master/Samples"

# Structured per (instrument, velocity_layer, round_robin_idx in {1, 2})
CC0_STUDIO_PACK_MAP: dict[str, dict[str, dict[int, dict[str, tuple[str, str] | None]]]] = {
    "kick": {
        "soft": {
            1: {
                "close": (VIRTUOSITY_CC0_BASE, "kickmic/kick/kickmic_kick_snoff_vl2_rr1.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/kick/oh_kick_snoff_vl2_rr1.flac"),
                "extra": None,
            },
            2: {
                "close": (VIRTUOSITY_CC0_BASE, "kickmic/kick/kickmic_kick_snoff_vl2_rr2.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/kick/oh_kick_snoff_vl2_rr2.flac"),
                "extra": None,
            },
        },
        "med": {
            1: {
                "close": (VIRTUOSITY_CC0_BASE, "kickmic/kick/kickmic_kick_snoff_vl3_rr1.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/kick/oh_kick_snoff_vl3_rr1.flac"),
                "extra": (VIRTUOSITY_CC0_BASE, "kickmic/kick/kickmic_kick_snon_vl3_rr1.flac"),
            },
            2: {
                "close": (VIRTUOSITY_CC0_BASE, "kickmic/kick/kickmic_kick_snoff_vl3_rr2.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/kick/oh_kick_snoff_vl3_rr2.flac"),
                "extra": (VIRTUOSITY_CC0_BASE, "kickmic/kick/kickmic_kick_snon_vl3_rr2.flac"),
            },
        },
        "hard": {
            1: {
                "close": (VIRTUOSITY_CC0_BASE, "kickmic/kick/kickmic_kick_snoff_vl4_rr1.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/kick/oh_kick_snoff_vl4_rr1.flac"),
                "extra": (VIRTUOSITY_CC0_BASE, "room/kick/room_kick_snoff_vl4_rr1.flac"),
            },
            2: {
                "close": (VIRTUOSITY_CC0_BASE, "kickmic/kick/kickmic_kick_snoff_vl4_rr2.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/kick/oh_kick_snoff_vl4_rr2.flac"),
                "extra": (VIRTUOSITY_CC0_BASE, "room/kick/room_kick_snoff_vl4_rr2.flac"),
            },
        },
    },
    "snare": {
        "soft": {
            1: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl11.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl11.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl4_rr1.flac"),
                "extra": None,
            },
            2: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl13.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl13.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl4_rr2.flac"),
                "extra": None,
            },
            3: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl10.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl10.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl4_rr3.flac"),
                "extra": None,
            },
            4: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl14.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl14.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl4_rr4.flac"),
                "extra": None,
            },
        },
        "med": {
            1: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl24.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl24.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl8_rr1.flac"),
                "extra": None,
            },
            2: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl26.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl26.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl8_rr2.flac"),
                "extra": None,
            },
            3: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl22.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl22.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl8_rr3.flac"),
                "extra": None,
            },
            4: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl28.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl28.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl8_rr4.flac"),
                "extra": None,
            },
        },
        "hard": {
            1: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl35.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl35.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl11_rr1.flac"),
                "extra": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_rimshot_vl10.flac"),
            },
            2: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl36.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl36.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl11_rr2.flac"),
                "extra": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_rimshot_vl12.flac"),
            },
            3: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl33.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl33.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl11_rr3.flac"),
                "extra": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_rimshot_vl9.flac"),
            },
            4: {
                "close": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_center_vl37.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/snare/oh_snare_center_vl37.flac"),
                "btm": (FRANKENSNARE_CC0_BASE, "14p_basic/btm/sn_vl11_rr4.flac"),
                "extra": (VIRTUOSITY_CC0_BASE, "snaremic/snare/snaremic_snare_rimshot_vl11.flac"),
            },
        },
    },
    "tom": {
        "high": {
            1: {
                "close": (VIRTUOSITY_CC0_BASE, "mid/htom/mid_htom_center_vl3.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/htom/oh_htom_center_vl3.flac"),
            },
            2: {
                "close": (VIRTUOSITY_CC0_BASE, "mid/htom/mid_htom_center_vl4.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/htom/oh_htom_center_vl4.flac"),
            },
        },
        "mid": {
            1: {
                "close": (VIRTUOSITY_CC0_BASE, "mid/htom/mid_htom_center_vl4.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/htom/oh_htom_center_vl4.flac"),
            },
            2: {
                "close": (VIRTUOSITY_CC0_BASE, "mid/ltom/mid_ltom_center_vl3.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/ltom/oh_ltom_center_vl3.flac"),
            },
        },
        "floor": {
            1: {
                "close": (VIRTUOSITY_CC0_BASE, "mid/ltom/mid_ltom_center_vl4.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/ltom/oh_ltom_center_vl4.flac"),
            },
            2: {
                "close": (VIRTUOSITY_CC0_BASE, "mid/ltom/mid_ltom_center_vl5.flac"),
                "oh": (VIRTUOSITY_CC0_BASE, "oh/ltom/oh_ltom_center_vl5.flac"),
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Pure-Python Audio EQ Cookbook Biquad Filters & Windowed-Sinc Resampler
# ---------------------------------------------------------------------------


def _biquad_coeffs_bandpass(fc: float, q: float, sr: int = SAMPLE_RATE) -> tuple[float, float, float, float, float]:
    """Constant 0 dB peak gain bandpass biquad."""
    fc = max(20.0, min(fc, 0.47 * sr))
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * max(q, 0.05))
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    inv_a0 = 1.0 / a0
    return b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0


def _biquad_coeffs_highpass(fc: float, q: float = 0.70710678, sr: int = SAMPLE_RATE) -> tuple[float, float, float, float, float]:
    fc = max(10.0, min(fc, 0.47 * sr))
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * max(q, 0.05))
    b0 = (1.0 + cos_w0) * 0.5
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    inv_a0 = 1.0 / a0
    return b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0


def _biquad_coeffs_lowpass(fc: float, q: float = 0.70710678, sr: int = SAMPLE_RATE) -> tuple[float, float, float, float, float]:
    fc = max(40.0, min(fc, 0.47 * sr))
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * max(q, 0.05))
    b0 = (1.0 - cos_w0) * 0.5
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    inv_a0 = 1.0 / a0
    return b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0


def _biquad_coeffs_peaking(fc: float, gain_db: float, q: float, sr: int = SAMPLE_RATE) -> tuple[float, float, float, float, float]:
    fc = max(20.0, min(fc, 0.47 * sr))
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * max(q, 0.05))
    b0 = 1.0 + alpha * amp
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * amp
    a0 = 1.0 + alpha / amp
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / amp
    inv_a0 = 1.0 / a0
    return b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0


def _biquad_coeffs_highshelf(fc: float, gain_db: float, sr: int = SAMPLE_RATE) -> tuple[float, float, float, float, float]:
    fc = max(200.0, min(fc, 0.46 * sr))
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 * 0.5 * math.sqrt(2.0)
    two_sqrt_a_alpha = 2.0 * math.sqrt(amp) * alpha
    b0 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_w0)
    b2 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (amp + 1.0) - (amp - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((amp - 1.0) - (amp + 1.0) * cos_w0)
    a2 = (amp + 1.0) - (amp - 1.0) * cos_w0 - two_sqrt_a_alpha
    inv_a0 = 1.0 / a0
    return b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0


def _apply_biquad(samples: list[float], coeffs: tuple[float, float, float, float, float]) -> list[float]:
    b0, b1, b2, a1, a2 = coeffs
    out = [0.0] * len(samples)
    z1 = 0.0
    z2 = 0.0
    for i, x in enumerate(samples):
        y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        out[i] = y
    return out


def _bessel_i0(x: float) -> float:
    """Modified Bessel function of the first kind I0(x) via rapidly converging power series."""
    ax = abs(x)
    s = 1.0
    term = 1.0
    half_x_sq = 0.25 * ax * ax
    for k in range(1, 28):
        term *= half_x_sq / (k * k)
        s += term
        if term < 1e-15 * s:
            break
    return s


_SINC_TABLE_CACHE: dict[tuple[float, int, float, int], list[list[float]]] = {}


def _build_kaiser_sinc_table(
    cutoff: float,
    half_lobes: int = 8,
    kaiser_beta: float = 8.0,
    phases: int = 512,
) -> list[list[float]]:
    """Build normalized polyphase Kaiser-windowed sinc kernel table in pure Python stdlib."""
    key = (round(cutoff, 4), half_lobes, round(kaiser_beta, 3), phases)
    cached = _SINC_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    inv_i0_beta = 1.0 / _bessel_i0(kaiser_beta)
    taps = 2 * half_lobes
    table: list[list[float]] = []

    for p in range(phases):
        frac = p / float(phases)
        row = [0.0] * taps
        row_sum = 0.0
        for m in range(taps):
            # Kernel offset relative to int(pos): k = m - (half_lobes - 1)
            k = m - (half_lobes - 1)
            dx = frac - float(k)
            norm_r = dx / float(half_lobes)
            if abs(norm_r) >= 1.0:
                w = 0.0
            else:
                w = _bessel_i0(kaiser_beta * math.sqrt(max(0.0, 1.0 - norm_r * norm_r))) * inv_i0_beta
                sx = math.pi * cutoff * dx
                sinc_v = 1.0 if abs(sx) < 1e-11 else (math.sin(sx) / sx)
                w *= cutoff * sinc_v
            row[m] = w
            row_sum += w
        inv_sum = (1.0 / row_sum) if abs(row_sum) > 1e-15 else 1.0
        table.append([v * inv_sum for v in row])

    _SINC_TABLE_CACHE[key] = table
    return table


def _resample_sinc_list(
    samples: list[float],
    ratio: float,
    half_lobes: int = 8,
    kaiser_beta: float = 8.0,
) -> list[float]:
    """
    High-definition polyphase Kaiser-windowed sinc resampler (pure Python stdlib).
    Preserves full 8-16 kHz snare-wire & cymbal transient brilliance without linear droop
    while preventing aliasing when pitching upward (`ratio > 1.0`).
    """
    n = len(samples)
    if n < 8 or abs(ratio - 1.0) < 1e-6:
        return list(samples)

    out_len = max(8, int(round(n / ratio)))
    cutoff = 0.96 if ratio <= 1.0 else max(0.20, 0.95 / ratio)
    phases = 512
    table = _build_kaiser_sinc_table(cutoff=cutoff, half_lobes=half_lobes, kaiser_beta=kaiser_beta, phases=phases)
    taps = 2 * half_lobes
    pad = half_lobes + 2
    padded = [0.0] * pad + list(samples) + [0.0] * pad
    offset_base = pad - (half_lobes - 1)

    out = [0.0] * out_len
    last_src = float(n - 1)
    for i in range(out_len):
        pos = i * ratio
        if pos > last_src:
            pos = last_src
        idx = int(pos)
        frac = pos - idx
        p_idx = int(frac * phases)
        if p_idx >= phases:
            p_idx = phases - 1
        row = table[p_idx]
        base = idx + offset_base
        acc = 0.0
        for m in range(taps):
            acc += row[m] * padded[base + m]
        out[i] = acc
    return out


def _resample_linear_list(samples: list[float], ratio: float) -> list[float]:
    """Backward-compatible alias upgraded to Kaiser-windowed sinc interpolation."""
    return _resample_sinc_list(samples, ratio)


# ---------------------------------------------------------------------------
# Optional CC0 Studio Acoustic Multi-Mic Loader (Pure Stdlib + FFmpeg)
# ---------------------------------------------------------------------------


def _locate_ffmpeg() -> str | None:
    ff = shutil.which("ffmpeg")
    if ff and os.path.isfile(ff):
        return ff
    local_app = os.environ.get("LOCALAPPDATA", "") or os.path.expanduser(r"~\AppData\Local")
    if local_app and os.path.isdir(local_app):
        winget_dir = os.path.join(local_app, "Microsoft", "WinGet", "Packages")
        for pat in ("*FFmpeg*", "*ffmpeg*"):
            pattern = os.path.join(winget_dir, pat, "**", "ffmpeg.exe")
            matches = sorted(glob.glob(pattern, recursive=True), reverse=True)
            for m in matches:
                if os.path.isfile(m):
                    return m
    for candidate in (
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _read_wav_pcm24_lists(path: str) -> tuple[list[float], list[float]]:
    """Read 24-bit or 16-bit WAV file into (left, right) float lists in [-1, 1]."""
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    total = nframes * nch
    vals = [0.0] * total
    if sw == 3:
        scale = 1.0 / 8388608.0
        idx = 0
        for k in range(total):
            v = raw[idx] | (raw[idx + 1] << 8) | (raw[idx + 2] << 16)
            if v & 0x800000:
                v -= 0x1000000
            vals[k] = v * scale
            idx += 3
    elif sw == 2:
        ints = struct.unpack(f"<{total}h", raw[: total * 2])
        vals = [x / 32768.0 for x in ints]
    else:
        raise ValueError(f"Unsupported sample width {sw}")

    if nch == 1:
        return vals, list(vals)
    left = vals[0::nch]
    right = vals[1::nch]
    return left, right


def _fetch_and_decode_cc0_flac(
    base_url: str,
    rel_path: str,
    ffmpeg_bin: str,
    work_dir: str,
    sr: int = SAMPLE_RATE,
) -> tuple[list[float], list[float]] | None:
    """Download a CC0 studio drum FLAC and decode to 44.1 kHz 24-bit WAV lists."""
    repo_tag = "frk_" if "frankensnare" in base_url else "vrt_"
    safe_name = repo_tag + rel_path.replace("/", "_")
    flac_path = os.path.join(work_dir, safe_name)
    wav_path = flac_path + ".wav"
    if not os.path.isfile(wav_path):
        url = f"{base_url}/{rel_path}"
        req = urllib.request.Request(url, headers={"User-Agent": "StudioEQ-MasterPro/2.0"})
        with urllib.request.urlopen(req, timeout=14) as resp:
            payload = resp.read()
        with open(flac_path, "wb") as fh:
            fh.write(payload)
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-i",
            flac_path,
            "-ar",
            str(sr),
            "-af",
            "aresample=resampler=swr:filter_size=32:phase_shift=10:cutoff=0.97",
            "-c:a",
            "pcm_s24le",
            wav_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=18)
    return _read_wav_pcm24_lists(wav_path)


def _find_primary_transient_peak(samples: list[float], max_window: int) -> tuple[int, float]:
    """
    Return (peak_index, polarity_sign) of the FIRST major transient half-cycle
    (>= 58% of initial window max) so soft/med/hard & rr1/rr2 never lock onto adjacent half-cycles.
    """
    lim = min(len(samples), max(16, max_window))
    win_max = 0.0
    best_global_i = 0
    for i in range(lim):
        a = abs(samples[i])
        if a > win_max:
            win_max = a
            best_global_i = i
    if win_max <= 1e-12:
        return 0, 1.0

    thr = 0.58 * win_max
    best_i = best_global_i
    for i in range(1, lim - 1):
        a = abs(samples[i])
        if a >= thr and a >= abs(samples[i - 1]) and a >= abs(samples[i + 1]):
            best_i = i
            break

    pol = 1.0 if samples[best_i] >= 0.0 else -1.0
    return best_i, pol


def _align_and_pad_layer(
    src: list[float],
    src_pk_idx: int,
    dst_pk_idx: int,
    dst_len: int,
) -> list[float]:
    """Time-align `src` so `src[src_pk_idx]` lands at `out[dst_pk_idx]` with smooth onset."""
    out = [0.0] * dst_len
    shift = dst_pk_idx - src_pk_idx
    src_n = len(src)
    start_dst = max(0, shift)
    start_src = max(0, -shift)
    count = min(dst_len - start_dst, src_n - start_src)
    if count <= 0:
        return out
    for k in range(count):
        out[start_dst + k] = src[start_src + k]
    fade = min(8, count // 8)
    for k in range(fade):
        w = 0.5 * (1.0 - math.cos(math.pi * k / max(1, fade)))
        out[start_dst + k] *= w
    return out


# ---------------------------------------------------------------------------
# Studio Room / Plate Early Reflections & Post-Processing Helpers
# ---------------------------------------------------------------------------


def _add_early_studio_room(
    mono_mid_high: list[float],
    sr: int,
    taps_ms_gains_l: list[tuple[float, float]],
    taps_ms_gains_r: list[tuple[float, float]],
    lp_hz: float = 4200.0,
) -> tuple[list[float], list[float]]:
    """Generate subtle decorrelated stereo early room reflections from mid/high content."""
    n = len(mono_mid_high)
    room_l = [0.0] * n
    room_r = [0.0] * n

    for delay_ms, gain in taps_ms_gains_l:
        d = int(round(delay_ms * 0.001 * sr))
        if 0 < d < n:
            for i in range(d, n):
                room_l[i] += gain * mono_mid_high[i - d]

    for delay_ms, gain in taps_ms_gains_r:
        d = int(round(delay_ms * 0.001 * sr))
        if 0 < d < n:
            for i in range(d, n):
                room_r[i] += gain * mono_mid_high[i - d]

    lp_coeffs = _biquad_coeffs_lowpass(lp_hz, q=0.707, sr=sr)
    room_l = _apply_biquad(room_l, lp_coeffs)
    room_r = _apply_biquad(room_r, lp_coeffs)
    return room_l, room_r


def _generate_snare_plate_85ms(
    source_l: list[float],
    source_r: list[float],
    rng: random.Random,
    sr: int = SAMPLE_RATE,
    decay_tau_s: float = 0.085,
) -> tuple[list[float], list[float]]:
    """
    Synthesize a smooth, dense 85 ms studio plate ambience, high-passed at 680 Hz
    and damped above 8.8 kHz so 280-520 Hz stays 100% mud-free.
    """
    n = len(source_l)
    plate_l = [0.0] * n
    plate_r = [0.0] * n

    num_impulses = 68
    span_s = 0.250
    grid = span_s / num_impulses
    for k in range(num_impulses):
        t_l = 0.004 + (k + rng.uniform(0.08, 0.92)) * grid
        t_r = 0.0045 + (k + rng.uniform(0.08, 0.92)) * grid
        env_l = math.exp(-t_l / decay_tau_s) * (1.0 if (k % 2 == 0) else -1.0) * rng.uniform(0.75, 1.0)
        env_r = math.exp(-t_r / decay_tau_s) * (-1.0 if ((k + 1) % 3 == 0) else 1.0) * rng.uniform(0.75, 1.0)
        dl = int(round(t_l * sr))
        dr = int(round(t_r * sr))
        if dl < n:
            gl = env_l * 0.14
            for i in range(dl, n):
                plate_l[i] += gl * (0.7 * source_l[i - dl] + 0.3 * source_r[i - dl])
        if dr < n:
            gr = env_r * 0.14
            for i in range(dr, n):
                plate_r[i] += gr * (0.7 * source_r[i - dr] + 0.3 * source_l[i - dr])

    hp_coeffs = _biquad_coeffs_highpass(680.0, q=0.707, sr=sr)
    lp_coeffs = _biquad_coeffs_lowpass(8800.0, q=0.707, sr=sr)
    plate_l = _apply_biquad(_apply_biquad(plate_l, hp_coeffs), lp_coeffs)
    plate_r = _apply_biquad(_apply_biquad(plate_r, hp_coeffs), lp_coeffs)
    return plate_l, plate_r


def _finalize_stereo_sample(
    left: list[float],
    right: list[float],
    hp_hz: float,
    sr: int = SAMPLE_RATE,
    target_peak: float = TARGET_PEAK_LINEAR,
) -> tuple[list[float], list[float]]:
    """
    1. High-pass sub-sonic DC rumble.
    2. Smooth cosine fade-out on final 28 ms so tail hits exact zero.
    3. Exact zero-DC window-weighted mean removal (preserving 0.0 endpoints).
    4. Enforce positive (+1) primary initial transient peak polarity.
    5. Stereo-linked peak normalization to `target_peak`.
    """
    n = len(left)
    hp_coeffs = _biquad_coeffs_highpass(hp_hz, q=0.70710678, sr=sr)
    left = _apply_biquad(left, hp_coeffs)
    right = _apply_biquad(right, hp_coeffs)

    fade_in = min(6, n // 16)
    for i in range(fade_in):
        w = 0.5 * (1.0 - math.cos(math.pi * i / max(1, fade_in)))
        left[i] *= w
        right[i] *= w

    fade_out = min(int(0.028 * sr), n // 4)
    start_fo = n - fade_out
    for k in range(fade_out):
        w = 0.5 * (1.0 + math.cos(math.pi * (k + 1) / fade_out))
        left[start_fo + k] *= w
        right[start_fo + k] *= w
    left[-1] = 0.0
    right[-1] = 0.0

    dc_win = [0.0] * n
    edge = max(32, min(256, n // 20))
    for i in range(n):
        if i < edge:
            dc_win[i] = 0.5 * (1.0 - math.cos(math.pi * i / edge))
        elif i >= n - edge:
            dc_win[i] = 0.5 * (1.0 - math.cos(math.pi * (n - 1 - i) / edge))
        else:
            dc_win[i] = 1.0
    w_sum = sum(dc_win) + 1e-18
    corr_l = sum(left) / w_sum
    corr_r = sum(right) / w_sum
    for i in range(n):
        left[i] -= corr_l * dc_win[i]
        right[i] -= corr_r * dc_win[i]

    mid_early = [0.5 * (left[i] + right[i]) for i in range(min(n, max(64, int(0.015 * sr))))]
    _, pol = _find_primary_transient_peak(mid_early, len(mid_early))

    peak = 0.0
    for l_val, r_val in zip(left, right):
        al = abs(l_val)
        ar = abs(r_val)
        if al > peak:
            peak = al
        if ar > peak:
            peak = ar

    gain = (pol * target_peak / peak) if peak > 1e-12 else 1.0
    left = [x * gain for x in left]
    right = [x * gain for x in right]
    return left, right


def _write_wav_24bit_stereo(
    path: str,
    left: list[float],
    right: list[float],
    seed: int = 101,
    sr: int = SAMPLE_RATE,
) -> None:
    """Write stereo float [-1, 1] lists as 24-bit little-endian PCM WAV with gentle TPDF dither."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = len(left)
    rng = random.Random(seed)
    buf = bytearray(n * 6)
    dither_scale = 0.5 / MAX_INT24

    silent_tail = max(0, n - 16)
    idx = 0
    for i in range(n):
        lv = left[i]
        rv = right[i]
        if 2 <= i < silent_tail and (abs(lv) > 1e-9 or abs(rv) > 1e-9):
            lv += (rng.random() - rng.random()) * dither_scale
            rv += (rng.random() - rng.random()) * dither_scale

        li = int(round(max(-1.0, min(1.0, lv)) * MAX_INT24))
        ri = int(round(max(-1.0, min(1.0, rv)) * MAX_INT24))
        lu = li & 0xFFFFFF
        ru = ri & 0xFFFFFF
        buf[idx] = lu & 0xFF
        buf[idx + 1] = (lu >> 8) & 0xFF
        buf[idx + 2] = (lu >> 16) & 0xFF
        buf[idx + 3] = ru & 0xFF
        buf[idx + 4] = (ru >> 8) & 0xFF
        buf[idx + 5] = (ru >> 16) & 0xFF
        idx += 6

    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(3)
        wf.setframerate(sr)
        wf.writeframes(bytes(buf))


# ---------------------------------------------------------------------------
# Acoustic Studio Kick Drum Synthesis + CC0 Real Multi-Mic Hybridization
# ---------------------------------------------------------------------------


def synthesize_studio_kick(
    velocity_layer: str = "med",
    sr: int = SAMPLE_RATE,
    cc0_bundle: dict | None = None,
    rr_idx: int = 1,
) -> tuple[list[float], list[float]]:
    """
    Model a 22"x18" studio maple/birch acoustic kick drum, hybridized
    phase-coherently with real CC0 studio close + overhead + room recordings:
      - Fast batter membrane pitch envelope: 155 Hz -> 52 Hz over 38 ms
      - Warm 82 Hz resonant front-head / shell body decay
      - Natural felt/wood beater transient (2.4 - 5.2 kHz) monotonically scaled by velocity
      - Distinct Round-Robin (`rr_idx=1` vs `rr_idx=2`) acoustic takes & micro-resonances
      - Zero DC offset, <150 Hz strictly mono, velocity-calibrated studio peak target
    """
    rr_detune_hz = 0.0 if rr_idx == 1 else 0.75
    params = {
        "soft": {
            "seed": 4101 if rr_idx == 1 else 4201,
            "duration_s": 0.54,
            "f_start": 144.0 + rr_detune_hz,
            "f_end": 52.0,
            "tau_pitch_s": 0.0118,
            "tau_sub_s": 0.142,
            "body_82_gain": 0.32,
            "tau_body_82_s": 0.080,
            "shell_knock_gain": 0.07,
            "beater_gain": 0.09,
            "beater_hi_gain": 0.04,
            "sat_drive": 1.15,
            "room_gain": 0.040,
            "cc0_close_mix": 0.34,
            "cc0_oh_mix": 0.10,
            "cc0_extra_mix": 0.0,
        },
        "med": {
            "seed": 4102 if rr_idx == 1 else 4202,
            "duration_s": 0.58,
            "f_start": 155.0 + rr_detune_hz,
            "f_end": 52.0,
            "tau_pitch_s": 0.0116,
            "tau_sub_s": 0.160,
            "body_82_gain": 0.39,
            "tau_body_82_s": 0.090,
            "shell_knock_gain": 0.13,
            "beater_gain": 0.22,
            "beater_hi_gain": 0.14,
            "sat_drive": 1.34,
            "room_gain": 0.055,
            "cc0_close_mix": 0.42,
            "cc0_oh_mix": 0.14,
            "cc0_extra_mix": 0.06,
        },
        "hard": {
            "seed": 4103 if rr_idx == 1 else 4203,
            "duration_s": 0.62,
            "f_start": 168.0 + rr_detune_hz,
            "f_end": 52.0,
            "tau_pitch_s": 0.0112,
            "tau_sub_s": 0.178,
            "body_82_gain": 0.46,
            "tau_body_82_s": 0.098,
            "shell_knock_gain": 0.19,
            "beater_gain": 0.39,
            "beater_hi_gain": 0.30,
            "sat_drive": 1.56,
            "room_gain": 0.072,
            "cc0_close_mix": 0.48,
            "cc0_oh_mix": 0.18,
            "cc0_extra_mix": 0.08,
        },
    }[velocity_layer]

    rng = random.Random(params["seed"])
    n = int(round(params["duration_s"] * sr))
    inv_sr = 1.0 / sr
    two_pi = 2.0 * math.pi

    f_start = params["f_start"]
    f_end = params["f_end"]
    f_span = f_start - f_end
    tau_p = params["tau_pitch_s"]
    tau_sub = params["tau_sub_s"]
    body_82_gain = params["body_82_gain"]
    tau_b82 = params["tau_body_82_s"]
    knock_gain = params["shell_knock_gain"]
    drive = params["sat_drive"]
    norm_tanh = 1.0 / math.tanh(drive)

    membrane = [0.0] * n
    upper_body = [0.0] * n
    noise_raw = [0.0] * n

    phi_sub = 0.0
    phi_82 = 0.0
    phi_knock = 0.0

    for i in range(n):
        t = i * inv_sr
        att = 1.0 - math.exp(-t / 0.00055)

        f_inst = f_end + f_span * math.exp(-t / tau_p) + 2.6 * math.exp(-t / 0.110)
        phi_sub += two_pi * f_inst * inv_sr

        env_sub = att * (0.80 * math.exp(-t / tau_sub) + 0.26 * math.exp(-t / 0.042))
        sub_sig = math.sin(phi_sub) + 0.14 * math.sin(2.0 * phi_sub) * math.exp(-t / 0.055)

        f_82 = (82.0 + 0.4 * rr_detune_hz) + 24.0 * math.exp(-t / 0.015)
        phi_82 += two_pi * f_82 * inv_sr
        env_82 = att * math.exp(-t / tau_b82)
        body_82_sig = body_82_gain * env_82 * math.sin(phi_82)

        f_kn = 104.0 + 34.0 * math.exp(-t / 0.008)
        phi_knock += two_pi * f_kn * inv_sr
        knock_sig = knock_gain * att * math.exp(-t / 0.018) * math.sin(phi_knock)

        low_sum = env_sub * sub_sig + body_82_sig + knock_sig
        membrane[i] = math.tanh(drive * low_sum) * norm_tanh
        upper_body[i] = knock_sig

        if t < 0.025:
            env_click = (1.0 - math.exp(-t / 0.00035)) * (
                0.72 * math.exp(-t / 0.0028) + 0.28 * math.exp(-t / 0.0068)
            )
            noise_raw[i] = (rng.uniform(-1.0, 1.0) + 0.45 * math.exp(-t / 0.0011)) * env_click

    skin_mid = _apply_biquad(noise_raw, _biquad_coeffs_bandpass(1150.0, q=1.35, sr=sr))
    beater_lo = _apply_biquad(noise_raw, _biquad_coeffs_bandpass(2950.0, q=1.20, sr=sr))
    beater_hi = _apply_biquad(noise_raw, _biquad_coeffs_bandpass(4650.0, q=1.35, sr=sr))

    b_gain = params["beater_gain"]
    bh_gain = params["beater_hi_gain"]
    transient_mono = [0.0] * n
    for i in range(n):
        tr = 0.36 * b_gain * skin_mid[i] + b_gain * beater_lo[i] + bh_gain * beater_hi[i]
        transient_mono[i] = tr
        upper_body[i] += tr

    upper_clean = _apply_biquad(upper_body, _biquad_coeffs_peaking(360.0, gain_db=-7.5, q=1.4, sr=sr))
    room_src = _apply_biquad(upper_clean, _biquad_coeffs_highpass(175.0, q=0.707, sr=sr))
    if rr_idx == 1:
        taps_l = [(7.4, 0.46), (13.8, -0.31), (21.5, 0.22), (31.2, 0.14), (43.0, -0.08)]
        taps_r = [(8.3, 0.44), (15.1, -0.29), (23.4, 0.21), (33.8, -0.13), (46.5, 0.08)]
    else:
        taps_l = [(7.9, 0.45), (14.4, -0.30), (22.3, 0.21), (32.4, 0.13), (44.8, -0.08)]
        taps_r = [(7.1, 0.45), (15.6, -0.28), (24.1, 0.22), (34.5, -0.13), (47.2, 0.08)]
    room_l, room_r = _add_early_studio_room(room_src, sr=sr, taps_ms_gains_l=taps_l, taps_ms_gains_r=taps_r, lp_hz=3800.0)

    rg = params["room_gain"]
    left = [membrane[i] + transient_mono[i] + rg * room_l[i] for i in range(n)]
    right = [membrane[i] + transient_mono[i] + rg * room_r[i] for i in range(n)]

    # Phase-coherent hybridization with real CC0 studio close + overhead (+ room/snon) kick recording
    if cc0_bundle and cc0_bundle.get("close") and cc0_bundle.get("oh"):
        syn_mid = [0.5 * (left[i] + right[i]) for i in range(n)]
        pk_syn, pol_syn = _find_primary_transient_peak(syn_mid, int(0.018 * sr))

        close_raw = cc0_bundle["close"][0]
        oh_l_raw, oh_r_raw = cc0_bundle["oh"]

        pk_close_abs = max((abs(x) for x in close_raw), default=0.0)
        if pk_close_abs > 1e-6:
            ons = 0
            thr = 0.07 * pk_close_abs
            for i, x in enumerate(close_raw):
                if abs(x) >= thr:
                    ons = i
                    break
            trim = max(0, ons - 10)
            cl = close_raw[trim : trim + n + 512]
            ohl = oh_l_raw[trim : trim + n + 512]
            ohr = oh_r_raw[trim : trim + n + 512]

            # High-pass close mic at 66 Hz (52 Hz sub comes cleanly from 22" membrane); carve 360 Hz boxiness
            cl_eq = _apply_biquad(
                _apply_biquad(cl, _biquad_coeffs_highpass(66.0, q=0.707, sr=sr)),
                _biquad_coeffs_peaking(365.0, gain_db=-8.5, q=1.35, sr=sr),
            )
            if velocity_layer == "hard":
                cl_eq = _apply_biquad(cl_eq, _biquad_coeffs_peaking(3400.0, gain_db=2.8, q=1.1, sr=sr))
            elif velocity_layer == "soft":
                cl_eq = _apply_biquad(cl_eq, _biquad_coeffs_lowpass(5800.0, q=0.707, sr=sr))

            # High-pass stereo overheads at 185 Hz (<150 Hz remains 100% mono!) and carve 375 Hz
            hp185 = _biquad_coeffs_highpass(185.0, q=0.707, sr=sr)
            pk375 = _biquad_coeffs_peaking(375.0, gain_db=-6.5, q=1.35, sr=sr)
            ohl_eq = _apply_biquad(_apply_biquad(ohl, hp185), pk375)
            ohr_eq = _apply_biquad(_apply_biquad(ohr, hp185), pk375)

            pk_cc0, pol_cc0 = _find_primary_transient_peak(cl_eq, int(0.024 * sr))
            cl_peak = max((abs(x) for x in cl_eq[: int(0.06 * sr)]), default=1e-9) + 1e-9
            oh_peak = max(
                max((abs(x) for x in ohl_eq[: int(0.08 * sr)]), default=0.0),
                max((abs(x) for x in ohr_eq[: int(0.08 * sr)]), default=0.0),
            ) + 1e-9

            pol_match = pol_syn * pol_cc0
            cl_aligned = _align_and_pad_layer(cl_eq, pk_cc0, pk_syn, n)
            ohl_aligned = _align_and_pad_layer(ohl_eq, pk_cc0, pk_syn, n)
            ohr_aligned = _align_and_pad_layer(ohr_eq, pk_cc0, pk_syn, n)

            cmix = params["cc0_close_mix"] * pol_match / cl_peak
            omix = params["cc0_oh_mix"] * pol_match / oh_peak

            ex_l_aligned = None
            ex_r_aligned = None
            emix = 0.0
            if params["cc0_extra_mix"] > 0.0 and cc0_bundle.get("extra"):
                ex_l_raw, ex_r_raw = cc0_bundle["extra"]
                ex_l_s = ex_l_raw[trim : trim + n + 512]
                ex_r_s = ex_r_raw[trim : trim + n + 512]
                hp240 = _biquad_coeffs_highpass(240.0, q=0.707, sr=sr)
                pk380 = _biquad_coeffs_peaking(380.0, gain_db=-7.0, q=1.4, sr=sr)
                ex_l_eq = _apply_biquad(_apply_biquad(ex_l_s, hp240), pk380)
                ex_r_eq = _apply_biquad(_apply_biquad(ex_r_s, hp240), pk380)
                ex_pk = max(
                    max((abs(x) for x in ex_l_eq[: int(0.08 * sr)]), default=0.0),
                    max((abs(x) for x in ex_r_eq[: int(0.08 * sr)]), default=0.0),
                ) + 1e-9
                ex_l_aligned = _align_and_pad_layer(ex_l_eq, pk_cc0, pk_syn, n)
                ex_r_aligned = _align_and_pad_layer(ex_r_eq, pk_cc0, pk_syn, n)
                emix = params["cc0_extra_mix"] * pol_match / ex_pk

            for i in range(n):
                t = i * inv_sr
                gate = 1.0 if t <= 0.120 else math.exp(-(t - 0.120) / 0.110)
                c_val = cmix * gate * cl_aligned[i]
                el = (emix * gate * ex_l_aligned[i]) if ex_l_aligned is not None else 0.0
                er = (emix * gate * ex_r_aligned[i]) if ex_r_aligned is not None else 0.0
                left[i] += c_val + omix * gate * ohl_aligned[i] + el
                right[i] += c_val + omix * gate * ohr_aligned[i] + er

    target_pk = 10.0 ** (LAYER_TARGET_PEAK_DBFS[velocity_layer] / 20.0)
    return _finalize_stereo_sample(left, right, hp_hz=24.0, sr=sr, target_peak=target_pk)


# ---------------------------------------------------------------------------
# Acoustic Studio Snare Drum Synthesis + CC0 Multi-Mic Hybridization
# ---------------------------------------------------------------------------


def synthesize_studio_snare(
    velocity_layer: str = "med",
    sr: int = SAMPLE_RATE,
    cc0_bundle: dict | None = None,
    rr_idx: int = 1,
) -> tuple[list[float], list[float]]:
    """
    Model a crisp tuned 14"x6.5" studio maple snare drum, hybridized phase-coherently
    with real CC0 studio close + bottom snare-wire (`btm`) + stereo overhead + rimshot mics:
      - Tuned fundamental & warm shell overtone: 188 Hz + 252 Hz
        with tight decay and surgical anti-mud notch across 280-520 Hz
      - Crisp hickory stick impact (2.2 - 5.5 kHz) monotonically scaled by velocity
      - Realistic stereo decorrelated 20-strand snare-wire rattle (2.4 - 14.5 kHz)
      - Real CC0 coated-head top mic, bottom snare-wire mic, stereo overheads & rimshot
      - Subtle 85 ms studio plate ambience, velocity-calibrated studio peak target
    """
    rr_detune = {1: 0.0, 2: 2.2, 3: -1.8, 4: 3.4}.get(rr_idx, 0.0)
    rr_seed_off = {1: 0, 2: 100, 3: 200, 4: 300}.get(rr_idx, 0)
    rr_pitch_off = {1: 0.0, 2: 0.008, 3: -0.006, 4: 0.013}.get(rr_idx, 0.0)
    params = {
        "soft": {
            "seed": 5201 + rr_seed_off,
            "duration_s": 0.45,
            "m188_gain": 0.56,
            "tau_188_s": 0.036,
            "m252_gain": 0.16,
            "tau_252_s": 0.020,
            "rim_gain": 0.03,
            "stick_gain": 0.36,
            "wire_gain": 0.44,
            "wire_lp_hz": 9200.0,
            "wire_air_db": 0.0,
            "tau_wire_s": 0.052,
            "plate_wet": 0.048,
            "sat_drive": 1.12,
            "cc0_close_mix": 0.48,
            "cc0_oh_mix": 0.18,
            "cc0_btm_mix": 0.20,
            "cc0_rim_mix": 0.0,
            "cc0_tau_gate_s": 0.084,
            "pitch_ratio": 0.875 + rr_pitch_off,
        },
        "med": {
            "seed": 5202 + rr_seed_off,
            "duration_s": 0.51,
            "m188_gain": 0.65,
            "tau_188_s": 0.044,
            "m252_gain": 0.20,
            "tau_252_s": 0.024,
            "rim_gain": 0.08,
            "stick_gain": 0.56,
            "wire_gain": 0.68,
            "wire_lp_hz": 12800.0,
            "wire_air_db": 2.2,
            "tau_wire_s": 0.070,
            "plate_wet": 0.064,
            "sat_drive": 1.32,
            "cc0_close_mix": 0.54,
            "cc0_oh_mix": 0.24,
            "cc0_btm_mix": 0.30,
            "cc0_rim_mix": 0.0,
            "cc0_tau_gate_s": 0.098,
            "pitch_ratio": 0.880 + rr_pitch_off,
        },
        "hard": {
            "seed": 5203 + rr_seed_off,
            "duration_s": 0.56,
            "m188_gain": 0.60,
            "tau_188_s": 0.045,
            "m252_gain": 0.21,
            "tau_252_s": 0.025,
            "rim_gain": 0.16,
            "stick_gain": 0.82,
            "wire_gain": 0.92,
            "wire_lp_hz": 15800.0,
            "wire_air_db": 4.8,
            "tau_wire_s": 0.082,
            "plate_wet": 0.078,
            "sat_drive": 1.44,
            "cc0_close_mix": 0.58,
            "cc0_oh_mix": 0.30,
            "cc0_btm_mix": 0.44,
            "cc0_rim_mix": 0.28,
            "cc0_tau_gate_s": 0.112,
            "pitch_ratio": 0.885 + rr_pitch_off,
        },
    }[velocity_layer]

    rng_l = random.Random(params["seed"])
    rng_r = random.Random(params["seed"] + 77)
    rng_plate = random.Random(params["seed"] + 199)

    n = int(round(params["duration_s"] * sr))
    inv_sr = 1.0 / sr
    two_pi = 2.0 * math.pi

    shell_mono = [0.0] * n
    stick_exc = [0.0] * n
    wire_raw_l = [0.0] * n
    wire_raw_r = [0.0] * n

    phi_188 = 0.0
    phi_252 = 0.0
    phi_1120 = 0.0
    phi_2180 = 0.0

    m188_g = params["m188_gain"]
    tau_188 = params["tau_188_s"]
    m252_g = params["m252_gain"]
    tau_252 = params["tau_252_s"]
    rim_g = params["rim_gain"]
    tau_wire = params["tau_wire_s"]

    for i in range(n):
        t = i * inv_sr
        att = 1.0 - math.exp(-t / 0.00030)

        f_188 = (188.0 + rr_detune) + 48.0 * math.exp(-t / 0.0030)
        phi_188 += two_pi * f_188 * inv_sr
        s188 = math.sin(phi_188)

        f_252 = (252.0 + 1.2 * rr_detune) + 34.0 * math.exp(-t / 0.0022)
        phi_252 += two_pi * f_252 * inv_sr
        s252 = math.sin(phi_252)

        phi_1120 += two_pi * 1120.0 * inv_sr
        phi_2180 += two_pi * 2180.0 * inv_sr
        rim_sig = rim_g * math.exp(-t / 0.014) * (0.55 * math.sin(phi_1120) + 0.45 * math.sin(phi_2180))

        punch_env = 1.0 + 0.48 * math.exp(-t / 0.0032)
        shell_mono[i] = att * punch_env * (
            m188_g * math.exp(-t / tau_188) * s188
            + m252_g * math.exp(-t / tau_252) * s252
            + rim_sig
        )

        if t < 0.018:
            env_st = (1.0 - math.exp(-t / 0.00018)) * (
                0.82 * math.exp(-t / 0.0019) + 0.18 * math.exp(-t / 0.0046)
            )
            stick_exc[i] = (0.58 * rng_l.uniform(-1.0, 1.0) + 0.82 * math.exp(-t / 0.00080)) * env_st

        if t >= 0.0012:
            tw = t - 0.0012
            wire_rise = 1.0 - math.exp(-tw / 0.0022)
            wire_body = 0.60 * math.exp(-tw / tau_wire) + 0.40 * math.exp(-tw / 0.024)
            chatter = 1.0 + 0.26 * math.exp(-tw / 0.028) * max(0.0, s188)
            env_w = wire_rise * wire_body * chatter

            n_mid = rng_l.gauss(0.0, 0.58)
            n_side = rng_r.gauss(0.0, 0.58)
            if tw < 0.035 and rng_l.random() < 0.06:
                n_mid += rng_l.uniform(-0.85, 0.85)
            wire_raw_l[i] = env_w * (0.86 * n_mid + 0.26 * n_side)
            wire_raw_r[i] = env_w * (0.86 * n_mid - 0.26 * n_side)

    # Keep 280-520 Hz super clean on the shell
    shell_clean = _apply_biquad(shell_mono, _biquad_coeffs_peaking(385.0, gain_db=-10.5, q=1.25, sr=sr))

    stick_lo = _apply_biquad(stick_exc, _biquad_coeffs_bandpass(2750.0, q=1.10, sr=sr))
    stick_hi = _apply_biquad(stick_exc, _biquad_coeffs_bandpass(4350.0, q=1.25, sr=sr))
    st_g = params["stick_gain"]
    stick_mono = [st_g * (0.58 * stick_lo[i] + 0.56 * stick_hi[i]) for i in range(n)]

    hp_w1 = _biquad_coeffs_highpass(2350.0, q=0.707, sr=sr)
    hp_w2 = _biquad_coeffs_highpass(2350.0, q=0.85, sr=sr)
    lp_w1 = _biquad_coeffs_lowpass(params["wire_lp_hz"], q=0.707, sr=sr)
    pk_w = _biquad_coeffs_peaking(4600.0, gain_db=3.2, q=1.05, sr=sr)
    hs_w = _biquad_coeffs_highshelf(8200.0, gain_db=params["wire_air_db"], sr=sr)

    wire_l = _apply_biquad(_apply_biquad(_apply_biquad(_apply_biquad(_apply_biquad(wire_raw_l, hp_w1), hp_w2), lp_w1), pk_w), hs_w)
    wire_r = _apply_biquad(_apply_biquad(_apply_biquad(_apply_biquad(_apply_biquad(wire_raw_r, hp_w1), hp_w2), lp_w1), pk_w), hs_w)

    wg = params["wire_gain"]
    drive = params["sat_drive"]
    norm_tanh = 1.0 / math.tanh(drive)

    dry_l = [0.0] * n
    dry_r = [0.0] * n
    for i in range(n):
        dl = shell_clean[i] + stick_mono[i] + wg * wire_l[i]
        dr = shell_clean[i] + stick_mono[i] + wg * wire_r[i]
        dry_l[i] = math.tanh(drive * dl) * norm_tanh
        dry_r[i] = math.tanh(drive * dr) * norm_tanh

    plate_l, plate_r = _generate_snare_plate_85ms(dry_l, dry_r, rng=rng_plate, sr=sr, decay_tau_s=0.085)
    pw = params["plate_wet"]

    left = [dry_l[i] + pw * plate_l[i] for i in range(n)]
    right = [dry_r[i] + pw * plate_r[i] for i in range(n)]

    # Phase-coherent hybridization with real CC0 multi-mic studio snare recordings
    if cc0_bundle and cc0_bundle.get("close") and cc0_bundle.get("oh"):
        syn_mid = [0.5 * (left[i] + right[i]) for i in range(n)]
        pk_syn, pol_syn = _find_primary_transient_peak(syn_mid, int(0.015 * sr))

        close_raw = cc0_bundle["close"][0]
        oh_l_raw, oh_r_raw = cc0_bundle["oh"]

        pk_close_abs = max((abs(x) for x in close_raw), default=0.0)
        if pk_close_abs > 1e-6:
            ons = 0
            thr = 0.06 * pk_close_abs
            for i, x in enumerate(close_raw):
                if abs(x) >= thr:
                    ons = i
                    break
            trim = max(0, ons - 12)
            pitch_ratio = params["pitch_ratio"]
            need_src = int((n + 512) * pitch_ratio) + 96
            cl_p = _resample_sinc_list(close_raw[trim : trim + need_src], pitch_ratio)
            ohl_p = _resample_sinc_list(oh_l_raw[trim : trim + need_src], pitch_ratio)
            ohr_p = _resample_sinc_list(oh_r_raw[trim : trim + need_src], pitch_ratio)

            # Console anti-boxiness EQ: HP 105 Hz, deep -10.5 dB carve at 385 Hz, +3.5 dB 3.8 kHz crack
            crack_boost = {"soft": 1.6, "med": 3.4, "hard": 4.8}[velocity_layer]
            air_shelf = {"soft": -1.5, "med": 1.8, "hard": 4.2}[velocity_layer]
            cl_eq = _apply_biquad(
                _apply_biquad(
                    _apply_biquad(
                        _apply_biquad(cl_p, _biquad_coeffs_highpass(105.0, q=0.707, sr=sr)),
                        _biquad_coeffs_peaking(385.0, gain_db=-10.5, q=1.20, sr=sr),
                    ),
                    _biquad_coeffs_peaking(3650.0, gain_db=crack_boost, q=1.0, sr=sr),
                ),
                _biquad_coeffs_highshelf(7800.0, gain_db=air_shelf, sr=sr),
            )
            hp240 = _biquad_coeffs_highpass(240.0, q=0.707, sr=sr)
            pk395 = _biquad_coeffs_peaking(395.0, gain_db=-8.0, q=1.25, sr=sr)
            hs_oh = _biquad_coeffs_highshelf(8000.0, gain_db=air_shelf, sr=sr)
            ohl_eq = _apply_biquad(_apply_biquad(_apply_biquad(ohl_p, hp240), pk395), hs_oh)
            ohr_eq = _apply_biquad(_apply_biquad(_apply_biquad(ohr_p, hp240), pk395), hs_oh)

            pk_cc0, pol_cc0 = _find_primary_transient_peak(cl_eq, int(0.018 * sr))
            oh_mid_early = [0.5 * (ohl_eq[i] + ohr_eq[i]) for i in range(min(len(ohl_eq), int(0.022 * sr)))]
            _, pol_oh = _find_primary_transient_peak(oh_mid_early, len(oh_mid_early))

            cl_peak = max((abs(x) for x in cl_eq[: int(0.05 * sr)]), default=1e-9) + 1e-9
            oh_peak = max(
                max((abs(x) for x in ohl_eq[: int(0.06 * sr)]), default=0.0),
                max((abs(x) for x in ohr_eq[: int(0.06 * sr)]), default=0.0),
            ) + 1e-9

            cl_aligned = _align_and_pad_layer(cl_eq, pk_cc0, pk_syn, n)
            ohl_aligned = _align_and_pad_layer(ohl_eq, pk_cc0, pk_syn, n)
            ohr_aligned = _align_and_pad_layer(ohr_eq, pk_cc0, pk_syn, n)

            cmix = params["cc0_close_mix"] * (pol_syn * pol_cc0) / cl_peak
            omix = params["cc0_oh_mix"] * (pol_syn * pol_oh) / oh_peak
            tau_gate = params["cc0_tau_gate_s"]

            # Real dedicated 14"x6.5" Maple bottom snare-wire mic (`btm` from Frankensnare CC0)
            btm_aligned = None
            bmix = 0.0
            if params["cc0_btm_mix"] > 0.0 and cc0_bundle.get("btm"):
                btm_raw = cc0_bundle["btm"][0]
                pk_b_abs = max((abs(x) for x in btm_raw), default=0.0)
                if pk_b_abs > 1e-6:
                    b_ons = next((i for i, x in enumerate(btm_raw) if abs(x) >= 0.06 * pk_b_abs), 0)
                    b_slice = btm_raw[max(0, b_ons - 10) : max(0, b_ons - 10) + n + 512]
                    # High-pass bottom mic at 1850 Hz (zero mud!) + boost 3.8 kHz crispness & 9 kHz wire sizzle
                    b_eq = _apply_biquad(
                        _apply_biquad(
                            _apply_biquad(b_slice, _biquad_coeffs_highpass(1850.0, q=0.707, sr=sr)),
                            _biquad_coeffs_peaking(3800.0, gain_db=2.8, q=1.0, sr=sr),
                        ),
                        _biquad_coeffs_highshelf(8500.0, gain_db=air_shelf, sr=sr),
                    )
                    pk_b, pol_b = _find_primary_transient_peak(b_eq, int(0.018 * sr))
                    b_peak = max((abs(x) for x in b_eq[: int(0.05 * sr)]), default=1e-9) + 1e-9
                    btm_aligned = _align_and_pad_layer(b_eq, pk_b, pk_syn, n)
                    bmix = params["cc0_btm_mix"] * (pol_syn * pol_b) / b_peak

            # Studio hickory rimshot crack layer on "hard" velocity (high-passed above 620 Hz -> pure crack, zero mud!)
            rim_aligned = None
            rmix = 0.0
            if params["cc0_rim_mix"] > 0.0 and cc0_bundle.get("extra"):
                rim_raw = cc0_bundle["extra"][0]
                pk_r_abs = max((abs(x) for x in rim_raw), default=0.0)
                if pk_r_abs > 1e-6:
                    r_ons = next((i for i, x in enumerate(rim_raw) if abs(x) >= 0.07 * pk_r_abs), 0)
                    r_p = _resample_sinc_list(rim_raw[max(0, r_ons - 10) : max(0, r_ons - 10) + need_src], 0.92)
                    r_eq = _apply_biquad(
                        _apply_biquad(r_p, _biquad_coeffs_highpass(680.0, q=0.707, sr=sr)),
                        _biquad_coeffs_peaking(3400.0, gain_db=3.5, q=1.0, sr=sr),
                    )
                    pk_r, pol_r = _find_primary_transient_peak(r_eq, int(0.015 * sr))
                    r_peak = max((abs(x) for x in r_eq[: int(0.04 * sr)]), default=1e-9) + 1e-9
                    rim_aligned = _align_and_pad_layer(r_eq, pk_r, pk_syn, n)
                    rmix = params["cc0_rim_mix"] * (pol_syn * pol_r) / r_peak

            syn_scale = 0.68
            for i in range(n):
                t = i * inv_sr
                gate = 1.0 if t <= 0.072 else math.exp(-(t - 0.072) / tau_gate)
                c_val = cmix * gate * cl_aligned[i]
                if btm_aligned is not None:
                    b_gate = 1.0 if t <= 0.085 else math.exp(-(t - 0.085) / (tau_gate * 1.08))
                    c_val += bmix * b_gate * btm_aligned[i]
                if rim_aligned is not None:
                    r_gate = 1.0 if t <= 0.042 else math.exp(-(t - 0.042) / 0.048)
                    c_val += rmix * r_gate * rim_aligned[i]
                left[i] = syn_scale * left[i] + c_val + omix * gate * ohl_aligned[i]
                right[i] = syn_scale * right[i] + c_val + omix * gate * ohr_aligned[i]

    target_pk = 10.0 ** (LAYER_TARGET_PEAK_DBFS[velocity_layer] / 20.0)
    return _finalize_stereo_sample(left, right, hp_hz=72.0, sr=sr, target_peak=target_pk)


# ---------------------------------------------------------------------------
# Acoustic Studio Rack & Floor Tom Synthesis + CC0 Multi-Mic Hybridization
# ---------------------------------------------------------------------------


def synthesize_studio_tom(
    tom_pitch: str = "mid",
    sr: int = SAMPLE_RATE,
    cc0_bundle: dict | None = None,
    rr_idx: int = 1,
) -> tuple[list[float], list[float]]:
    """
    Model studio maple/birch High (10" ~168 Hz), Mid (12" ~126 Hz), and Floor (16" ~88 Hz) toms,
    hybridized phase-coherently with CC0 studio close + overhead recordings (`virtuosity_drums`).
    """
    rr_det = 0.0 if rr_idx == 1 else 1.4
    cfg = {
        "high": {
            "seed": 6101 if rr_idx == 1 else 6201,
            "duration_s": 0.56,
            "f_end": 168.0 + rr_det,
            "f_drop": 44.0,
            "tau_body": 0.145,
            "tau_over": 0.070,
            "stick_gain": 0.34,
            "pitch_ratio": 1.00 if rr_idx == 1 else 1.012,
            "hp_hz": 75.0,
        },
        "mid": {
            "seed": 6102 if rr_idx == 1 else 6202,
            "duration_s": 0.64,
            "f_end": 126.0 + rr_det,
            "f_drop": 38.0,
            "tau_body": 0.175,
            "tau_over": 0.082,
            "stick_gain": 0.32,
            "pitch_ratio": 0.80 if rr_idx == 1 else 1.02,
            "hp_hz": 58.0,
        },
        "floor": {
            "seed": 6103 if rr_idx == 1 else 6203,
            "duration_s": 0.74,
            "f_end": 88.0 + 0.8 * rr_det,
            "f_drop": 32.0,
            "tau_body": 0.215,
            "tau_over": 0.095,
            "stick_gain": 0.30,
            "pitch_ratio": 0.84 if rr_idx == 1 else 0.855,
            "hp_hz": 42.0,
        },
    }[tom_pitch]

    rng = random.Random(cfg["seed"])
    n = int(round(cfg["duration_s"] * sr))
    inv_sr = 1.0 / sr
    two_pi = 2.0 * math.pi

    shell = [0.0] * n
    stick_raw = [0.0] * n
    phi_f0 = 0.0
    phi_f1 = 0.0
    f_end = cfg["f_end"]
    f_drop = cfg["f_drop"]

    for i in range(n):
        t = i * inv_sr
        att = 1.0 - math.exp(-t / 0.00050)
        f0 = f_end + f_drop * math.exp(-t / 0.016)
        phi_f0 += two_pi * f0 * inv_sr
        phi_f1 += two_pi * (1.58 * f0) * inv_sr
        s_main = 0.84 * math.exp(-t / cfg["tau_body"]) * math.sin(phi_f0)
        s_over = 0.24 * math.exp(-t / cfg["tau_over"]) * math.sin(phi_f1)
        shell[i] = math.tanh(1.24 * att * (s_main + s_over)) / math.tanh(1.24)
        if t < 0.020:
            env_st = (1.0 - math.exp(-t / 0.00025)) * (0.75 * math.exp(-t / 0.0024) + 0.25 * math.exp(-t / 0.0055))
            stick_raw[i] = (0.65 * rng.uniform(-1.0, 1.0) + 0.55 * math.exp(-t / 0.0010)) * env_st

    shell_clean = _apply_biquad(shell, _biquad_coeffs_peaking(395.0, gain_db=-8.5, q=1.35, sr=sr))
    st_mid = _apply_biquad(stick_raw, _biquad_coeffs_bandpass(2650.0, q=1.15, sr=sr))
    st_hi = _apply_biquad(stick_raw, _biquad_coeffs_bandpass(4400.0, q=1.30, sr=sr))
    sg = cfg["stick_gain"]
    dry_mono = [shell_clean[i] + sg * (0.62 * st_mid[i] + 0.45 * st_hi[i]) for i in range(n)]

    room_src = _apply_biquad(dry_mono, _biquad_coeffs_highpass(190.0, q=0.707, sr=sr))
    taps_l = [(8.2, 0.36), (16.4, -0.22), (26.5, 0.14)]
    taps_r = [(9.1, 0.34), (17.8, -0.21), (28.2, 0.13)]
    rm_l, rm_r = _add_early_studio_room(room_src, sr=sr, taps_ms_gains_l=taps_l, taps_ms_gains_r=taps_r, lp_hz=4500.0)

    left = [dry_mono[i] + 0.075 * rm_l[i] for i in range(n)]
    right = [dry_mono[i] + 0.075 * rm_r[i] for i in range(n)]

    if cc0_bundle and cc0_bundle.get("close") and cc0_bundle.get("oh"):
        syn_mid = [0.5 * (left[i] + right[i]) for i in range(n)]
        pk_syn, pol_syn = _find_primary_transient_peak(syn_mid, int(0.018 * sr))
        close_raw = cc0_bundle["close"][0]
        oh_l_raw, oh_r_raw = cc0_bundle["oh"]
        pk_c_abs = max((abs(x) for x in close_raw), default=0.0)
        if pk_c_abs > 1e-6:
            ons = next((i for i, x in enumerate(close_raw) if abs(x) >= 0.06 * pk_c_abs), 0)
            trim = max(0, ons - 12)
            pr = cfg["pitch_ratio"]
            need = int((n + 512) * pr) + 96
            cl_p = _resample_sinc_list(close_raw[trim : trim + need], pr)
            ohl_p = _resample_sinc_list(oh_l_raw[trim : trim + need], pr)
            ohr_p = _resample_sinc_list(oh_r_raw[trim : trim + need], pr)

            cl_eq = _apply_biquad(
                _apply_biquad(
                    _apply_biquad(cl_p, _biquad_coeffs_highpass(cfg["hp_hz"], q=0.707, sr=sr)),
                    _biquad_coeffs_peaking(410.0, gain_db=-8.5, q=1.30, sr=sr),
                ),
                _biquad_coeffs_peaking(3400.0, gain_db=2.8, q=1.05, sr=sr),
            )
            hp200 = _biquad_coeffs_highpass(200.0, q=0.707, sr=sr)
            pk410 = _biquad_coeffs_peaking(410.0, gain_db=-6.5, q=1.30, sr=sr)
            ohl_eq = _apply_biquad(_apply_biquad(ohl_p, hp200), pk410)
            ohr_eq = _apply_biquad(_apply_biquad(ohr_p, hp200), pk410)

            pk_cc0, pol_cc0 = _find_primary_transient_peak(cl_eq, int(0.020 * sr))
            cl_pk = max((abs(x) for x in cl_eq[: int(0.06 * sr)]), default=1e-9) + 1e-9
            oh_pk = max(
                max((abs(x) for x in ohl_eq[: int(0.07 * sr)]), default=0.0),
                max((abs(x) for x in ohr_eq[: int(0.07 * sr)]), default=0.0),
            ) + 1e-9

            cl_al = _align_and_pad_layer(cl_eq, pk_cc0, pk_syn, n)
            ohl_al = _align_and_pad_layer(ohl_eq, pk_cc0, pk_syn, n)
            ohr_al = _align_and_pad_layer(ohr_eq, pk_cc0, pk_syn, n)
            pol_m = pol_syn * pol_cc0
            cmix = 0.50 * pol_m / cl_pk
            omix = 0.20 * pol_m / oh_pk
            for i in range(n):
                t = i * inv_sr
                gate = 1.0 if t <= 0.14 else math.exp(-(t - 0.14) / 0.16)
                c_val = cmix * gate * cl_al[i]
                left[i] = 0.72 * left[i] + c_val + omix * gate * ohl_al[i]
                right[i] = 0.72 * right[i] + c_val + omix * gate * ohr_al[i]

    target_pk = 10.0 ** (-1.50 / 20.0)
    return _finalize_stereo_sample(left, right, hp_hz=cfg["hp_hz"], sr=sr, target_peak=target_pk)


# ---------------------------------------------------------------------------
# Public Directory & Auto-Generation Helper
# ---------------------------------------------------------------------------


def _load_optional_cc0_studio_packs(sr: int = SAMPLE_RATE) -> dict:
    """
    Fetch & decode CC0-1.0 studio multi-mic recordings (`virtuosity_drums` +
    `karoryfer.frankensnare`) into memory for RR1..RR4 hybrid studio WAVs & Toms.
    Gracefully skips individual unavailable files without aborting the rest.
    """
    ffmpeg_bin = _locate_ffmpeg()
    if not ffmpeg_bin:
        return {}

    packs: dict = {}
    tmp_dir = tempfile.mkdtemp(prefix="studioeq_cc0_drums_v3_")
    try:
        for inst, layers in CC0_STUDIO_PACK_MAP.items():
            packs[inst] = {}
            for layer, rr_dict in layers.items():
                packs[inst][layer] = {}
                for rr_idx, mic_specs in rr_dict.items():
                    bundle: dict[str, tuple[list[float], list[float]] | None] = {}
                    for mic_key, spec in mic_specs.items():
                        if spec is None:
                            bundle[mic_key] = None
                        else:
                            base_url, rel_path = spec
                            try:
                                bundle[mic_key] = _fetch_and_decode_cc0_flac(
                                    base_url, rel_path, ffmpeg_bin, tmp_dir, sr=sr
                                )
                            except Exception:
                                bundle[mic_key] = None
                    packs[inst][layer][rr_idx] = bundle
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return packs


def _is_valid_wav_file(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) < 512:
            return False
        with wave.open(path, "rb") as wf:
            return wf.getnframes() > 32 and wf.getnchannels() >= 1
    except wave.Error:
        try:
            with open(path, "rb") as fh:
                hdr = fh.read(12)
            return len(hdr) == 12 and hdr[:4] == b"RIFF" and hdr[8:12] == b"WAVE"
        except OSError:
            return False
    except OSError:
        return False


def generate_all_studio_drum_samples(samples_dir: str = DEFAULT_SAMPLES_DIR, force: bool = False) -> list[str]:
    """
    Generate all studio 24-bit 44.1 kHz stereo WAV files in `samples_dir`:
      - Kick RR1 + RR2 (`soft`, `med`, `hard`)
      - Snare RR1 + RR2 + RR3 + RR4 (`soft`, `med`, `hard` — 4-way Round-Robin for fast rolls!)
      - Studio Toms RR1 + RR2 (`high`, `mid`, `floor`)
    """
    samples_dir = os.path.abspath(samples_dir)
    os.makedirs(samples_dir, exist_ok=True)

    missing_targets: list[tuple[str, str, int, str]] = []
    for rr_idx, table in ((1, STUDIO_SAMPLE_FILES), (2, STUDIO_SAMPLE_FILES_RR2)):
        for instrument, layers in table.items():
            for layer, filename in layers.items():
                out_path = os.path.join(samples_dir, filename)
                if force or not _is_valid_wav_file(out_path):
                    missing_targets.append((instrument, layer, rr_idx, out_path))

    for rr_idx, layers in STUDIO_SNARE_ROLL_RR.items():
        for layer, filename in layers.items():
            out_path = os.path.join(samples_dir, filename)
            if force or not _is_valid_wav_file(out_path):
                missing_targets.append(("snare", layer, rr_idx, out_path))

    for rr_idx, pitches in STUDIO_TOM_FILES.items():
        for tom_pitch, filename in pitches.items():
            out_path = os.path.join(samples_dir, filename)
            if force or not _is_valid_wav_file(out_path):
                missing_targets.append(("tom", tom_pitch, rr_idx, out_path))

    if not missing_targets:
        return []

    cc0_packs = _load_optional_cc0_studio_packs(sr=SAMPLE_RATE)
    generated: list[str] = []

    seed_base = 700
    for instrument, layer, rr_idx, out_path in missing_targets:
        seed_base += 13
        bundle = cc0_packs.get(instrument, {}).get(layer, {}).get(rr_idx)
        if instrument == "kick":
            left, right = synthesize_studio_kick(
                velocity_layer=layer, sr=SAMPLE_RATE, cc0_bundle=bundle, rr_idx=rr_idx
            )
        elif instrument == "tom":
            left, right = synthesize_studio_tom(
                tom_pitch=layer, sr=SAMPLE_RATE, cc0_bundle=bundle, rr_idx=rr_idx
            )
        else:
            left, right = synthesize_studio_snare(
                velocity_layer=layer, sr=SAMPLE_RATE, cc0_bundle=bundle, rr_idx=rr_idx
            )
        _write_wav_24bit_stereo(out_path, left, right, seed=seed_base, sr=SAMPLE_RATE)
        generated.append(out_path)

    return generated


def ensure_studio_drum_samples(samples_dir: str = DEFAULT_SAMPLES_DIR) -> dict:
    """
    Ensure all canonical RR1, alternate RR2..RR4, and Studio Tom WAV samples
    exist in `samples_dir` (auto-generating any missing ones) and detect optional
    user custom samples (`custom_kick.wav`, `custom_snare.wav`).
    """
    if not samples_dir:
        samples_dir = DEFAULT_SAMPLES_DIR
    samples_dir = os.path.abspath(samples_dir)
    generated = generate_all_studio_drum_samples(samples_dir=samples_dir, force=False)

    kick_paths = {
        layer: os.path.join(samples_dir, fname)
        for layer, fname in STUDIO_SAMPLE_FILES["kick"].items()
    }
    snare_paths = {
        layer: os.path.join(samples_dir, fname)
        for layer, fname in STUDIO_SAMPLE_FILES["snare"].items()
    }
    kick_rr2_paths = {
        layer: os.path.join(samples_dir, fname)
        for layer, fname in STUDIO_SAMPLE_FILES_RR2["kick"].items()
    }
    snare_rr2_paths = {
        layer: os.path.join(samples_dir, fname)
        for layer, fname in STUDIO_SAMPLE_FILES_RR2["snare"].items()
    }
    snare_rr3_paths = {
        layer: os.path.join(samples_dir, fname)
        for layer, fname in STUDIO_SNARE_ROLL_RR[3].items()
    }
    snare_rr4_paths = {
        layer: os.path.join(samples_dir, fname)
        for layer, fname in STUDIO_SNARE_ROLL_RR[4].items()
    }
    tom_rr1_paths = {
        tp: os.path.join(samples_dir, fname)
        for tp, fname in STUDIO_TOM_FILES[1].items()
    }
    tom_rr2_paths = {
        tp: os.path.join(samples_dir, fname)
        for tp, fname in STUDIO_TOM_FILES[2].items()
    }

    custom_kick_candidate = os.path.join(samples_dir, "custom_kick.wav")
    custom_snare_candidate = os.path.join(samples_dir, "custom_snare.wav")
    custom_kick = custom_kick_candidate if _is_valid_wav_file(custom_kick_candidate) else None
    custom_snare = custom_snare_candidate if _is_valid_wav_file(custom_snare_candidate) else None

    return {
        "samples_dir": samples_dir,
        "kick": kick_paths,
        "snare": snare_paths,
        "kick_rr2": kick_rr2_paths,
        "snare_rr2": snare_rr2_paths,
        "snare_rr3": snare_rr3_paths,
        "snare_rr4": snare_rr4_paths,
        "toms": tom_rr1_paths,
        "toms_rr2": tom_rr2_paths,
        "custom_kick": custom_kick,
        "custom_snare": custom_snare,
        "generated": generated,
    }


if __name__ == "__main__":
    created = generate_all_studio_drum_samples(DEFAULT_SAMPLES_DIR, force=True)
    info = ensure_studio_drum_samples(DEFAULT_SAMPLES_DIR)
    print(f"[drum_samples_synth v2] Ready in: {info['samples_dir']} (generated {len(created)} hybrid studio WAVs)")
    for inst in ("kick", "snare"):
        for lyr in ("soft", "med", "hard"):
            p1 = info[inst][lyr]
            p2 = info[f"{inst}_rr2"][lyr]
            print(
                f"  - {inst:5s} ({lyr:4s}): {os.path.basename(p1)} ({os.path.getsize(p1)/1024:.1f} KB) | "
                f"{os.path.basename(p2)} ({os.path.getsize(p2)/1024:.1f} KB)"
            )
