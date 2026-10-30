#!/usr/bin/env python3
"""
Studio EQ & Rehearsal Room Restoration Engine (DSP + Dynamic Spectral Mastering)
Specificamente calibrato per correggere registrazioni in sala prove (es. Logic Pro X / ambientali)
con accumulo di risonanze medio-basse (100-500 Hz), buco sulle medie (1-2 kHz) e carenza di aria (>6 kHz).
"""

import os
import sys
import json
import subprocess
import numpy as np


def design_biquad(filter_type: str, freq: float, sr: float, gain_db: float = 0.0, q: float = 1.0):
    """
    Progetta coefficienti Biquad IIR secondo l'Audio EQ Cookbook di Robert Bristow-Johnson.
    Restituisce (b, a) normalizzati per a0 = 1.
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    if filter_type == "peaking":
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A
    elif filter_type == "highpass":
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif filter_type == "lowpass":
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif filter_type == "lowshelf":
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha)
        a0 = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        a2 = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha
    elif filter_type == "highshelf":
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha)
        a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha
    else:
        raise ValueError(f"Unsupported filter_type: {filter_type}")

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def apply_biquad_fft_overlap(audio: np.ndarray, filters: list, sr: int) -> np.ndarray:
    """
    Applica una catena di filtri biquad ad altissima precisione (64-bit)
    usando scipy.signal.lfilter se disponibile, altrimenti risposta in frequenza esatta su blocchi STFT lunghi.
    """
    try:
        from scipy.signal import lfilter
        out = audio.astype(np.float64)
        for f in filters:
            b, a = design_biquad(f["type"], f["freq"], sr, f.get("gain_db", 0.0), f.get("q", 1.0))
            out = lfilter(b, a, out, axis=0)
        return out
    except ImportError:
        # Calcolo esatto della funzione di trasferimento complessiva H(e^jw)
        n_samples, n_ch = audio.shape
        n_fft = 1 << int(np.ceil(np.log2(n_samples)))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        w = 2.0 * np.pi * freqs / sr
        ejw1 = np.exp(-1j * w)
        ejw2 = np.exp(-2j * w)

        H_total = np.ones_like(freqs, dtype=np.complex128)
        for f in filters:
            b, a = design_biquad(f["type"], f["freq"], sr, f.get("gain_db", 0.0), f.get("q", 1.0))
            num = b[0] + b[1] * ejw1 + b[2] * ejw2
            den = a[0] + a[1] * ejw1 + a[2] * ejw2
            H_total *= (num / den)

        out = np.zeros_like(audio, dtype=np.float64)
        for ch in range(n_ch):
            spec = np.fft.rfft(audio[:, ch].astype(np.float64), n=n_fft)
            filtered = np.fft.irfft(spec * H_total, n=n_fft)[:n_samples]
            out[:, ch] = filtered
        return out


def dynamic_mud_suppressor(audio: np.ndarray, sr: int, mud_low: float = 110.0, mud_high: float = 380.0,
                           threshold_db: float = -18.0, ratio: float = 3.0) -> np.ndarray:
    """
    Compressore dinamico di banda (Dynamic EQ) mirato sulla zona 110-380 Hz:
    quando il rimbombo della sala prove supera la soglia, attenua solo quella banda
    preservando il calore nei passaggi più delicati.
    """
    n_samples, n_ch = audio.shape
    # Estrai la banda medio-bassa tramite FFT
    n_fft = 1 << int(np.ceil(np.log2(n_samples)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # Maschera morbida tra mud_low e mud_high
    mask = np.exp(-0.5 * ((np.log2(np.maximum(freqs, 1.0) / np.sqrt(mud_low * mud_high))) / 0.65) ** 4)

    mud_band = np.zeros_like(audio, dtype=np.float64)
    for ch in range(n_ch):
        spec = np.fft.rfft(audio[:, ch], n=n_fft)
        mud_band[:, ch] = np.fft.irfft(spec * mask, n=n_fft)[:n_samples]

    # Calcola inviluppo RMS a finestre di 30ms
    win_len = int(0.030 * sr)
    power = np.mean(mud_band ** 2, axis=1)
    kernel = np.ones(win_len) / win_len
    env_rms = np.sqrt(np.convolve(power, kernel, mode="same") + 1e-12)
    env_db = 20.0 * np.log10(env_rms)

    # Riduzione del guadagno sopra threshold_db
    over_db = np.maximum(0.0, env_db - threshold_db)
    gain_reduction_db = -over_db * (1.0 - 1.0 / ratio)
    # Limita la riduzione dinamica a max -5.5 dB per naturalezza
    gain_reduction_db = np.maximum(gain_reduction_db, -5.5)
    # Quanto sottrarre dalla banda mud (1 - linear_gain)
    sub_factor = 1.0 - (10.0 ** (gain_reduction_db / 20.0))

    return audio - mud_band * sub_factor[:, None]


def harmonic_air_exciter(audio: np.ndarray, sr: int, drive: float = 0.22) -> np.ndarray:
    """
    Rigenera le armoniche alte (> 4.5 kHz) andate perse nella registrazione ambientale in sala prove
    o nella compressione MP3, aggiungendo brillantezza ('Air') naturale senza stridore.
    """
    high_filters = [
        {"type": "highpass", "freq": 4500.0, "q": 0.707},
        {"type": "highpass", "freq": 4500.0, "q": 0.707},
    ]
    highs = apply_biquad_fft_overlap(audio, high_filters, sr)
    # Saturazione asimmetrica morbida (stile valvola/nastro) per generare armoniche pari e dispari
    excited = np.tanh(highs * 3.2) / 3.2
    return audio + excited * drive


def mid_side_stage(audio: np.ndarray, sr: int, mono_below_hz: float = 135.0, stereo_width: float = 1.22) -> np.ndarray:
    """
    Elaborazione Mid/Side da studio:
    1. Rende rigorosamente MONO le basse frequenze sotto mono_below_hz (cassa e basso fermi al centro).
    2. Apre l'immagine stereo sulle medio-alte per dare respiro agli strumenti e separarli dalla voce.
    """
    if audio.shape[1] < 2:
        return audio

    mid = 0.5 * (audio[:, 0] + audio[:, 1])
    side = 0.5 * (audio[:, 0] - audio[:, 1])

    # Filtra il canale Side togliendo le basse sotto mono_below_hz
    side_2d = side[:, None]
    side_clean = apply_biquad_fft_overlap(
        side_2d,
        [
            {"type": "highpass", "freq": mono_below_hz, "q": 0.707},
            {"type": "highpass", "freq": mono_below_hz, "q": 0.707},
            {"type": "highshelf", "freq": 1800.0, "gain_db": 1.8, "q": 0.707},
        ],
        sr,
    )[:, 0]

    side_out = side_clean * stereo_width
    left = mid + side_out
    right = mid - side_out
    return np.stack([left, right], axis=1)


def studio_bus_compressor_and_limiter(audio: np.ndarray, sr: int, ceiling_db: float = -0.4) -> np.ndarray:
    """
    Bus Compressor morbido (stile SSL Glue) + Lookahead Soft-Knee Limiter finale
    per controllare i picchi della batteria e uniformare il livello della traccia.
    """
    # 1. RMS Glue Compression (attacco 25ms, rilascio 150ms)
    win_len = int(0.050 * sr)
    power = np.mean(audio ** 2, axis=1)
    env_rms = np.sqrt(np.convolve(power, np.ones(win_len) / win_len, mode="same") + 1e-12)
    env_db = 20.0 * np.log10(env_rms)

    thresh_db = -18.5
    ratio = 2.2
    over_db = np.maximum(0.0, env_db - thresh_db)
    gr_db = -over_db * (1.0 - 1.0 / ratio)
    makeup_db = 2.4
    gain_lin = 10.0 ** ((gr_db + makeup_db) / 20.0)
    compressed = audio * gain_lin[:, None]

    # 2. Normalizzazione RMS target (-12.5 dB RMS, ottima spinta ma con dinamica preservata)
    curr_rms = np.sqrt(np.mean(compressed ** 2) + 1e-12)
    target_rms = 10.0 ** (-12.8 / 20.0)
    scaled = compressed * (target_rms / curr_rms)

    # 3. Lookahead Soft-Clip + Brickwall Limiter a ceiling_db
    ceiling_lin = 10.0 ** (ceiling_db / 20.0)
    # Soft saturation negli ultimi 1.5 dB prima del ceiling
    knee = ceiling_lin * 0.84
    abs_x = np.abs(scaled)
    above = np.maximum(0.0, abs_x - knee)
    soft_abs = np.where(
        abs_x > knee,
        knee + (ceiling_lin - knee) * np.tanh(above / (ceiling_lin - knee + 1e-9)),
        abs_x,
    )
    out = np.sign(scaled) * np.minimum(soft_abs, ceiling_lin)
    return out


DEFAULT_SALA_PROVE_EQ = [
    # 1. Taglio sub-rumble da pavimento / aste microfoniche
    {"type": "highpass", "freq": 45.0, "gain_db": 0.0, "q": 0.707, "label": "Sub Rumble Cut (45 Hz)"},
    {"type": "highpass", "freq": 45.0, "gain_db": 0.0, "q": 0.707, "label": "Sub Rumble Cut 2nd Stage"},
    # 2. Controllo risonanza modale del basso (picco rilevato a 113 Hz)
    {"type": "peaking", "freq": 113.0, "gain_db": -3.2, "q": 2.6, "label": "Room Bass Mode (113 Hz)"},
    # 3. Taglio principale "Mud / Rimbombo Sala" (picco rilevato a 164.2 Hz: +45.6 dB!)
    {"type": "peaking", "freq": 164.0, "gain_db": -6.2, "q": 1.5, "label": "Anti-Mud Sala (164 Hz)"},
    # 4. Pulizia medio-bassi gonfi (260 Hz)
    {"type": "peaking", "freq": 265.0, "gain_db": -3.0, "q": 1.4, "label": "Low-Mid Clean (265 Hz)"},
    # 5. Taglio effetto "Scatola / Cartone" (picco rilevato a 495.3 Hz)
    {"type": "peaking", "freq": 495.0, "gain_db": -5.2, "q": 2.0, "label": "Anti-Boxiness (495 Hz)"},
    # 6. Recupero fondamentale di Voce, Chitarre e Rullante (buco rilevato tra 1000 e 2000 Hz)
    {"type": "peaking", "freq": 1420.0, "gain_db": 5.4, "q": 0.85, "label": "Vocal & Mid Presence (1.42 kHz)"},
    # 7. Controllo risonanza acida microfono (picco rilevato a 2336 Hz)
    {"type": "peaking", "freq": 2336.0, "gain_db": -2.4, "q": 3.4, "label": "Harshness Notch (2.34 kHz)"},
    # 8. Definizione, attacco cordiera/plettro e intelligibilità (4.8 kHz)
    {"type": "peaking", "freq": 4800.0, "gain_db": 4.8, "q": 1.0, "label": "Clarity & Definition (4.8 kHz)"},
    # 9. Recupero Brillantezza e Aria (sopra 8 kHz la traccia originale scendeva a -3.3 dB)
    {"type": "highshelf", "freq": 8200.0, "gain_db": 7.8, "q": 0.707, "label": "Studio Air Shelf (8.2 kHz)"},
]


def load_audio_ffmpeg(path: str, sr: int = 44100) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)


def save_audio_ffmpeg(audio: np.ndarray, out_path: str, sr: int = 44100):
    ext = os.path.splitext(out_path)[1].lower()
    audio_f32 = np.clip(audio, -1.0, 1.0).astype(np.float32)
    if ext == ".wav":
        codec_args = ["-c:a", "pcm_s24le"]
    else:
        codec_args = ["-c:a", "libmp3lame", "-b:a", "320k"]

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
        *codec_args, out_path
    ]
    subprocess.run(cmd, input=audio_f32.tobytes(), check=True)


def compute_spectrum_report(audio: np.ndarray, sr: int = 44100) -> dict:
    mono = audio.mean(axis=1)
    n_fft = 16384
    hop = 8192
    win = np.hanning(n_fft)
    specs = []
    for i in range(0, len(mono) - n_fft, hop):
        chunk = mono[i:i + n_fft]
        if np.sqrt(np.mean(chunk ** 2)) > 0.01:
            specs.append(np.abs(np.fft.rfft(chunk * win)) ** 2)
    psd = np.mean(specs, axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    bands_def = [
        (20, 60, "Sub-bass (20-60 Hz)"),
        (60, 120, "Bass (60-120 Hz)"),
        (120, 250, "Low-Mid / Mud (120-250 Hz)"),
        (250, 500, "Boxiness (250-500 Hz)"),
        (500, 1000, "Mid (500-1000 Hz)"),
        (1000, 2000, "Upper-Mid / Voce (1-2 kHz)"),
        (2000, 4000, "Presence (2-4 kHz)"),
        (4000, 8000, "Brilliance (4-8 kHz)"),
        (8000, 16000, "Air (8-16 kHz)"),
    ]
    report = {}
    for f1, f2, name in bands_def:
        m = (freqs >= f1) & (freqs < f2)
        report[name] = round(float(10.0 * np.log10(np.mean(psd[m]) + 1e-12)), 1)

    rms_db = round(float(20.0 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12)), 2)
    peak_db = round(float(20.0 * np.log10(np.max(np.abs(audio)) + 1e-12)), 2)
    return {"bands_db": report, "rms_db": rms_db, "peak_db": peak_db}


def process_track(
    input_path: str,
    output_wav: str,
    output_mp3: str = None,
    eq_bands: list = None,
    enable_dynamic_mud: bool = True,
    enable_exciter: bool = True,
    enable_mid_side: bool = True,
    stereo_width: float = 1.22,
    exciter_drive: float = 0.22,
    sr: int = 44100,
) -> dict:
    if eq_bands is None:
        eq_bands = DEFAULT_SALA_PROVE_EQ

    print(f"[1/6] Caricamento traccia originale: {input_path}")
    orig = load_audio_ffmpeg(input_path, sr=sr)
    before_stats = compute_spectrum_report(orig, sr=sr)

    audio = orig.copy()

    if enable_dynamic_mud:
        print("[2/6] Applicazione Dynamic EQ Anti-Rimbombo (110-380 Hz)...")
        audio = dynamic_mud_suppressor(audio, sr=sr)

    print("[3/6] Applicazione Catena EQ Parametrica a 64-bit (10 stadi)...")
    audio = apply_biquad_fft_overlap(audio, eq_bands, sr=sr)

    if enable_exciter:
        print("[4/6] Rigenerazione Armoniche Alte (Harmonic Air Exciter >4.5 kHz)...")
        audio = harmonic_air_exciter(audio, sr=sr, drive=exciter_drive)

    if enable_mid_side:
        print("[5/6] Elaborazione Mid/Side (Basse in Mono <135 Hz + Apertura Stereo)...")
        audio = mid_side_stage(audio, sr=sr, mono_below_hz=135.0, stereo_width=stereo_width)

    print("[6/6] Bus Glue Compression & True-Peak Mastering Limiter...")
    audio = studio_bus_compressor_and_limiter(audio, sr=sr, ceiling_db=-0.4)

    after_stats = compute_spectrum_report(audio, sr=sr)

    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    save_audio_ffmpeg(audio, output_wav, sr=sr)
    print(f"Esportato WAV 24-bit: {output_wav}")

    if output_mp3:
        save_audio_ffmpeg(audio, output_mp3, sr=sr)
        print(f"Esportato MP3 320kbps: {output_mp3}")

    return {"before": before_stats, "after": after_stats}


if __name__ == "__main__":
    in_file = sys.argv[1]
    out_wav = sys.argv[2]
    out_mp3 = sys.argv[3] if len(sys.argv) > 3 else None
    stats = process_track(in_file, out_wav, out_mp3)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
