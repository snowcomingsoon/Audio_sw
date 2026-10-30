#!/usr/bin/env python3
"""
Pipeline AI Multi-Stem & Mastering Professionale basata sulle migliori repository GitHub:
1. facebookresearch/demucs (htdemucs_ft) -> Separazione AI ad alta precisione in 4 Stems (Voce, Batteria, Basso, Chitarre/Altro)
2. timsainb/noisereduce                 -> Riduzione spettrale del fruscio/rientri di sala su Voce e Chitarre
3. spotify/pedalboard                   -> Channel Strip C++ dedicato per ogni singolo strumento + Master Bus
4. csteinmetz1/pyloudnorm               -> Normalizzazione Loudness ITU-R BS.1770-4 a -11.5 LUFS da standard streaming
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_MININTERVAL", "0.15")

import sys
import json
import time
import logging
import warnings
import subprocess

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from demucs.pretrained import get_model
from demucs.apply import apply_model
import noisereduce as nr
import pyloudnorm as pyln
from pedalboard import (
    Pedalboard,
    NoiseGate,
    HighpassFilter,
    LowpassFilter,
    PeakFilter,
    HighShelfFilter,
    Compressor,
    Reverb,
    Gain,
    Limiter,
)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def emit_progress(pct: float, stage_idx: int, stage: str, detail: str = "", gpu: str = ""):
    print(
        f"@@AI_PROGRESS@@ {json.dumps({'pct': round(pct, 1), 'stage_idx': stage_idx, 'stage': stage, 'detail': detail, 'gpu': gpu}, ensure_ascii=False)}",
        flush=True,
    )


def detect_gpu_label() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.cuda.get_device_name(0) if device == "cuda" else "CPU Multi-Core"


def load_audio_stereo_44k(path: str, sr: int = 44100) -> np.ndarray:
    """Carica qualsiasi formato audio tramite FFmpeg restituendo float32 shape (2, N)."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", path,
        "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"
    ]
    raw = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ).stdout
    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)
    return arr.T.copy()  # shape (2, samples)


def save_audio_ffmpeg(audio_ch_first: np.ndarray, out_path: str, sr: int = 44100):
    """Salva array shape (2, N) in WAV 24-bit o MP3 320 kbps tramite FFmpeg."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    interleaved = np.clip(audio_ch_first.T, -1.0, 1.0).astype(np.float32)
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".wav":
        codec = ["-c:a", "pcm_s24le"]
    else:
        codec = ["-c:a", "libmp3lame", "-b:a", "320k"]

    cmd = [
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
        *codec, out_path
    ]
    subprocess.run(
        cmd,
        input=interleaved.tobytes(),
        stdout=subprocess.DEVNULL,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def separate_with_demucs(audio_2xN: np.ndarray, model_name: str = "htdemucs_ft") -> dict:
    """
    Esegue facebookresearch/demucs (Hybrid Transformer Fine-Tuned) su GPU CUDA (se disponibile) o CPU.
    Restituisce dizionario { 'vocals': (2,N), 'drums': (2,N), 'bass': (2,N), 'other': (2,N) }.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_label = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        gpu_label = "Apple Silicon GPU (Metal M1/M2/M3)"
    else:
        device = "cpu"
        gpu_label = "CPU Multi-Core"
    emit_progress(
        pct=6.0,
        stage_idx=1,
        stage="Caricamento Modello AI",
        detail=f"Caricamento {model_name} su {gpu_label}...",
        gpu=gpu_label,
    )
    print(f"[Demucs AI] Caricamento modello '{model_name}' su device={device.upper()} ({gpu_label})...", flush=True)

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    model = get_model(model_name)
    model.to(device)
    model.eval()

    wav = torch.from_numpy(audio_2xN).to(torch.float32)
    ref = wav.mean(0)
    ref_mean = ref.mean()
    ref_std = ref.std() + 1e-8
    wav_norm = (wav - ref_mean) / ref_std

    emit_progress(
        pct=12.0,
        stage_idx=2,
        stage="Separazione Reti Neurali (Demucs v4)",
        detail="Avvio separazione in 4 tracce (Voce, Batteria, Basso, Chitarre)...",
        gpu=gpu_label,
    )

    t0 = time.time()
    with torch.no_grad():
        sources = apply_model(
            model,
            wav_norm[None].to(device),
            device=device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True,
        )[0]
    sources = sources.cpu() * ref_std + ref_mean
    elapsed = time.time() - t0

    emit_progress(
        pct=76.0,
        stage_idx=2,
        stage="Separazione Reti Neurali Completata",
        detail=f"4 Stems estratti in {elapsed:.1f}s",
        gpu=gpu_label,
    )
    print(f"[Demucs AI] Separazione 4-Stems completata in {elapsed:.1f}s!", flush=True)

    stems = {}
    for idx, name in enumerate(model.sources):
        stems[name] = sources[idx].numpy().astype(np.float32)
    return stems


def process_vocals_stem(vocals: np.ndarray, sr: int = 44100) -> np.ndarray:
    """
    Catena Studio Voce (noisereduce + Spotify Pedalboard):
    - Pulizia spettrale leggera del fruscio/rientri di sala
    - Taglio netto sotto 115 Hz (elimina rimbombo cassa/basso dal microfono voce)
    - Scavo effetto scatola (210 Hz e 490 Hz)
    - Recupero corpo, intelligibilità e Aria da studio (1.65 kHz, 4.5 kHz, 8.5 kHz)
    - Compressore vocale + velo (5%) di Plate Reverb corto per naturalezza
    """
    print("[Stem 1/4 - VOCE] Spectral De-Noise (noisereduce) + Pedalboard Vocal Strip...", flush=True)
    cleaned = nr.reduce_noise(
        y=vocals,
        sr=sr,
        stationary=True,
        prop_decrease=0.42,
        n_fft=2048,
    ).astype(np.float32)

    board = Pedalboard([
        NoiseGate(threshold_db=-44.0, ratio=2.2, attack_ms=4.0, release_ms=180.0),
        HighpassFilter(cutoff_frequency_hz=115.0),
        PeakFilter(cutoff_frequency_hz=210.0, gain_db=-3.5, q=1.6),
        PeakFilter(cutoff_frequency_hz=490.0, gain_db=-4.6, q=2.1),
        PeakFilter(cutoff_frequency_hz=1650.0, gain_db=4.0, q=0.9),
        PeakFilter(cutoff_frequency_hz=2350.0, gain_db=-2.2, q=3.2),
        PeakFilter(cutoff_frequency_hz=4500.0, gain_db=4.4, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=8500.0, gain_db=5.8, q=0.707),
        Compressor(threshold_db=-22.0, ratio=3.2, attack_ms=12.0, release_ms=110.0),
        Reverb(room_size=0.24, damping=0.70, wet_level=0.04, dry_level=0.98, width=0.75),
        Gain(gain_db=1.0),
    ])
    return board(cleaned, sr)


def process_drums_stem(drums: np.ndarray, sr: int = 44100) -> np.ndarray:
    """
    Catena Studio Batteria (Spotify Pedalboard):
    - Punch cassa a 62 Hz, scavo profondo del rimbombo sala su fusti (168 Hz) e inscatolamento (485 Hz)
    - Attacco rullante (1.8 kHz / 5.2 kHz) e brillantezza piatti (8.5 kHz)
    - Slow-attack Bus Compressor per preservare il transiente
    """
    print("[Stem 2/4 - BATTERIA] Pedalboard Drum Punch & Anti-Box Strip...", flush=True)
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=34.0),
        PeakFilter(cutoff_frequency_hz=62.0, gain_db=3.2, q=1.4),
        PeakFilter(cutoff_frequency_hz=168.0, gain_db=-5.8, q=1.6),
        PeakFilter(cutoff_frequency_hz=485.0, gain_db=-5.0, q=2.0),
        PeakFilter(cutoff_frequency_hz=1800.0, gain_db=2.8, q=1.1),
        PeakFilter(cutoff_frequency_hz=5200.0, gain_db=3.8, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=8500.0, gain_db=6.5, q=0.707),
        Compressor(threshold_db=-19.0, ratio=2.8, attack_ms=25.0, release_ms=90.0),
        Gain(gain_db=1.2),
    ])
    return board(drums, sr)


def process_bass_stem(bass: np.ndarray, sr: int = 44100) -> np.ndarray:
    """
    Catena Studio Basso (Mono Sub + Spotify Pedalboard):
    - Allineamento rigoroso in Mono per compattezza con la cassa
    - Eliminazione risonanze modali della sala a 113 Hz e 164 Hz (-6.8 dB)
    - Definizione corde a 780 Hz + compressione livellatrice 4:1
    """
    print("[Stem 3/4 - BASSO] Mono Alignment + Anti-Resonance (113/164 Hz) + Leveling...", flush=True)
    mono = 0.5 * (bass[0] + bass[1])
    bass_centered = np.stack([mono, mono], axis=0).astype(np.float32)

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=38.0),
        PeakFilter(cutoff_frequency_hz=78.0, gain_db=2.5, q=1.3),
        PeakFilter(cutoff_frequency_hz=113.0, gain_db=-4.2, q=2.8),
        PeakFilter(cutoff_frequency_hz=164.0, gain_db=-6.8, q=1.7),
        PeakFilter(cutoff_frequency_hz=310.0, gain_db=-3.0, q=1.5),
        PeakFilter(cutoff_frequency_hz=780.0, gain_db=2.8, q=1.2),
        LowpassFilter(cutoff_frequency_hz=4200.0),
        Compressor(threshold_db=-20.0, ratio=4.0, attack_ms=18.0, release_ms=120.0),
        Gain(gain_db=1.0),
    ])
    return board(bass_centered, sr)


def process_other_stem(other: np.ndarray, sr: int = 44100) -> np.ndarray:
    """
    Catena Studio Chitarre / Tastiere (noisereduce + Spotify Pedalboard + Mid/Side Widening):
    - Taglio High-Pass a 105 Hz per lasciare spazio totale a Cassa e Basso
    - Rimozione mud (165 Hz) e scatola (495 Hz)
    - Corpo e presenza (1.4 kHz e 4.2 kHz) + allargamento immagine Stereo (1.30x) per aprire il mix
    """
    print("[Stem 4/4 - CHITARRE/ALTRO] De-Buzz + Guitar EQ + Mid/Side Stereo Widening...", flush=True)
    cleaned = nr.reduce_noise(
        y=other,
        sr=sr,
        stationary=True,
        prop_decrease=0.28,
        n_fft=2048,
    ).astype(np.float32)

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=105.0),
        PeakFilter(cutoff_frequency_hz=165.0, gain_db=-5.5, q=1.6),
        PeakFilter(cutoff_frequency_hz=495.0, gain_db=-4.8, q=2.0),
        PeakFilter(cutoff_frequency_hz=1400.0, gain_db=4.2, q=0.9),
        PeakFilter(cutoff_frequency_hz=2340.0, gain_db=-2.5, q=3.5),
        PeakFilter(cutoff_frequency_hz=4200.0, gain_db=3.8, q=1.1),
        HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=5.2, q=0.707),
        Compressor(threshold_db=-20.0, ratio=2.4, attack_ms=20.0, release_ms=130.0),
        Gain(gain_db=1.2),
    ])
    eq_out = board(cleaned, sr)

    # Apertura Stereo Mid/Side (lascia il centro libero per la Voce!)
    mid = 0.5 * (eq_out[0] + eq_out[1])
    side = 0.5 * (eq_out[0] - eq_out[1]) * 1.32
    return np.stack([mid + side, mid - side], axis=0).astype(np.float32)


def master_bus_finalize(mix: np.ndarray, sr: int = 44100, target_lufs: float = -11.2) -> tuple[np.ndarray, dict]:
    """
    Master Bus finale con Spotify Pedalboard Glue Compressor + pyloudnorm (ITU-R BS.1770-4) + Soft-Knee Ceiling -0.5 dBTP.
    """
    print(f"[MASTER BUS] Bus Glue Compression + Calibrazione Loudness ITU-R BS.1770-4 ({target_lufs} LUFS)...", flush=True)
    peak_in = np.max(np.abs(mix)) + 1e-8
    if peak_in > 0.80:
        mix = mix * (0.80 / peak_in)

    glue_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30.0),
        PeakFilter(cutoff_frequency_hz=162.0, gain_db=-1.5, q=1.8),
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=1.8, q=0.707),
        Compressor(threshold_db=-17.5, ratio=2.0, attack_ms=30.0, release_ms=140.0),
    ])
    glued = glue_board(mix, sr).astype(np.float64)

    # Calibrazione esatta LUFS con pyloudnorm + Soft-Knee True-Peak Limiter a -0.50 dBTP (zero clipping garantito)
    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(glued.T)
    gain_lin = 10.0 ** (np.clip(target_lufs - current_lufs, -8.0, 10.0) / 20.0)
    scaled = glued * gain_lin

    ceiling = 10.0 ** (-0.50 / 20.0)  # ~0.94406 (-0.50 dBTP)
    knee = ceiling * 0.82
    abs_x = np.abs(scaled)
    above = np.maximum(0.0, abs_x - knee)
    span = ceiling - knee
    limited_abs = np.where(
        abs_x > knee,
        knee + span * np.tanh(above / span),
        abs_x,
    )
    master = (np.sign(scaled) * np.minimum(limited_abs, ceiling)).astype(np.float32)

    final_lufs = float(meter.integrated_loudness(master.T.astype(np.float64)))
    final_peak_db = float(20.0 * np.log10(np.max(np.abs(master)) + 1e-12))
    final_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(master ** 2)) + 1e-12))

    return master, {
        "lufs": round(final_lufs, 2),
        "rms_db": round(final_rms_db, 2),
        "peak_db": round(final_peak_db, 2),
    }


def run_full_pipeline(input_audio: str, out_dir: str):
    sr = 44100
    os.makedirs(out_dir, exist_ok=True)
    stems_dir = os.path.join(out_dir, "stems_studio")
    raw_cache_dir = os.path.join(out_dir, "stems_raw")
    os.makedirs(stems_dir, exist_ok=True)
    os.makedirs(raw_cache_dir, exist_ok=True)

    emit_progress(
        pct=2.0,
        stage_idx=0,
        stage="Decodifica Audio",
        detail="Lettura e conversione stereo 44.1 kHz...",
    )
    orig = load_audio_stereo_44k(input_audio, sr=sr)
    meter = pyln.Meter(sr)
    orig_lufs = round(float(meter.integrated_loudness(orig.T.astype(np.float64))), 2)

    gpu_label = detect_gpu_label()

    # 1. Separazione AI con Demucs Fine-Tuned (con cache WAV 24-bit per remix istantanei)
    expected_raw = ["vocals", "drums", "bass", "other"]
    if all(os.path.exists(os.path.join(raw_cache_dir, f"{k}.wav")) for k in expected_raw):
        emit_progress(
            pct=6.0,
            stage_idx=1,
            stage="Caricamento Modello AI",
            detail=f"Cache Stems rilevata ({gpu_label})...",
            gpu=gpu_label,
        )
        print("[Demucs AI] Caricamento stems raw dalla cache...", flush=True)
        raw_stems = {k: load_audio_stereo_44k(os.path.join(raw_cache_dir, f"{k}.wav"), sr=sr) for k in expected_raw}
        emit_progress(
            pct=76.0,
            stage_idx=2,
            stage="Separazione Reti Neurali Completata",
            detail="4 Stems caricati istantaneamente dalla cache WAV 24-bit",
            gpu=gpu_label,
        )
    else:
        raw_stems = separate_with_demucs(orig, model_name="htdemucs_ft")
        for k, v in raw_stems.items():
            save_audio_ffmpeg(v, os.path.join(raw_cache_dir, f"{k}.wav"), sr=sr)

    # 2. Equalizzazione e trattamento individuale di ciascuno strumento
    emit_progress(
        pct=78.0,
        stage_idx=3,
        stage="Restauro & Channel Strip Stems",
        detail="1/4 Voce — De-Noise Spettrale + Vocal Strip",
        gpu=gpu_label,
    )
    vocals_eq = process_vocals_stem(raw_stems["vocals"], sr=sr)

    emit_progress(
        pct=82.0,
        stage_idx=3,
        stage="Restauro & Channel Strip Stems",
        detail="2/4 Batteria — Transient Punch & Anti-Box",
        gpu=gpu_label,
    )
    drums_eq = process_drums_stem(raw_stems["drums"], sr=sr)

    emit_progress(
        pct=86.0,
        stage_idx=3,
        stage="Restauro & Channel Strip Stems",
        detail="3/4 Basso — Allineamento Mono & Anti-Risonanza",
        gpu=gpu_label,
    )
    bass_eq = process_bass_stem(raw_stems["bass"], sr=sr)

    emit_progress(
        pct=89.0,
        stage_idx=3,
        stage="Restauro & Channel Strip Stems",
        detail="4/4 Chitarre & Altro — EQ Presenza & Mid/Side Stereo",
        gpu=gpu_label,
    )
    other_eq = process_other_stem(raw_stems["other"], sr=sr)

    # Salva gli stems separati equalizzati (WAV + MP3 per l'app Rust, Web App e Logic Pro X)
    emit_progress(
        pct=92.0,
        stage_idx=4,
        stage="Mastering & Esportazione 24-Bit",
        detail="Salvataggio 4 Stems WAV 24-bit e MP3 320 kbps...",
        gpu=gpu_label,
    )
    stem_exports = {
        "01_Voce_StudioEQ": vocals_eq,
        "02_Batteria_StudioEQ": drums_eq,
        "03_Basso_StudioEQ": bass_eq,
        "04_Chitarre_Altro_StudioEQ": other_eq,
    }
    for name, audio_arr in stem_exports.items():
        save_audio_ffmpeg(audio_arr, os.path.join(stems_dir, f"{name}.wav"), sr=sr)
        save_audio_ffmpeg(audio_arr, os.path.join(stems_dir, f"{name}.mp3"), sr=sr)

    # 3. Somma Multi-Traccia e Master Bus finale
    summed = vocals_eq + drums_eq + bass_eq + other_eq
    emit_progress(
        pct=96.0,
        stage_idx=4,
        stage="Mastering & Esportazione 24-Bit",
        detail="Calibrazione Master ITU-R BS.1770-4 (-11.2 LUFS)...",
        gpu=gpu_label,
    )
    master_audio, master_metrics = master_bus_finalize(summed, sr=sr, target_lufs=-11.2)

    out_wav = os.path.join(out_dir, "Master_AI_Demucs_Pedalboard.wav")
    out_mp3 = os.path.join(out_dir, "Master_AI_Demucs_Pedalboard.mp3")
    save_audio_ffmpeg(master_audio, out_wav, sr=sr)
    save_audio_ffmpeg(master_audio, out_mp3, sr=sr)

    report = {
        "original_lufs": orig_lufs,
        "master_ai": master_metrics,
        "output_wav": out_wav,
        "output_mp3": out_mp3,
        "stems_dir": stems_dir,
    }
    with open(os.path.join(out_dir, "ai_master_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    emit_progress(
        pct=100.0,
        stage_idx=5,
        stage="Completato",
        detail=f"Master pronto ({master_metrics['lufs']} LUFS, Peak {master_metrics['peak_db']} dBTP)",
        gpu=gpu_label,
    )

    print("\n=== COMPLETATO CON SUCCESSO ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_folder = sys.argv[2]
    run_full_pipeline(in_path, out_folder)
