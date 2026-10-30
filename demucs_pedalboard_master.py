#!/usr/bin/env python3
"""
Pipeline AI Multi-Stem & Mastering Professionale (StudioEQ Master Pro):
1. facebookresearch/demucs (htdemucs_ft) -> Separazione AI 4 Stems con cache SHA-256 per-traccia e
                                            supporto CUDA / ROCm / Apple MPS / AMD DirectML (con offload
                                            STFT complesso su CPU e fallback automatico CPU Multi-Core)
2. audio_modules/time_stretch_bpm        -> Rilevamento BPM multibanda + Time-Stretching sincrono zero-distorsione
                                            (WSOLA stereo phase-locked per Batteria + Identity Phase-Locked Vocoder)
3. audio_modules/vocal_autotune          -> Rilevamento Tonalità Krumhansl-Kessler + Auto-Tune vocale con
                                            preservazione formanti cepstrali (jurihock/stftpitchshift) e crossover 4.2 kHz
4. audio_modules/drum_replacer & bass    -> Rinforzo transiente Cassa/Rullante & Ri-sintesi/Sidechain Basso
5. timsainb/noisereduce + pedalboard     -> De-Noise spettrale + Channel Strip C++ dedicato + Master Bus
6. csteinmetz1/pyloudnorm                -> Normalizzazione Loudness ITU-R BS.1770-4 a -11.2 LUFS (-0.50 dBTP)
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_MININTERVAL", "0.15")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import argparse
import hashlib
import inspect
import json
import logging
import subprocess
import sys
import time
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

_CPU_THREADS = max(1, os.cpu_count() or 8)
try:
    torch.set_num_threads(_CPU_THREADS)
except Exception:
    pass

try:
    import torch_directml  # type: ignore

    _HAS_DIRECTML = True
except Exception:
    torch_directml = None
    _HAS_DIRECTML = False

from demucs.apply import apply_model
from demucs.pretrained import get_model
import noisereduce as nr
import pyloudnorm as pyln
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowpassFilter,
    NoiseGate,
    PeakFilter,
    Pedalboard,
    Reverb,
)

from audio_modules.time_stretch_bpm import detect_bpm, stretch_stems_to_bpm
from audio_modules.vocal_autotune import apply_vocal_autotune, detect_musical_key

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

_WMI_ADAPTER_CACHE: str | None = None
_LAST_EFFECTIVE_GPU_LABEL: str | None = None


def emit_progress(pct: float, stage_idx: int, stage: str, detail: str = "", gpu: str = ""):
    print(
        f"@@AI_PROGRESS@@ {json.dumps({'pct': round(pct, 1), 'stage_idx': stage_idx, 'stage': stage, 'detail': detail, 'gpu': gpu}, ensure_ascii=False)}",
        flush=True,
    )


def _query_windows_gpu_wmi() -> str:
    """Query Windows WMI/CIM video adapter name once (fast, non-blocking, safe fallback)."""
    global _WMI_ADAPTER_CACHE
    if _WMI_ADAPTER_CACHE is not None:
        return _WMI_ADAPTER_CACHE

    if os.name != "nt":
        _WMI_ADAPTER_CACHE = ""
        return _WMI_ADAPTER_CACHE

    try:
        res = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ' | '",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=1.8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        names = [p.strip() for p in (res.stdout or "").split("|") if p.strip()]
        # Prefer discrete AMD Radeon / NVIDIA / Intel Arc names over basic display adapters
        preferred = [
            n for n in names if any(k in n.lower() for k in ("radeon", "rx ", "geforce", "rtx", "gtx", "arc "))
        ]
        _WMI_ADAPTER_CACHE = preferred[0] if preferred else (names[0] if names else "")
    except Exception:
        _WMI_ADAPTER_CACHE = ""
    return _WMI_ADAPTER_CACHE


def _select_compute_backend() -> tuple[object, str, bool]:
    """
    Detect compute device in priority order:
    CUDA/ROCm -> Apple MPS -> DirectML (`torch_directml`) -> WMI/CPU Multi-Core.

    Returns:
        (torch_device, human_gpu_label, is_directml)
    """
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        if getattr(torch.version, "hip", None):
            return torch.device("cuda"), f"ROCm • {dev_name}", False
        return torch.device("cuda"), dev_name, False

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon GPU (Metal MPS)", False

    if _HAS_DIRECTML and torch_directml is not None:
        try:
            if torch_directml.is_available():
                dml_dev = torch_directml.device()
                try:
                    dml_name = str(torch_directml.device_name(0)).strip()
                except Exception:
                    dml_name = _query_windows_gpu_wmi() or "AMD/DirectX12 GPU"
                return dml_dev, f"{dml_name} (DirectML)", True
        except Exception:
            pass

    wmi_gpu = _query_windows_gpu_wmi()
    if wmi_gpu:
        return torch.device("cpu"), f"CPU Multi-Core ({_CPU_THREADS} Threads — {wmi_gpu})", False
    return torch.device("cpu"), f"CPU Multi-Core ({_CPU_THREADS} Threads)", False


def detect_gpu_label() -> str:
    if _LAST_EFFECTIVE_GPU_LABEL:
        return _LAST_EFFECTIVE_GPU_LABEL
    _, label, _ = _select_compute_backend()
    return label


def compute_track_hash(file_path: str) -> str:
    """
    Compute 16-hex-char SHA-256 fingerprint from input file size + first/middle/last 128 KB.
    Prevents wrong-song stem cache collisions while remaining instantaneous (< 1 ms).
    """
    chunk_size = 128 * 1024
    file_size = os.path.getsize(file_path)
    hasher = hashlib.sha256()
    hasher.update(f"size:{file_size}:".encode("utf-8"))

    with open(file_path, "rb") as f:
        if file_size <= chunk_size * 3:
            hasher.update(f.read())
        else:
            hasher.update(f.read(chunk_size))
            mid_offset = max(0, (file_size // 2) - (chunk_size // 2))
            f.seek(mid_offset)
            hasher.update(f.read(chunk_size))
            f.seek(max(0, file_size - chunk_size))
            hasher.update(f.read(chunk_size))

    return hasher.hexdigest()[:16]


_CACHED_FFMPEG_BIN: str | None = None


def _resolve_ffmpeg_bin() -> str:
    """Risolve il percorso reale di ffmpeg.exe (inclusi i pacchetti WinGet su Windows) e arricchisce PATH."""
    global _CACHED_FFMPEG_BIN
    if _CACHED_FFMPEG_BIN is not None:
        return _CACHED_FFMPEG_BIN

    import glob
    import shutil

    # Installazione autonoma: usa prima FFmpeg incluso accanto all'app.
    bundled_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin")
    bundled_ffmpeg = os.path.join(bundled_bin, "ffmpeg.exe")
    if os.path.isfile(bundled_ffmpeg):
        os.environ["PATH"] = bundled_bin + os.pathsep + os.environ.get("PATH", "")
        _CACHED_FFMPEG_BIN = bundled_ffmpeg
        return bundled_ffmpeg

    if os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            patterns = [
                os.path.join(local_app, "Microsoft", "WinGet", "Packages", "*FFmpeg*", "**", "ffmpeg.exe"),
                os.path.join(local_app, "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
            ]
            for pat in patterns:
                matches = sorted(glob.glob(pat, recursive=True), reverse=True)
                for m in matches:
                    if os.path.isfile(m):
                        bin_dir = os.path.dirname(m)
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                        _CACHED_FFMPEG_BIN = m
                        return m

    found = shutil.which("ffmpeg")
    _CACHED_FFMPEG_BIN = found if found else "ffmpeg"
    return _CACHED_FFMPEG_BIN


def load_audio_stereo_44k(path: str, sr: int = 44100) -> np.ndarray:
    """Carica qualsiasi formato audio tramite FFmpeg restituendo float32 shape (2, N)."""
    ffmpeg_bin = _resolve_ffmpeg_bin()
    cmd = [
        ffmpeg_bin, "-nostdin", "-v", "error", "-i", path,
        "-f", "f32le", "-ac", "2", "-ar", str(sr), "-",
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
    ffmpeg_bin = _resolve_ffmpeg_bin()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    interleaved = np.clip(audio_ch_first.T, -1.0, 1.0).astype(np.float32)
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".wav":
        codec = ["-c:a", "pcm_s24le"]
    else:
        codec = ["-c:a", "libmp3lame", "-b:a", "320k"]

    cmd = [
        ffmpeg_bin, "-nostdin", "-y", "-v", "error",
        "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
        *codec, out_path,
    ]
    subprocess.run(
        cmd,
        input=interleaved.tobytes(),
        stdout=subprocess.DEVNULL,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _patch_htdemucs_spectro_for_directml(model: object, dml_device: object) -> list[tuple[object, object, object]]:
    """
    Offload complex64 `_spectro` / `_ispectro` (`torch.stft` / `torch.istft`) onto CPU while
    keeping heavy Conv/Transformer blocks on DirectML (facebookresearch/demucs DirectML pattern).
    Returns saved method handles for clean restoration if fallback to CPU is needed.
    """
    saved: list[tuple[object, object, object]] = []
    sub_models = getattr(model, "models", [model])
    for sub in sub_models:
        orig_s = getattr(sub, "_spectro", None)
        orig_is = getattr(sub, "_ispectro", None)
        saved.append((sub, orig_s, orig_is))

        if callable(orig_s):
            def _make_s(fn):
                def _spectro_cpu(x, *args, **kwargs):
                    return fn(x.to("cpu"), *args, **kwargs).to(dml_device)
                return _spectro_cpu

            sub._spectro = _make_s(orig_s)

        if callable(orig_is):
            def _make_is(fn):
                def _ispectro_cpu(z, *args, **kwargs):
                    return fn(z.to("cpu"), *args, **kwargs).to(dml_device)
                return _ispectro_cpu

            sub._ispectro = _make_is(orig_is)

    return saved


def _restore_htdemucs_spectro(saved: list[tuple[object, object, object]]) -> None:
    for sub, orig_s, orig_is in saved:
        if orig_s is not None:
            sub._spectro = orig_s
        if orig_is is not None:
            sub._ispectro = orig_is


def separate_with_demucs(audio_2xN: np.ndarray, model_name: str = "htdemucs_ft") -> dict[str, np.ndarray]:
    """
    Esegue facebookresearch/demucs (Hybrid Transformer Fine-Tuned) su CUDA/ROCm, MPS,
    DirectML (AMD Radeon RX 9070 / DirectX 12 con fallback automatico CPU Multi-Core) o CPU.
    Restituisce dizionario { 'vocals': (2,N), 'drums': (2,N), 'bass': (2,N), 'other': (2,N) }.
    """
    global _LAST_EFFECTIVE_GPU_LABEL
    torch.set_num_threads(_CPU_THREADS)

    device, gpu_label, is_dml = _select_compute_backend()
    _LAST_EFFECTIVE_GPU_LABEL = gpu_label

    emit_progress(
        pct=6.0,
        stage_idx=1,
        stage="Caricamento Modello AI",
        detail=f"Caricamento {model_name} su {gpu_label}...",
        gpu=gpu_label,
    )
    print(f"[Demucs AI] Caricamento modello '{model_name}' su {gpu_label}...", flush=True)

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    model = get_model(model_name)
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
    sources = None

    if is_dml:
        saved_hooks: list[tuple[object, object, object]] = []
        try:
            saved_hooks = _patch_htdemucs_spectro_for_directml(model, device)
            model.to(device)
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
        except Exception as dml_err:
            _restore_htdemucs_spectro(saved_hooks)
            wmi_gpu = _query_windows_gpu_wmi()
            cpu_fallback_label = (
                f"CPU Multi-Core ({_CPU_THREADS} Threads — Fallback da DirectML {wmi_gpu})".strip()
                if wmi_gpu
                else f"CPU Multi-Core ({_CPU_THREADS} Threads)"
            )
            _LAST_EFFECTIVE_GPU_LABEL = cpu_fallback_label
            gpu_label = cpu_fallback_label
            print(
                f"[Demucs AI] Nota: DirectML non supporta tutte le op STFT/tensor complesse di {model_name} "
                f"su questa build PyTorch ({type(dml_err).__name__}). Passaggio automatico a {cpu_fallback_label}...",
                flush=True,
            )
            emit_progress(
                pct=14.0,
                stage_idx=2,
                stage="Separazione Reti Neurali (Demucs v4)",
                detail=f"Fallback automatico su {cpu_fallback_label}...",
                gpu=cpu_fallback_label,
            )
            device = torch.device("cpu")
            torch.set_num_threads(_CPU_THREADS)
            model.to(device)
            with torch.no_grad():
                sources = apply_model(
                    model,
                    wav_norm[None],
                    device=device,
                    shifts=1,
                    split=True,
                    overlap=0.25,
                    progress=True,
                )[0]
    else:
        model.to(device)
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
        pct=72.0,
        stage_idx=2,
        stage="Separazione Reti Neurali Completata",
        detail=f"4 Stems estratti in {elapsed:.1f}s",
        gpu=gpu_label,
    )
    print(f"[Demucs AI] Separazione 4-Stems completata in {elapsed:.1f}s!", flush=True)

    stems: dict[str, np.ndarray] = {}
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
    - Compressore vocale + velo (4%) di Plate Reverb corto per naturalezza
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


def process_drums_stem(drums: np.ndarray, sr: int = 44100, drum_mode: str = "off") -> np.ndarray:
    """
    Catena Studio Batteria (Spotify Pedalboard):
    - Quando `drum_mode != "off"` ("natural" / "punch"), `drums` proviene da `enhance_and_reinforce_drums`
      con sample da studio Cassa/Rullante già scolpiti e anti-mud dinamico già applicato: esegue un
      Gentle Studio Drum Bus Polish equilibrato senza sovra-enfatizzare i piatti né comprimere due volte.
    - Quando `drum_mode == "off"`, applica la catena completa di recupero per batteria grezza da sala prove.
    """
    enhanced_active = (drum_mode or "off").strip().lower() != "off"
    if enhanced_active:
        print(
            f"[Stem 2/4 - BATTERIA] Gentle Studio Drum Bus Polish (drum_mode={drum_mode})...",
            flush=True,
        )
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=32.0),
            PeakFilter(cutoff_frequency_hz=168.0, gain_db=-1.5, q=1.8),
            PeakFilter(cutoff_frequency_hz=420.0, gain_db=-2.2, q=1.8),
            PeakFilter(cutoff_frequency_hz=4800.0, gain_db=1.2, q=1.0),
            HighShelfFilter(cutoff_frequency_hz=9000.0, gain_db=2.4, q=0.707),
            Compressor(threshold_db=-16.0, ratio=2.0, attack_ms=28.0, release_ms=95.0),
            Gain(gain_db=0.5),
        ])
    else:
        print(
            f"[Stem 2/4 - BATTERIA] Pedalboard Full Drum Recovery & Anti-Box Strip (drum_mode={drum_mode})...",
            flush=True,
        )
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
    return board(drums.astype(np.float32), sr)


def process_bass_stem(bass: np.ndarray, sr: int = 44100, bass_mode: str = "off") -> np.ndarray:
    """
    Catena Studio Basso (Mono Sub + Spotify Pedalboard):
    - Quando `bass_mode != "off"`, applica interventi più gentili sulle basse/medio-basse
      (62–78 Hz / 164–168 Hz) già scolpite da `bass_enhancer`.
    """
    print(f"[Stem 3/4 - BASSO] Mono Alignment + Anti-Resonance + Leveling (bass_mode={bass_mode})...", flush=True)
    mono = 0.5 * (bass[0] + bass[1])
    bass_centered = np.stack([mono, mono], axis=0).astype(np.float32)

    enhanced_active = (bass_mode or "off").strip().lower() != "off"
    gain_low = 1.2 if enhanced_active else 2.5
    gain_113 = -2.6 if enhanced_active else -4.2
    gain_168 = -3.4 if enhanced_active else -6.8

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=38.0),
        PeakFilter(cutoff_frequency_hz=62.0, gain_db=gain_low * 0.6, q=1.3),
        PeakFilter(cutoff_frequency_hz=78.0, gain_db=gain_low, q=1.3),
        PeakFilter(cutoff_frequency_hz=113.0, gain_db=gain_113, q=2.8),
        PeakFilter(cutoff_frequency_hz=168.0, gain_db=gain_168, q=1.7),
        PeakFilter(cutoff_frequency_hz=310.0, gain_db=-2.6 if enhanced_active else -3.0, q=1.5),
        PeakFilter(cutoff_frequency_hz=780.0, gain_db=2.4 if enhanced_active else 2.8, q=1.2),
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
    - Corpo e presenza (1.4 kHz e 4.2 kHz) + allargamento immagine Stereo (1.32x) per aprire il mix
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
    Master Bus finale con Spotify Pedalboard Glue Compressor + pyloudnorm (ITU-R BS.1770-4) + Soft-Knee Ceiling -0.50 dBTP.
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
    if np.isfinite(current_lufs):
        gain_lin = 10.0 ** (np.clip(target_lufs - current_lufs, -8.0, 10.0) / 20.0)
    else:
        gain_lin = 1.0
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
    if not np.isfinite(final_lufs):
        final_lufs = target_lufs
    final_peak_db = float(20.0 * np.log10(np.max(np.abs(master)) + 1e-12))
    final_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(master ** 2)) + 1e-12))

    return master, {
        "lufs": round(final_lufs, 2),
        "rms_db": round(final_rms_db, 2),
        "peak_db": round(final_peak_db, 2),
    }


def _invoke_drum_enhancer(
    drums_2xN: np.ndarray,
    sr: int,
    drum_mode: str,
    kick_blend: float | None,
    snare_blend: float | None,
) -> tuple[np.ndarray, dict]:
    """Call `audio_modules.drum_replacer.enhance_and_reinforce_drums` adhering strictly to contract."""
    from audio_modules.drum_replacer import enhance_and_reinforce_drums

    sig = inspect.signature(enhance_and_reinforce_drums)
    kwargs: dict[str, object] = {"sr": sr, "drum_mode": drum_mode}

    if "kick_blend" in sig.parameters:
        if kick_blend is not None:
            kwargs["kick_blend"] = float(kick_blend)
        elif sig.parameters["kick_blend"].default is inspect.Parameter.empty or sig.parameters["kick_blend"].default is None:
            kwargs["kick_blend"] = None

    if "snare_blend" in sig.parameters:
        if snare_blend is not None:
            kwargs["snare_blend"] = float(snare_blend)
        elif sig.parameters["snare_blend"].default is inspect.Parameter.empty or sig.parameters["snare_blend"].default is None:
            kwargs["snare_blend"] = None

    drums_clean, drum_stats = enhance_and_reinforce_drums(drums_2xN, **kwargs)
    return np.asarray(drums_clean, dtype=np.float32), (drum_stats if isinstance(drum_stats, dict) else {})


def _invoke_bass_enhancer(
    bass_2xN: np.ndarray,
    drums_for_sc: np.ndarray,
    sr: int,
    bass_mode: str,
    kick_positions: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Call `audio_modules.bass_enhancer.enhance_and_resynthesize_bass` adhering strictly to contract."""
    from audio_modules.bass_enhancer import enhance_and_resynthesize_bass

    sig = inspect.signature(enhance_and_resynthesize_bass)
    kwargs: dict[str, object] = {
        "drums_2xN": drums_for_sc,
        "sr": sr,
        "bass_mode": bass_mode,
    }
    if "kick_positions" in sig.parameters:
        kwargs["kick_positions"] = kick_positions

    bass_clean, bass_stats = enhance_and_resynthesize_bass(
        bass_2xN,
        **kwargs,
    )
    return np.asarray(bass_clean, dtype=np.float32), (bass_stats if isinstance(bass_stats, dict) else {})


def _align_stems_length(*stems_2xN: np.ndarray) -> list[np.ndarray]:
    """Ensure all 4 processed stems share exact sample count before multi-track summation."""
    target_n = min(s.shape[1] for s in stems_2xN)
    return [s[:, :target_n].astype(np.float32, copy=False) for s in stems_2xN]


def run_full_pipeline(
    input_audio: str,
    out_dir: str,
    autotune_mode: str = "off",
    drum_mode: str = "natural",
    bass_mode: str = "natural",
    target_bpm: float | None = None,
    force_reseparate: bool = False,
    kick_blend: float | None = None,
    snare_blend: float | None = None,
):
    sr = 44100
    autotune_mode = (autotune_mode or "off").strip().lower()
    drum_mode = (drum_mode or "natural").strip().lower()
    bass_mode = (bass_mode or "natural").strip().lower()
    if target_bpm is not None and float(target_bpm) <= 0.0:
        target_bpm = None

    os.makedirs(out_dir, exist_ok=True)
    stems_dir = os.path.join(out_dir, "stems_studio")
    os.makedirs(stems_dir, exist_ok=True)

    gpu_label = detect_gpu_label()

    emit_progress(
        pct=2.0,
        stage_idx=0,
        stage="Decodifica Audio",
        detail="Lettura traccia e calcolo impronta SHA-256...",
        gpu=gpu_label,
    )
    track_hash = compute_track_hash(input_audio)
    raw_cache_dir = os.path.join(out_dir, "stems_raw", track_hash)
    os.makedirs(raw_cache_dir, exist_ok=True)

    orig = load_audio_stereo_44k(input_audio, sr=sr)
    meter = pyln.Meter(sr)
    raw_orig_lufs = float(meter.integrated_loudness(orig.T.astype(np.float64)))
    orig_lufs = round(raw_orig_lufs, 2) if np.isfinite(raw_orig_lufs) else -24.0

    # 1. Separazione AI con Demucs Fine-Tuned + Cache per-traccia SHA-256 (stems_raw/<track_hash>)
    expected_raw = ["vocals", "drums", "bass", "other"]
    cached_paths = {k: os.path.join(raw_cache_dir, f"{k}.wav") for k in expected_raw}

    if (not force_reseparate) and all(os.path.exists(p) for p in cached_paths.values()):
        cache_hit = True
        emit_progress(
            pct=8.0,
            stage_idx=1,
            stage="Cache Stems SHA-256",
            detail=f"Cache rilevata [{track_hash[:8]}] — caricamento istantaneo...",
            gpu=gpu_label,
        )
        print(f"[Demucs AI] Cache SHA-256 rilevata ({track_hash}): caricamento istantaneo 4 stems raw...", flush=True)
        raw_stems = {k: load_audio_stereo_44k(p, sr=sr) for k, p in cached_paths.items()}
        emit_progress(
            pct=72.0,
            stage_idx=2,
            stage="Separazione Reti Neurali Completata",
            detail=f"4 Stems caricati istantaneamente dalla cache ({track_hash[:8]})",
            gpu=gpu_label,
        )
    else:
        cache_hit = False
        raw_stems = separate_with_demucs(orig, model_name="htdemucs_ft")
        gpu_label = detect_gpu_label()
        for k, v in raw_stems.items():
            save_audio_ffmpeg(v, cached_paths[k], sr=sr)

    # 2. Analisi Musicale: Rilevamento BPM & Tonalità Armonica (Krumhansl-Kessler)
    emit_progress(
        pct=74.0,
        stage_idx=3,
        stage="Analisi Musicale BPM & Tonalità",
        detail="Stima BPM multibanda e profilo armonico Krumhansl-Kessler...",
        gpu=gpu_label,
    )
    detected_bpm = detect_bpm(raw_stems["drums"], fallback_mix_2xN=orig, sr=sr)
    key_info = detect_musical_key(raw_stems["other"], bass_2xN=raw_stems["bass"], sr=sr)
    print(
        f"[Analisi AI] BPM Rilevato: {detected_bpm:.1f} BPM | Tonalità: {key_info['label']}",
        flush=True,
    )

    # 3. Time-Stretching Multi-Stem sincrono (se target_bpm attivo)
    if target_bpm is not None and target_bpm > 0 and abs(float(target_bpm) - detected_bpm) >= 0.2:
        emit_progress(
            pct=76.0,
            stage_idx=3,
            stage="Time-Stretching BPM Multi-Stem",
            detail=f"Adattamento tempo {detected_bpm:.1f} -> {float(target_bpm):.1f} BPM (WSOLA + Phase-Locked Vocoder)...",
            gpu=gpu_label,
        )
        print(
            f"[Time-Stretch] Stretching 4 stems da {detected_bpm:.1f} BPM a {float(target_bpm):.1f} BPM...",
            flush=True,
        )
        raw_stems, effective_bpm = stretch_stems_to_bpm(
            raw_stems,
            detected_bpm=detected_bpm,
            target_bpm=float(target_bpm),
            sr=sr,
        )
    else:
        effective_bpm = round(float(detected_bpm), 1)

    # 4. Voce: Auto-Tune Formant-Preserving opzionale + Vocal Channel Strip
    emit_progress(
        pct=79.0,
        stage_idx=3,
        stage="Restauro & Channel Strip Stems",
        detail=f"1/4 Voce — Auto-Tune ({autotune_mode}) + De-Noise & Vocal Strip [{key_info['label']}]",
        gpu=gpu_label,
    )
    vocals_tuned, autotune_stats = apply_vocal_autotune(
        raw_stems["vocals"],
        sr=sr,
        mode=autotune_mode,
        allowed_pcs=key_info["allowed_pcs"],
    )
    if autotune_mode != "off":
        print(
            f"[Auto-Tune Voce] Mode={autotune_mode} | Key={key_info['label']} | "
            f"Frame corretti={autotune_stats.get('corrected_frames_pct', 0.0)}% "
            f"(media {autotune_stats.get('mean_cents_correction', 0.0)} cents)",
            flush=True,
        )
    vocals_eq = process_vocals_stem(vocals_tuned, sr=sr)

    # 5. Batteria: Drum Replacer / Transient Reinforcement + Drum Strip
    emit_progress(
        pct=83.0,
        stage_idx=3,
        stage="Restauro & Channel Strip Stems",
        detail=f"2/4 Batteria — Drum Enhancer ({drum_mode}) & Anti-Box Strip",
        gpu=gpu_label,
    )
    drums_clean, drum_stats = _invoke_drum_enhancer(
        raw_stems["drums"],
        sr=sr,
        drum_mode=drum_mode,
        kick_blend=kick_blend,
        snare_blend=snare_blend,
    )
    drums_eq = process_drums_stem(drums_clean, sr=sr, drum_mode=drum_mode)

    # 6. Basso: Bass Enhancer & Kick-to-Bass Sidechain (usando drums_clean rinforzato + kick_positions) + Bass Strip
    emit_progress(
        pct=87.0,
        stage_idx=3,
        stage="Restauro & Channel Strip Stems",
        detail=f"3/4 Basso — Bass Re-Amp/Sidechain ({bass_mode}) & Mono Sub Strip",
        gpu=gpu_label,
    )
    raw_kick_pos = drum_stats.get("kick_positions") if isinstance(drum_stats, dict) else None
    kick_pos_for_sc = raw_kick_pos if (drum_mode != "off" or raw_kick_pos) else None
    bass_clean, bass_stats = _invoke_bass_enhancer(
        raw_stems["bass"],
        drums_for_sc=drums_clean,
        sr=sr,
        bass_mode=bass_mode,
        kick_positions=kick_pos_for_sc,
    )
    bass_eq = process_bass_stem(bass_clean, sr=sr, bass_mode=bass_mode)

    # 7. Chitarre / Altro: De-Buzz + EQ + Mid/Side Stereo Widening
    emit_progress(
        pct=90.0,
        stage_idx=3,
        stage="Restauro & Channel Strip Stems",
        detail="4/4 Chitarre & Altro — EQ Presenza & Mid/Side Stereo",
        gpu=gpu_label,
    )
    other_eq = process_other_stem(raw_stems["other"], sr=sr)

    vocals_eq, drums_eq, bass_eq, other_eq = _align_stems_length(vocals_eq, drums_eq, bass_eq, other_eq)

    # 8. Salva gli stems separati equalizzati (WAV 24-bit + MP3 320 kbps)
    emit_progress(
        pct=93.0,
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

    # 9. Somma Multi-Traccia e Master Bus finale (-11.2 LUFS, -0.50 dBTP)
    summed = vocals_eq + drums_eq + bass_eq + other_eq
    emit_progress(
        pct=96.0,
        stage_idx=4,
        stage="Mastering & Esportazione 24-Bit",
        detail="Calibrazione Master ITU-R BS.1770-4 (-11.2 LUFS, -0.50 dBTP)...",
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
        "detected_bpm": round(float(detected_bpm), 1),
        "effective_bpm": round(float(effective_bpm), 1),
        "detected_key": key_info["label"],
        "autotune_mode": autotune_mode,
        "drum_mode": drum_mode,
        "bass_mode": bass_mode,
        "drum_hits": {
            "kicks": int(drum_stats.get("kicks", 0)),
            "snares": int(drum_stats.get("snares", 0)),
        },
        "cache_hit": bool(cache_hit),
        "gpu": gpu_label,
        "output_wav": out_wav,
        "output_mp3": out_mp3,
        "stems_dir": stems_dir,
    }
    with open(os.path.join(out_dir, "ai_master_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    emit_progress(
        pct=100.0,
        stage_idx=5,
        stage="Completato",
        detail=(
            f"Master pronto ({master_metrics['lufs']} LUFS, Peak {master_metrics['peak_db']} dBTP | "
            f"{effective_bpm:.1f} BPM | {key_info['label']})"
        ),
        gpu=gpu_label,
    )

    print("\n=== COMPLETATO CON SUCCESSO ===", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def _parse_cli_and_options(argv: list[str] | None = None) -> tuple[str, str, dict]:
    parser = argparse.ArgumentParser(
        description="StudioEQ Master Pro — AI 4-Stem Separation, Vocal Auto-Tune, BPM Time-Stretch & Mastering"
    )
    parser.add_argument("in_path", help="Percorso del file audio di input")
    parser.add_argument("out_folder", help="Cartella di output per Master e Stems")
    parser.add_argument("--config-file", default=None, help="Percorso opzionale a ai_pipeline_options.json")
    parser.add_argument(
        "--autotune",
        choices=["off", "natural", "modern"],
        default=None,
        help="Modalità Auto-Tune vocale (default: off)",
    )
    parser.add_argument(
        "--drum-mode",
        choices=["off", "natural", "punch"],
        default=None,
        help="Modalità Drum Enhancer & Sample Reinforcement (default: natural)",
    )
    parser.add_argument(
        "--bass-mode",
        choices=["off", "natural", "reamp"],
        default=None,
        help="Modalità Bass Enhancer & Kick Sidechain (default: natural)",
    )
    parser.add_argument(
        "--target-bpm",
        type=float,
        default=None,
        help="BPM target opzionale per time-stretching sincrono dei 4 stems",
    )
    parser.add_argument(
        "--force-reseparate",
        action="store_true",
        default=False,
        help="Forza nuova separazione Demucs ignorando la cache SHA-256",
    )

    args = parser.parse_args(argv)

    options: dict[str, object] = {
        "autotune_mode": "off",
        "drum_mode": "natural",
        "bass_mode": "natural",
        "target_bpm": None,
        "force_reseparate": False,
        "kick_blend": None,
        "snare_blend": None,
    }

    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r", encoding="utf-8") as cf:
            loaded = json.load(cf)
        if isinstance(loaded, dict):
            if loaded.get("autotune_mode") is not None:
                options["autotune_mode"] = str(loaded["autotune_mode"]).strip().lower()
            elif loaded.get("autotune") is not None:
                options["autotune_mode"] = str(loaded["autotune"]).strip().lower()

            if loaded.get("drum_mode") is not None:
                options["drum_mode"] = str(loaded["drum_mode"]).strip().lower()
            if loaded.get("bass_mode") is not None:
                options["bass_mode"] = str(loaded["bass_mode"]).strip().lower()

            if "target_bpm" in loaded:
                tb = loaded["target_bpm"]
                options["target_bpm"] = float(tb) if (tb is not None and float(tb) > 0.0) else None

            if "force_reseparate" in loaded and loaded["force_reseparate"] is not None:
                options["force_reseparate"] = bool(loaded["force_reseparate"])

            if loaded.get("kick_blend") is not None:
                options["kick_blend"] = float(loaded["kick_blend"])
            if loaded.get("snare_blend") is not None:
                options["snare_blend"] = float(loaded["snare_blend"])

    # Explicit CLI flags override JSON file values
    if args.autotune is not None:
        options["autotune_mode"] = args.autotune
    if args.drum_mode is not None:
        options["drum_mode"] = args.drum_mode
    if args.bass_mode is not None:
        options["bass_mode"] = args.bass_mode
    if args.target_bpm is not None:
        options["target_bpm"] = float(args.target_bpm) if args.target_bpm > 0.0 else None
    if args.force_reseparate:
        options["force_reseparate"] = True

    return args.in_path, args.out_folder, options


if __name__ == "__main__":
    input_path, output_folder, pipeline_opts = _parse_cli_and_options()
    run_full_pipeline(
        input_path,
        output_folder,
        autotune_mode=str(pipeline_opts["autotune_mode"]),
        drum_mode=str(pipeline_opts["drum_mode"]),
        bass_mode=str(pipeline_opts["bass_mode"]),
        target_bpm=pipeline_opts["target_bpm"],  # type: ignore[arg-type]
        force_reseparate=bool(pipeline_opts["force_reseparate"]),
        kick_blend=pipeline_opts["kick_blend"],  # type: ignore[arg-type]
        snare_blend=pipeline_opts["snare_blend"],  # type: ignore[arg-type]
    )
