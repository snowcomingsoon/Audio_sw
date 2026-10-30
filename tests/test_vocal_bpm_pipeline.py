#!/usr/bin/env python3
"""
Real-Stem Verification Suite for StudioEQ Master Pro:
- BPM & Musical Key Detection (`99.9 BPM`, `Mi minore / E min`) on cached Demucs stems
- Formant-Preserving Vocal Auto-Tune (`off`, `natural`, `modern`): zero NaN/clipping,
  >4.2 kHz sibilance preservation, scale cent-deviation reduction, and anti-metallic formant check
- Synchronous 4-Stem Time-Stretching (`99.9 -> 92.0 BPM` & `110.0 BPM`): sample-exact alignment,
  drum transient Crest Factor preservation, and stereo phase coherence
- Gentle Studio Drum Bus Polish & Kick-to-Bass Sidechain integration in `demucs_pedalboard_master.py`
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy import signal

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import demucs_pedalboard_master as dpm
from audio_modules.time_stretch_bpm import detect_bpm, stretch_stems_to_bpm
from audio_modules.vocal_autotune import (
    _cepstral_formant_envelope,
    _track_voiced_f0_yin,
    apply_vocal_autotune,
    detect_musical_key,
)

CACHE_STEMS_DIR = os.path.join(REPO_ROOT, "output", "stems_raw", "2b1cf6f94fd422b1")
SR = 44100


def _compute_crest_db(audio_2xN: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio_2xN)))
    rms = float(np.sqrt(np.mean(audio_2xN.astype(np.float64) ** 2)) + 1e-12)
    return 20.0 * float(np.log10(peak / rms + 1e-12))


def _compute_lr_corr(audio_2xN: np.ndarray) -> float:
    l_ch = audio_2xN[0].astype(np.float64)
    r_ch = audio_2xN[1].astype(np.float64)
    if np.std(l_ch) < 1e-9 or np.std(r_ch) < 1e-9:
        return 1.0
    return float(np.corrcoef(l_ch, r_ch)[0, 1])


def _measure_scale_cents_deviation(audio_2xN: np.ndarray, allowed_pcs: list[int], sr: int = SR) -> tuple[float, float]:
    nyq = 0.5 * float(sr)
    sos_lp = signal.butter(4, 4200.0 / nyq, btype="low", output="sos")
    mono_body = signal.sosfiltfilt(sos_lp, 0.5 * (audio_2xN[0].astype(np.float64) + audio_2xN[1].astype(np.float64)))
    f0_hz, voiced, _ = _track_voiced_f0_yin(mono_body, sr=sr, hop=max(1, int(round(0.010 * sr))))

    allowed_set = set(int(pc) % 12 for pc in allowed_pcs)
    cents_devs: list[float] = []
    for f0, is_v in zip(f0_hz, voiced):
        if not is_v or f0 <= 0.0:
            continue
        midi = 69.0 + 12.0 * float(np.log2(f0 / 440.0))
        cands = [
            abs(float(m) - midi) * 100.0
            for m in range(int(np.floor(midi)) - 3, int(np.ceil(midi)) + 4)
            if (m % 12) in allowed_set
        ]
        if cands:
            cents_devs.append(min(cands))

    arr = np.asarray(cents_devs, dtype=np.float64)
    return float(np.mean(arr)), float(np.median(arr))


class TestVocalBpmRealStemsPipeline(unittest.TestCase):
    stems_excerpt: dict[str, np.ndarray] = {}
    mix_excerpt: np.ndarray = np.empty((2, 0), dtype=np.float32)
    initial_cache_stat: dict[str, tuple[int, float]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        stem_names = ("vocals", "drums", "bass", "other")
        for name in stem_names:
            path = os.path.join(CACHE_STEMS_DIR, f"{name}.wav")
            if not os.path.isfile(path):
                raise unittest.SkipTest(f"Real stem cache file not found: {path}")
            st = os.stat(path)
            cls.initial_cache_stat[name] = (st.st_size, st.st_mtime)

        # Extract 25-second active vocal + rhythm section (150s..175s) in memory without touching disk cache
        s0, s1 = 150 * SR, 175 * SR
        cls.stems_excerpt = {}
        for name in stem_names:
            full = dpm.load_audio_stereo_44k(os.path.join(CACHE_STEMS_DIR, f"{name}.wav"), sr=SR)
            cls.stems_excerpt[name] = full[:, s0:s1].copy()

        cls.mix_excerpt = (
            cls.stems_excerpt["vocals"]
            + cls.stems_excerpt["drums"]
            + cls.stems_excerpt["bass"]
            + cls.stems_excerpt["other"]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # Strict immutability guarantee on output/stems_raw/2b1cf6f94fd422b1/
        for name, (orig_size, orig_mtime) in cls.initial_cache_stat.items():
            path = os.path.join(CACHE_STEMS_DIR, f"{name}.wav")
            st = os.stat(path)
            assert st.st_size == orig_size, f"Cache file size modified for {name}.wav!"
            assert abs(st.st_mtime - orig_mtime) < 1e-6, f"Cache file mtime modified for {name}.wav!"

    def test_01_real_stem_bpm_and_key_detection(self) -> None:
        bpm = detect_bpm(self.stems_excerpt["drums"], fallback_mix_2xN=self.mix_excerpt, sr=SR)
        key_info = detect_musical_key(
            self.stems_excerpt["other"],
            bass_2xN=self.stems_excerpt["bass"],
            sr=SR,
        )

        self.assertAlmostEqual(bpm, 99.9, delta=1.2, msg=f"Expected ~99.9 BPM, got {bpm}")
        self.assertEqual(key_info["root_pc"], 4, msg=f"Expected root E (pc=4), got {key_info}")
        self.assertEqual(key_info["scale"], "minor", msg=f"Expected minor scale, got {key_info}")
        self.assertIn("Mi minore", key_info["label"])
        self.assertIn("E min", key_info["label"])
        self.assertEqual(key_info["allowed_pcs"], [0, 2, 4, 6, 7, 9, 11])

    def test_02_vocal_autotune_off_natural_modern_real_vocals(self) -> None:
        vocals = self.stems_excerpt["vocals"]
        allowed_pcs = [0, 2, 4, 6, 7, 9, 11]  # E minor

        raw_mean_c, raw_med_c = _measure_scale_cents_deviation(vocals, allowed_pcs, sr=SR)

        # 1) Mode OFF: bit-exact passthrough
        vt_off, st_off = apply_vocal_autotune(vocals, sr=SR, mode="off", allowed_pcs=allowed_pcs)
        self.assertEqual(vt_off.shape, vocals.shape)
        self.assertEqual(vt_off.dtype, np.float32)
        self.assertTrue(np.allclose(vt_off, vocals, atol=1e-7))
        self.assertEqual(st_off["autotune_mode"], "off")
        self.assertEqual(st_off["corrected_frames_pct"], 0.0)

        # 2) Modes NATURAL and MODERN
        vt_nat, st_nat = apply_vocal_autotune(vocals, sr=SR, mode="natural", allowed_pcs=allowed_pcs)
        vt_mod, st_mod = apply_vocal_autotune(vocals, sr=SR, mode="modern", allowed_pcs=allowed_pcs)

        for label, vt, st in (("natural", vt_nat, st_nat), ("modern", vt_mod, st_mod)):
            self.assertEqual(vt.shape, vocals.shape)
            self.assertEqual(vt.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(vt)), f"NaN/Inf found in {label} vocal output")
            self.assertLessEqual(float(np.max(np.abs(vt))), 1.0, f"Clipping detected in {label} vocal output")
            self.assertGreater(st["corrected_frames_pct"], 25.0)

            # Verify >4.2 kHz sibilance & air preservation (5 kHz..16 kHz STFT band)
            freqs, _, z_ref = signal.stft(vocals[0], fs=SR, nperseg=4096)
            _, _, z_out = signal.stft(vt[0], fs=SR, nperseg=4096)
            sib_mask = (freqs >= 5000.0) & (freqs <= 16000.0)
            mag_corr = float(np.corrcoef(np.abs(z_ref[sib_mask]).ravel(), np.abs(z_out[sib_mask]).ravel())[0, 1])
            diff_rms_db = 20.0 * float(
                np.log10(np.sqrt(np.mean(np.abs(z_out[sib_mask] - z_ref[sib_mask]) ** 2)) + 1e-12)
            )
            self.assertGreaterEqual(mag_corr, 0.985, f"Sibilance >5 kHz correlation degraded in {label}: {mag_corr}")
            self.assertLess(diff_rms_db, -95.0, f"High-band sibilance leakage too high in {label}: {diff_rms_db} dB")

            # Verify absence of metallic/buzzy artifacts via spectral flatness & formant envelope correlation
            body_mask = (freqs >= 300.0) & (freqs <= 3800.0)
            m_in = np.abs(z_ref[body_mask]) + 1e-9
            m_out = np.abs(z_out[body_mask]) + 1e-9
            sfm_in = float(np.mean(np.exp(np.mean(np.log(m_in), axis=0)) / np.mean(m_in, axis=0)))
            sfm_out = float(np.mean(np.exp(np.mean(np.log(m_out), axis=0)) / np.mean(m_out, axis=0)))
            self.assertLessEqual(sfm_out, sfm_in * 1.08, f"Metallic spectral flattening detected in {label}")

            env_in = _cepstral_formant_envelope(np.abs(z_ref), n_fft=4096, sr=SR)[: int(3800 * 4096 / SR), :]
            env_out = _cepstral_formant_envelope(np.abs(z_out), n_fft=4096, sr=SR)[: int(3800 * 4096 / SR), :]
            formant_corr = float(np.corrcoef(env_in.ravel(), env_out.ravel())[0, 1])
            self.assertGreaterEqual(formant_corr, 0.95, f"Formant envelope shifted unnaturally in {label}: {formant_corr}")

        nat_mean_c, nat_med_c = _measure_scale_cents_deviation(vt_nat, allowed_pcs, sr=SR)
        mod_mean_c, mod_med_c = _measure_scale_cents_deviation(vt_mod, allowed_pcs, sr=SR)

        # Progressive cent-deviation reduction: raw > natural > modern
        self.assertLess(nat_mean_c, raw_mean_c - 8.0)
        self.assertLess(mod_mean_c, nat_mean_c - 5.0)
        self.assertLess(nat_med_c, raw_med_c - 8.0)
        self.assertLess(mod_med_c, nat_med_c - 5.0)

    def test_03_time_stretch_92_and_110_bpm_sync_and_transients(self) -> None:
        detected = detect_bpm(self.stems_excerpt["drums"], fallback_mix_2xN=self.mix_excerpt, sr=SR)
        raw_drum_crest = _compute_crest_db(self.stems_excerpt["drums"])
        raw_drum_corr = _compute_lr_corr(self.stems_excerpt["drums"])
        raw_voc_corr = _compute_lr_corr(self.stems_excerpt["vocals"])

        for target_bpm in (92.0, 110.0):
            stretched, eff_bpm = stretch_stems_to_bpm(
                self.stems_excerpt,
                detected_bpm=detected,
                target_bpm=target_bpm,
                sr=SR,
            )
            self.assertAlmostEqual(eff_bpm, target_bpm, places=1)

            expected_len = int(round(self.stems_excerpt["drums"].shape[1] / (target_bpm / detected)))
            shapes = {k: arr.shape for k, arr in stretched.items()}
            self.assertEqual(len(set(shapes.values())), 1, f"Stem lengths out of sync at {target_bpm} BPM: {shapes}")
            self.assertEqual(stretched["drums"].shape, (2, expected_len))

            for k, arr in stretched.items():
                self.assertEqual(arr.dtype, np.float32)
                self.assertTrue(np.all(np.isfinite(arr)), f"Non-finite samples in stretched {k}")
                self.assertLessEqual(float(np.max(np.abs(arr))), 1.05, f"Runaway peak in stretched {k}")

            # Verify stretched BPM matches target
            stretched_mix = stretched["vocals"] + stretched["drums"] + stretched["bass"] + stretched["other"]
            redetected_bpm = detect_bpm(stretched["drums"], fallback_mix_2xN=stretched_mix, sr=SR)
            self.assertAlmostEqual(
                redetected_bpm,
                target_bpm,
                delta=1.5,
                msg=f"Stretched BPM mismatch: target={target_bpm}, detected={redetected_bpm}",
            )

            # Verify drum transient Crest Factor & stereo phase coherence
            st_drum_crest = _compute_crest_db(stretched["drums"])
            self.assertLessEqual(
                abs(st_drum_crest - raw_drum_crest),
                1.2,
                msg=f"Drum transient crest degraded at {target_bpm} BPM: {raw_drum_crest:.2f} -> {st_drum_crest:.2f} dB",
            )
            self.assertLessEqual(abs(_compute_lr_corr(stretched["drums"]) - raw_drum_corr), 0.02)
            self.assertLessEqual(abs(_compute_lr_corr(stretched["vocals"]) - raw_voc_corr), 0.02)

    def test_04_drum_bus_polish_and_bass_sidechain_integration(self) -> None:
        # Use 12s excerpt for fast channel-strip & end-to-end verification
        n_12s = 12 * SR
        drums_12s = self.stems_excerpt["drums"][:, :n_12s]
        bass_12s = self.stems_excerpt["bass"][:, :n_12s]

        drums_clean, drum_stats = dpm._invoke_drum_enhancer(
            drums_12s,
            sr=SR,
            drum_mode="punch",
            kick_blend=None,
            snare_blend=None,
        )
        self.assertGreater(drum_stats.get("kicks", 0), 5)
        self.assertIsInstance(drum_stats.get("kick_positions"), list)

        polished = dpm.process_drums_stem(drums_clean, sr=SR, drum_mode="punch")
        aggressive_off = dpm.process_drums_stem(drums_clean, sr=SR, drum_mode="off")

        nyq = 0.5 * SR
        sos_hi = signal.butter(4, 7000.0 / nyq, btype="high", output="sos")
        hi_in = float(np.sqrt(np.mean(signal.sosfiltfilt(sos_hi, drums_clean.astype(np.float64), axis=-1) ** 2)))
        hi_pol = float(np.sqrt(np.mean(signal.sosfiltfilt(sos_hi, polished.astype(np.float64), axis=-1) ** 2)))
        hi_off = float(np.sqrt(np.mean(signal.sosfiltfilt(sos_hi, aggressive_off.astype(np.float64), axis=-1) ** 2)))

        gain_pol_db = 20.0 * float(np.log10((hi_pol + 1e-12) / (hi_in + 1e-12)))
        gain_off_db = 20.0 * float(np.log10((hi_off + 1e-12) / (hi_in + 1e-12)))
        # Gentle Studio Drum Bus Polish keeps >7 kHz air silky (< 3.5 dB) vs full recovery (+6.4 dB)
        self.assertLess(gain_pol_db, 3.5)
        self.assertLess(gain_pol_db, gain_off_db - 2.5)

        # Verify live _invoke_bass_enhancer works cleanly with drums_clean + kick_positions
        bass_out, bass_stats = dpm._invoke_bass_enhancer(
            bass_12s,
            drums_for_sc=drums_clean,
            sr=SR,
            bass_mode="natural",
            kick_positions=drum_stats.get("kick_positions"),
        )
        self.assertEqual(bass_out.shape, bass_12s.shape)
        self.assertTrue(np.all(np.isfinite(bass_out)))
        self.assertEqual(bass_stats.get("bass_mode"), "natural")

        # Verify inspect.signature forwarding when enhance_and_resynthesize_bass accepts kick_positions
        captured_call: dict[str, object] = {}

        def _fake_bass_enhancer(
            bass_2xN: np.ndarray,
            drums_2xN: np.ndarray | None = None,
            sr: int = 44100,
            bass_mode: str = "natural",
            kick_positions: list[int] | None = None,
        ) -> tuple[np.ndarray, dict]:
            captured_call["drums_2xN"] = drums_2xN
            captured_call["kick_positions"] = kick_positions
            return bass_2xN.astype(np.float32), {"bass_mode": bass_mode}

        with mock.patch("audio_modules.bass_enhancer.enhance_and_resynthesize_bass", _fake_bass_enhancer):
            dpm._invoke_bass_enhancer(
                bass_12s,
                drums_for_sc=drums_clean,
                sr=SR,
                bass_mode="reamp",
                kick_positions=[4410, 26460],
            )
        self.assertIs(captured_call["drums_2xN"], drums_clean)
        self.assertEqual(captured_call["kick_positions"], [4410, 26460])

        # Verify isolated TemporaryDirectory end-to-end pipeline run without touching real cache dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_wav = os.path.join(tmp_dir, "excerpt_input.wav")
            out_dir = os.path.join(tmp_dir, "master_out")
            mix_12s = self.mix_excerpt[:, :n_12s]
            dpm.save_audio_ffmpeg(mix_12s, in_wav, sr=SR)

            t_hash = dpm.compute_track_hash(in_wav)
            tmp_cache = os.path.join(out_dir, "stems_raw", t_hash)
            os.makedirs(tmp_cache, exist_ok=True)
            for k in ("vocals", "drums", "bass", "other"):
                dpm.save_audio_ffmpeg(self.stems_excerpt[k][:, :n_12s], os.path.join(tmp_cache, f"{k}.wav"), sr=SR)

            report = dpm.run_full_pipeline(
                input_audio=in_wav,
                out_dir=out_dir,
                autotune_mode="natural",
                drum_mode="natural",
                bass_mode="natural",
                target_bpm=None,
            )
            self.assertTrue(report["cache_hit"])
            self.assertEqual(report["autotune_mode"], "natural")
            self.assertEqual(report["drum_mode"], "natural")
            self.assertEqual(report["bass_mode"], "natural")
            self.assertLessEqual(report["master_ai"]["peak_db"], -0.45)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "ai_master_report.json")))
            with open(os.path.join(out_dir, "ai_master_report.json"), "r", encoding="utf-8") as rf:
                saved_report = json.load(rf)
            self.assertEqual(saved_report["drum_hits"]["kicks"], report["drum_hits"]["kicks"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
