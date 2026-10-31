#!/usr/bin/env python3
"""
Verification Suite for StudioEQ Master Pro v3.0:
  1. Fast Snare Roll (`rullate veloci` IOI 38-50 ms) 100% Recall + Room-Slapback Rejection
  2. Full 4-Sub-Stem `DrumSep` v3 Engine (90% Kick/Snare 4xRR/Toms + Apollo Cymbals + Studio Room IR)
  3. Non-Manipulative Guitar Restoration (`htdemucs_6s` + `JusperLee/Apollo`, core waveform corr >= 0.99)
  4. Closed-Loop AI Self-Listening Critic (`audio_modules/ai_audio_critic.py`)
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import soundfile as sf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from audio_modules.ai_audio_critic import audit_drum_upgrade_quality, audit_guitar_non_manipulation
from audio_modules.apollo_restorer import restore_guitar_transparent
from audio_modules.drum_samples_synth import ensure_studio_drum_samples
from audio_modules.drumsep_v3 import (
    detect_isolated_snare_hits_with_rolls,
    enhance_drums_v3_drumsep,
    load_studio_bank_v3,
    master_standalone_drum_stem,
)

TRACK_HASH = "2b1cf6f94fd422b1"
DRUMSEP_DIR = os.path.join(ROOT_DIR, "output", "stems_drumsep", TRACK_HASH)
RAW_DRUMS_WAV = os.path.join(ROOT_DIR, "output", "stems_raw", TRACK_HASH, "drums.wav")
GUITAR_6S_WAV = os.path.join(ROOT_DIR, "output", "stems_6s", TRACK_HASH, "guitar.wav")


class TestStudioV3DrumSepGuitarCritic(unittest.TestCase):
    def test_01_synthetic_fast_snare_roll_vs_lone_room_slapback(self) -> None:
        sr = 44100
        dur_s = 4.0
        n = int(dur_s * sr)
        left = np.zeros(n, dtype=np.float64)
        right = np.zeros(n, dtype=np.float64)
        rng = np.random.default_rng(42)

        def make_snare_stroke(amp: float) -> np.ndarray:
            m = int(0.16 * sr)
            t = np.arange(m, dtype=np.float64) / sr
            body = np.sin(2.0 * np.pi * 196.0 * t) * np.exp(-t / 0.028)
            wire = rng.standard_normal(m) * np.exp(-t / 0.042) * 0.55
            click = np.exp(-t / 0.0022) * 0.8
            return amp * (0.65 * body + 0.45 * wire + 0.35 * click)

        # 1. Lone groove backbeat at 0.60s + weak +56ms rehearsal-room slapback echo (no stick attack, no 3rd hit)
        p_back = int(0.60 * sr)
        s_main = make_snare_stroke(0.75)
        left[p_back : p_back + s_main.size] += s_main
        right[p_back : p_back + s_main.size] += s_main

        p_echo = p_back + int(0.056 * sr)
        m_ec = int(0.09 * sr)
        t_ec = np.arange(m_ec, dtype=np.float64) / sr
        room_echo = 0.18 * np.sin(2.0 * np.pi * 205.0 * t_ec) * np.exp(-t_ec / 0.030)
        left[p_echo : p_echo + m_ec] += room_echo
        right[p_echo : p_echo + m_ec] += room_echo

        # 2. Blazing 10-stroke 32nd-note snare roll at 1.80s (IOI = 43 ms, crescendo 0.28 -> 0.92!)
        roll_gt: list[int] = []
        for k in range(10):
            pos = int((1.80 + k * 0.043) * sr)
            amp = 0.28 + 0.64 * (k / 9.0)
            st = make_snare_stroke(amp)
            left[pos : pos + st.size] += st
            right[pos : pos + st.size] += st
            roll_gt.append(pos)

        snare_2xn = np.vstack([left, right]).astype(np.float32)
        hits, roll_spans = detect_isolated_snare_hits_with_rolls(snare_2xn, sr=sr)
        det_pos = [h[0] for h in hits]

        # Must reject the +56ms room echo (only 1 hit in 0.50..0.80s)
        backbeat_hits = [p for p in det_pos if int(0.50 * sr) <= p <= int(0.80 * sr)]
        self.assertEqual(len(backbeat_hits), 1, f"Expected 1 clean backbeat (echo rejected), got {len(backbeat_hits)}")

        # Must capture 10/10 strokes in the 43ms fast snare roll!
        roll_hits = [p for p in det_pos if int(1.75 * sr) <= p <= int(2.35 * sr)]
        self.assertGreaterEqual(len(roll_hits), 10, f"Expected all 10 fast-roll strokes, got {len(roll_hits)}")
        self.assertGreaterEqual(len(roll_spans), 1, "Expected continuous roll_span across the fast snare roll")

    def test_02_studio_bank_v3_round_robin_and_toms(self) -> None:
        info = ensure_studio_drum_samples()
        for rr_key in ("snare", "snare_rr2", "snare_rr3", "snare_rr4"):
            for layer in ("soft", "med", "hard"):
                self.assertTrue(os.path.isfile(info[rr_key][layer]), f"Missing {rr_key}/{layer}")
        for tom_key in ("toms", "toms_rr2"):
            for pitch in ("high", "mid", "floor"):
                self.assertTrue(os.path.isfile(info[tom_key][pitch]), f"Missing {tom_key}/{pitch}")

        bank = load_studio_bank_v3(None, 44100)
        self.assertEqual(len(bank["snare"]), 4)
        self.assertEqual(len(bank["toms"][1]), 3)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(DRUMSEP_DIR, "snare.wav")) and os.path.isfile(RAW_DRUMS_WAV),
        "Real rehearsal DrumSep cache not present",
    )
    def test_03_real_rehearsal_1m49s_fast_roll_and_critic(self) -> None:
        sr = 44100
        # Load 1:40 -> 2:25 slice (`100s -> 145s`) covering the 1:49.5 fill AND 1:59 & 2:18 fills!
        s0, s1 = 100 * sr, 145 * sr
        raw_d, _ = sf.read(RAW_DRUMS_WAV, start=s0, stop=s1, dtype="float32", always_2d=True)
        raw_drums = raw_d.T

        sub_stems = {}
        for k in ("kick", "snare", "cymbals", "toms"):
            arr, _ = sf.read(os.path.join(DRUMSEP_DIR, f"{k}.wav"), start=s0, stop=s1, dtype="float32", always_2d=True)
            sub_stems[k] = arr.T

        v3_res = enhance_drums_v3_drumsep(
            raw_drums, sr=sr, sub_stems=sub_stems, use_apollo_cymbals=False
        )
        self.assertIsNotNone(v3_res)
        v3_drums, v3_stats = v3_res  # type: ignore[misc]

        # Check the exact user-reported fast snare roll at 1:50.7 - 1:51.4 (`10.6s - 11.5s` inside slice)
        sn_pos = v3_stats["snare_positions"]
        fill_1m50_hits = [p for p in sn_pos if int(10.6 * sr) <= p <= int(11.5 * sr)]
        print(
            f"\n[REAL 1:49.5 FILL CHECK] Snare strokes in 1:50.6..1:51.5 = {len(fill_1m50_hits)} | "
            f"Slice kicks={v3_stats['kicks']} snares={v3_stats['snares']} toms={v3_stats['toms']} "
            f"fast_roll_strokes={v3_stats['fast_roll_strokes']}"
        )
        self.assertGreaterEqual(
            len(fill_1m50_hits),
            9,
            f"Must preserve the blazing 32nd-note snare roll at 1:50.8 (got {len(fill_1m50_hits)} strokes)!",
        )
        self.assertGreaterEqual(v3_stats["toms"], 3, "Must detect and replace the tom lead-in hits at 1:49.5!")

        audit = audit_drum_upgrade_quality(
            raw_drums, v3_drums, sr=sr, isolated_snare_2xn=sub_stems["snare"]
        )
        print(f"[AI CRITIC DRUMS] {audit}")
        self.assertTrue(audit["passed"], f"AI Drum Critic failed: {audit}")
        self.assertGreaterEqual(audit["fast_roll_recall_pct"], 95.0)

        mastered_d, m_stats = master_standalone_drum_stem(v3_drums, sr=sr, target_lufs=-14.0)
        self.assertFalse(np.any(np.isnan(mastered_d)))
        self.assertAlmostEqual(m_stats["active_lufs"], -14.0, delta=0.8)

    @unittest.skipUnless(os.path.isfile(GUITAR_6S_WAV), "Real htdemucs_6s guitar.wav not present")
    def test_04_guitar_non_manipulation_and_critic(self) -> None:
        sr = 44100
        g_arr, _ = sf.read(GUITAR_6S_WAV, start=85 * sr, stop=105 * sr, dtype="float32", always_2d=True)
        g_raw = g_arr.T
        g_res, g_st = restore_guitar_transparent(g_raw, sr=sr)
        g_audit = audit_guitar_non_manipulation(g_raw, g_res, sr=sr)
        print(f"\n[AI CRITIC GUITAR] stats={g_st} | audit={g_audit}")
        self.assertTrue(g_audit["unmanipulated_verified"])
        self.assertGreaterEqual(g_audit["core_waveform_corr_85_3400hz"], 0.99)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(DRUMSEP_DIR, "kick.wav")) and os.path.isfile(RAW_DRUMS_WAV),
        "Real rehearsal DrumSep cache not present",
    )
    def test_05_real_rehearsal_2m25_to_end_double_pedal_bass_and_clarity(self) -> None:
        sr = 44100
        # Load exact `2:25 -> end` slice (`145s -> end`) matching `raw_drums_2m25_to_end`!
        s0 = 145 * sr
        raw_d, _ = sf.read(RAW_DRUMS_WAV, start=s0, dtype="float32", always_2d=True)
        raw_drums = raw_d.T

        sub_stems = {}
        for k in ("kick", "snare", "cymbals", "toms"):
            arr, _ = sf.read(os.path.join(DRUMSEP_DIR, f"{k}.wav"), start=s0, dtype="float32", always_2d=True)
            sub_stems[k] = arr.T

        v3_res = enhance_drums_v3_drumsep(
            raw_drums, sr=sr, sub_stems=sub_stems, use_apollo_cymbals=False
        )
        self.assertIsNotNone(v3_res)
        v3_drums, v3_stats = v3_res  # type: ignore[misc]

        audit = audit_drum_upgrade_quality(
            raw_drums, v3_drums, sr=sr, isolated_snare_2xn=sub_stems["snare"]
        )
        print(
            f"\n[REAL 2:25->END DOUBLE PEDAL, BASS & CLARITY CHECK] "
            f"kicks={v3_stats['kicks']} double_pedal_strokes={v3_stats['double_pedal_strokes']} "
            f"snares={v3_stats['snares']} fast_roll_strokes={v3_stats['fast_roll_strokes']} | "
            f"sub_to_box_gain_db=+{audit['sub_to_box_gain_db']} dB | "
            f"clarity_hi_to_mud_gain_db=+{audit['clarity_hi_to_mud_gain_db']} dB"
        )
        self.assertGreaterEqual(
            v3_stats["kicks"],
            65,
            f"Must detect kicks & double-pedal strokes in 2:25->end (got {v3_stats['kicks']})!",
        )
        self.assertGreaterEqual(
            v3_stats["double_pedal_strokes"],
            40,
            f"Must preserve fast double-pedal strokes in 2:25->end (got {v3_stats['double_pedal_strokes']})!",
        )
        self.assertGreaterEqual(audit["sub_to_box_gain_db"], 15.0)
        self.assertGreaterEqual(audit["clarity_hi_to_mud_gain_db"], 6.0)
        self.assertTrue(audit["passed"], f"AI Critic failed on 2:25->end: {audit}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
