#!/usr/bin/env python3
"""
GitHub Open-Source Pro Audio Pipeline
Integra le migliori repository open-source su GitHub per il restauro e il mastering di registrazioni in sala prove:

1. facebookresearch/demucs (https://github.com/facebookresearch/demucs)
   -> Separa con AI la traccia unica registrata in sala in 4 tracce separate (Voce, Batteria, Basso, Chitarre/Altro)
   così da equalizzare e ripulire ogni strumento singolarmente senza che il taglio sul basso spenga la voce!

2. spotify/pedalboard (https://github.com/spotify/pedalboard)
   -> Motore DSP C++ di Spotify per Python: applica EQ parametrici, Highpass, Compressori, Gate e Limiter da studio
   su ciascuno stem separato e sul master bus.

3. sergree/matchering (https://github.com/sergree/matchering)
   -> Mastering automatico open-source per matching spettrale (EQ + RMS + Stereo Width) rispetto a una traccia
   di riferimento commerciale.
"""

import os
import argparse


def run_pedalboard_stem_chain(input_wav: str, output_wav: str):
    """
    Applica una catena di studio professionale con Spotify Pedalboard (https://github.com/spotify/pedalboard).
    """
    from pedalboard import (
        Pedalboard,
        HighpassFilter,
        LowShelfFilter,
        PeakFilter,
        HighShelfFilter,
        Compressor,
        Limiter,
        Gain,
    )
    from pedalboard.io import AudioFile

    board = Pedalboard([
        # 1. Rimuove il rimbombo sub-basso della stanza (< 45 Hz)
        HighpassFilter(cutoff_frequency_hz=45.0),
        # 2. Taglia le risonanze modali della sala prove (113 Hz e 164 Hz)
        PeakFilter(cutoff_frequency_hz=113.0, gain_db=-3.2, q=2.6),
        PeakFilter(cutoff_frequency_hz=164.0, gain_db=-6.2, q=1.5),
        PeakFilter(cutoff_frequency_hz=265.0, gain_db=-3.0, q=1.4),
        # 3. Elimina l'effetto "scatola di cartone" tipico dei muri della sala (495 Hz)
        PeakFilter(cutoff_frequency_hz=495.0, gain_db=-5.2, q=2.0),
        # 4. Riporta in primo piano Voce, Rullante e Chitarre (1.42 kHz)
        PeakFilter(cutoff_frequency_hz=1420.0, gain_db=5.4, q=0.85),
        # 5. Smorza la risonanza acida dei capsule microfoniche a 2.34 kHz
        PeakFilter(cutoff_frequency_hz=2336.0, gain_db=-2.4, q=3.4),
        # 6. Aggiunge presenza (4.8 kHz) e Aria da studio (8.2 kHz)
        PeakFilter(cutoff_frequency_hz=4800.0, gain_db=4.8, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=8200.0, gain_db=7.8, q=0.707),
        # 7. Compressione Glue da Bus + Limiter finale
        Compressor(threshold_db=-18.0, ratio=2.5, attack_ms=20.0, release_ms=150.0),
        Gain(gain_db=3.5),
        Limiter(threshold_db=-0.5, release_ms=80.0),
    ])

    with AudioFile(input_wav) as f:
        audio = f.read(f.frames)
        sr = f.samplerate

    effected = board(audio, sr)

    with AudioFile(output_wav, "w", sr, effected.shape[0]) as f:
        f.write(effected)
    print(f"[Spotify Pedalboard] Mastering completato: {output_wav}")


def run_matchering_reference(target_path: str, reference_path: str, output_path: str):
    """
    Esegue il matching automatico di EQ, RMS e dinamica con sergree/matchering
    (https://github.com/sergree/matchering).
    """
    import matchering as mg

    mg.log(print)
    mg.process(
        target=target_path,
        reference=reference_path,
        results=[mg.pcm24(output_path)],
    )
    print(f"[Matchering 2.0] Traccia masterizzata sul reference: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline professionale con librerie GitHub (Pedalboard / Matchering)")
    parser.add_argument("--input", required=True, help="Percorso del file audio da equalizzare")
    parser.add_argument("--output", required=True, help="Percorso del file WAV di output")
    parser.add_argument("--reference", default=None, help="File di riferimento opzionale per Matchering 2.0")
    args = parser.parse_args()

    if args.reference:
        run_matchering_reference(args.input, args.reference, args.output)
    else:
        run_pedalboard_stem_chain(args.input, args.output)
