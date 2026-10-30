# StudioEQ Master Pro — Multi-Track AI Stem Mixer & Equalizer

**StudioEQ Master Pro** è una workstation audio nativa **Windows 64-bit scritta in Rust** (`StudioEQ_Master_Pro.exe`, `#![forbid(unsafe_code)]`) con pipeline integrata di **separazione AI su GPU NVIDIA CUDA e mastering automatico multi-stem**, progettata per trasformare registrazioni grezze da sala prove o live in produzioni pulite, bilanciate e pronte per lo streaming.

---

## Architettura Completa

### 1. DAW Multi-Traccia Nativa Rust 64-Bit (`StudioEQ_Master_Pro.exe`)
- **4 Tracce Waveform Separate ad Alta Risoluzione**: visualizzazione dedicata per **`Voce`**, **`Batteria`**, **`Basso`** e **`Chitarre & Altro`** con rendering **Peak + RMS** stratificato a 60 FPS (`eframe` / `egui`).
- **Mixer Completo per Singola Traccia**: controlli indipendenti di **Mute (`M`)**, **Solo (`S`)**, **Volume (`dB`)**, **Pan (`L/R` a potenza costante)** e meter di livello stereo in tempo reale.
- **Trasporto, Loop & Zoom Fluido**: righello temporale interattivo con **Zoom orizzontale fluido (`Ctrl + Rotellina`)**, riposizionamento istantaneo della playhead, selezione di intervalli sulla timeline e riproduzione ciclica in **Loop (`L`)**.
- **Motore Audio a Bassa Latenza**: streaming nativo **WASAPI (`cpal`)** con elaborazione DSP interna a doppia precisione **64-bit floating point (`f64`)** e commutazione istantanea tra traccia originale processata live, mixer multi-stem e master finale.

### 2. Monitor Integrato In-App di Separazione AI GPU & Mastering
- **Zero Finestre Console Esterne**: l'intera pipeline Python/CUDA viene orchestrata in background direttamente dall'applicazione Rust, senza aprire finestre `cmd.exe` o terminali esterni.
- **Avanzamento Live `htdemucs_ft` su CUDA**: pannello di monitoraggio in-app in tempo reale con stato dettagliato delle fasi, utilizzo **GPU NVIDIA CUDA** (con fallback automatico su CPU) e caricamento automatico degli stem a fine elaborazione.
- **Catena Studio Per-Stem & Loudness Mastering**:
  - **[`facebookresearch/demucs`](https://github.com/facebookresearch/demucs) (`htdemucs_ft`)**: separazione Hybrid Transformer Fine-Tuned ad alta precisione nei 4 stem (`Voce`, `Batteria`, `Basso`, `Chitarre & Altro`) con caching automatico in WAV 24-bit.
  - **[`timsainb/noisereduce`](https://github.com/timsainb/noisereduce)**: riduzione spettrale selettiva del fruscio microfonico e dei rientri ambientali su *Voce* e *Chitarre & Altro*.
  - **[`spotify/pedalboard`](https://github.com/spotify/pedalboard)**: channel strip C++ dedicato per ogni strumento (*Noise Gate*, EQ chirurgico anti-mud/anti-box, *Compressor*, *Plate Reverb* vocale, *Mono Sub Alignment* sul basso, *Mid/Side Stereo Widening* sulle chitarre) più *Master Bus Glue Compressor*.
  - **[`csteinmetz1/pyloudnorm`](https://github.com/csteinmetz1/pyloudnorm)**: normalizzazione integrata ITU-R BS.1770-4 calibrata a **`-11.2 LUFS`** con *Soft-Knee True-Peak Limiter* a **`-0.50 dBTP`** (zero clipping digitale).

### 3. Equalizzatore Parametrico Stile FabFilter, Auto-EQ `<15 ms` & Strumenti Pro
- **Cassetto EQ Parametrico a Scomparsa (`E`)**: interfaccia ispirata a FabFilter Pro-Q con **Analizzatore di Spettro FFT** a 4096 punti in tempo reale, **8 bande Biquad interattive** trascinabili direttamente sul grafico (frequenza, guadagno, fattore `Q`) e filtro **Sub-Cut High-Pass 24 dB/oct**.
- **4 Smart Knobs da Studio**:
  1. **Pulizia Sala** — *Dynamic Anti-Mud* per attenuare dinamicamente rimbombi e risonanze tra 130 Hz e 350 Hz.
  2. **Voce & Presenza** — messa a fuoco ed esaltazione dell'intelligibilità sulle medio-alte.
  3. **Aria & Alti** — *Harmonic Exciter* per restituire apertura e brillantezza alle alte frequenze.
  4. **Spinta Master** — gestione *Mid/Side Stereo Width* (con basse ancorate in mono) + *Bus Glue Compressor & Ceiling Limiter*.
- **Auto-EQ Spettrale Istantaneo (`<15 ms`, tasto `A`)**: analisi **Welch PSD** (finestra di Hann a 8192 punti con doppio smoothing logaritmico su blocchi musicali attivi) che individua risonanze modali di sala, inscatolamento (*boxiness*) e carenze d'aria, calcolando in meno di **15 ms** una curva EQ correttiva su misura.
- **Undo / Redo a 40 Livelli (`Ctrl+Z` / `Ctrl+Y`)**: storico circolare fino a 40 snapshot completi di tutti i parametri EQ e Master DSP, con confronto rapido **A/B Bypass (`B`)**.
- **Drag & Drop, Preset `.seqpreset` & Export WAV 24-Bit**:
  - **Drag & Drop** diretto nella finestra per aprire file `WAV`, `MP3`, `FLAC`, `OGG` e `M4A`.
  - Salvataggio e richiamo istantaneo delle catene di equalizzazione e mastering nel formato nativo **`.seqpreset`**.
  - **Esportazione Stem in WAV 24-bit PCM** (`01_Voce_Mix.wav`, `02_Batteria_Mix.wav`, `03_Basso_Mix.wav`, `04_Chitarre_Altro_Mix.wav`) con i livelli di Volume, Pan e soft-limiting applicati dal mixer, oltre all'esportazione del mix masterizzato completo.

---

## Avvio Rapido (Quick Start)

### Opzione A — Avvio immediato (consigliato)
Fai doppio clic su **`Avvia_StudioEQ.bat`** nella cartella del progetto (oppure usa il collegamento **`StudioEQ Master Pro`** sul Desktop):
- Se `StudioEQ_Master_Pro.exe` è già presente, l'applicazione parte istantaneamente **senza lasciare alcuna finestra `cmd.exe` aperta**.
- Se è il primo avvio dopo aver clonato la repository, `Avvia_StudioEQ.bat` avvia automaticamente lo script di configurazione e compilazione.

### Opzione B — Setup automatico ambiente AI + Build nativa Rust
Per verificare i prerequisiti, creare/aggiornare l'ambiente virtuale `.venv` (con rilevamento automatico GPU NVIDIA CUDA `cu126` o fallback CPU), compilare `StudioEQ_Master_Pro.exe` in modalità `--release` e aggiornare il collegamento sul Desktop:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_and_build.ps1
```

Parametri utili dello script `scripts\setup_and_build.ps1`:
- `-CheckOnly` — verifica solo la presenza di Rust (`cargo`), Python (3.11–3.13), FFmpeg e GPU NVIDIA senza modificare file.
- `-Launch` — avvia automaticamente `StudioEQ_Master_Pro.exe` al termine della compilazione.
- `-SkipVenv` / `-ForceVenv` — salta la verifica di `.venv` oppure forza la ricreazione pulita dell'ambiente Python.
- `-ForceCpu` — installa PyTorch in versione CPU anche in presenza di una scheda NVIDIA.
- `-NoDesktopShortcut` — salta la creazione/aggiornamento del collegamento sul Desktop.

È inoltre possibile forzare una ricompilazione completa da batch eseguendo:

```bat
Avvia_StudioEQ.bat --rebuild
```

### Prerequisiti di Sistema (per compilazione da zero)
- **Sistema Operativo**: Windows 10 / 11 (64-bit)
- **Rust Toolchain**: `rustup` + `cargo` (con *Visual Studio C++ Build Tools*)
- **Python**: versione **3.11, 3.12 o 3.13**
- **FFmpeg**: presente nel `PATH` di sistema (`winget install -e --id Gyan.FFmpeg`)
- **GPU (Consigliata)**: scheda **NVIDIA compatibile CUDA** per separazione `htdemucs_ft` accelerata via hardware (supportato fallback automatico su CPU)

---

## Scorciatoie da Tastiera DAW

| Tasto | Funzione |
| :--- | :--- |
| `Spazio` | Avvia / Metti in pausa la riproduzione |
| `Home` | Riporta la playhead all'inizio del brano |
| `L` | Attiva / Disattiva la riproduzione in **Loop** |
| `E` | Mostra / Nascondi il cassetto **EQ Parametrico & Master Bus** |
| `A` | Esegue l'**Auto-EQ Welch PSD (`<15 ms`)** sulla traccia attiva |
| `B` | Confronto istantaneo **A/B Bypass** del motore DSP |
| `1` / `2` / `3` / `4` | **Solo** rapido su *Voce* (`1`), *Batteria* (`2`), *Basso* (`3`), *Chitarre & Altro* (`4`) |
| `Ctrl + Z` | **Annulla (Undo)** ultima modifica DSP/EQ (fino a 40 livelli) |
| `Ctrl + Y` / `Ctrl + Shift + Z` | **Ripristina (Redo)** modifica DSP/EQ |
| `Ctrl + Rotellina` | **Zoom** fluido avanti/indietro sulla timeline |
| `F1` | Apre / Chiude la guida rapida alle scorciatoie |

---

## Struttura del Progetto

```text
Audio SW/
├── Avvia_StudioEQ.bat             # Launcher rapido Windows (zero finestre cmd residue)
├── StudioEQ_Master_Pro.exe        # Binario nativo Rust 64-bit (Release)
├── app_icon.ico                   # Icona applicazione ad alta risoluzione
├── demucs_pedalboard_master.py    # Pipeline AI GPU (Demucs v4 + Noisereduce + Pedalboard + PyLoudnorm)
├── requirements.txt               # Dipendenze Python per l'ambiente .venv
├── scripts/
│   └── setup_and_build.ps1        # Setup automatico .venv CUDA/CPU, build Cargo e shortcut Desktop
├── studio-eq-rust/                # Sorgenti Rust della DAW multi-traccia e del motore DSP a 64-bit
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                # Applicazione egui, trasporto, FFT real-time e monitor AI in-app
│       ├── audio_io.rs            # Motore WASAPI/cpal, decodifica Symphonia, mixer 4 stem
│       ├── dsp.rs                 # Catena DSP 64-bit (Biquad 8 bande, Anti-Mud, Exciter, M/S, Glue Comp)
│       ├── auto_eq.rs             # Analizzatore spettrale Welch PSD (<15 ms) per Auto-EQ
│       ├── timeline_view.rs       # Timeline multi-traccia con forme d'onda Peak+RMS, Loop e Zoom
│       ├── eq_drawer.rs           # Cassetto EQ FabFilter-style + 4 Smart Knobs
│       ├── project_io.rs          # Drag & Drop, preset .seqpreset ed export WAV 24-bit
│       ├── shortcuts_and_meters.rs# Undo/Redo a 40 livelli, scorciatoie DAW e meter LED
│       ├── ui_widgets.rs          # Palette grafica Studio e widget custom
│       └── icon.rs                # Icona finestra embedded
└── output/                        # Cartella di lavoro predefinita per tracce, stem e report JSON
```
