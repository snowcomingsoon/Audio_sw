#!/usr/bin/env bash
# ==============================================================================
# StudioEQ Master Pro v3.1.0 — Builder Nativo macOS Apple Silicon (M1 / M2 / M3 / M4)
# Target: aarch64-apple-darwin (arm64 nativo, CoreAudio + Metal Retina + EQ 3 Bande per Stem)
# Output: StudioEQ_Master_Pro_v3.1.0_macOS_M1_arm64.dmg
# ==============================================================================
set -euo pipefail

export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export MACOSX_DEPLOYMENT_TARGET="11.0"

if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

APP_VERSION="3.1.0"
APP_NAME="StudioEQ Master Pro"
BIN_NAME="StudioEQ_Master_Pro"
RUST_TARGET="aarch64-apple-darwin"
DIST_DIR="${REPO_ROOT}/dist"
APP_BUNDLE="${DIST_DIR}/${APP_NAME}.app"
STAGING_DIR="${DIST_DIR}/dmg_staging"
DMG_FILENAME="StudioEQ_Master_Pro_v${APP_VERSION}_macOS_M1_arm64.dmg"
DMG_OUTPUT="${REPO_ROOT}/${DMG_FILENAME}"

echo "======================================================================"
echo "  StudioEQ Master Pro v${APP_VERSION} — Packaging Nativo Apple Silicon (${RUST_TARGET})"
echo "======================================================================"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "[ERRORE] Questo script richiede macOS (hdiutil, codesign, CoreAudio SDK)." >&2
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "[ERRORE] Rust/Cargo non trovato nel PATH." >&2
    echo "Installa Rust con: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" >&2
    exit 1
fi

if command -v rustup >/dev/null 2>&1; then
    if ! rustup target list --installed | grep -q "^${RUST_TARGET}$"; then
        echo "[0/6] Installazione target Rust ${RUST_TARGET}..."
        rustup target add "${RUST_TARGET}"
    fi
fi

echo "[1/6] Esecuzione test unitari DSP/EQ e compilazione Release Apple Silicon (${RUST_TARGET})..."
pushd "${REPO_ROOT}/studio-eq-rust" >/dev/null
cargo test --release --target "${RUST_TARGET}"
cargo build --release --target "${RUST_TARGET}"
popd >/dev/null

BUILT_BIN="${REPO_ROOT}/studio-eq-rust/target/${RUST_TARGET}/release/${BIN_NAME}"
if [ ! -f "${BUILT_BIN}" ]; then
    echo "[ERRORE] Binario compilato non trovato: ${BUILT_BIN}" >&2
    exit 1
fi

echo "      Architettura binario:"
file "${BUILT_BIN}" || true
if command -v lipo >/dev/null 2>&1; then
    lipo -info "${BUILT_BIN}" || true
fi

echo "[2/6] Assemblaggio App Bundle Retina v${APP_VERSION}: ${APP_BUNDLE}..."
rm -rf "${APP_BUNDLE}" "${STAGING_DIR}"
mkdir -p "${APP_BUNDLE}/Contents/MacOS"
mkdir -p "${APP_BUNDLE}/Contents/Resources/audio_modules"
mkdir -p "${APP_BUNDLE}/Contents/Resources/assets/drum_samples"

cp "${REPO_ROOT}/macos/Info.plist" "${APP_BUNDLE}/Contents/Info.plist"
printf "APPL????" > "${APP_BUNDLE}/Contents/PkgInfo"

cp "${BUILT_BIN}" "${APP_BUNDLE}/Contents/MacOS/${BIN_NAME}"
chmod 755 "${APP_BUNDLE}/Contents/MacOS/${BIN_NAME}"

cp "${REPO_ROOT}/macos/AppIcon.icns" "${APP_BUNDLE}/Contents/Resources/AppIcon.icns"

# Includi l'intera pipeline Python AI + moduli DSP + campioni batteria 24-bit studio + installer AI M1
if [ -f "${REPO_ROOT}/demucs_pedalboard_master.py" ]; then
    cp "${REPO_ROOT}/demucs_pedalboard_master.py" "${APP_BUNDLE}/Contents/Resources/demucs_pedalboard_master.py"
fi
if [ -d "${REPO_ROOT}/audio_modules" ]; then
    cp "${REPO_ROOT}/audio_modules/"*.py "${APP_BUNDLE}/Contents/Resources/audio_modules/"
fi
if [ -d "${REPO_ROOT}/assets/drum_samples" ]; then
    cp "${REPO_ROOT}/assets/drum_samples/"*.wav "${APP_BUNDLE}/Contents/Resources/assets/drum_samples/"
fi
cp "${REPO_ROOT}/macos/requirements-macos-m1.txt" "${APP_BUNDLE}/Contents/Resources/requirements-macos-m1.txt"
cp "${REPO_ROOT}/macos/setup_ai_macos_m1.sh" "${APP_BUNDLE}/Contents/Resources/setup_ai_macos_m1.sh"
chmod 755 "${APP_BUNDLE}/Contents/Resources/setup_ai_macos_m1.sh"

echo "[3/6] Rimozione attributi di quarantena e firma nativa Ad-Hoc (Apple Silicon amfid)..."
xattr -cr "${APP_BUNDLE}" || true
codesign --force --deep --sign - "${APP_BUNDLE}"
xattr -cr "${APP_BUNDLE}" || true
codesign --verify --deep --strict --verbose=2 "${APP_BUNDLE}"

echo "[4/6] Preparazione contenuto installer DMG con Setup Guidato 1-Click..."
mkdir -p "${STAGING_DIR}"
ditto "${APP_BUNDLE}" "${STAGING_DIR}/${APP_NAME}.app"
ln -s /Applications "${STAGING_DIR}/Applications"

# Script 1-Click: Installa App in /Applications + sblocca Gatekeeper + setup opzionale Motore AI M1
cat > "${STAGING_DIR}/Installa StudioEQ su Mac M1.command" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_APP="${SRC_DIR}/StudioEQ Master Pro.app"
DST_APP="/Applications/StudioEQ Master Pro.app"

clear || true
echo "======================================================================"
echo "  INSTALLAZIONE AUTOMATICA — StudioEQ Master Pro v3.1.0 (Apple M1/M2/M3/M4)"
echo "======================================================================"
echo ""

if [ ! -d "${SRC_APP}" ]; then
    echo "[ERRORE] 'StudioEQ Master Pro.app' non trovato accanto a questo installer."
    read -r -p "Premi INVIO per chiudere..."
    exit 1
fi

echo "[1/3] Copia di 'StudioEQ Master Pro.app' nella cartella /Applications..."
if [ -w "/Applications" ]; then
    rm -rf "${DST_APP}"
    ditto "${SRC_APP}" "${DST_APP}"
else
    echo "      (Richiesto permesso amministratore macOS per scrivere in /Applications)"
    sudo rm -rf "${DST_APP}"
    sudo ditto "${SRC_APP}" "${DST_APP}"
fi

echo "[2/3] Rimozione blocco quarantena Gatekeeper (sblocco avvio immediato)..."
xattr -cr "${DST_APP}" 2>/dev/null || sudo xattr -cr "${DST_APP}" || true
mkdir -p "$HOME/Music/StudioEQ_Master_Pro/output"
echo "      [OK] App installata e sbloccata in: ${DST_APP}"
echo ""

VENV_PY="$HOME/Library/Application Support/StudioEQ_Master_Pro/.venv/bin/python3"
AI_SETUP_SH="${DST_APP}/Contents/Resources/setup_ai_macos_m1.sh"

if [ -x "${VENV_PY}" ]; then
    echo "[3/3] Motore AI Apple Silicon (Metal MPS) già presente e attivo!"
elif [ "${STUDIOEQ_AUTO_AI:-0}" = "1" ]; then
    echo "[3/3] Configurazione automatica Motore AI Apple Silicon..."
    bash "${AI_SETUP_SH}"
elif [ -t 0 ]; then
    echo "[3/3] Vuoi attivare ora anche il Motore AI locale per 'Separa AI' in 4 Tracce"
    echo "      con accelerazione GPU Apple Metal M1? (Consigliato, ~2 minuti la prima volta)"
    read -r -p "      Installare il Motore AI adesso? [S/n]: " RISPOSTA
    RISPOSTA="${RISPOSTA:-S}"
    case "${RISPOSTA}" in
        [sSyY]* )
            echo ""
            bash "${AI_SETUP_SH}"
            ;;
        * )
            echo "      [Info] Potrai attivare il Motore AI in qualsiasi momento aprendo"
            echo "             'Attiva Motore AI (Apple M1).command'."
            ;;
    esac
fi

echo ""
echo "Avvio di StudioEQ Master Pro v3.1.0..."
open "${DST_APP}" || true

if [ -t 0 ]; then
    echo ""
    read -r -p "Installazione completata! Premi INVIO per chiudere il Terminale..."
fi
EOF
chmod 755 "${STAGING_DIR}/Installa StudioEQ su Mac M1.command"
codesign --force --sign - "${STAGING_DIR}/Installa StudioEQ su Mac M1.command" || true

# Script standalone per attivare/riparare solo il Motore AI in qualsiasi momento
cat > "${STAGING_DIR}/Attiva Motore AI (Apple M1).command" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
for cand in \
    "/Applications/StudioEQ Master Pro.app/Contents/Resources/setup_ai_macos_m1.sh" \
    "${SRC_DIR}/StudioEQ Master Pro.app/Contents/Resources/setup_ai_macos_m1.sh"
do
    if [ -f "${cand}" ]; then
        bash "${cand}"
        if [ -t 0 ]; then
            echo ""
            read -r -p "Premi INVIO per chiudere..."
        fi
        exit 0
    fi
done

echo "[ERRORE] Trascina prima 'StudioEQ Master Pro.app' in Applications."
if [ -t 0 ]; then
    read -r -p "Premi INVIO per chiudere..."
fi
exit 1
EOF
chmod 755 "${STAGING_DIR}/Attiva Motore AI (Apple M1).command"
codesign --force --sign - "${STAGING_DIR}/Attiva Motore AI (Apple M1).command" || true

cat > "${STAGING_DIR}/LEGGIMI_INSTALLAZIONE_MAC.txt" <<'EOF'
================================================================================
  STUDIOEQ MASTER PRO v3.1.0 — EDIZIONE NATIVA APPLE SILICON (M1 / M2 / M3 / M4)
================================================================================

NOVITÀ v3.1.0:
  • Equalizzatore dedicato a 3 Bande (BASSI 95 Hz / MEDI / ALTI 6.5 kHz, ±12 dB)
    per ciascuna delle 4 tracce isolate (Voce, Batteria, Basso, Chitarre), sia
    direttamente nelle card della Timeline sia nel cassetto inferiore EQ!
  • Pipeline AI completa con accelerazione GPU Apple Metal (MPS): Demucs v4,
    DrumSep v3 (4 sub-stems batteria + Fast-Roll Engine), Apollo 44.1 kHz
    e Pedalboard C++.

--------------------------------------------------------------------------------
METODO A — INSTALLAZIONE AUTOMATICA 1-CLICK (CONSIGLIATA):
--------------------------------------------------------------------------------
  1. Apri il Terminale (Cmd + Spazio -> scrivi Terminale -> Invio)
  2. Incolla questo comando per avviare l'installer completo (copia l'app in
     /Applications, sblocca Gatekeeper e configura la GPU Metal M1 per l'AI):

     bash "/Volumes/StudioEQ Master Pro (Apple M1)/Installa StudioEQ su Mac M1.command"

     (Oppure fai Ctrl + Clic / Tasto Destro su "Installa StudioEQ su Mac M1.command"
      nel disco aperto -> clicca "Apri").

--------------------------------------------------------------------------------
METODO B — INSTALLAZIONE MANUALE DRAG & DROP:
--------------------------------------------------------------------------------
  1. Trascina "StudioEQ Master Pro.app" sopra la cartella "Applications".
  2. Poiché l'app usa firma nativa Ad-Hoc (senza abbonamento Apple Developer
     a pagamento), sblocca Gatekeeper aprendo Terminale e incollando:

     xattr -cr "/Applications/StudioEQ Master Pro.app"

  3. Se desideri usare anche la funzione "Separa AI" in 4 tracce su Mac M1:

     bash "/Applications/StudioEQ Master Pro.app/Contents/Resources/setup_ai_macos_m1.sh"

--------------------------------------------------------------------------------
REQUISITI DI SISTEMA:
--------------------------------------------------------------------------------
  • macOS 11.0 Big Sur o successivo (Ventura / Sonoma / Sequoia consigliati)
  • Processore Apple Silicon M1, M1 Pro/Max/Ultra, M2, M3 o M4 (arm64 nativo)
  • CoreAudio + Metal Retina: l'app DAW (EQ 7 bande Master, EQ 3 bande per Stem,
    Auto-EQ, Analizzatore FFT, Mixer 4 Stems, Time-Stretch BPM, Export WAV 24-bit)
    funziona istantaneamente a zero latenza senza dipendenze esterne!
================================================================================
EOF

echo "[5/6] Creazione immagine disco compressa ${DMG_FILENAME}..."
rm -f "${DMG_OUTPUT}"

for attempt in 1 2 3; do
    if hdiutil create \
        -volname "StudioEQ Master Pro (Apple M1)" \
        -srcfolder "${STAGING_DIR}" \
        -ov \
        -format UDZO \
        -imagekey zlib-level=9 \
        "${DMG_OUTPUT}"; then
        break
    fi
    echo "      [Avviso] Volume temporaneo impegnato da Spotlight (tentativo ${attempt}/3), riprovo tra 3s..."
    sleep 3
done

if [ ! -f "${DMG_OUTPUT}" ]; then
    echo "[ERRORE] Creazione del file DMG fallita." >&2
    exit 1
fi

xattr -cr "${DMG_OUTPUT}" || true
codesign --force --sign - "${DMG_OUTPUT}" || true
mkdir -p "${DIST_DIR}"
cp -f "${DMG_OUTPUT}" "${DIST_DIR}/${DMG_FILENAME}"
rm -rf "${STAGING_DIR}"

echo "[6/6] Verifica finale pacchetto DMG:"
ls -lh "${DMG_OUTPUT}"
if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${DMG_OUTPUT}"
fi

echo "======================================================================"
echo "  COMPLETATO CON SUCCESSO!"
echo "  Installer DMG pronto: ${DMG_OUTPUT}"
echo "======================================================================"
