#!/usr/bin/env bash
# ==============================================================================
# StudioEQ Master Pro — Builder Nativo macOS Apple Silicon (M1 / M2 / M3 / M4)
# Target: aarch64-apple-darwin (arm64 nativo, CoreAudio + Metal Retina)
# Output: StudioEQ_Master_Pro_macOS_M1_arm64.dmg
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

APP_NAME="StudioEQ Master Pro"
BIN_NAME="StudioEQ_Master_Pro"
RUST_TARGET="aarch64-apple-darwin"
DIST_DIR="${REPO_ROOT}/dist"
APP_BUNDLE="${DIST_DIR}/${APP_NAME}.app"
STAGING_DIR="${DIST_DIR}/dmg_staging"
DMG_FILENAME="StudioEQ_Master_Pro_macOS_M1_arm64.dmg"
DMG_OUTPUT="${REPO_ROOT}/${DMG_FILENAME}"

echo "======================================================================"
echo "  StudioEQ Master Pro — Packaging Nativo Apple Silicon (${RUST_TARGET})"
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
        echo "[1/6] Installazione target Rust ${RUST_TARGET}..."
        rustup target add "${RUST_TARGET}"
    fi
fi

echo "[1/6] Compilazione Release nativa Apple Silicon (${RUST_TARGET})..."
pushd "${REPO_ROOT}/studio-eq-rust" >/dev/null
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

echo "[2/6] Assemblaggio App Bundle Retina: ${APP_BUNDLE}..."
rm -rf "${APP_BUNDLE}" "${STAGING_DIR}"
mkdir -p "${APP_BUNDLE}/Contents/MacOS"
mkdir -p "${APP_BUNDLE}/Contents/Resources"

cp "${REPO_ROOT}/macos/Info.plist" "${APP_BUNDLE}/Contents/Info.plist"
printf "APPL????" > "${APP_BUNDLE}/Contents/PkgInfo"

cp "${BUILT_BIN}" "${APP_BUNDLE}/Contents/MacOS/${BIN_NAME}"
chmod 755 "${APP_BUNDLE}/Contents/MacOS/${BIN_NAME}"

cp "${REPO_ROOT}/macos/AppIcon.icns" "${APP_BUNDLE}/Contents/Resources/AppIcon.icns"

if [ -f "${REPO_ROOT}/demucs_pedalboard_master.py" ]; then
    cp "${REPO_ROOT}/demucs_pedalboard_master.py" "${APP_BUNDLE}/Contents/Resources/demucs_pedalboard_master.py"
fi

echo "[3/6] Rimozione attributi di quarantena e firma nativa Ad-Hoc (Apple Silicon amfid)..."
xattr -cr "${APP_BUNDLE}" || true
codesign --force --deep --sign - "${APP_BUNDLE}"
xattr -cr "${APP_BUNDLE}" || true
codesign --verify --deep --strict --verbose=2 "${APP_BUNDLE}"

echo "[4/6] Preparazione contenuto installer DMG..."
mkdir -p "${STAGING_DIR}"
ditto "${APP_BUNDLE}" "${STAGING_DIR}/${APP_NAME}.app"
ln -s /Applications "${STAGING_DIR}/Applications"

cat > "${STAGING_DIR}/LEGGIMI_INSTALLAZIONE_MAC.txt" <<'EOF'
================================================================================
  STUDIOEQ MASTER PRO v1.0.0 — EDIZIONE NATIVA APPLE SILICON (M1 / M2 / M3 / M4)
================================================================================

INSTALLAZIONE RAPIDA:
  1. Trascina l'icona "StudioEQ Master Pro.app" sopra la cartella "Applications".
  2. Apri il Launchpad o la cartella Applicazioni e avvia "StudioEQ Master Pro".

--------------------------------------------------------------------------------
NOTA IMPORTANTE PER IL PRIMO AVVIO SU MAC (Gatekeeper / Apple Silicon):
--------------------------------------------------------------------------------
Questa applicazione è compilata nativamente a 64-bit per chip Apple Silicon
(arm64: M1, M2, M3, M4) con firma locale Ad-Hoc, senza notarizzazione annuale
Apple Developer a pagamento ($99/anno).

Se scarichi il file .dmg da un browser (Safari / Chrome), macOS potrebbe
mostrare al primo avvio un avviso ("Impossibile verificare lo sviluppatore"
oppure "L'app è danneggiata e non può essere aperta").

Per sbloccarla definitivamente in 2 secondi:
  1. Apri l'app "Terminale" (Cmd + Spazio -> scrivi Terminale -> Invio)
  2. Incolla questo comando e premi Invio:

     xattr -cr "/Applications/StudioEQ Master Pro.app"

  3. Ora puoi aprire "StudioEQ Master Pro" normalmente con doppio clic!
     (In alternativa: fai Ctrl + Clic / Tasto Destro sull'app in Applicazioni
      e seleziona "Apri").

--------------------------------------------------------------------------------
REQUISITI E FUNZIONALITÀ:
--------------------------------------------------------------------------------
  • macOS 11.0 Big Sur o successivo (macOS Sonoma / Sequoia consigliati)
  • Chip Apple Silicon M1 / M2 / M3 / M4 (esecuzione 100% nativa arm64)
  • Motore DSP Parametrico, Auto-EQ, Analizzatore FFT, Mixer 4 Stems ed Export
    WAV 24-bit funzionano immediatamente senza alcuna dipendenza esterna.
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
