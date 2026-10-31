#!/usr/bin/env bash
# ==============================================================================
# StudioEQ Master Pro v3.1.0 — Builder SETUP COMPLETO OFFLINE macOS Apple Silicon
# Target: aarch64-apple-darwin (M1 / M2 / M3 / M4 — 100% Offline All-In-One)
# Include:
#   - App Nativa Rust CoreAudio + Metal Retina (v3.1.0 con EQ 3 Bande per Stem)
#   - Runtime Python 3.12 arm64 standalone rilocabile (zero dipendenza dal Mac ospite)
#   - PyTorch Metal MPS + Demucs 4.1 + Pedalboard + SciPy + Librosa + SoundFile
#   - FFmpeg arm64 standalone
#   - Tutti i Modelli AI pre-scaricati offline:
#       * HTDemucs-ft (4-stem Fine-Tuned) + HTDemucs-6s (Chitarra pura)
#       * DrumSep v3 (49469ca8.th — Kick / Snare / Cymbals / Toms)
#       * Apollo 44.1 kHz (pytorch_model.bin — Restauro Codec Lossless)
#       * 24 Campioni Batteria Studio 24-bit WAV
# Output: StudioEQ_Master_Pro_v3.1.0_Setup_Completo_macOS_M1_arm64.dmg
# ==============================================================================
set -euo pipefail

export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
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
PKG_ROOT="${DIST_DIR}/pkg_root"
APP_BUNDLE="${PKG_ROOT}/Applications/${APP_NAME}.app"
SUPPORT_STAGE="${PKG_ROOT}/Library/Application Support/StudioEQ_Master_Pro"
PKG_SCRIPTS="${DIST_DIR}/pkg_scripts"
PKG_RESOURCES="${DIST_DIR}/pkg_resources"
STAGING_DIR="${DIST_DIR}/dmg_staging"

DMG_FILENAME="StudioEQ_Master_Pro_v${APP_VERSION}_Setup_Completo_macOS_M1_arm64.dmg"
DMG_OUTPUT="${REPO_ROOT}/${DMG_FILENAME}"

echo "======================================================================"
echo "  StudioEQ Master Pro v${APP_VERSION} — SETUP COMPLETO OFFLINE Apple Silicon"
echo "======================================================================"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "[ERRORE] Questo script richiede macOS (pkgbuild, productbuild, hdiutil, codesign)." >&2
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "[ERRORE] Rust/Cargo non trovato nel PATH." >&2
    exit 1
fi

if command -v rustup >/dev/null 2>&1; then
    if ! rustup target list --installed | grep -q "^${RUST_TARGET}$"; then
        echo "[0/7] Installazione target Rust ${RUST_TARGET}..."
        rustup target add "${RUST_TARGET}"
    fi
fi

rm -rf "${DIST_DIR}"
mkdir -p "${APP_BUNDLE}/Contents/MacOS"
mkdir -p "${APP_BUNDLE}/Contents/Resources"
mkdir -p "${SUPPORT_STAGE}/audio_modules"
mkdir -p "${SUPPORT_STAGE}/assets/drum_samples"
mkdir -p "${SUPPORT_STAGE}/assets/models/drumsep"
mkdir -p "${SUPPORT_STAGE}/assets/models/apollo"
mkdir -p "${SUPPORT_STAGE}/models"
mkdir -p "${SUPPORT_STAGE}/ffmpeg/bin"
mkdir -p "${PKG_SCRIPTS}" "${PKG_RESOURCES}" "${STAGING_DIR}"

echo "[1/7] Test unitari e compilazione Release nativa Apple Silicon (${RUST_TARGET})..."
pushd "${REPO_ROOT}/studio-eq-rust" >/dev/null
cargo test --release --target "${RUST_TARGET}"
cargo build --release --target "${RUST_TARGET}"
popd >/dev/null

BUILT_BIN="${REPO_ROOT}/studio-eq-rust/target/${RUST_TARGET}/release/${BIN_NAME}"
file "${BUILT_BIN}" || true

echo "[2/7] Creazione e firma App Bundle Retina: ${APP_BUNDLE}..."
cp "${REPO_ROOT}/macos/Info.plist" "${APP_BUNDLE}/Contents/Info.plist"
printf "APPL????" > "${APP_BUNDLE}/Contents/PkgInfo"
cp "${BUILT_BIN}" "${APP_BUNDLE}/Contents/MacOS/${BIN_NAME}"
chmod 755 "${APP_BUNDLE}/Contents/MacOS/${BIN_NAME}"
cp "${REPO_ROOT}/macos/AppIcon.icns" "${APP_BUNDLE}/Contents/Resources/AppIcon.icns"

xattr -cr "${APP_BUNDLE}" || true
codesign --force --deep --sign - "${APP_BUNDLE}"
codesign --verify --deep --strict --verbose=2 "${APP_BUNDLE}"

echo "[3/7] Provisioning Python 3.12 arm64 standalone + PyTorch Metal MPS + Librerie DSP..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

PY_RAW_DIR="${DIST_DIR}/py_standalone_raw"
UV_PYTHON_INSTALL_DIR="${PY_RAW_DIR}" uv python install 3.12
PY_EXTRACTED="$(find "${PY_RAW_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'cpython-3.12*' | head -n 1)"
if [ -z "${PY_EXTRACTED}" ] || [ ! -d "${PY_EXTRACTED}" ]; then
    echo "[ERRORE] Distribuzione Python 3.12 standalone non trovata in ${PY_RAW_DIR}" >&2
    exit 1
fi

cp -R "${PY_EXTRACTED}" "${SUPPORT_STAGE}/python"
rm -rf "${PY_RAW_DIR}"
rm -f "${SUPPORT_STAGE}/python/lib/python3.12/EXTERNALLY-MANAGED"

BUNDLED_PY="${SUPPORT_STAGE}/python/bin/python3.12"
uv pip install --system --no-cache \
    --python "${BUNDLED_PY}" \
    -r "${REPO_ROOT}/macos/requirements-macos-m1.txt"

# Estrai FFmpeg standalone arm64 e rimuovi il duplicato dentro site-packages/imageio_ffmpeg
"${BUNDLED_PY}" - "${SUPPORT_STAGE}/ffmpeg/bin/ffmpeg" <<'PYEOF'
import os
import shutil
import sys
import imageio_ffmpeg

dst = sys.argv[1]
src = imageio_ffmpeg.get_ffmpeg_exe()
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy2(src, dst)
os.chmod(dst, 0o755)
print(f"      FFmpeg standalone arm64 copiato in: {dst} ({os.path.getsize(dst)/1048576:.1f} MB)")
PYEOF
codesign --force --sign - "${SUPPORT_STAGE}/ffmpeg/bin/ffmpeg" || true
rm -rf "${SUPPORT_STAGE}/python/lib/python3.12/site-packages/imageio_ffmpeg/binaries"

# Normalizza eventuali shebang negli script CLI di python/bin per portabilità assoluta
"${BUNDLED_PY}" - "${SUPPORT_STAGE}/python/bin" <<'PYEOF'
import os
import sys

bin_dir = sys.argv[1]
target_shebang = b"#!/Library/Application Support/StudioEQ_Master_Pro/python/bin/python3\n"
for fname in os.listdir(bin_dir):
    fpath = os.path.join(bin_dir, fname)
    if os.path.islink(fpath) or not os.path.isfile(fpath):
        continue
    with open(fpath, "rb") as f:
        head = f.read(4)
        if not head.startswith(b"#!"):
            continue
        f.seek(0)
        lines = f.readlines()
    if lines and b"python" in lines[0]:
        lines[0] = target_shebang
        with open(fpath, "wb") as f:
            f.writelines(lines)
        os.chmod(fpath, 0o755)
PYEOF

# Pulizia file non necessari a runtime (header C, suite di test, tkinter, __pycache__)
rm -rf \
    "${SUPPORT_STAGE}/python/include" \
    "${SUPPORT_STAGE}/python/share" \
    "${SUPPORT_STAGE}/python/lib/python3.12/test" \
    "${SUPPORT_STAGE}/python/lib/python3.12/tkinter" \
    "${SUPPORT_STAGE}/python/lib/python3.12/idlelib" \
    "${SUPPORT_STAGE}/python/lib/python3.12/turtledemo"
find "${SUPPORT_STAGE}/python" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "${SUPPORT_STAGE}/python" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true

echo "[4/7] Copia pipeline StudioEQ e download OFFLINE di tutti i Modelli Neurali AI..."
cp "${REPO_ROOT}/demucs_pedalboard_master.py" "${SUPPORT_STAGE}/demucs_pedalboard_master.py"
cp "${REPO_ROOT}/audio_modules/"*.py "${SUPPORT_STAGE}/audio_modules/"
cp "${REPO_ROOT}/assets/drum_samples/"*.wav "${SUPPORT_STAGE}/assets/drum_samples/"

HF_HOME="${SUPPORT_STAGE}/models" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONNOUSERSITE=1 \
"${BUNDLED_PY}" - "${SUPPORT_STAGE}" <<'PYEOF'
import hashlib
import os
import shutil
import sys
from huggingface_hub import hf_hub_download
from demucs.pretrained import get_model

stage = sys.argv[1]
print("      [AI 1/4] Pre-download Demucs v4 Fine-Tuned (htdemucs_ft)...")
m_ft = get_model("htdemucs_ft")
print(f"               OK ({len(m_ft.models)} sub-models)")

print("      [AI 2/4] Pre-download Demucs 6-Stems Chitarra (htdemucs_6s)...")
m_6s = get_model("htdemucs_6s")
print(f"               OK ({len(m_6s.models)} model)")

print("      [AI 3/4] Pre-download JusperLee/Apollo 44.1 kHz Codec Restorer...")
apollo_dir = os.path.join(stage, "assets", "models", "apollo")
apollo_bin = hf_hub_download(
    repo_id="JusperLee/Apollo",
    filename="pytorch_model.bin",
    local_dir=apollo_dir,
)
assert os.path.isfile(os.path.join(apollo_dir, "pytorch_model.bin")), "Apollo model missing!"
shutil.rmtree(os.path.join(apollo_dir, ".cache"), ignore_errors=True)
print(f"               OK ({os.path.getsize(os.path.join(apollo_dir, 'pytorch_model.bin'))/1048576:.1f} MB)")

print("      [AI 4/4] Pre-download DrumSep v3 4-Sub-Stems (49469ca8.th)...")
drumsep_dir = os.path.join(stage, "assets", "models", "drumsep")
drumsep_target = os.path.join(drumsep_dir, "49469ca8.th")
dl_tmp = hf_hub_download(
    repo_id="Eddycrack864/Drumsep",
    filename="modelo_final.th",
    local_dir=drumsep_dir,
)
shutil.move(dl_tmp, drumsep_target)
shutil.rmtree(os.path.join(drumsep_dir, ".cache"), ignore_errors=True)
sha = hashlib.sha256(open(drumsep_target, "rb").read()).hexdigest()
assert sha == "aefaa8543c9b9c75e22f5f32b53ab86dfe416457849af1383ff1aef83401423f", f"Unexpected SHA256: {sha}"
print(f"               OK ({os.path.getsize(drumsep_target)/1048576:.1f} MB, SHA-256 verificato)")
PYEOF

rm -rf "${SUPPORT_STAGE}/models/hub/.locks"
find "${SUPPORT_STAGE}" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
chmod -R a+rX "${SUPPORT_STAGE}"
chmod 755 "${SUPPORT_STAGE}/python/bin/"* "${SUPPORT_STAGE}/ffmpeg/bin/ffmpeg" 2>/dev/null || true
xattr -cr "${PKG_ROOT}" || true

echo "      Dimensione payload non compresso:"
du -sh "${APP_BUNDLE}" "${SUPPORT_STAGE}"

echo "[5/7] Creazione Installer Grafico Nativo macOS (.pkg) tramite pkgbuild + productbuild..."
cat > "${PKG_SCRIPTS}/postinstall" <<'EOF'
#!/bin/bash
xattr -cr "/Applications/StudioEQ Master Pro.app" 2>/dev/null || true
xattr -cr "/Library/Application Support/StudioEQ_Master_Pro" 2>/dev/null || true
chmod -R a+rX "/Library/Application Support/StudioEQ_Master_Pro" 2>/dev/null || true
chmod 755 "/Library/Application Support/StudioEQ_Master_Pro/python/bin/"* 2>/dev/null || true
chmod 755 "/Library/Application Support/StudioEQ_Master_Pro/ffmpeg/bin/ffmpeg" 2>/dev/null || true
if [ -n "${USER:-}" ] && [ "${USER}" != "root" ] && [ -d "/Users/${USER}" ]; then
    sudo -u "${USER}" mkdir -p "/Users/${USER}/Music/StudioEQ_Master_Pro/output" 2>/dev/null || true
fi
exit 0
EOF
chmod 755 "${PKG_SCRIPTS}/postinstall"

cat > "${PKG_RESOURCES}/welcome.html" <<'EOF'
<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; font-size: 13px; line-height: 1.5; padding: 8px 4px; }
  h2 { margin-top: 0; font-size: 18px; }
  ul { padding-left: 20px; }
  li { margin-bottom: 5px; }
</style>
</head>
<body>
  <h2>StudioEQ Master Pro v3.1.0 — Setup Completo Offline (Apple Silicon)</h2>
  <p>Questo programma di installazione configura sul tuo Mac (M1 / M2 / M3 / M4) l'intero ambiente di produzione audio <b>100% autonomo e funzionante anche senza connessione Internet</b>:</p>
  <ul>
    <li><b>StudioEQ Master Pro.app</b> in <code>/Applications</code> (CoreAudio + Metal Retina, EQ Master 7 bande + EQ a 3 bande <i>Bassi / Medi / Alti</i> per ogni traccia isolata).</li>
    <li><b>Motore AI Completo incluso offline</b> (Python 3.12 <code>arm64</code> + PyTorch GPU Apple Metal MPS + FFmpeg standalone).</li>
    <li><b>Tutti i 4 Modelli Neurali già inclusi</b>: <i>Demucs v4 Fine-Tuned</i>, <i>DrumSep v3</i> (Cassa/Rullante/Piatti/Toms + 24 campioni studio 24-bit), <i>HTDemucs-6s</i> e <i>Apollo 44.1 kHz</i>.</li>
  </ul>
  <p>Clicca su <b>Continua</b> per installare.</p>
</body>
</html>
EOF

pkgbuild --analyze --root "${PKG_ROOT}" "${DIST_DIR}/component.plist"
"${BUNDLED_PY}" - "${DIST_DIR}/component.plist" <<'PYEOF'
import plistlib
import sys

p = sys.argv[1]
with open(p, "rb") as f:
    items = plistlib.load(f)
for entry in items:
    if isinstance(entry, dict) and "BundleIsRelocatable" in entry:
        entry["BundleIsRelocatable"] = False
with open(p, "wb") as f:
    plistlib.dump(items, f)
PYEOF

pkgbuild \
    --root "${PKG_ROOT}" \
    --component-plist "${DIST_DIR}/component.plist" \
    --scripts "${PKG_SCRIPTS}" \
    --identifier "com.studioeq.masterpro.full" \
    --version "${APP_VERSION}" \
    --install-location "/" \
    "${DIST_DIR}/StudioEQ_Component.pkg"

# Libera subito lo spazio disco di PKG_ROOT prima di productbuild e hdiutil
rm -rf "${PKG_ROOT}"

cat > "${DIST_DIR}/distribution.xml" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>StudioEQ Master Pro v${APP_VERSION} — Setup Completo (Apple Silicon)</title>
    <welcome file="welcome.html" mime-type="text/html"/>
    <options customize="never" require-scripts="false" hostArchitectures="arm64"/>
    <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
    <choices-outline>
        <line choice="default">
            <line choice="com.studioeq.masterpro.full"/>
        </line>
    </choices-outline>
    <choice id="default"/>
    <choice id="com.studioeq.masterpro.full" visible="false">
        <pkg-ref id="com.studioeq.masterpro.full"/>
    </choice>
    <pkg-ref id="com.studioeq.masterpro.full" version="${APP_VERSION}" onConclusion="none">StudioEQ_Component.pkg</pkg-ref>
</installer-gui-script>
EOF

productbuild \
    --distribution "${DIST_DIR}/distribution.xml" \
    --resources "${PKG_RESOURCES}" \
    --package-path "${DIST_DIR}" \
    "${STAGING_DIR}/Installa StudioEQ Master Pro.pkg"

rm -f "${DIST_DIR}/StudioEQ_Component.pkg"

# Launcher rapido 1-Click che installa il .pkg sbloccando automaticamente Gatekeeper anche su macOS Sequoia 15+
cat > "${STAGING_DIR}/Installa Subito (1-Click).command" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_FILE="${SRC_DIR}/Installa StudioEQ Master Pro.pkg"

clear || true
echo "======================================================================"
echo "  INSTALLAZIONE COMPLETA OFFLINE — StudioEQ Master Pro v3.1.0 (Mac M1)"
echo "======================================================================"
echo ""
echo "Questo setup installa StudioEQ Master Pro.app + Python 3.12 arm64 +"
echo "PyTorch GPU Metal M1 + FFmpeg + tutti i Modelli AI (100% offline)."
echo ""
echo "Inserisci la password del Mac quando richiesta per completare il setup:"
sudo installer -pkg "${PKG_FILE}" -target /
sudo xattr -cr "/Applications/StudioEQ Master Pro.app" 2>/dev/null || true
mkdir -p "$HOME/Music/StudioEQ_Master_Pro/output"

echo ""
echo "[OK] Installazione completata! Avvio di StudioEQ Master Pro..."
open "/Applications/StudioEQ Master Pro.app" || true

if [ -t 0 ]; then
    echo ""
    read -r -p "Fatto! Premi INVIO per chiudere questa finestra..."
fi
EOF
chmod 755 "${STAGING_DIR}/Installa Subito (1-Click).command"
codesign --force --sign - "${STAGING_DIR}/Installa Subito (1-Click).command" || true

cat > "${STAGING_DIR}/LEGGIMI_INSTALLAZIONE_MAC.txt" <<'EOF'
================================================================================
  STUDIOEQ MASTER PRO v3.1.0 — SETUP COMPLETO OFFLINE (MAC M1 / M2 / M3 / M4)
================================================================================

QUESTO PACCHETTO È 100% AUTONOMO E COMPLETO DI TUTTE LE DIPENDENZE:
  Include già al suo interno:
  • StudioEQ Master Pro.app nativa Apple Silicon (CoreAudio + Metal Retina)
  • Python 3.12 arm64 standalone + PyTorch GPU Metal MPS + Pedalboard + SciPy
  • FFmpeg arm64 standalone per import/export MP3 320 kbps e WAV 24-bit
  • TUTTI i Modelli AI già pre-scaricati offline (Demucs v4 htdemucs_ft,
    DrumSep v3 a 4 sub-stems, HTDemucs-6s Chitarra, Apollo 44.1 kHz e i
    24 campioni batteria studio 24-bit).
  NON serve Python sul Mac, NON serve Homebrew e NON serve Internet!

--------------------------------------------------------------------------------
COME INSTALLARE SU QUALSIASI MAC APPLE SILICON (M1 / M2 / M3 / M4):
--------------------------------------------------------------------------------
OPZIONE 1 — PROCEDURA GUIDATA GRAFICA (.pkg):
  1. Fai Ctrl + Clic (Tasto Destro) su "Installa StudioEQ Master Pro.pkg"
     e scegli "Apri" -> clicca "Apri" -> segui la finestra guidata (Continua ->
     Installa).
  2. (Se su macOS 15 Sequoia compare un avviso di sicurezza per pacchetti non
     notarizzati Apple: vai in Impostazioni di Sistema -> Privacy e sicurezza ->
     scorri in basso e clicca "Apri comunque", oppure usa l'Opzione 2 qui sotto!).

OPZIONE 2 — INSTALLAZIONE RAPIDA DA TERMINALE (ZERO BLOCCHI GATEKEEPER):
  Apri Terminale (Cmd + Spazio -> Terminale) e incolla:

  sudo installer -pkg "/Volumes/StudioEQ Master Pro Setup/Installa StudioEQ Master Pro.pkg" -target /

  Poi apri "StudioEQ Master Pro" dal Launchpad o dalla cartella Applicazioni!
================================================================================
EOF

echo "[6/7] Creazione immagine disco ${DMG_FILENAME}..."
rm -f "${DMG_OUTPUT}"

for attempt in 1 2 3; do
    if hdiutil create \
        -volname "StudioEQ Master Pro Setup" \
        -srcfolder "${STAGING_DIR}" \
        -ov \
        -format UDZO \
        -imagekey zlib-level=1 \
        "${DMG_OUTPUT}"; then
        break
    fi
    echo "      [Avviso] Volume impegnato (tentativo ${attempt}/3), riprovo tra 3s..."
    sleep 3
done

if [ ! -f "${DMG_OUTPUT}" ]; then
    echo "[ERRORE] Creazione del file DMG fallita." >&2
    exit 1
fi

xattr -cr "${DMG_OUTPUT}" || true
codesign --force --sign - "${DMG_OUTPUT}" || true
rm -rf "${STAGING_DIR}" "${PKG_SCRIPTS}" "${PKG_RESOURCES}"

echo "[7/7] Verifica finale pacchetto Setup Completo DMG:"
ls -lh "${DMG_OUTPUT}"
if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${DMG_OUTPUT}"
fi

echo "======================================================================"
echo "  SETUP COMPLETO OFFLINE PRONTO: ${DMG_OUTPUT}"
echo "======================================================================"
