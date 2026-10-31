#!/usr/bin/env bash
# ==============================================================================
# StudioEQ Master Pro v3.1.0 — Setup Automatico Motore AI Apple Silicon (M1/M2/M3/M4)
# Configura Python 3.12 (arm64 nativo) + PyTorch Metal MPS + Demucs v4 + DrumSep v3
# + Pedalboard + FFmpeg standalone in ~/Library/Application Support/StudioEQ_Master_Pro
# ZERO sudo richiesto, ZERO Homebrew obbligatorio!
# ==============================================================================
set -euo pipefail

export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPPORT_DIR="$HOME/Library/Application Support/StudioEQ_Master_Pro"
VENV_DIR="${SUPPORT_DIR}/.venv"
BIN_DIR="${SUPPORT_DIR}/bin"

# Trova requirements-macos-m1.txt (accanto allo script, dentro /Applications/.../Resources o nel repo)
REQ_FILE=""
for cand in \
    "${SCRIPT_DIR}/requirements-macos-m1.txt" \
    "/Applications/StudioEQ Master Pro.app/Contents/Resources/requirements-macos-m1.txt" \
    "${SCRIPT_DIR}/../Resources/requirements-macos-m1.txt" \
    "${SCRIPT_DIR}/macos/requirements-macos-m1.txt"
do
    if [ -f "${cand}" ]; then
        REQ_FILE="${cand}"
        break
    fi
done

if [ -z "${REQ_FILE}" ]; then
    echo "[ERRORE] File requirements-macos-m1.txt non trovato." >&2
    exit 1
fi

echo "======================================================================"
echo "  StudioEQ Master Pro v3.1.0 — Configurazione Motore AI Apple Silicon"
echo "======================================================================"
echo "  Architettura Mac : $(uname -m)"
echo "  Ambiente AI      : ${VENV_DIR}"
echo ""

mkdir -p "${SUPPORT_DIR}" "${BIN_DIR}" "$HOME/Music/StudioEQ_Master_Pro/output"

# 1. Installa `uv` in ~/.local/bin se non presente (scarica Python 3.12 arm64 senza Xcode/sudo!)
if ! command -v uv >/dev/null 2>&1; then
    echo "[1/4] Installazione gestore pacchetti ultra-rapido 'uv' (arm64 nativo)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo "[1/4] Gestore 'uv' rilevato: $(command -v uv)"
fi

# 2. Crea ambiente virtuale Python 3.12 nativo Apple Silicon (aarch64-apple-darwin)
echo "[2/4] Creazione ambiente Python 3.12 nativo Apple Silicon..."
uv venv --python 3.12 --clear "${VENV_DIR}"

# 3. Installa PyTorch (Metal MPS), Demucs v4, Pedalboard, SciPy, FFmpeg standalone
echo "[3/4] Download e installazione librerie AI & DSP (GPU Metal MPS)..."
uv pip install --python "${VENV_DIR}/bin/python3" -r "${REQ_FILE}"

# 4. Configura FFmpeg nativo arm64 (usa Homebrew se presente, altrimenti estrae il binario arm64 da imageio-ffmpeg)
echo "[4/4] Configurazione encoder/decoder FFmpeg nativo arm64 e verifica GPU Metal M1..."
if command -v ffmpeg >/dev/null 2>&1; then
    ln -sf "$(command -v ffmpeg)" "${BIN_DIR}/ffmpeg"
else
    "${VENV_DIR}/bin/python3" - "${BIN_DIR}/ffmpeg" <<'PYEOF'
import os
import shutil
import sys
import imageio_ffmpeg

dst = sys.argv[1]
src = imageio_ffmpeg.get_ffmpeg_exe()
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy2(src, dst)
os.chmod(dst, 0o755)
print(f"      FFmpeg standalone arm64 installato in: {dst}")
PYEOF
fi

"${VENV_DIR}/bin/python3" - <<'PYEOF'
import platform
import torch
import demucs
import pedalboard
import scipy
import soundfile

mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"      Python     : {platform.python_version()} ({platform.machine()})")
print(f"      PyTorch    : {torch.__version__} | Accelerazione GPU Apple Metal (MPS): {'ATTIVA' if mps_ok else 'CPU Multi-Core'}")
print(f"      Demucs     : {demucs.__version__} | Pedalboard: {pedalboard.__version__}")
PYEOF

echo ""
echo "======================================================================"
echo "  MOTORE AI CONFIGURATO CON SUCCESSO SU APPLE SILICON!"
echo "  Ora puoi usare il pulsante 'Separa AI' e 'Guidata' in StudioEQ."
echo "======================================================================"
