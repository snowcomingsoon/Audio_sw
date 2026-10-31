#!/usr/bin/env bash
# ==============================================================================
# Doppio clic nel Finder su Mac M1 / M2 / M3 / M4 per creare automaticamente
# l'installer nativo StudioEQ_Master_Pro_macOS_M1_arm64.dmg
# ==============================================================================
set -euo pipefail

export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "======================================================================"
echo "  StudioEQ Master Pro — Creazione Automatica DMG per Mac Apple M1"
echo "======================================================================"
echo ""

if ! xcode-select -p >/dev/null 2>&1; then
    echo "[INFO] Strumenti da riga di comando Xcode non rilevati."
    echo "       Avvio installazione guidata Apple Xcode Command Line Tools..."
    xcode-select --install || true
    echo ""
    echo "Completa la finestra di installazione Apple e poi fai di nuovo"
    echo "doppio clic su Crea_DMG_su_Mac_M1.command."
    if [ -t 0 ]; then
        read -r -p "Premi INVIO per chiudere..."
    fi
    exit 1
fi

if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "[INFO] Compilatore Rust non trovato. Installazione automatica per Apple Silicon..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --target aarch64-apple-darwin
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

bash "${SCRIPT_DIR}/build_macos_m1_dmg.sh"

DMG_FILE="${REPO_ROOT}/StudioEQ_Master_Pro_v3.1.0_Setup_Completo_macOS_M1_arm64.dmg"
if [ -f "${DMG_FILE}" ] && command -v open >/dev/null 2>&1; then
    echo ""
    echo "Apertura del Finder con il file DMG evidenziato..."
    open -R "${DMG_FILE}" || true
fi

if [ -t 0 ]; then
    echo ""
    read -r -p "Fatto! Premi INVIO per chiudere questa finestra del Terminale..."
fi
