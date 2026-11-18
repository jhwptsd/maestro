#!/usr/bin/env bash
# Project setup for Linux and macOS: installs ffmpeg and git if missing, Maestro with the default CREPE backend, and
# VioPose (the 3D pose model behind the movement analysis, in its own Python 3.9 environment under third_party/).
# Usage: ./setup.sh [--skip-viopose]
# Install the CUDA-specific PyTorch build first if GPU acceleration is desired.
set -euo pipefail
cd "$(dirname "$0")"

skip_viopose=0
for arg in "$@"; do
    [ "$arg" = "--skip-viopose" ] && skip_viopose=1
done

install_package() {   # install_package <command> <apt package> <brew formula>
    if command -v "$1" >/dev/null 2>&1; then
        echo "$1 already installed."
    elif command -v brew >/dev/null 2>&1; then
        brew install "$3"
    elif command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install -y "$2"
    else
        echo "Please install $1 and re-run." >&2
        exit 1
    fi
}

install_package ffmpeg ffmpeg ffmpeg
install_package git git git
command -v audiveris >/dev/null 2>&1 || echo "Note: PDF/image scores need Audiveris (https://github.com/Audiveris/audiveris); MusicXML and MIDI scores work without it."

python3 -m pip install -e ".[crepe,dev]"

if [ "$skip_viopose" -eq 1 ]; then
    echo "Skipping VioPose: the movement analysis stays unavailable until you run: python3 src/scripts/setup_viopose.py"
elif ! python3 src/scripts/setup_viopose.py; then
    echo "WARNING: VioPose setup did not finish (see above). Everything else works; re-run: python3 src/scripts/setup_viopose.py" >&2
fi
