#!/usr/bin/env bash
# Project setup for Linux and macOS: installs ffmpeg and git if missing, Maestro with the default CREPE backend (torch, ONNX
# Runtime), and VioPose (the 3D pose model behind the movement analysis, in its own Python 3.9
# environment under third_party/).
# Usage: ./setup.sh [--skip-viopose] [--neural] [--no-venv] [--local-model]
#   --neural     also install transformers for the optional CLAP tone/expression judgement (analyze(neural="clap"))
#   --no-venv    install into the Python you ran this with instead of a virtual environment (not recommended)
# Maestro is installed into its own virtual environment (.venv in this folder), which keeps it apart from packages already
# installed in your global Python (pip cannot always replace those). Activate it afterwards: source .venv/bin/activate
# Install the CUDA-specific PyTorch build first if GPU acceleration is desired.
set -euo pipefail
cd "$(dirname "$0")"

skip_viopose=0
use_venv=1
extras="crepe,dev"
for arg in "$@"; do
    case "$arg" in
        --skip-viopose) skip_viopose=1 ;;
        --neural) extras="$extras,neural" ;;
        --no-venv) use_venv=0 ;;
        --local-model) local_model=1 ;;
        *) echo "Unknown option $arg (use --skip-viopose, --neural, --no-venv, --local-model)" >&2; exit 1 ;;
    esac
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

PY=python3
if [ "$use_venv" -eq 1 ]; then
    if [ ! -x .venv/bin/python ]; then
        python3 -m venv .venv || { echo "Could not create .venv (on Debian/Ubuntu: sudo apt-get install python3-venv)." >&2; exit 1; }
    fi
    PY=.venv/bin/python
fi
"$PY" -m pip install -e ".[$extras]"

# GPU hints (nothing is installed automatically: the right PyTorch build depends on the card and driver)
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU found: the default PyTorch wheels on Linux include CUDA, so CREPE and the other networks use it."
elif [ "$(uname)" = "Darwin" ]; then
    echo "macOS: PyTorch uses the Apple GPU (MPS) automatically."
elif command -v lspci >/dev/null 2>&1 && lspci | grep -qiE "(vga|3d|display).*(amd|advanced micro|radeon)"; then
    echo "AMD GPU found: install the ROCm build of PyTorch into this environment to use it, see https://pytorch.org/get-started/locally/ (choose ROCm)."
fi

if [ "${local_model:-0}" -eq 1 ]; then   # Ollama runs the small local model for the written feedback
    if command -v ollama >/dev/null 2>&1; then echo "Ollama already installed."
    elif command -v brew >/dev/null 2>&1; then brew install ollama
    else curl -fsSL https://ollama.com/install.sh | sh; fi
fi

if [ "$skip_viopose" -eq 1 ]; then
    echo "Skipping VioPose: the movement analysis stays unavailable until you run: python3 src/scripts/setup_viopose.py"
elif ! "$PY" src/scripts/setup_viopose.py; then
    echo "WARNING: VioPose setup did not finish (see above). Everything else works; re-run: python3 src/scripts/setup_viopose.py" >&2
fi


echo
echo "Compute devices Maestro will use:"
"$PY" -c "from maestro.utils.device import describe; print('  ' + describe())"
if [ "$use_venv" -eq 1 ]; then
    echo
    echo "Done. Activate the environment in every new terminal before using Maestro:  source .venv/bin/activate"
fi
