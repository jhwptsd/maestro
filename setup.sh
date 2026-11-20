#!/usr/bin/env bash
# Project setup for Linux and macOS: installs ffmpeg and git if missing, Maestro with the default CREPE backend and the
# fusion encoders' runtime (torch), and VioPose (the 3D pose model behind the movement analysis, in its own Python 3.9
# environment under third_party/).
# Usage: ./setup.sh [--skip-viopose] [--training] [--neural]
#   --training   also install what training the models needs (music21, yt-dlp) and show the data / training commands
#   --neural     also install transformers for the optional CLAP tone/expression judgement (analyze(neural="clap"))
# Install the CUDA-specific PyTorch build first if GPU acceleration is desired.
set -euo pipefail
cd "$(dirname "$0")"

skip_viopose=0
extras="crepe,fusion,dev"
training=0
for arg in "$@"; do
    case "$arg" in
        --skip-viopose) skip_viopose=1 ;;
        --training) extras="$extras,training"; training=1 ;;
        --neural) extras="$extras,neural" ;;
        *) echo "Unknown option $arg (use --skip-viopose, --training, --neural)" >&2; exit 1 ;;
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

python3 -m pip install -e ".[$extras]"

if [ "$skip_viopose" -eq 1 ]; then
    echo "Skipping VioPose: the movement analysis stays unavailable until you run: python3 src/scripts/setup_viopose.py"
elif ! python3 src/scripts/setup_viopose.py; then
    echo "WARNING: VioPose setup did not finish (see above). Everything else works; re-run: python3 src/scripts/setup_viopose.py" >&2
fi

if [ "$training" -eq 1 ]; then
    cat <<'EOF'

Training the models (see README.md): data goes to data/, trained models to models/.
  python3 src/training/install_data.py audio            # MusicNet violin recordings (large download)
  python3 src/training/install_data.py video            # Creative Commons violin videos -> VioPose poses (needs VioPose)
  python3 src/training/train_fusion.py audio --dataset musicnet
  python3 src/training/train_fusion.py video --dataset youtube-cc
  python3 src/training/train_fusion.py score --music21
EOF
fi
