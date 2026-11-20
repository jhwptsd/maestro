# Project setup for Windows: installs ffmpeg (if missing), Maestro with the default CREPE backend and the fusion encoders'
# runtime (torch), and VioPose (the 3D pose model behind the movement analysis, in its own Python 3.9 environment under
# third_party\).
# Usage: .\setup.ps1 [-SkipViopose] [-Training] [-Neural]
#   -Training   also install what training the models needs (music21, yt-dlp) and show the data / training commands
#   -Neural     also install transformers for the optional CLAP tone/expression judgement (analyze(neural="clap"))
# Install the CUDA-specific PyTorch build first if GPU acceleration is desired.
param([switch]$SkipViopose, [switch]$Training, [switch]$Neural)
$ErrorActionPreference = "Stop"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg not found; installing via winget..."
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    Write-Host "ffmpeg installed. Open a new terminal (and reload VS Code) so PATH updates take effect."
} else {
    Write-Host "ffmpeg already installed."
}

if (-not (Get-Command audiveris -ErrorAction SilentlyContinue) -and
    -not (Test-Path "$env:ProgramFiles\Audiveris\Audiveris.exe")) {
    Write-Host "Audiveris not found; installing via winget (needed for PDF/image scores)..."
    winget install --id audiveris.org.Audiveris -e --accept-source-agreements --accept-package-agreements
} else {
    Write-Host "Audiveris already installed."
}

$extras = "crepe,fusion,dev"
if ($Training) { $extras += ",training" }
if ($Neural) { $extras += ",neural" }
python -m pip install -e ".[$extras]"

if ($SkipViopose) {
    Write-Host "Skipping VioPose: the movement analysis stays unavailable until you run: python src/scripts/setup_viopose.py"
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "git not found; installing via winget (needed to download VioPose)..."
        winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements
        # make the fresh install visible to this session
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    }
    Write-Host "Setting up VioPose (large download; creates its own Python 3.9 environment)..."
    try {
        python src/scripts/setup_viopose.py
        if ($LASTEXITCODE -ne 0) { throw "setup_viopose.py exited with code $LASTEXITCODE" }
    } catch {
        Write-Warning "VioPose setup did not finish: $_"
        Write-Warning "Everything else works. Re-run later with: python src/scripts/setup_viopose.py"
    }
}

if ($Training) {
    Write-Host ""
    Write-Host "Training the models (see README.md): data goes to data\, trained models to models\."
    Write-Host "  python src/training/install_data.py audio     # MusicNet violin recordings (large download)"
    Write-Host "  python src/training/install_data.py video     # Creative Commons violin videos -> VioPose poses (needs VioPose)"
    Write-Host "  python src/training/train_fusion.py audio --dataset musicnet"
    Write-Host "  python src/training/train_fusion.py video --dataset youtube-cc"
    Write-Host "  python src/training/train_fusion.py score --music21"
}
