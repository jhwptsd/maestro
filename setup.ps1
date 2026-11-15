# Project setup for Windows: installs ffmpeg (if missing) and the maestro package.
# Usage: .\setup.ps1
$ErrorActionPreference = "Stop"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg not found; installing via winget..."
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    Write-Host "ffmpeg installed. Open a new terminal (and reload VS Code) so PATH updates take effect."
} else {
    Write-Host "ffmpeg already installed."
}

python -m pip install -e ".[dev]"
