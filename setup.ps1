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

if (-not (Get-Command audiveris -ErrorAction SilentlyContinue) -and
    -not (Test-Path "$env:ProgramFiles\Audiveris\Audiveris.exe")) {
    Write-Host "Audiveris not found; installing via winget (needed for PDF/image scores)..."
    winget install --id audiveris.org.Audiveris -e --accept-source-agreements --accept-package-agreements
} else {
    Write-Host "Audiveris already installed."
}

python -m pip install -e ".[dev]"
