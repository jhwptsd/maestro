# Project setup for Windows: installs ffmpeg (if missing), Maestro with default CREPE backend, and VioPose (the 3D pose
# model behind the movement analysis, in its own Python 3.9 environment under third_party\).
# Usage: .\setup.ps1 [-SkipViopose]
# Install the CUDA-specific PyTorch build first if GPU acceleration is desired.
param([switch]$SkipViopose)
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

python -m pip install -e ".[crepe,dev]"

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
