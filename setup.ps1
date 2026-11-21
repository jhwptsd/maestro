# Project setup for Windows: installs ffmpeg (if missing), Maestro with the default CREPE backend and the fusion encoders'
# runtime (torch), and VioPose (the 3D pose model behind the movement analysis, in its own Python 3.9 environment under
# third_party\).
# Usage: .\setup.ps1 [-SkipViopose] [-Training] [-Neural] [-NoVenv] [-NoGpu]
#   -Training   also install what training the models needs (music21, yt-dlp) and show the data / training commands
#   -Neural     also install transformers for the optional CLAP tone/expression judgement (analyze(neural="clap"))
#   -NoVenv     install into the Python you ran this with instead of a virtual environment (not recommended)
#   -NoGpu      do not install the GPU runtime (ONNX Runtime with DirectML, which runs the CREPE pitch network on any
#               DirectX 12 GPU: AMD, Intel or NVIDIA); the CPU runtime is used instead
# Maestro is installed into its own virtual environment (.venv in this folder, or %LOCALAPPDATA%\maestro\venv when the folder
# is inside OneDrive, which would otherwise sync thousands of files). That keeps it apart from packages already installed in
# your global Python (an old torch there makes pip fail with "WinError 2 ... .exe.deleteme"). Activate it afterwards with the
# command printed at the end.
# Install the CUDA-specific PyTorch build first if GPU acceleration is desired.
param([switch]$SkipViopose, [switch]$Training, [switch]$Neural, [switch]$NoVenv, [switch]$NoGpu)
$ErrorActionPreference = "Stop"

function Test-Ffmpeg {            # both programs exist AND run (a stale or half-installed copy on the PATH does not count)
    foreach ($tool in "ffmpeg", "ffprobe") {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { return $false }
        & $tool -version *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
    }
    return $true
}

if (Test-Ffmpeg) {
    Write-Host "ffmpeg already installed."
} else {
    Write-Host "ffmpeg / ffprobe not found or not working; installing via winget..."
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    # winget only updates the PATH of NEW terminals: make the fresh install visible to this session
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (Test-Ffmpeg)) {
        # still not found: winget did install it, but into its own package folder that the PATH does not list (yet).
        # Find it there and add it to this session and, permanently, to the user's PATH.
        $packages = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
        $probe = Get-ChildItem $packages -Filter ffprobe.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($probe) {
            $bin = $probe.DirectoryName
            $env:Path = "$bin;$env:Path"
            $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
            if (($userPath -split ";") -notcontains $bin) {
                [Environment]::SetEnvironmentVariable("Path", "$userPath;$bin", "User")
            }
            Write-Host "Added $bin to your PATH."
        }
    }
    if (Test-Ffmpeg) {
        Write-Host "ffmpeg is ready. Open a NEW terminal (and reload VS Code) before running Maestro, so that its PATH includes ffmpeg."
    } else {
        Write-Warning "ffmpeg could not be made available. Run 'winget install --id Gyan.FFmpeg -e' yourself, open a new terminal and re-run this script."
    }
}

if (-not (Get-Command audiveris -ErrorAction SilentlyContinue) -and
    -not (Test-Path "$env:ProgramFiles\Audiveris\Audiveris.exe")) {
    Write-Host "Audiveris not found; installing via winget (needed for PDF/image scores)..."
    winget install --id audiveris.org.Audiveris -e --accept-source-agreements --accept-package-agreements
} else {
    Write-Host "Audiveris already installed."
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10 or newer was not found on PATH. Install it from python.org (tick 'Add python.exe to PATH') and re-run."
}
$python = "python"
$activate = $null
if (-not $NoVenv) {
    $venv = if ((Get-Location).Path -like "*OneDrive*") { Join-Path $env:LOCALAPPDATA "maestro\venv" } else { Join-Path (Get-Location).Path ".venv" }
    if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
        Write-Host "Creating the Python environment $venv ..."
        python -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment $venv" }
    }
    $python = Join-Path $venv "Scripts\python.exe"
    $activate = Join-Path $venv "Scripts\Activate.ps1"
}

$extras = "crepe,fusion,dev"
if ($Training) { $extras += ",training" }
if ($Neural) { $extras += ",neural" }
& $python -m pip install -e ".[$extras]"
if ($LASTEXITCODE -ne 0) {
    throw "pip could not install Maestro (see the error above). If it names a file in Python's Scripts folder, packages in your global Python are in the way: re-run without -NoVenv, or delete the leftovers it names and any ~orch folder in site-packages."
}

# GPU: PyTorch cannot use an AMD or Intel GPU on Windows, but ONNX Runtime's DirectML provider runs the CREPE network (the main
# neural cost of the audio analysis) on any DirectX 12 GPU. It replaces the CPU build of onnxruntime that the install above brought.
$gpus = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object { $_.Name } |
          Where-Object { $_ -and $_ -notmatch "Microsoft Basic|Remote Display|Virtual|Parsec" })
if ($NoGpu) {
    Write-Host "Skipping the GPU runtime (-NoGpu): the pitch network runs on the CPU."
} elseif ($gpus.Count -eq 0) {
    Write-Host "No GPU found: the pitch network runs on the CPU."
} else {
    Write-Host "GPU: $($gpus -join ', '). Installing ONNX Runtime with DirectML so the pitch network can use it..."
    & $python -m pip uninstall -y onnxruntime
    & $python -m pip install --force-reinstall --no-deps onnxruntime-directml      # (uninstalling onnxruntime removed files DirectML shares)
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "onnxruntime-directml could not be installed; restoring the CPU runtime. Maestro still works, on the CPU."
        & $python -m pip install onnxruntime
    }
    if ($gpus -match "NVIDIA") {
        Write-Host "NVIDIA GPU: for the other networks (CLAP, training) also install the CUDA build of PyTorch into this environment: https://pytorch.org/get-started/locally/"
    }
}

if ($SkipViopose) {
    Write-Host "Skipping VioPose: the movement analysis stays unavailable until you run: python src/scripts/setup_viopose.py (after activating the environment)"
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "git not found; installing via winget (needed to download VioPose)..."
        winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements
        # make the fresh install visible to this session
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    }
    Write-Host "Setting up VioPose (large download; creates its own Python 3.9 environment)..."
    try {
        & $python src/scripts/setup_viopose.py
        if ($LASTEXITCODE -ne 0) { throw "setup_viopose.py exited with code $LASTEXITCODE" }
    } catch {
        Write-Warning "VioPose setup did not finish: $_"
        Write-Warning "Everything else works. Re-run later with: python src/scripts/setup_viopose.py (after activating the environment)"
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

Write-Host ""
Write-Host "Compute devices Maestro will use:"
& $python -c "from maestro.utils.device import describe; print('  ' + describe())"
Write-Host "(VioPose, the pose model, runs on the CPU on Windows: TensorFlow has no GPU support there.)"
if ($activate) {
    Write-Host "Done. Activate the environment in every new terminal before using Maestro:"
    Write-Host "  & `"$activate`""
} else {
    Write-Host "Done."
}
