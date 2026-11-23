Scripts (run from the repository root; training scripts are in `src/training/`)
- analyze.py VIDEO SCORE [OUT_DIR] [--pitch-backend crepe|yin|pyin|basic_pitch] [--neural clap] [--feedback NAME] [--no-feedback] [--model NAME]: command-line analysis (prints its progress stage by stage; when an AI provider is connected an AI model also writes feedback.json: see the README); writes report.html (a player with live charts; the media is saved next to it) and assessment.json.
- serve.py [PORT]: starts the web GUI (upload form, then the player page) at http://127.0.0.1:8000.
- setup_viopose.py [--dry-run] [--skip-weights] [--python PATH] [--dir DIR]: installs VioPose (clone, Python 3.9 environment, the packages inference.py imports, gdown, weights) under third_party/; called by setup.ps1 / setup.sh, safe to re-run.
- convert_viopose.py RESULT.npy OUT.json: converts a VioPose result.npy you produced yourself to Maestro's pose JSON (for `analyze(..., pose_file=...)`).
- bench_pitch.py [FILE]: times every installed pitch backend (first call vs steady state, x real time) so you can compare CREPE, pYIN, YIN and Basic Pitch on your hardware.
- tune_denoise.py: picks the audio-denoising defaults from synthetic noisy/clean pairs, rejecting any setting that dulls vibrato, bow noise or pitch (run with PYTHONPATH=src:src/tests).
