Scripts
- analyze.py VIDEO SCORE [OUT_DIR]: command-line analysis; writes score.svg and report.html.
- serve.py [PORT]: starts the web GUI (upload form + charts) at http://127.0.0.1:8000.
- tune_denoise.py: picks the audio-denoising defaults from synthetic noisy/clean pairs, rejecting any setting that dulls vibrato, bow noise or pitch (run with PYTHONPATH=src:src/tests).
- bench_pitch.py [FILE]: times every installed pitch backend (first call vs steady state, x real time) so you can compare CREPE, pYIN, YIN and Basic Pitch on your hardware.
