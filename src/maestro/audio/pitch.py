# Use CREPE to estimate fundamental-frequency (f0) and confidence from audio
#
# Called by audio/features.py. Its f0 output feeds assessment/intonation.py (via features) and
# audio/vibrato.py, audio/transcription.py.
# Optionally use Spotify Basic Pitch
# Use Librosa