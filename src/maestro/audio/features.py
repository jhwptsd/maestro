# Audio feature extraction and analysis
#
# Aggregates the outputs of pitch.py, onset_detection.py, vibrato.py, and tone_quality.py into
# the common per-frame JSON below, stored in Performance.audio_features. Consumed by alignment/*,
# assessment/*, and fusion/audio_encoder.py.

# Should format into a common JSON
# ex. {
#    "time": 12.35,
#    "pitch_confidence": 0.97,
#    "rms_energy": 0.18,
#    "spectral_centroid": 3200.0,
#    "onset_probability": 0.82
#    "estimated_pitch": 440.0,
#    "estimated_tempo": 120.0
# }