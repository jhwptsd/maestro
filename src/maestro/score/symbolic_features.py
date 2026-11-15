# Detect pitch, rhythm, dynamics, and other symbolic features from a processed score
#
# Derives features from score_representation notes; stored in Performance.score_features. Used
# by alignment/ (expected pitch/timing per note), assessment/ (expected values), and fusion/score_encoder.py.
from maestro.utils.audio_math import midi_to_hz


def extract_symbolic_features(score):
    """Score -> list of per-note dicts (the note fields plus expected frequency `hz`)."""
    return [{**n.to_dict(), "hz": float(midi_to_hz(n.midi_pitch))} for n in score.notes]
