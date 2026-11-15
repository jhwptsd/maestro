# Take in processed audio and processed score and align them
#
# Uses dtw.py to match Performance.audio_features to score_features; writes the mapping into
# Performance.alignment. Required by assessment/intonation.py, rhythym.py, tone.py, expression.py.