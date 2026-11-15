# Standard internal representation of a musical score
#
# The canonical score format. Produced by score/musicxml.py; consumed by
# score/symbolic_features.py, alignment/*, and fusion/score_encoder.py. Keep the schema stable:
# every other score-aware module depends on these field names.

# Should format into a common JSON
# ex. {
#     "measure": 12,
#     "beat": 2.0,
#     "pitch": "A5",
#     "midi_pitch": 81,
#     "duration": 0.5,
#     "articulation": "staccato"
# }