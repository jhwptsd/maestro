# Standard internal representation of a musical score
#
# The canonical score format. Produced by score/musicxml.py; consumed by
# score/symbolic_features.py, alignment/*, and fusion/score_encoder.py. Keep the schema stable:
# every other score-aware module depends on these field names.
#
# Should format into a common JSON
# ex. {
#     "measure": 12,
#     "beat": 2.0,
#     "pitch": "A5",
#     "midi_pitch": 81,
#     "duration": 0.5,
#     "articulation": "staccato"
# }
# `duration` is in quarter notes. onset/offset (seconds) are added so that alignment can compare
# against audio directly; they assume one constant tempo (the first tempo marking, default 120).
import json
from dataclasses import dataclass, asdict, field
from typing import List, Optional

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: int) -> str:
    """60 -> 'C4'."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


@dataclass
class Note:
    measure: int                  # 1-based measure number
    beat: float                   # 1-based beat position inside the measure
    pitch: str                    # scientific pitch name, e.g. "A5"
    midi_pitch: int
    duration: float               # quarter notes
    articulation: Optional[str]   # e.g. "staccato", "accent" or None
    onset: float                  # seconds from the start of the score
    offset: float                 # seconds

    def to_dict(self):
        return asdict(self)


@dataclass
class Score:
    notes: List[Note] = field(default_factory=list)  # sorted by (onset, midi_pitch)
    tempo: float = 120.0                             # quarter notes per minute

    def to_json(self, **kw):
        return json.dumps({"tempo": self.tempo, "notes": [n.to_dict() for n in self.notes]}, **kw)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        return cls([Note(**n) for n in d["notes"]], d["tempo"])
