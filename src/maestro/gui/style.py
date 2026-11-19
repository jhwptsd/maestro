# Shared colours and grading thresholds for the score viewer (new file; what is left of the old chart module).
#
# gui/viewer.py colours notes with these; the thresholds come from the assessment modules so the colours always agree
# with the events. The live charts of the player page (gui/player.html) are drawn in the browser, not here.
from maestro.assessment.intonation import CENTS_GOOD, CENTS_OK
from maestro.score.score_representation import NOTE_NAMES

GOOD, OK, BAD = "#2a9d3f", "#f08c00", "#d62828"


def note_name(midi):
    midi = int(round(midi))
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"
