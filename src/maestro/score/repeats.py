# Repeat / navigation handling for scores (new file)
#
# Written music is not played in page order: repeat barlines, 1st/2nd endings (voltas),
# D.C./D.S. (da capo / dal segno), "To Coda" and "Fine" all reorder or repeat measures. The student's
# audio follows the *played* order, so alignment/ needs the score unrolled. score/musicxml.py reads
# the raw markings into `Measure` records and calls `playback_order` to get the played sequence of
# measure indices; it then lays the notes out on the played timeline.
#
# Conventions: on the D.C./D.S. pass repeats are not taken and the *last* ending is played; "Fine"
# and "To Coda" only act after a D.C./D.S. jump.
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Measure:
    number: int                                   # written measure number
    length: float                                 # quarter notes
    notes: list = field(default_factory=list)     # raw note dicts (onsets relative to the measure)
    tempos: List[Tuple[float, float]] = field(default_factory=list)  # (offset in quarters, bpm)
    fwd: bool = False                             # forward repeat barline
    back: int = 0                                 # backward repeat: total plays of the section (0 = none)
    endings: Tuple[int, ...] = ()                 # volta numbers this measure belongs to
    segno: Optional[str] = None                   # ids are "" when the marking has no id
    coda: Optional[str] = None
    dacapo: bool = False
    dalsegno: Optional[str] = None
    tocoda: Optional[str] = None
    fine: bool = False


def words_to_navigation(text):
    """Map direction text like 'D.C. al Fine' / 'D.S. al Coda' / 'To Coda' / 'Fine' to Measure kwargs."""
    t = re.sub(r"[^a-z]", "", text.lower())
    nav = {}
    if t.startswith("dc") or "dacapo" in t:
        nav["dacapo"] = True
    elif t.startswith("ds") or "dalsegno" in t:
        nav["dalsegno"] = ""
    elif "tocoda" in t:
        nav["tocoda"] = ""
    elif t == "fine":
        nav["fine"] = True
    return nav


def playback_order(measures: List[Measure], max_len=10000) -> List[int]:
    """Indices of `measures` in the order they are played."""
    segnos = {m.segno: i for i, m in enumerate(measures) if m.segno is not None}
    codas = {m.coda: i for i, m in enumerate(measures) if m.coda is not None}
    last_ending = max((e for m in measures for e in m.endings), default=0)
    order, i = [], 0
    start, passes = 0, 1       # current repeat section: first measure, which pass we are on
    taken = {}                 # backward-repeat measure -> times we already jumped back
    jumped = False             # True after a D.C./D.S.

    while i < len(measures) and len(order) < max_len:
        m = measures[i]
        if i and measures[i - 1].endings and not m.endings:
            start, passes = i, 1                       # volta group finished: new implicit section
        if m.fwd and start != i:
            start, passes = i, 1
        if m.endings and (last_ending not in m.endings if jumped else passes not in m.endings):
            i += 1                                     # not the ending for this pass
            continue
        order.append(i)

        if jumped:
            if m.fine:
                break
            if m.tocoda is not None and m.tocoda in codas:
                i = codas[m.tocoda]
                continue
        elif m.dacapo:
            jumped, i = True, 0
            continue
        elif m.dalsegno is not None and m.dalsegno in segnos:
            jumped, i = True, segnos[m.dalsegno]
            continue

        if m.back and not jumped:
            if taken.get(i, 0) < m.back - 1:
                taken[i] = taken.get(i, 0) + 1
                passes += 1
                i = start
                continue
            start, passes = i + 1, 1                   # section done
        i += 1
    return order
