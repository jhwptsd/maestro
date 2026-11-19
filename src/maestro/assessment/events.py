# Common event format and shared context for every assessment module (new file).
#
# Each assessment file (intonation, rhythm, expression, tone, bowing, posture) returns a plain JSON-compatible dict
#     {"module", "available", "score" (0-100 or None), "summary": {...}, "events": [event, ...]}
# whose events all have the same shape (`make_event`), so assessment/synthesize.py can merge them into one report and
# gui/ can draw them without knowing which module made them:
#     {"module", "category", "start", "end" (seconds in the performance), "measure", "beat", "score_idx", "perf_idx",
#      "severity" (0 fine ... 1 severe), "value", "expected", "unit", "message",
#      "causes": [{"cause" (slug), "explanation", "confidence" (0-1), "evidence": {...}}, ...] (most likely first)}
# Causes are hypotheses read off the measurements (always with the evidence they rest on), not certainties.
#
# `Context` is a read-only view of the Performance (score, notes, alignment, audio, pose/motion, visual pitch, optional
# neural results) that the modules share for locating events and finding causes, so each module takes the same input.
import numpy as np


def jsonable(x):
    """Recursively turn numpy scalars/arrays into plain Python so json.dumps accepts the result."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def ramp(value, good, bad):
    """Severity 0 at `good`, 1 at `bad` (either order), linear in between."""
    return float(np.clip((value - good) / (bad - good), 0.0, 1.0))


def cause(slug, explanation, confidence, **evidence):
    """One root-cause hypothesis with the measurements it rests on."""
    return {"cause": slug, "explanation": explanation, "confidence": round(float(np.clip(confidence, 0, 1)), 2),
            "evidence": jsonable(evidence)}


def make_event(module, category, start, end, severity, message, causes=(), ctx=None, score_idx=None, perf_idx=None,
               value=None, expected=None, unit=None):
    """An event in the common format. measure/beat are looked up from score_idx (or the time) when ctx is given;
    causes are sorted most likely first."""
    measure = beat = None
    if ctx is not None:
        measure, beat = ctx.position(score_idx, start)
    return jsonable({"module": module, "category": category, "start": start, "end": end,
                     "measure": measure, "beat": beat, "score_idx": score_idx, "perf_idx": perf_idx,
                     "severity": round(float(severity), 3), "value": value, "expected": expected, "unit": unit,
                     "message": message, "causes": sorted(causes, key=lambda c: -c["confidence"])})


def module_result(module, events, n_units, summary=None, available=True):
    """Wrap events with a 0-100 score: 100 minus the severity summed over events per unit judged (notes or windows)."""
    score = 100 * (1 - min(1.0, sum(e["severity"] for e in events) / max(n_units, 1))) if available and n_units else None
    return jsonable({"module": module, "available": available, "score": None if score is None else round(score, 1),
                     "summary": summary or {}, "events": events})


def unavailable(module, reason):
    return {"module": module, "available": False, "score": None, "summary": {"reason": reason}, "events": []}


class Context:
    """Read-only view of a Performance for the assessment modules. Build once: Context(perf)."""

    def __init__(self, perf):
        self.score = perf.score_features or []
        feats = perf.audio_features or {}
        self.notes = feats.get("notes", [])
        frames = feats.get("frames", [])
        self.frame_times = np.array([f["time"] for f in frames])
        self.f0 = np.array([f["estimated_pitch"] for f in frames])
        self.alignment = perf.alignment or []
        self.y = perf.audio["samples"] if perf.audio else None
        self.sr = perf.audio["sr"] if perf.audio else None
        self.motion = perf.motion_features                  # vision/motion_features output or None
        self.pose = perf.pose
        self.assessment = perf.assessment or {}
        self.neural = (perf.assessment or {}).get("neural") or {}      # assessment/neural.py windows, if enabled
        self.matches = [(a["score_idx"], a["perf_idx"]) for a in self.alignment if a["perf_idx"] is not None]
        self._by_score = {s: p for s, p in self.matches}

    @classmethod
    def from_parts(cls, score, notes, alignment):
        """Minimal context when only the score, performed notes and alignment are at hand (no audio, no pose)."""
        from maestro.utils.Performance import Performance
        perf = Performance()
        perf.score_features, perf.audio_features, perf.alignment = score, {"notes": notes}, alignment
        return cls(perf)

    def played(self, score_idx):
        """The performed note (dict) matched to a score note, or None."""
        p = self._by_score.get(score_idx)
        return None if p is None else self.notes[p]

    def position(self, score_idx, time=None):
        """(measure, beat) of a score note; when only a time is known, of the nearest matched performed note."""
        if score_idx is None and time is not None and self.matches:
            onsets = np.array([self.notes[p]["onset"] for _, p in self.matches])
            score_idx = self.matches[int(np.argmin(np.abs(onsets - time)))][0]
        if score_idx is None:
            return None, None
        n = self.score[score_idx]
        return n.get("measure"), n.get("beat")

    def strokes(self):
        """Bow strokes from the pose analysis ([] when there is no pose)."""
        bow = (self.motion or {}).get("bow")
        return bow["strokes"] if bow else []

    def bow_sync_offset(self, window=0.3):
        """Median lag (s) of the bow direction changes behind the note onsets they belong to. A constant offset is the
        video/audio latency and pose smoothing, not a fault, so judging code subtracts it. 0 without a pose."""
        if not hasattr(self, "_bow_offset"):
            ends = np.array([s["end"] for s in self.strokes()])
            lags = []
            if len(ends):
                for _, p in self.matches:
                    lag = ends[np.argmin(np.abs(ends - self.notes[p]["onset"]))] - self.notes[p]["onset"]
                    if abs(lag) <= window:
                        lags.append(lag)
            self._bow_offset = float(np.median(lags)) if len(lags) >= 5 else 0.0
        return self._bow_offset

    def stroke_at(self, time):
        """The bow stroke sounding at `time`, or None."""
        return next((s for s in self.strokes() if s["start"] <= time <= s["end"]), None)

    def series(self, key, start, end):
        """Slice of a per-frame pose-time series (e.g. "shoulder_tilt_deg") between two times, or None."""
        values = (self.motion or {}).get(key)
        if values is None:
            return None
        t = np.asarray(self.motion["times"])
        return np.asarray(values)[(t >= start) & (t <= end)]
