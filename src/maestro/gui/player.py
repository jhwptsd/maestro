# Data for the player page (new file): everything the browser needs to play the recording with live charts.
#
# Input: the Performance filled by maestro.analyze() (audio_features, alignment, assessment["synthesis"] and the
# module results, perf.pose / perf.motion_features when VioPose ran). Output: one JSON-compatible dict, embedded in the
# page by gui/report.py and drawn by gui/player.html, in which every series is indexed by time in seconds of the
# recording (video, audio and pose share one clock):
#   duration, scores {skill: 0-100}, events [...]  (assessment/synthesize.py, top three causes as [slug, confidence])
#   traces: pitch {t, midi (played, null = unvoiced), score [{start, end, midi}]}, loudness {t, db},
#           notes [{start, end, measure, beat, cents, ms, wrong}]   (one per matched note: pitch and timing error)
#           bow {t, pos, speed, strokes}, posture {t, tilt, left, right}, hand {t, semitones}   (only with a pose)
#   pose {fps, joints, edges, xyz (cm, [frame][joint] = [x, y, z] or null)}  drawn as a skeleton next to the video
#   reference: the professional comparison (assessment/reference: available, reason | source, clip, summary, gaps, reference_choices) or null
#   run (server id or null), digest (what feedback would send to a model), feedback (written feedback or null): the feedback card
# Values are rounded so the page stays small (a 3-minute recording is a few MB).
import numpy as np

from maestro.assessment.events import jsonable
from maestro.utils.audio_math import hz_to_midi

SKELETON = [("l_shoulder", "r_shoulder"), ("l_pelvis", "r_pelvis"), ("l_shoulder", "l_pelvis"), ("r_shoulder", "r_pelvis"),
            ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("l_wrist", "l_hand"),
            ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("r_wrist", "r_hand"),
            ("l_pelvis", "l_knee"), ("l_knee", "l_ankle"), ("r_pelvis", "r_knee"), ("r_knee", "r_ankle")]


def _round(x, digits):
    """Array -> list with NaN as None (JSON has no NaN)."""
    x = np.round(np.asarray(x, dtype=float), digits)
    return [None if np.isnan(v) else float(v) for v in x]


def _pitch_and_loudness(perf):
    frames = perf.audio_features["frames"]
    t = np.array([f["time"] for f in frames])
    f0 = np.array([f["estimated_pitch"] for f in frames])
    rms = np.array([f["rms_energy"] for f in frames])
    midi = np.where(f0 > 0, hz_to_midi(np.where(f0 > 0, f0, 440.0)), np.nan)
    return ({"t": _round(t, 3), "midi": _round(midi, 2),
             "score": [{"start": round(perf.audio_features["notes"][p]["onset"], 3),
                        "end": round(perf.audio_features["notes"][p].get("offset", perf.audio_features["notes"][p]["onset"]), 3),
                        "midi": perf.score_features[i]["midi_pitch"]} for i, p in _matches(perf)]},
            {"t": _round(t, 3), "db": _round(20 * np.log10(rms + 1e-6), 1)})


def _matches(perf):
    return [(a["score_idx"], a["perf_idx"]) for a in perf.alignment if a["perf_idx"] is not None]


def _notes(perf):
    """One record per matched note: where it is, and its pitch (cents) and timing (ms) error."""
    cents = {n["score_idx"]: n for n in perf.assessment["intonation"]["notes"]}
    timing = {n["score_idx"]: n["error"] for n in perf.assessment["rhythm"].get("notes", [])}
    out = []
    for i, p in _matches(perf):
        n, s = perf.audio_features["notes"][p], perf.score_features[i]
        out.append({"start": round(n["onset"], 3), "end": round(n.get("offset", n["onset"]), 3),
                    "measure": s.get("measure"), "beat": s.get("beat"),
                    "cents": round(cents[i]["cents"], 1) if i in cents else None,
                    "ms": round(timing[i] * 1000) if i in timing else None,
                    "wrong": bool(cents[i]["wrong_note"]) if i in cents else False})
    return out


def _movement(perf):
    """Bow, posture and left-hand traces from the motion features (empty without a pose)."""
    motion, out = perf.motion_features, {}
    if not motion:
        return out
    t = _round(motion["times"], 3)
    bow = motion.get("bow")
    if bow:
        out["bow"] = {"t": t, "pos": _round(bow["s"], 3), "speed": _round(np.abs(bow["v"]), 3),
                      "strokes": [{"start": round(s["start"], 3), "end": round(s["end"], 3), "dir": int(s["dir"])}
                                  for s in bow["strokes"]]}
    out["posture"] = {"t": t, "tilt": _round(motion["shoulder_tilt_deg"], 1) if motion["shoulder_tilt_deg"] is not None else None,
                      "left": _round(motion["elbow_angle"]["left"], 1) if "left" in motion["elbow_angle"] else None,
                      "right": _round(motion["elbow_angle"]["right"], 1) if "right" in motion["elbow_angle"] else None}
    vp = perf.assessment.get("visual_pitch")
    if vp and vp.get("available"):
        out["hand"] = {"t": _round(vp["times"], 3), "semitones": _round(vp["semitones_above_open"], 2)}
    return out


def _y_is_down(pose):
    """True if the pose's y axis points down (image convention): the shoulders then have a smaller y than the pelvis or
    the legs. False when y points up, and when no lower-body joint is tracked to tell."""
    lower = next((j for j in ("l_pelvis", "r_pelvis", "l_knee", "r_knee", "l_ankle", "r_ankle") if pose.has(j)), None)
    shoulders = [j for j in ("l_shoulder", "r_shoulder") if pose.has(j)]
    if lower is None or not shoulders:
        return False
    return bool(np.nanmean([np.nanmean(pose.joint(j)[:, 1]) for j in shoulders]) < np.nanmean(pose.joint(lower)[:, 1]))


def _pose(perf):
    """Joints in centimetres, y up (flipped if the pose has y down), for the skeleton view (None without a pose)."""
    pose = perf.pose
    if pose is None:
        return None
    xyz = pose.xyz / 10.0 * (np.array([1, -1, 1]) if _y_is_down(pose) else 1)
    return {"fps": pose.fps, "joints": list(pose.joints),
            "edges": [[a, b] for a, b in SKELETON if a in pose.joints and b in pose.joints],
            "xyz": [[None if np.isnan(j).any() else [round(float(c), 1) for c in j] for j in frame] for frame in xyz]}


def _reference(perf):
    """The reference comparison for the page's card (the events themselves are in the events list like every other module's), or None when
    no reference was asked for."""
    ref = (perf.assessment or {}).get("reference")
    if not ref:
        return None
    if not ref.get("available"):
        return {"available": False, "reason": (ref.get("summary") or {}).get("reason", "")}
    s = ref["summary"]
    return {"available": True, "source": s["source"], "clip": s["clip"], "summary": s["by_category"], "gaps": s["gaps"], "choices": s["choices"]}


def player_data(perf, media_url, media_kind, run=None, digest=None):
    """The page's data (see the module comment). media_kind: "video" or "audio". `run` is the server's id of this analysis (the
    page then offers to write feedback through /api/feedback), `digest` what would be sent to the model, perf.feedback the written
    feedback if there already is some."""
    synthesis = perf.assessment["synthesis"]
    pitch, loudness = _pitch_and_loudness(perf)
    events = [{**{k: e[k] for k in ("id", "module", "category", "start", "end", "measure", "beat", "severity", "value",
                                    "expected", "unit")},
               "causes": [[c["cause"], c["confidence"]] for c in e["causes"][:3]]} for e in synthesis["events"]]
    duration = len(perf.audio["samples"]) / perf.audio["sr"]
    return jsonable({"duration": duration, "media": {"url": media_url, "kind": media_kind},
                     "scores": synthesis["scores"], "unavailable": synthesis["unavailable"],
                     "events": events, "links": synthesis["links"], "causes": synthesis["causes"][:6],
                     "traces": {"pitch": pitch, "loudness": loudness, "notes": _notes(perf), **_movement(perf)},
                     "pose": _pose(perf), "reference": _reference(perf), "run": run, "digest": digest, "feedback": perf.feedback})
