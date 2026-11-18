# Evaluate bowing based on processed data
#
# Uses right-hand motion from vision/ aligned to score articulations. Writes to
# Performance.assessment.
# Use Bowing Gestures Classification in Violin Performance: A Machine Learning Approach (Dalmazzo & Ramírez, 2019)
#
# Input: the bow strokes found by vision/motion_features.py (right-hand travel along the bow axis). Output:
# assessment["bowing"]. For now it uses the pose alone, no audio or score: it describes how steady and balanced
# the bow arm is. Aligning bow changes with note onsets and classifying gestures (detaché, legato, staccato, as in
# Dalmazzo & Ramírez, who train a model on motion capture) needs the audio tools and comes later.
# Evenness metrics are coefficients of variation (spread / mean: 0 = perfectly even), so they are unit-free.
# Stroke length varies with note length in real music, so evenness is judged on bow *speed* per stroke.
import numpy as np


def _cv(values):
    values = np.asarray(values, dtype=float)
    return float(values.std() / values.mean()) if len(values) >= 2 and values.mean() > 0 else None


def assess_bowing(motion):
    """Bow strokes -> consistency summary ({"available": False, "reason"} if fewer than two strokes were found)."""
    bow = motion.get("bow")
    strokes = bow["strokes"] if bow else []
    if len(strokes) < 2:
        return {"available": False, "reason": "fewer than two bow strokes found (right hand not tracked or not moving)"}
    fps = motion["fps"]
    lengths = np.array([s["length"] for s in strokes])
    durations = np.array([s["end"] - s["start"] for s in strokes])
    speed = np.abs(bow["v"])
    within = []                                # speed variation inside a stroke, away from its turnarounds
    for s in strokes:
        a, b = int(s["start"] * fps), int(s["end"] * fps)
        middle = speed[a + (b - a) // 5:b - (b - a) // 5]
        if len(middle) >= 3 and middle.mean() > 0:
            within.append(middle.std() / middle.mean())
    by_direction = [np.mean([s["length"] for s in strokes if s["dir"] == d]) for d in (1, -1) if any(s["dir"] == d for s in strokes)]
    return {"available": True, "n_strokes": len(strokes), "strokes": strokes,
            "strokes_per_second": float(len(strokes) / (strokes[-1]["end"] - strokes[0]["start"])),
            "mean_stroke_length": float(lengths.mean()), "stroke_length_cv": _cv(lengths),
            "stroke_duration_cv": _cv(durations), "mean_speed": float(np.mean([s["mean_speed"] for s in strokes])),
            "stroke_speed_cv": _cv([s["mean_speed"] for s in strokes]),
            "speed_cv_within_stroke": float(np.mean(within)) if within else None,
            "direction_balance": float(min(by_direction) / max(by_direction)) if len(by_direction) == 2 else None,
            "direction_source": bow["axis_source"]}
