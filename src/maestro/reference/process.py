# Process a professional reference recording into a baseline (rewritten from the outline).
#
# `process_reference(perf, source, out_dir)` takes the student's analysed Performance (its score and the stretch of it that was played) and a
# professional recording (an audio or video file) and stores the result in `perf.reference` (a dict, see the end of this
# comment). Steps, each reusing the pipeline that analyses the student:
#   1. load_video.load_reference: the file.
#   2. Pitch, notes and alignment with the score over the whole recording (preprocessing/audio.extract_audio, audio/, alignment/): this finds
#      where each bar is played. A recording that does not contain the student's bars is refused (MismatchError). This first pass uses a
#      single-voice pitch tracker, so it assumes the soloist is the loudest voice, as in a normal concerto recording.
#   3. Clip to the student's section: the bars the student played, with a short lead-in and tail; the audio and the video (cut in the
#      one re-encode to the common format) are clipped, so only that part is processed further.
#   4. isolate.py: score-informed harmonic masking (and, if asked, Demucs first) keeps the soloist and drops accompaniment and noise;
#      stretches where the soloist cannot be heard are reported as gaps. The clip is then analysed again from the isolated audio.
#   5. Video, if there is one: pose with VioPose (vision/pose_estimation.py, as for the student) and gaps.py: camera cuts, lost pose and a
#      different framing become gaps, and bow strokes are measured only between them.
#   6. baseline.py: the profile of the soloist (the same measurements as the student's, baseline.profile) and the choices the soloist visibly made.
# perf.reference = {"source": {"origin", "title"}, "clip": {"first_measure", "last_measure", "start", "end"}, "audio": {"samples", "sr"},
#   "audio_features", "alignment", "pose" (or None), "visual_status", "gaps": {"audio": [...], "video": [...]} (seconds in the clip, each with a
#   reason), "profile", "choices", "separation", "timings"}. Times in the clip start at 0 (the clip's start in the original recording is
#   clip["start"]). reference/compare.py compares a student's profile against it.
import time
from pathlib import Path

import numpy as np

from maestro.alignment.align_audio_score import align_audio_score
from maestro.assessment.events import jsonable
from maestro.audio.features import extract_features
from maestro.audio.pitch import DEFAULT_PITCH_BACKEND, estimate_pitch
from maestro.audio.transcription import transcribe
from maestro.preprocessing.audio import extract_audio
from maestro.reference import baseline, gaps, isolate, load_video
from maestro.utils.convert_video_type import probe, to_common_video
from maestro.utils.errors import InputError, MismatchError, PoseUnavailableError
from maestro.vision.pose_estimation import estimate_pose

MIN_SECTION_MATCH = 0.6          # share of the written notes of the student's bars that the reference must contain
LEAD_IN_S, TAIL_S = 0.3, 0.5     # kept before the first and after the last note of the section


def clip_range(score, notes, alignment, first, last, duration):
    """Seconds (start, end) of the bars first..last (inclusive) in a recording whose notes were aligned with the score. InputError if the
    recording does not cover them."""
    hit = [notes[a["perf_idx"]] for a in alignment if a["perf_idx"] is not None and first <= score[a["score_idx"]]["measure"] <= last]
    if not hit:
        raise InputError(f"The reference does not seem to contain bars {first} to {last} of the score.")
    return max(0.0, min(n["onset"] for n in hit) - LEAD_IN_S), min(duration, max(n["offset"] for n in hit) + TAIL_S)


def section_match(score, notes, alignment, first, last, tol=1.0):
    """Share (0-1) of the written notes of bars first..last that were found in the recording within `tol` semitones. The notes the
    recording has beyond the score (a band under the soloist) do not count against it, unlike in alignment/correspondence.check_correspondence."""
    wanted = [i for i, n in enumerate(score) if n.get("measure") is not None and first <= n["measure"] <= last]
    found = {a["score_idx"]: a["perf_idx"] for a in alignment if a["perf_idx"] is not None}
    return sum(i in found and abs(notes[found[i]]["midi"] - score[i]["midi_pitch"]) <= tol for i in wanted) / max(len(wanted), 1)


def _analyze(y, sr, score, pitch_backend, pitch_options):
    """Pitch track, notes, frame features and alignment of a mono signal."""
    track = estimate_pitch(y, sr, backend=pitch_backend, **(pitch_options or {}))
    notes = transcribe(y, sr, pitch_track=track)
    return track, notes, extract_features(y, sr, pitch_track=track), align_audio_score(score, notes)


def process_reference(perf, source, out_dir, measures=None, pitch_backend=DEFAULT_PITCH_BACKEND, pitch_options=None, separate=False,
                      pose_file=None, progress=None):
    """Process the reference `source` (a file) against `perf`'s score and fill perf.reference (see the module comment).
    measures: (first, last) bars to clip to; default the bars the student played (perf.assessment["section"]), else the whole score.
    separate: run Demucs first (needs `pip install demucs`; skipped with a note in "separation" when missing).
    pose_file: a saved pose.json of the clipped reference, instead of running VioPose.
    Raises InputError / MismatchError for unusable references, ToolMissingError for a missing ffmpeg."""
    step = progress or (lambda stage: None)
    score = perf.score_features
    work = Path(out_dir) / "reference"
    work.mkdir(parents=True, exist_ok=True)
    timings = {}

    def timed(name, fn, *args, **kw):
        start = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            timings[name] = round(time.perf_counter() - start, 3)

    step("reference")
    loaded = timed("load", load_video.load_reference, source, work)
    y, sr = timed("decode", extract_audio, loaded["path"])
    if not np.any(np.abs(y) > 1e-3):
        raise InputError("The reference recording is silent.")
    separation = "harmonic_mask"
    y_search = y
    if separate:
        other = timed("demucs", isolate.preseparate, y, sr)
        if other is None:
            separation = "harmonic_mask (demucs is not installed: pip install demucs)"
        else:
            y_search, separation = other, "demucs+harmonic_mask"

    track, notes, _, alignment = timed("find_section", _analyze, y_search, sr, score, pitch_backend, pitch_options)
    section = (perf.assessment or {}).get("section") or {}
    all_bars = [n["measure"] for n in score if n.get("measure") is not None]
    first, last = measures or (section.get("first_measure"), section.get("last_measure"))
    first, last = first if first is not None else min(all_bars), last if last is not None else max(all_bars)
    match = section_match(score, notes, alignment, first, last)
    if match < MIN_SECTION_MATCH:
        raise MismatchError(f"The reference does not match the score: only {match:.0%} of the written notes of bars {first} to {last} were "
                            "found in it. Check that it is a recording of this piece (the soloist must be the loudest voice).")
    t0, t1 = clip_range(score, notes, alignment, first, last, len(y) / sr)

    info = probe(loaded["path"])
    clip_video = str(timed("cut_video", to_common_video, loaded["path"], work, start=t0, end=t1)) if info["has_video"] else None
    lo, hi = int(t0 * sr), int(t1 * sr)
    sel = (track.times >= t0) & (track.times <= t1)
    guide = isolate.guide_f0(track.times, np.nan_to_num(track.f0), notes, alignment, score)
    isolated = timed("isolate", isolate.isolate_solo, y_search[lo:hi], sr, track.times[sel] - t0, guide[sel])
    track2, notes2, frames2, alignment2 = timed("analyze_clip", _analyze, isolated["samples"], sr, score, pitch_backend, pitch_options)

    pose, status, video_gaps, strokes = None, {"available": False, "reason": "audio-only reference"}, [], []
    if clip_video:
        cuts = gaps.scene_cuts(clip_video)
        try:
            pose = timed("pose", estimate_pose, clip_video, None, pose_file)
            video_gaps = gaps.find_gaps(pose, cuts)
            strokes = gaps.segment_motion(pose, video_gaps)["strokes"]
            status = {"available": True, "reason": ""}
        except PoseUnavailableError as e:
            status = {"available": False, "reason": str(e)}
            video_gaps = [{"start": c, "end": c + 0.1, "reason": "camera_cut"} for c in cuts]
    prof = baseline.profile(score, notes2, alignment2, isolated["samples"], sr, frames2, strokes,
                            lambda a, b: isolate.presence_in(isolated, a, b))
    perf.reference = {
        "source": {k: loaded[k] for k in ("origin", "title")},
        "clip": {"first_measure": first, "last_measure": last, "start": t0, "end": t1},
        "audio": {"samples": isolated["samples"], "sr": sr},
        "audio_features": {"notes": notes2, "frames": frames2, "pitch_backend": track2.backend},
        "alignment": alignment2, "pose": pose, "visual_status": status, "bow_strokes": strokes,
        "gaps": {"audio": isolated["gaps"], "video": video_gaps}, "profile": prof,
        "choices": baseline.reference_choices(prof, score), "separation": separation, "timings": timings}
    return perf.reference


def reference_summary(ref):
    """The reference without its audio and pose, as plain JSON types (for saving next to the other results)."""
    return jsonable({k: ref[k] for k in ("source", "clip", "gaps", "visual_status", "profile", "choices", "separation", "timings")})
