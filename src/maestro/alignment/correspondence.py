# Check that a score and a recording are the same piece (new file).
#
# Runs in maestro.analyze() after alignment/align_audio_score.py. A wrong score would otherwise
# produce meaningless "deviations", so we assert an error instead (utils/errors.MismatchError).
#
# Method: after alignment, count how many score events (a chord = one event) have a performed note
# within `tol` semitones, and how many performed notes were explained by the score. Mistakes, bad
# tuning and tempo changes still leave most of both high; a different piece leaves them low.
# Calibrated on synthetic cases: right piece with mistakes ~0.8, half the piece played ~0.7, a
# different melody in the same key ~0.57, unrelated material < 0.45, so the default threshold is 0.6.
# Limitation: it is pitch-based, so a very similar piece (e.g. a variation) can still pass, and
# playing less than ~60% of the score is rejected.
from maestro.alignment.align_audio_score import score_events
from maestro.utils.errors import InputError, MismatchError


def check_correspondence(score_notes, perf_notes, alignment, min_match=0.4, tol=1.0):
    """Raise MismatchError / InputError if the recording does not match the score.

    Returns {"score_match", "perf_match"} (fractions in [0, 1]) when it does.
    """
    if not perf_notes:
        raise InputError("No notes were detected in the audio (silence, noise or too quiet).")
    by_score = {a["score_idx"]: a["perf_idx"] for a in alignment}
    ok_events, used = 0, set()
    for ev in score_events(score_notes):
        hits = [by_score[i] for i in ev if by_score[i] is not None
                and abs(perf_notes[by_score[i]]["midi"] - score_notes[i]["midi_pitch"]) <= tol]
        ok_events += bool(hits)
        used.update(hits)
    score_match = ok_events / len(score_events(score_notes))
    perf_match = len(used) / len(perf_notes)
    if min(score_match, perf_match) < min_match:
        raise MismatchError(
            f"The recording does not appear to match the score: only {score_match:.0%} of the written notes "
            f"and {perf_match:.0%} of the played notes correspond. Check that you uploaded the right score.")
    return {"score_match": score_match, "perf_match": perf_match}
