"""Assessment modules: common event format, root causes, the synthesized report and the optional neural backend.

Every scene is built from a spec of notes (pitch error, timing, loudness, vibrato, noise ...) so each fault has a known
expected event and cause."""
import json
import unittest

import numpy as np

from maestro.assessment.bowing import assess_bowing
from maestro.assessment.events import Context
from maestro.assessment.expression import assess_expression
from maestro.assessment.intonation import assess_intonation
from maestro.assessment.neural import analyze_windows, window_spans
from maestro.assessment.posture import assess_posture
from maestro.assessment.rhythm import assess_rhythm
from maestro.assessment.synthesize import synthesize_assessment
from maestro.assessment.tone import assess_tone
from maestro.audio.tone_quality import note_tone
from maestro.audio.vibrato import note_vibrato
from maestro.utils.Performance import Performance
from maestro.utils.audio_math import midi_to_hz

SR, HOP = 16000, 256
EVENT_KEYS = {"module", "category", "start", "end", "measure", "beat", "score_idx", "perf_idx", "severity", "value",
              "expected", "unit", "message", "causes"}


def scene(spec, spacing=0.5, motion=None):
    """Performance from note specs: dicts with midi and optional cents, shift (s), db, vib (Hz, cents), noise, dynamic,
    hairpin, art, dur. Score is metronomic (one note per `spacing`); audio is a synthesized violin-like tone."""
    n_samples = int((len(spec) * spacing + 0.5) * SR)
    y, frames_f0 = np.zeros(n_samples), np.zeros(n_samples // HOP + 1)
    score, notes, align = [], [], []
    for i, n in enumerate(spec):
        s_on, dur = i * spacing, n.get("dur", spacing * 0.9)
        on = s_on + n.get("shift", 0.0)
        midi = n["midi"] + n.get("cents", 0) / 100
        t = np.arange(int(dur * SR)) / SR
        rate, ext = n.get("vib", (0, 0))
        inst = midi_to_hz(midi) * 2 ** (ext * np.sin(2 * np.pi * rate * t) / 1200)
        phase = 2 * np.pi * np.cumsum(inst) / SR
        tone = sum(np.sin(k * phase) / k for k in range(1, 8)) * 10 ** (n.get("db", -20) / 20) * 0.5
        tone = tone * np.minimum(1, t / 0.02)                         # 20 ms attack
        tone = tone + np.random.default_rng(i).normal(0, n.get("noise", 0), len(t))
        a = int(on * SR)
        y[a:a + len(t)] += tone
        f = np.arange(int(on * SR / HOP), int((on + dur) * SR / HOP))
        frames_f0[f] = inst[np.minimum(((f * HOP - a)).clip(0), len(t) - 1)]
        score.append({"measure": 1 + i // 4, "beat": 1.0 + i % 4, "pitch": str(n["midi"]), "midi_pitch": n["midi"],
                      "duration": spacing * 2, "articulation": n.get("art"), "onset": s_on, "offset": s_on + dur,
                      "hz": float(midi_to_hz(n["midi"])), "dynamic": n.get("dynamic"), "hairpin": n.get("hairpin")})
        notes.append({"onset": on, "offset": on + dur, "hz": float(midi_to_hz(midi)), "midi": float(midi)})
        align.append({"score_idx": i, "perf_idx": i})
    perf = Performance()
    perf.audio, perf.score_features, perf.alignment = {"samples": y, "sr": SR}, score, align
    perf.audio_features = {"notes": notes, "frames": [{"time": k * HOP / SR, "estimated_pitch": float(f)}
                                                      for k, f in enumerate(frames_f0)]}
    perf.assessment, perf.motion_features = {}, motion
    return perf


def notes_of(perf):
    return perf.score_features, perf.audio_features["notes"], perf.alignment


def categories(result):
    return [e["category"] for e in result["events"]]


def top_cause(event):
    return event["causes"][0]["cause"]


class Intonation(unittest.TestCase):
    def test_leap_overshoot_is_blamed_on_the_shift(self):
        spec = [{"midi": m} for m in (60, 62, 60, 62, 72, 62, 60, 62)]
        spec[4]["cents"] = 40                                       # lands sharp after an octave leap up
        perf = scene(spec)
        r = assess_intonation(*notes_of(perf), ctx=Context(perf))
        event, = r["events"]
        self.assertEqual((event["category"], event["score_idx"]), ("sharp", 4))
        self.assertEqual(top_cause(event), "shift_overshoot")
        self.assertEqual(set(event) , EVENT_KEYS)
        self.assertEqual(event["measure"], 2)

    def test_whole_performance_flat_is_a_tuning_offset(self):
        perf = scene([{"midi": 60 + i % 3, "cents": -30} for i in range(8)])
        r = assess_intonation(*notes_of(perf), ctx=Context(perf))
        self.assertEqual(len(r["events"]), 8)
        self.assertTrue(all(top_cause(e) == "global_tuning_offset" for e in r["events"]))
        self.assertAlmostEqual(r["tuning_offset_cents"], -30, delta=1)

    def test_short_notes_get_a_wider_pitch_tolerance(self):
        def events(dur):
            spec = [{"midi": 60 + i % 3, "dur": dur, "cents": 33 if i == 4 else 0} for i in range(8)]
            perf = scene(spec, spacing=0.3)
            return assess_intonation(*notes_of(perf), ctx=Context(perf))["events"]
        self.assertEqual(len(events(0.27)), 1)                        # 33 cents on a normal note is out of tune
        self.assertEqual(events(0.1), [])                             # ... but not on a 100 ms note (limit 37.5 cents)

    def test_wrong_note_is_maximally_severe_and_in_tune_playing_is_clean(self):
        spec = [{"midi": 60 + i % 3} for i in range(8)]
        spec[3]["midi"] = 66
        perf = scene(spec)
        perf.score_features[3]["midi_pitch"], perf.score_features[3]["hz"] = 61, float(midi_to_hz(61))
        r = assess_intonation(*notes_of(perf), ctx=Context(perf))
        self.assertEqual([(e["category"], e["severity"]) for e in r["events"]], [("wrong_note", 1.0)])
        clean = scene([{"midi": 60 + i % 3} for i in range(8)])
        r = assess_intonation(*notes_of(clean), ctx=Context(clean))
        self.assertEqual((r["events"], r["score"]), ([], 100.0))

    def test_hand_position_from_video_is_used_as_evidence(self):
        spec = [{"midi": 60 + i % 3} for i in range(8)]
        spec[5]["cents"] = 40
        perf = scene(spec)
        t = np.arange(0, 4.5, 1 / 30)
        perf.assessment["visual_pitch"] = {"available": True, "times": t, "candidates": np.full((len(t), 4), 90.0)}
        r = assess_intonation(*notes_of(perf), ctx=Context(perf))
        self.assertIn("hand_position_mismatch", [c["cause"] for c in r["events"][0]["causes"]])


class Rhythm(unittest.TestCase):
    def test_late_entrance_after_a_rest(self):
        spec = [{"midi": 60 + i % 3} for i in range(10)]
        spec[5]["shift"] = 0.2
        perf = scene(spec)
        for n in perf.score_features[5:]:                           # a rest before note 5 in the score
            n["onset"] += 0.0
        perf.score_features[4]["offset"] = perf.score_features[4]["onset"] + 0.2
        r = assess_rhythm(*notes_of(perf), ctx=Context(perf))
        late = [e for e in r["events"] if e["category"] == "late"]
        self.assertTrue(late)
        self.assertIn("late_after_rest", [c["cause"] for c in late[0]["causes"]])

    def test_a_change_of_tempo_is_one_mild_event_not_dozens_of_late_notes(self):
        spec = [{"midi": 60 + i % 3, "shift": -0.2 * (min(max(i, 10), 30) - 10)} for i in range(40)]   # 20 notes at 0.3 s, not 0.5 s
        perf = scene(spec)
        r = assess_rhythm(*notes_of(perf), ctx=Context(perf))
        changes = [e for e in r["events"] if e["category"] == "tempo_change"]
        self.assertTrue(changes)                                    # reported, relative to the average tempo ...
        self.assertTrue(all(e["severity"] < 0.4 for e in changes))   # ... as an often-musical, never severe, event
        self.assertEqual([e for e in r["events"] if e["category"] in ("early", "late")], [])
        clean = scene([{"midi": 60 + i % 3} for i in range(12)])
        self.assertEqual(assess_rhythm(*notes_of(clean))["events"], [])

    def test_gradual_rubato_is_forgiven_but_a_single_slip_inside_it_is_found(self):
        drift = lambda i: 0.4 * np.sin(i / 24 * np.pi)              # the player breathes by 0.4 s over the piece
        spec = [{"midi": 60 + i % 3, "shift": drift(i) + (0.25 if i == 12 else 0)} for i in range(24)]
        perf = scene(spec)
        r = assess_rhythm(*notes_of(perf), ctx=Context(perf))
        self.assertEqual([(e["category"], e["score_idx"]) for e in r["events"] if e["category"] in ("early", "late")],
                         [("late", 12)])

    def test_the_same_error_means_different_things_in_slow_and_fast_music(self):
        def events(spacing, shift):
            spec = [{"midi": 60 + i % 3, "dur": spacing * 0.9, "shift": shift if i == 8 else 0} for i in range(16)]
            perf = scene(spec, spacing=spacing)
            return [e for e in assess_rhythm(*notes_of(perf), ctx=Context(perf))["events"] if e["category"] in ("early", "late")]
        self.assertEqual(events(0.5, 0.09), [])                      # 90 ms in a slow passage is within the 120 ms allowed
        slow = events(0.5, 0.2)
        self.assertEqual(len(slow), 1)
        self.assertNotIn("possible_misalignment", [c["cause"] for c in slow[0]["causes"]])
        fast = events(0.15, 0.09)                                    # 60% of a note: reported, but may be a wrong match
        self.assertEqual(len(fast), 1)
        self.assertIn("possible_misalignment", [c["cause"] for c in fast[0]["causes"]])
        self.assertEqual(events(0.15, 0.05), [])                     # under the 70 ms floor onsets cannot be resolved

    def test_tempo_far_from_the_written_one_is_an_event(self):
        perf = scene([{"midi": 60 + i % 3} for i in range(10)], spacing=0.5)
        for i, n in enumerate(perf.audio_features["notes"]):
            n["onset"] = i * 0.8                                    # 62% of the written tempo
        r = assess_rhythm(*notes_of(perf))
        self.assertIn("tempo", categories(r))


class Expression(unittest.TestCase):
    def dyn_spec(self, loud_db):
        return [{"midi": 60 + i % 3, "dynamic": "p" if i < 6 else "f", "db": loud_db[i >= 6], "vib": (6, 25), "dur": 0.7}
                for i in range(12)]

    def test_marks_played_at_one_level_are_flat_dynamics(self):
        perf = scene(self.dyn_spec((-20, -20)), spacing=0.8)
        r = assess_expression(*notes_of(perf), ctx=Context(perf))
        self.assertIn("flat_dynamics", categories(r))
        good = scene(self.dyn_spec((-32, -14)), spacing=0.8)
        self.assertNotIn("flat_dynamics", categories(assess_expression(*notes_of(good), ctx=Context(good))))

    def test_flat_dynamics_with_constant_bow_speed_blames_the_bow(self):
        strokes = [{"start": i * 0.8, "end": i * 0.8 + 0.7, "dir": 1 if i % 2 else -1, "length": 100.0,
                    "mean_speed": 140.0 + i % 2} for i in range(12)]
        motion = {"bow": {"strokes": strokes}, "fps": 30, "times": np.arange(0, 10, 1 / 30)}
        perf = scene(self.dyn_spec((-20, -20)), spacing=0.8, motion=motion)
        event = next(e for e in assess_expression(*notes_of(perf), ctx=Context(perf))["events"] if e["category"] == "flat_dynamics")
        self.assertEqual(top_cause(event), "constant_bow_speed")

    def test_missing_wobbling_and_normal_vibrato(self):
        plain = scene([{"midi": 60 + i % 3, "dur": 0.7} for i in range(8)], spacing=0.8)
        self.assertIn("no_vibrato", categories(assess_expression(*notes_of(plain), ctx=Context(plain))))
        spec = [{"midi": 60 + i % 3, "dur": 0.7, "vib": (6, 25)} for i in range(8)]
        spec[3]["vib"] = (3, 25)                                    # one slow vibrato
        perf = scene(spec, spacing=0.8)
        events = assess_expression(*notes_of(perf), ctx=Context(perf))["events"]
        self.assertEqual([(e["category"], e["score_idx"]) for e in events], [("vibrato_slow", 3)])

    def test_staccato_that_rings_on(self):
        spec = [{"midi": 60 + i % 3, "art": "staccato" if i == 2 else None, "dur": 0.45} for i in range(6)]
        perf = scene(spec)
        self.assertIn("staccato_too_long", categories(assess_expression(*notes_of(perf), ctx=Context(perf))))

    def test_without_audio_the_module_says_so(self):
        perf = scene([{"midi": 60}] * 4)
        perf.audio = None
        self.assertFalse(assess_expression(*notes_of(perf), ctx=Context(perf))["available"])


class Tone(unittest.TestCase):
    def stroke_motion(self, speed):
        strokes = [{"start": i * 0.5, "end": i * 0.5 + 0.45, "dir": 1, "length": 100.0, "mean_speed": speed if i == 4 else 100.0}
                   for i in range(10)]
        return {"bow": {"strokes": strokes}, "fps": 30, "times": np.arange(0, 6, 1 / 30)}

    def test_noisy_note_with_slow_bow_is_too_much_pressure(self):
        spec = [{"midi": 60 + i % 3, "noise": 0.25 if i == 4 else 0} for i in range(10)]
        perf = scene(spec, motion=self.stroke_motion(40.0))
        r = assess_tone(*notes_of(perf), ctx=Context(perf))
        event, = [e for e in r["events"] if e["category"] == "noisy_tone"]
        self.assertEqual((event["score_idx"], top_cause(event)), (4, "too_much_bow_pressure"))
        fast = scene(spec, motion=self.stroke_motion(180.0))
        event, = [e for e in assess_tone(*notes_of(fast), ctx=Context(fast))["events"] if e["category"] == "noisy_tone"]
        self.assertEqual(top_cause(event), "bow_too_fast_or_light")

    def test_clean_tone_has_no_events(self):
        perf = scene([{"midi": 60 + i % 3} for i in range(10)])
        self.assertEqual(assess_tone(*notes_of(perf), ctx=Context(perf))["events"], [])

    def test_descriptors_separate_clean_from_noisy(self):
        perf = scene([{"midi": 69}, {"midi": 69, "noise": 0.25}])
        y = perf.audio["samples"]
        clean, noisy = (note_tone(y, SR, i * 0.5, i * 0.5 + 0.45, 440.0) for i in (0, 1))
        self.assertGreater(clean["harmonic_ratio"], 0.9)
        self.assertLess(noisy["harmonic_ratio"], 0.7)


class VibratoDetector(unittest.TestCase):
    def test_rate_extent_and_absence(self):
        t = np.arange(0, 1.5, HOP / SR)
        v = note_vibrato(t, 440 * 2 ** (25 * np.sin(2 * np.pi * 6 * t) / 1200), 0, 1.5)
        self.assertTrue(v["present"])
        self.assertAlmostEqual(v["rate_hz"], 6, delta=0.3)
        self.assertAlmostEqual(v["extent_cents"], 25, delta=4)
        self.assertFalse(note_vibrato(t, np.full_like(t, 440.0), 0, 1.5)["present"])
        self.assertIsNone(note_vibrato(t, np.zeros_like(t), 0, 1.5)["rate_hz"])


def stroke_motion(strokes, n=300):
    return {"bow": {"strokes": strokes, "v": np.full(n, 100.0), "axis_source": "test"}, "fps": 30,
            "times": np.arange(n) / 30}


def strokes(speeds, length=100.0):
    return [{"start": i * 0.5, "end": i * 0.5 + 0.5, "dir": 1 if i % 2 else -1, "length": length, "mean_speed": s}
            for i, s in enumerate(speeds)]


class Bowing(unittest.TestCase):
    def test_speed_outlier_and_balance(self):
        sp = strokes([100.0] * 9 + [400.0])
        r = assess_bowing(stroke_motion(sp))
        self.assertEqual([e["category"] for e in r["events"]], ["bow_too_fast"])
        for s in sp[::2]:
            s["length"] = 40.0                                       # one direction much shorter
        self.assertIn("unbalanced_bow_use", categories(assess_bowing(stroke_motion(sp))))

    def test_bow_change_late_against_note_onset_and_technique_labels(self):
        sp = strokes([100.0] * 10)
        perf = scene([{"midi": 60 + i % 3} for i in range(10)], spacing=0.5, motion=stroke_motion(sp))
        perf.audio_features["notes"][4]["onset"] -= 0.15            # note sounds 150 ms before the bow changes
        r = assess_bowing(perf.motion_features, Context(perf))
        late = [e for e in r["events"] if e["category"] == "bow_change_late"]
        self.assertEqual([e["perf_idx"] for e in late], [4])
        self.assertTrue(set(r["stroke_techniques"]) <= {"detache", "staccato", "legato", None})

    def test_notes_inside_a_slur_are_not_blamed_for_a_missing_bow_change(self):
        sp = strokes([100.0] * 10)                                  # a bow change every 0.5 s, so every second note is slurred
        perf = scene([{"midi": 60 + i % 3, "dur": 0.2} for i in range(20)], spacing=0.25, motion=stroke_motion(sp))
        self.assertEqual(assess_bowing(perf.motion_features, Context(perf))["events"], [])

    def test_in_running_notes_a_far_away_bow_change_is_a_mispairing_not_a_fault(self):
        sp = strokes([100.0] * 12)
        for st in sp:
            st["start"], st["end"] = st["start"] * 0.4, st["end"] * 0.4 + 0.08       # a bow change every 0.2 s, 80 ms after onsets
        perf = scene([{"midi": 60 + i % 3, "dur": 0.1} for i in range(24)], spacing=0.2, motion=stroke_motion(sp))
        ctx = Context(perf)
        self.assertEqual(assess_bowing(perf.motion_features, ctx)["events"], [])   # constant offset removed, nothing left

    def test_a_constant_video_audio_offset_is_not_a_fault(self):
        sp = strokes([100.0] * 10)
        for st in sp:
            st["end"] += 0.15                                       # the whole pose is 150 ms behind the audio
        perf = scene([{"midi": 60 + i % 3} for i in range(10)], spacing=0.5, motion=stroke_motion(sp))
        ctx = Context(perf)
        self.assertAlmostEqual(ctx.bow_sync_offset(), 0.15, delta=0.01)
        self.assertEqual(assess_bowing(perf.motion_features, ctx)["events"], [])

    def test_too_few_strokes_is_unavailable(self):
        self.assertFalse(assess_bowing(stroke_motion(strokes([100.0])))["available"])


class FakePose:
    def __init__(self, joints):
        self.joints = joints

    def has(self, *names):
        return all(n in self.joints for n in names)

    def joint(self, name):
        return self.joints[name]


class Posture(unittest.TestCase):
    def setup_motion(self, tilt, head_amp=0.0, seconds=20):
        n, fps = seconds * 30, 30
        t = np.arange(n) / fps
        head = np.zeros((n, 3))
        head[:, 0] = head_amp * np.sin(2 * np.pi * 1.0 * t)          # shoulder widths of sway
        pose = FakePose({"head": head, "l_shoulder": np.zeros((n, 3)), "r_shoulder": np.ones((n, 3)), "r_elbow": np.zeros((n, 3))})
        return {"fps": fps, "scale": 1.0, "times": t, "shoulder_tilt_deg": np.full(n, float(tilt)), "elbow_angle": {}}, pose

    def test_persistent_tilt_is_a_setup_problem_and_level_shoulders_are_clean(self):
        motion, pose = self.setup_motion(15)
        r = assess_posture(motion, pose)
        tilts = [e for e in r["events"] if e["category"] == "shoulders_uneven"]
        self.assertGreaterEqual(len(tilts), 15)
        self.assertEqual(top_cause(tilts[0]), "setup_height")
        motion, pose = self.setup_motion(2)
        self.assertEqual(assess_posture(motion, pose)["events"], [])

    def test_head_movement_is_found(self):
        motion, pose = self.setup_motion(0, head_amp=0.4)
        self.assertIn("head_movement", categories(assess_posture(motion, pose)))


class Synthesis(unittest.TestCase):
    def build(self):
        spec = [{"midi": m, "cents": 40 if i == 4 else 0, "noise": 0.25 if i == 4 else 0, "vib": (6, 25), "dur": 0.4}
                for i, m in enumerate((60, 62, 60, 62, 72, 62, 60, 62))]
        perf = scene(spec, motion=stroke_motion(strokes([100.0] * 12)))
        ctx = Context(perf)
        a = perf.assessment
        a["intonation"] = assess_intonation(*notes_of(perf), ctx=ctx)
        a["rhythm"] = assess_rhythm(*notes_of(perf), ctx=ctx)
        a["expression"] = assess_expression(*notes_of(perf), ctx=ctx)
        a["tone"] = assess_tone(*notes_of(perf), ctx=ctx)
        a["bowing"] = assess_bowing(perf.motion_features, ctx)
        return perf

    def test_single_json_with_ids_causes_links_and_scores(self):
        perf = self.build()
        report = synthesize_assessment(perf.assessment)
        again = json.loads(json.dumps(report))                       # plain JSON types all the way through
        self.assertEqual(again, report)
        self.assertEqual(len({e["id"] for e in report["events"]}), report["n_events"])
        self.assertEqual([e["start"] for e in report["events"]], sorted(e["start"] for e in report["events"]))
        self.assertTrue(all(e.keys() >= EVENT_KEYS for e in report["events"]))
        self.assertIn("shift_overshoot", [c["cause"] for c in report["causes"]])
        self.assertTrue(any({"intonation", "tone"} == set(l["modules"]) for l in report["links"]))
        self.assertIn("2", report["by_measure"])
        self.assertLess(report["scores"]["intonation"], 100)
        self.assertIsNotNone(report["scores"]["overall"])
        self.assertEqual(set(report["unavailable"]), {"posture"})

    def test_unavailable_modules_are_reported_not_dropped(self):
        report = synthesize_assessment({"bowing": {"available": False, "reason": "no pose", "events": []}})
        self.assertEqual(report["unavailable"]["bowing"], "no pose")
        self.assertIsNone(report["scores"]["overall"])


class FakeBackend:
    name = "fake"

    def p_first(self, windows, sr, prompts):
        return [0.9 if i == 0 else 0.1 for i in range(len(windows))]    # first window good, the rest poor


class Neural(unittest.TestCase):
    def test_windows_cover_the_recording(self):
        spans = window_spans(14.0)
        self.assertEqual((spans[0][0], spans[-1][1]), (0.0, 14.0))
        self.assertEqual(window_spans(3.0), [(0.0, 3.0)])

    def test_poor_windows_become_events_explained_by_measurements(self):
        perf = scene([{"midi": 60 + i % 3, "dur": 0.4, "noise": 0.25} for i in range(40)], spacing=0.5)
        perf.assessment["neural"] = analyze_windows(perf.audio["samples"], SR, FakeBackend())
        ctx = Context(perf)
        tone = assess_tone(*notes_of(perf), ctx=ctx)
        harsh = [e for e in tone["events"] if e["category"] == "neural_harsh_tone"]
        self.assertGreaterEqual(len(harsh), 2)                      # every window but the first
        self.assertEqual(harsh[0]["unit"], "p(good tone)")
        expr = assess_expression(*notes_of(perf), ctx=ctx)
        self.assertIn("mechanical_sound", categories(expr))

    def test_unavailable_backend_is_reported_cleanly(self):
        from maestro.assessment.neural import ClapBackend, NeuralUnavailable, load_backend
        self.assertIsNone(load_backend(None))
        with self.assertRaises(ValueError):
            load_backend("nope")
        try:
            import transformers  # noqa: F401
        except ImportError:
            with self.assertRaises(NeuralUnavailable):
                ClapBackend()


if __name__ == "__main__":
    unittest.main()
