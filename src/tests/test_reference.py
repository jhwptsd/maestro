"""The professional reference: loading, clipping to the student's bars, isolating the soloist, gaps, the baseline with its interpretive
choices, and the comparison (synthetic recordings: the Minuet with an 'orchestra' under it)."""
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

from helpers import SR, make_video, synth, write_wav
from minuet import BARS, MIDIS, minuet_musicxml, play_minuet
from pose_fixture import minuet_notes, synth_pose
from test_gui import HAS_BROWSER, multipart, page_data, performance
from maestro import analyze
from maestro.gui import server as gui_server
from maestro.gui.report import build_page
from maestro.reference import baseline, gaps, isolate, load_video
from maestro.reference.compare import compare_to_reference
from maestro.reference.process import clip_range, reference_summary
from maestro.utils.errors import InputError, MismatchError, ToolMissingError
from maestro.vision.pose_io import Pose, save_pose

ORCHESTRA_HZ = (233.1, 311.1, 466.2)          # not partials of anything in the melody (G major), so they can be told apart


def with_orchestra(y, level=0.03):
    """The recording with a constant 'orchestra' (a few sustained tones) and room noise under it."""
    t = np.arange(len(y)) / SR
    under = sum(np.sin(2 * np.pi * f * t) for f in ORCHESTRA_HZ) * level
    return (y + under + 0.01 * np.random.default_rng(1).standard_normal(len(y))).astype(np.float32)


def band_energy(y, hz, width=6.0):
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    return spectrum[np.abs(freqs - hz) <= width].sum()


class Loading(unittest.TestCase):
    def test_a_file_is_probed_and_other_links_and_missing_files_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "pro.wav"
            write_wav(wav, synth([(69, 0.2, 1.0)]))
            found = load_video.load_reference(wav, tmp)
            self.assertEqual((found["origin"], found["title"], found["url"]), ("file", "pro", None))
            with self.assertRaisesRegex(InputError, "Only YouTube"):
                load_video.load_reference("https://example.com/video.mp4", tmp)
            with self.assertRaises(InputError):
                load_video.load_reference(Path(tmp) / "missing.mp4", tmp)

    def test_youtube_links_are_recognised_and_need_yt_dlp(self):
        self.assertTrue(all(load_video.is_youtube(u) for u in ("https://www.youtube.com/watch?v=abc", "https://youtu.be/abc", "http://m.youtube.com/watch?v=1")))
        self.assertFalse(load_video.is_youtube("https://notyoutube.com/watch?v=abc"))
        with mock.patch.dict("sys.modules", {"yt_dlp": None}), tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ToolMissingError, "yt-dlp"):
            load_video.load_reference("https://youtu.be/abc", tmp)

    def fake_yt_dlp(self, behaviour):
        """A stand-in for the yt_dlp module: YoutubeDL(options).extract_info(url, download=True) runs `behaviour(options)`."""
        class YoutubeDLError(Exception):
            pass

        class FakeDL:
            def __init__(self, options):
                FakeDL.options = options

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def extract_info(self, url, download):
                return behaviour(FakeDL.options)

        return mock.MagicMock(YoutubeDL=FakeDL, utils=mock.MagicMock(YoutubeDLError=YoutubeDLError, match_filter_func=lambda expr: expr)), FakeDL

    def test_a_youtube_video_is_downloaded_with_one_video_no_playlist_and_ffmpeg_found(self):
        def download(options):
            path = Path(options["outtmpl"].replace("%(ext)s", "wav"))
            write_wav(path, synth([(69, 0.2, 1.0)]))
            return {"title": "A Violin Concerto", "requested_downloads": [{"filepath": str(path)}]}

        fake, dl = self.fake_yt_dlp(download)
        with mock.patch.dict("sys.modules", {"yt_dlp": fake}), tempfile.TemporaryDirectory() as tmp:
            found = load_video.load_reference("https://youtu.be/abc", tmp)
        self.assertEqual((found["origin"], found["title"], found["url"]), ("youtube", "A Violin Concerto", "https://youtu.be/abc"))
        self.assertTrue(dl.options["noplaylist"] and "height<=720" in dl.options["format"])
        self.assertTrue(Path(dl.options["ffmpeg_location"]).is_dir())                    # merging needs ffmpeg, which is not always on PATH
        self.assertIn(str(load_video.MAX_SECONDS), dl.options["match_filter"])            # the length limit is applied before downloading

    def test_a_failed_merge_is_retried_with_one_ready_made_file(self):
        calls = []

        def behaviour(options):
            calls.append(options)
            if len(calls) == 1:
                path = Path(options["outtmpl"].replace("%(ext)s", "f137.mp4"))
                path.write_bytes(b"half")                                                               # what a failed merge leaves behind
                raise fake.utils.YoutubeDLError("ERROR: Postprocessing: Stream #1:0 -> #0:1 (copy)\nConversion failed!")
            path = Path(options["outtmpl"].replace("%(ext)s", "wav"))
            write_wav(path, synth([(69, 0.2, 1.0)]))
            return {"title": "T", "requested_downloads": [{"filepath": str(path)}]}

        fake, _ = self.fake_yt_dlp(behaviour)
        with mock.patch.dict("sys.modules", {"yt_dlp": fake}), tempfile.TemporaryDirectory() as tmp:
            found = load_video.load_reference("https://youtu.be/abc", tmp)
            self.assertFalse(list(Path(tmp).glob("reference.f137*")))                                   # the half-merged file was removed
        self.assertEqual(found["title"], "T")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["merge_output_format"], "mkv")                                        # merged into a container that takes any codecs
        self.assertNotIn("merge_output_format", calls[1])                                               # then no merge at all

    def test_a_video_skipped_by_the_length_filter_is_explained(self):
        fake, _ = self.fake_yt_dlp(lambda options: {"title": "Very long"})
        with mock.patch.dict("sys.modules", {"yt_dlp": fake}), tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(InputError, "longer than 60 minutes"):
            load_video.load_reference("https://youtu.be/abc", tmp)

    def test_download_failures_show_the_real_reason_and_what_to_try(self):
        for message, expected in (("ERROR: [youtube] abc: Sign in to confirm you're not a bot", "sign-in"),
                                  ("ERROR: [youtube] abc: Requested format is not available", "pip install -U yt-dlp"),
                                  ("ERROR: [youtube] abc: n challenge solving failed: no JavaScript runtime", "Node.js")):
            fake, _ = self.fake_yt_dlp(lambda options: (_ for _ in ()).throw(fake.utils.YoutubeDLError(message)))
            with mock.patch.dict("sys.modules", {"yt_dlp": fake}), tempfile.TemporaryDirectory() as tmp, self.assertRaises(InputError) as caught:
                load_video.load_reference("https://youtu.be/abc", tmp)
            text = str(caught.exception)
            self.assertIn(message.removeprefix("ERROR: ")[:40], text)                      # yt-dlp's own words are kept
            self.assertIn(expected, text)
            self.assertIn("pip install -U yt-dlp", text)


class Isolation(unittest.TestCase):
    notes = [(69, 0.3, 0.8), (71, 1.2, 0.8), (72, 2.1, 0.8)]          # the soloist: A4, B4, C5

    def run_isolation(self, y):
        times = np.arange(0, len(y) / SR, 256 / SR)
        guide = np.zeros(len(times))
        for midi, start, dur in self.notes:
            guide[(times >= start) & (times <= start + dur)] = 440 * 2 ** ((midi - 69) / 12)
        return isolate.isolate_solo(y, SR, times, guide)

    def test_the_orchestra_is_removed_and_the_soloist_kept(self):
        solo = synth(self.notes, total=3.2)
        mixed = with_orchestra(solo)
        out = self.run_isolation(mixed)
        drop_db = 10 * np.log10(band_energy(mixed, 311.1) / band_energy(out["samples"], 311.1))
        kept_db = 10 * np.log10(band_energy(solo, 440) / band_energy(out["samples"], 440))
        self.assertGreater(drop_db, 20)                                       # the 'orchestra' tone is gone
        self.assertLess(abs(kept_db), 3)                                      # the violin's A is as loud as before
        self.assertEqual(out["gaps"], [])
        self.assertEqual(len(out["samples"]), len(mixed))

    def test_a_stretch_where_the_soloist_should_play_but_only_the_orchestra_is_heard_is_a_gap_and_rests_are_not(self):
        y = with_orchestra(synth(self.notes[:1], total=3.2), 0.3)             # the soloist stops after the first note
        out = self.run_isolation(y)
        self.assertEqual([g["reason"] for g in out["gaps"]], ["accompaniment_dominates"] * 2)       # the two notes that should sound, not the rest between
        first, second = out["gaps"]
        self.assertTrue(1.1 < first["start"] < 1.4 and 1.9 < first["end"] < 2.2)
        self.assertTrue(2.0 < second["start"] < 2.3 and second["end"] > 2.8)
        self.assertGreater(isolate.presence_in(out, 0.4, 1.0), 0.9)
        self.assertLess(isolate.presence_in(out, 1.4, 2.8), 0.1)

    def test_guide_follows_the_tracker_when_close_and_the_score_when_not(self):
        times = np.arange(0, 1, 0.01)
        f0 = np.full(len(times), 441.0)
        f0[50:] = 600.0                                                       # the tracker latches onto something else halfway
        guide = isolate.guide_f0(times, f0, [{"onset": 0.0, "offset": 1.0}], [{"score_idx": 0, "perf_idx": 0}], [{"hz": 440.0}])
        self.assertAlmostEqual(guide[10], 441.0)
        self.assertAlmostEqual(guide[70], 440.0)
        self.assertEqual(isolate.guide_f0(times, f0, [], [{"score_idx": 0, "perf_idx": None}], [{"hz": 440.0}]).max(), 0.0)

    def test_demucs_is_optional(self):
        with mock.patch.dict("sys.modules", {"demucs": None, "demucs.apply": None, "demucs.pretrained": None}):
            self.assertIsNone(isolate.preseparate(synth([(69, 0.2, 1.0)]), SR))


class CameraGaps(unittest.TestCase):
    def pose(self):
        return synth_pose(minuet_notes(), with_bow=False)

    def test_lost_pose_and_cuts_become_gaps_and_nothing_is_measured_across_them(self):
        pose = self.pose()
        pose.xyz[int(3.0 * pose.fps):int(4.5 * pose.fps)] = np.nan             # the camera is on the audience for 1.5 s
        found = gaps.find_gaps(pose, cuts=[7.0])
        by_reason = {g["reason"]: g for g in found}
        self.assertAlmostEqual(by_reason["pose_lost"]["start"], 3.0, places=1)
        self.assertAlmostEqual(by_reason["pose_lost"]["end"], 4.5, places=1)
        self.assertAlmostEqual(by_reason["camera_cut"]["start"], 7.0, places=1)
        motion = gaps.segment_motion(pose, found)
        self.assertTrue(motion["strokes"])
        self.assertFalse([s for s in motion["strokes"] if 3.2 < (s["start"] + s["end"]) / 2 < 4.3])      # no stroke invented in the gap
        self.assertTrue(all(seg["end"] <= 3.1 or seg["start"] >= 4.4 for seg in motion["segments"] if seg["start"] < 4.4))

    def test_short_flicker_is_not_a_gap_but_a_different_framing_is(self):
        pose = self.pose()
        pose.xyz[30:36] = np.nan                                              # 0.2 s flicker
        self.assertEqual(gaps.find_gaps(pose), [])
        close_up = self.pose()
        a, b = int(5 * close_up.fps), int(9 * close_up.fps)
        close_up.xyz[a:b] *= 1.8                                              # a close-up: the shoulders look much wider
        found = gaps.find_gaps(close_up, cuts=[5.0, 9.0])                     # (a change of framing comes with cuts)
        self.assertEqual(sorted(g["reason"] for g in found), ["camera_cut", "camera_cut", "camera_framing_changed"])
        framing = next(g for g in found if g["reason"] == "camera_framing_changed")
        self.assertAlmostEqual(framing["start"], 5.0, delta=0.1)
        self.assertAlmostEqual(framing["end"], 9.0, delta=0.1)

    def test_scene_cuts_are_found_in_a_video(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "cuts.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2:r=30", "-f", "lavfi", "-i",
                            "color=c=white:s=64x64:d=2:r=30", "-filter_complex", "[0][1]concat=n=2:v=1[v]", "-map", "[v]", "-c:v", "mpeg4", str(video)],
                           check=True)
            cuts = gaps.scene_cuts(video)
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0], 2.0, delta=0.1)


class Clipping(unittest.TestCase):
    def test_the_bars_the_student_played_are_cut_out_with_a_lead_in_and_a_tail(self):
        score = [{"measure": m} for m in (1, 1, 2, 2, 3)]
        notes = [{"onset": 1.0 + k, "offset": 1.8 + k} for k in range(5)]
        alignment = [{"score_idx": k, "perf_idx": k} for k in range(5)]
        start, end = clip_range(score, notes, alignment, 2, 2, duration=10)
        self.assertAlmostEqual(start, 3.0 - 0.3)
        self.assertAlmostEqual(end, 4.8 + 0.5)
        self.assertEqual(clip_range(score, notes, alignment, 3, 3, duration=5.0)[1], 5.0)             # not past the end
        with self.assertRaisesRegex(InputError, "bars 7 to 8"):
            clip_range(score, notes, alignment, 7, 8, duration=10)


class Baseline(unittest.TestCase):
    def test_the_choices_are_listed_as_interpretive_with_numbers_only(self):
        score = [{"measure": 1, "articulation": None}, {"measure": 1, "articulation": "staccato"}, {"measure": 2, "articulation": None}]
        vib = {"present": True, "rate_hz": 5.5, "extent_cents": 25.0, "regularity": 0.9}
        note = lambda cents, art: {"cents": cents, "duration": 0.6, "articulation": art, "vibrato": vib}
        prof = {"tempo_ratio": 0.8, "median_loudness_db": -20.0,
                "notes": {0: note(20.0, 0.4), 1: note(3.0, 0.4), 2: note(-2.0, 0.9)},
                "measures": {1: {"tempo_ratio": 0.8, "loudness_db": -20.0, "bow_changes": 3, "bow_speed": 1.0},
                             2: {"tempo_ratio": 0.6, "loudness_db": -12.0, "bow_changes": None, "bow_speed": None}}}
        choices = baseline.reference_choices(prof, score)
        slugs = {(c["slug"], c.get("measure"), c.get("score_idx")) for c in choices}
        self.assertIn(("tempo_choice", None, None), slugs)
        self.assertIn(("local_tempo_shift", 2, None), slugs)                  # measure 2 is 25 % slower than usual
        self.assertIn(("dynamic_shaping", 2, None), slugs)                    # 8 dB louder than the median
        self.assertIn(("expressive_intonation", None, 0), slugs)              # 20 cents off equal temperament
        self.assertNotIn(("expressive_intonation", None, 1), slugs)
        self.assertIn(("played_short", None, 0), slugs)                       # unmarked and short
        self.assertNotIn(("played_short", None, 1), slugs)                    # staccato was written: not a choice
        self.assertIn("vibrato_use", [c["slug"] for c in choices])
        self.assertEqual([c["slug"] for c in choices if c["category"] == "bowing"], ["bowing_choice"])
        self.assertTrue(all(c["nature"] == "interpretive" or c["category"] == "pitch" for c in choices))
        self.assertEqual(baseline.NATURE["pitch"], "objective")
        json.dumps(choices)                                                   # plain JSON, no prose


class EndToEnd(unittest.TestCase):
    """The student plays bars 3-5 of the Minuet at the marked tempo; the 'professional' plays the whole piece slower, with vibrato of
    another depth, an orchestra and noise underneath."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        (d / "score.musicxml").write_text(minuet_musicxml(100))
        write_wav(d / "student.wav", play_minuet(bars=slice(2, 5), bpm=100, jitter=0.01))
        cls.pro = with_orchestra(play_minuet(bpm=80, vibrato_cents=25, jitter=0.01, seed=3))
        write_wav(d / "pro.wav", cls.pro)
        cls.perf = analyze(d / "student.wav", d / "score.musicxml", d / "out", pitch_backend="yin", reference=d / "pro.wav")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_it_is_clipped_to_the_students_bars_and_isolated(self):
        ref = self.perf.reference
        self.assertEqual((ref["clip"]["first_measure"], ref["clip"]["last_measure"]), (3, 5))
        beat = 60 / 80
        self.assertAlmostEqual(ref["clip"]["start"], 0.4 + 2 * 3 * beat - 0.3, delta=0.25)             # bar 3 starts after two 3-beat bars
        length = len(ref["audio"]["samples"]) / ref["audio"]["sr"]
        self.assertAlmostEqual(length, ref["clip"]["end"] - ref["clip"]["start"], delta=0.05)
        self.assertLess(length, len(self.pro) / SR / 2)                                                   # a part of the recording, not all of it
        self.assertEqual(ref["gaps"]["audio"], [])
        self.assertIn("harmonic_mask", ref["separation"])
        y = ref["audio"]["samples"]
        self.assertLess(band_energy(y, 311.1) / band_energy(y, 440, 40), 0.05)                           # the orchestra tone is far below the violin band

    def test_the_baseline_has_the_soloists_numbers_and_marks_the_choices(self):
        prof = self.perf.reference["profile"]
        self.assertAlmostEqual(prof["tempo_ratio"], 0.8, delta=0.05)                                      # 80 against the marked 100
        choices = {c["slug"]: c for c in self.perf.reference["choices"]}
        self.assertAlmostEqual(choices["tempo_choice"]["value"], 0.8, delta=0.05)
        self.assertEqual(choices["tempo_choice"]["nature"], "interpretive")
        self.assertGreater(choices["vibrato_use"]["value"], 0.5)
        self.assertTrue(6 < choices["vibrato_use"]["extent_cents"] < 25)
        measured = [e for e in prof["notes"].values() if "gap" not in e]
        self.assertGreaterEqual(len(measured), 10)
        self.assertLess(np.median([abs(e["cents"]) for e in measured]), 15)

    def test_the_comparison_is_differential_events_in_the_common_format_with_the_choice_as_cause(self):
        result = self.perf.assessment["reference"]
        self.assertEqual((result["module"], result["available"], result["score"]), ("reference", True, None))      # never scored
        self.assertEqual(set(result["summary"]), {"source", "clip", "gaps", "choices"})                    # no averages or tallies
        self.assertEqual(result["summary"]["gaps"]["notes"], 0)
        events = result["events"]
        self.assertTrue(events)
        fields = {"module", "category", "start", "end", "measure", "beat", "score_idx", "perf_idx", "severity", "value", "expected", "unit", "causes"}
        self.assertTrue(all(fields <= set(e) for e in events))                                           # the same shape as every other module's
        self.assertTrue(all(e["module"] == "reference" and e["category"].endswith("_vs_reference") for e in events))
        self.assertTrue(all(0.25 <= e["severity"] <= 0.55 for e in events))                               # emphasised, yet below a clear note error
        self.assertTrue(all(e["expected"] == 0.0 for e in events))                                        # the value is the differential from the soloist
        self.assertTrue(all(e["causes"][0]["cause"] == "soloist_interpretive_choice" and e["causes"][0]["confidence"] == 1.0 for e in events))
        tempo = [e for e in events if e["category"] == "tempo_vs_reference"]
        self.assertTrue(tempo)
        for e in tempo:                                                                                  # the student (~1.0) minus the soloist (~0.8)
            ev = e["causes"][0]["evidence"]
            self.assertTrue(e["measure"] in (3, 4, 5) and 0.1 < e["value"] < 0.35)
            self.assertAlmostEqual(e["value"], ev["student"] - ev["soloist"], places=3)
            self.assertAlmostEqual(ev["relative_differential"], e["value"] / ev["soloist"], places=3)
        self.assertEqual(tempo[0]["causes"][0]["evidence"]["soloist_choice"], "tempo_choice")
        self.assertFalse([e for e in events if e["category"] in ("attack_vs_reference", "pitch_vs_reference")])   # accuracy is judged elsewhere
        json.dumps(result)
        summary = reference_summary(self.perf.reference)
        self.assertNotIn("samples", summary)
        json.dumps(summary)

    def test_the_teacher_is_told_to_bring_in_the_professional_recording_when_there_are_reference_events(self):
        from maestro.feedback import generator, prompts
        digest = prompts.build_digest(self.perf)
        message = prompts.user_message(digest)
        self.assertIn("professional recording", message.split("Now talk to the student")[1])               # said in the turn the model answers
        plain = {k: v for k, v in digest.items() if k != "reference"}
        plain["problem_events"] = [e for e in digest["problem_events"] if e["skill"] != "reference"]
        self.assertNotIn("professional", prompts.user_message(plain).split("Now talk to the student")[1])
        self.assertIn("professional recording", prompts.SYSTEM)                                           # the system prompt says to talk about it
        self.assertTrue({3, 5} <= generator.known_bars(digest))

    def test_the_synthesis_lists_the_events_but_gives_no_score_and_no_problem_bars_or_causes_for_them(self):
        synthesis = self.perf.assessment["synthesis"]
        reference_ids = {e["id"] for e in synthesis["events"] if e["module"] == "reference"}
        self.assertTrue(reference_ids)
        self.assertNotIn("reference", synthesis["scores"])
        self.assertNotIn("soloist_interpretive_choice", [c["cause"] for c in synthesis["causes"]])
        self.assertEqual(synthesis["unavailable"].get("reference"), None)

    def test_the_feedback_model_gets_the_comparison_as_events_like_everything_else(self):
        from maestro.feedback import prompts
        digest = prompts.build_digest(self.perf)
        events = [e for e in digest["problem_events"] if e["skill"] == "reference"]
        self.assertTrue(events)
        self.assertLessEqual(len(events), prompts.MAX_REFERENCE_EVENTS)
        self.assertTrue(all(set(e) >= {"skill", "kind", "bar", "severity", "value", "expected", "unit", "causes"} for e in events))      # the same keys as every event
        self.assertEqual(events[0]["causes"][0][0], "soloist_interpretive_choice")
        self.assertNotIn("compared", digest["reference"])                                                 # no separate statistics table
        self.assertTrue(all(e["expected"] == 0 for e in events))                                          # values are differentials from the professional
        self.assertEqual(digest["reference"]["bars"], [3, 5])
        self.assertIn("tempo_choice", [c["slug"] for c in digest["reference"]["soloist_choices"]])
        small = prompts.compact(digest)
        self.assertTrue([e for e in small["problem_events"] if e["skill"] == "reference"])
        self.assertEqual(small["reference"], digest["reference"])
        self.assertNotIn("pro.wav", json.dumps(digest))                                                   # no file names
        self.assertIn("still come first", prompts.SYSTEM)                                                # real note errors come first
        self.assertIn("never as a mistake", prompts.SYSTEM)

    def test_a_reference_of_another_piece_is_reported_not_raised(self):
        other = [60 + (i * 5) % 17 for i in range(len(MIDIS))]                   # a different tune
        write_wav(self.d / "other.wav", synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(other)]))
        perf = analyze(self.d / "student.wav", self.d / "score.musicxml", pitch_backend="yin", reference=self.d / "other.wav")
        self.assertFalse(perf.assessment["reference"]["available"])
        self.assertIn("does not match the score", perf.assessment["reference"]["summary"]["reason"])
        self.assertIn("does not match the score", perf.assessment["synthesis"]["unavailable"]["reference"])
        self.assertIsNone(perf.reference)

    def test_progress_includes_the_reference_stage_only_when_asked_for(self):
        from maestro.utils.progress import Tracker
        names = []
        tracker = Tracker(lambda info: names.append(info["stage"]), skip=("neural",))
        analyze(self.d / "student.wav", self.d / "score.musicxml", pitch_backend="yin", reference=self.d / "pro.wav", progress=tracker)
        self.assertIn("reference", names)
        self.assertLess(names.index("visual"), names.index("reference"))           # after the student's own analysis, before the judging
        self.assertLess(names.index("reference"), names.index("assess"))


class WithVideo(unittest.TestCase):
    """A reference video: the clip is cut in one re-encode, camera gaps are kept, and bars without usable video carry no bowing numbers."""

    def test_gaps_stay_gaps_in_the_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "score.musicxml").write_text(minuet_musicxml(100))
            write_wav(d / "student.wav", play_minuet(bars=slice(2, 5), bpm=100, jitter=0.01))
            pro = with_orchestra(play_minuet(bpm=100, jitter=0.01, seed=3))
            write_wav(d / "pro.wav", pro)
            make_video(d / "pro.mp4", d / "pro.wav", len(pro) / SR)
            pose = synth_pose(minuet_notes(), with_bow=False)
            pose.xyz[int(3.15 * pose.fps):int(5.05 * pose.fps)] = np.nan            # the camera leaves the soloist for all of bar 5
            save_pose(pose, d / "pose.json")
            perf = analyze(d / "student.wav", d / "score.musicxml", d / "out", pitch_backend="yin", reference=d / "pro.mp4",
                           reference_options={"pose_file": d / "pose.json"})
            ref = perf.reference
            self.assertTrue(ref["visual_status"]["available"])
            self.assertTrue(Path(d / "out" / "reference" / "pro_common.mp4").is_file())
            self.assertAlmostEqual(probe_duration(d / "out" / "reference" / "pro_common.mp4"), ref["clip"]["end"] - ref["clip"]["start"], delta=0.3)
            self.assertEqual([g["reason"] for g in ref["gaps"]["video"]], ["pose_lost"])
            lost = ref["gaps"]["video"][0]
            self.assertTrue(all(st["end"] <= lost["start"] + 0.2 or st["start"] >= lost["end"] - 0.2 for st in ref["bow_strokes"]))
            blind = [m for m, v in ref["profile"]["measures"].items() if v["bow_changes"] is None]
            self.assertTrue(blind)                                                                    # bars inside the gap have no bowing numbers
            result = perf.assessment["reference"]
            self.assertEqual(sorted(result["summary"]["gaps"]["bars_without_video"]), sorted(blind))
            self.assertEqual(blind, [5])
            self.assertFalse([e for e in result["events"] if e["category"] == "bowing_vs_reference" and e["measure"] in blind])


class InTheGui(unittest.TestCase):
    """The start page's reference inputs, the server's handling of them and the report's reference card."""

    @classmethod
    def setUpClass(cls):
        cls.srv = gui_server.make_server(port=0)
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def start(self, files):
        ctype, body = multipart(files)
        request = urllib.request.Request(self.url + "/api/analyze", body, {"Content-Type": ctype})
        try:
            with urllib.request.urlopen(request) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def finish(self, job):
        for _ in range(3000):
            with urllib.request.urlopen(f"{self.url}/api/job/{job}") as r:
                status = json.loads(r.read())
            if status["state"] != "running":
                return status
            time.sleep(0.1)

    def uploads(self):
        return {"video": ("take.wav", write_bytes(play_minuet(bars=slice(2, 5), jitter=0.01))),
                "score": ("piece.musicxml", minuet_musicxml(100).encode())}

    def test_the_start_page_offers_a_youtube_link_and_a_file(self):
        with urllib.request.urlopen(self.url + "/") as r:
            page = r.read().decode()
        self.assertIn('name="reference_url"', page)
        self.assertIn('name="reference"', page)

    def test_a_reference_file_is_analysed_and_shown_in_the_report(self):
        status, data = self.start({**self.uploads(), "reference": ("pro.wav", write_bytes(with_orchestra(play_minuet(bpm=80, jitter=0.01, seed=3))))})
        self.assertEqual(status, 202)
        final = self.finish(data["job"])
        self.assertEqual(final["state"], "done", final)
        with urllib.request.urlopen(self.url + final["url"]) as r:
            page = r.read().decode()
        self.assertNotIn("reference", page_data(page))                                       # no separate section: the differences are events
        self.assertNotIn("Compared with a professional", page)
        self.assertTrue(any(e["module"] == "reference" for e in page_data(page)["events"]))                # the events are in the list like every other module's

    def test_the_bar_includes_the_reference_stage_only_when_a_reference_is_given(self):
        with mock.patch.object(gui_server, "analyze", side_effect=InputError("stop")) as fake:
            status, data = self.start({**self.uploads(), "reference_url": ("", b"https://youtu.be/abc123")})
            self.assertEqual(self.finish(data["job"])["state"], "error")
            self.assertEqual(fake.call_args.kwargs["reference"], "https://youtu.be/abc123")
            status, data = self.start(self.uploads())
            self.finish(data["job"])
            self.assertIsNone(fake.call_args.kwargs["reference"])

    def test_only_youtube_links_are_accepted(self):
        status, data = self.start({**self.uploads(), "reference_url": ("", b"https://example.com/a.mp4")})
        self.assertEqual(status, 400)
        self.assertIn("YouTube", data["error"])

    def test_there_is_no_professional_comparison_section_in_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            perf = performance(Path(tmp))
            page = build_page(perf, "media.mp4")
        self.assertNotIn('id="reference"', page)
        self.assertNotIn("Compared with a professional", page)
        self.assertNotIn("reference", page_data(page))


def write_bytes(y):
    import io
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def probe_duration(path):
    from maestro.utils.convert_video_type import probe
    return probe(path)["duration"]


if __name__ == "__main__":
    unittest.main()
