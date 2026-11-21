"""End to end: synthetic performance (video + MusicXML score) through maestro.analyze."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from helpers import make_video, musicxml, synth, write_wav
from minuet import minuet_musicxml, play_minuet
from test_gui import page_data
import maestro
from maestro import analyze
from maestro.utils.errors import InputError, MismatchError, PoseUnavailableError


class Concurrency(unittest.TestCase):
    """The pose is estimated while the audio is analysed, and cancelled if the analysis fails."""
    midis = [69, 71, 72, 74, 76, 74, 72, 71]

    def files(self, d, score=None):
        (d / "s.musicxml").write_text(musicxml(score or self.midis))
        write_wav(d / "a.wav", synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(self.midis)]))
        make_video(d / "v.mp4", d / "a.wav", 5)

    def test_the_pose_runs_in_the_background_and_the_stages_are_timed(self):
        events = []

        def fake_pose(video, audio, pose_file=None, cancel=None):
            events.append(("pose started", threading.current_thread().name))
            time.sleep(1.0)
            raise PoseUnavailableError("fake: no pose")

        real_transcribe = maestro.transcribe

        def transcribe(*a, **k):
            events.append(("transcribe", threading.current_thread().name))
            return real_transcribe(*a, **k)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(maestro, "estimate_pose", fake_pose), \
                mock.patch.object(maestro, "transcribe", transcribe):
            d = Path(tmp)
            self.files(d)
            perf = analyze(d / "v.mp4", d / "s.musicxml", pitch_backend="yin")
        names = [e[0] for e in events]
        self.assertEqual(names, ["pose started", "transcribe"])                        # the pose started first, did not block the audio
        self.assertTrue(events[0][1].startswith("pose"))                                # ... in its own thread
        self.assertFalse(perf.assessment["visual_status"]["available"])
        self.assertGreaterEqual(perf.timings["pose"], 1.0)
        for stage in ("preprocess", "pitch", "transcribe", "features", "align", "pose_wait", "visual", "assess"):
            self.assertIn(stage, perf.timings)

    def test_a_failed_analysis_cancels_the_pose(self):
        seen = {}

        def fake_pose(video, audio, pose_file=None, cancel=None):
            seen["cancelled"] = cancel.wait(20)                                        # blocks like VioPose until told to stop
            raise PoseUnavailableError("VioPose was cancelled")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(maestro, "estimate_pose", fake_pose):
            d = Path(tmp)
            self.files(d, score=[60, 62, 64, 65, 55, 57, 59, 60])                      # the wrong score
            started = time.time()
            with self.assertRaises(MismatchError):
                analyze(d / "v.mp4", d / "s.musicxml", pitch_backend="yin")
            self.assertLess(time.time() - started, 15)                                 # the error did not wait for the pose
            deadline = time.time() + 5
            while "cancelled" not in seen and time.time() < deadline:
                time.sleep(0.05)
        self.assertTrue(seen.get("cancelled"))


class Pipeline(unittest.TestCase):
    def test_analyze_detects_sharp_and_late_notes(self):
        midis = [69, 71, 72, 74, 76, 74, 72, 71]
        perf = [(m + (0.3 if i == 3 else 0), 0.3 + 0.5 * i + (0.12 if i == 5 else 0), 0.4)
                for i, m in enumerate(midis)]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "s.musicxml").write_text(musicxml(midis))
            write_wav(directory / "a.wav", synth(perf))
            make_video(directory / "v.mp4", directory / "a.wav", 5)
            performance = analyze(
                directory / "v.mp4", directory / "s.musicxml", directory / "out", pitch_backend="yin"
            )
            self.assertTrue((directory / "out" / "score.svg").exists())

        intonation = {note["score_idx"]: note["cents"] for note in performance.assessment["intonation"]["notes"]}
        self.assertEqual(len(intonation), len(midis))
        self.assertAlmostEqual(intonation[3], 30, delta=8)
        self.assertLess(max(abs(c) for i, c in intonation.items() if i != 3), 10)

        timing = {note["score_idx"]: note["error"] for note in performance.assessment["rhythm"]["notes"]}
        self.assertEqual(max(timing, key=lambda key: abs(timing[key])), 5)
        self.assertAlmostEqual(performance.assessment["rhythm"]["tempo_ratio"], 1.0, delta=0.1)

        report = performance.assessment["synthesis"]                  # one JSON report from every module
        self.assertEqual(json.loads(json.dumps(report)), report)
        modules = ("intonation", "rhythm", "expression", "tone")
        self.assertEqual(report["n_events"], sum(len(performance.assessment[m]["events"]) for m in modules))
        self.assertTrue(set(modules) <= set(report["scores"]))          # (posture/bowing need VioPose: not installed here)

    def test_realistic_minuet_through_video_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "minuet.musicxml").write_text(minuet_musicxml())
            write_wav(directory / "a.wav", play_minuet())
            make_video(directory / "v.mp4", directory / "a.wav", 18)
            performance = analyze(
                directory / "v.mp4", directory / "minuet.musicxml", pitch_backend="yin"
            )
        self.assertGreater(min(performance.assessment["correspondence"].values()), 0.8)
        self.assertLess(performance.assessment["intonation"]["mean_abs_cents"], 15)
        self.assertEqual(performance.assessment["intonation"]["n_wrong_notes"], 0)
        self.assertEqual(performance.audio_features["pitch_backend"], "yin")

    def test_recording_that_starts_halfway_down_the_page(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "minuet.musicxml").write_text(minuet_musicxml())
            write_wav(directory / "a.wav", play_minuet(bars=slice(4, 8)))
            make_video(directory / "v.mp4", directory / "a.wav", 9)
            performance = analyze(
                directory / "v.mp4", directory / "minuet.musicxml", pitch_backend="yin"
            )
        section = performance.assessment["section"]
        self.assertEqual((section["first_measure"], section["last_measure"]), (5, 8))
        self.assertAlmostEqual(section["coverage"], 0.5, delta=0.05)
        matched = [item["score_idx"] for item in performance.alignment if item["perf_idx"] is not None]
        self.assertGreater(min(matched), 14)
        self.assertGreater(min(performance.assessment["correspondence"].values()), 0.8)
        self.assertLess(performance.assessment["intonation"]["mean_abs_cents"], 15)
        self.assertLess(performance.assessment["rhythm"]["mean_abs_error"], 0.08)

    def test_middle_section_and_report_card(self):
        from maestro.gui.report import build_page
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "minuet.musicxml").write_text(minuet_musicxml())
            write_wav(directory / "a.wav", play_minuet(bars=slice(2, 6), seed=3))
            make_video(directory / "v.mp4", directory / "a.wav", 9)
            performance = analyze(
                directory / "v.mp4", directory / "minuet.musicxml", pitch_backend="yin"
            )
        section = performance.assessment["section"]
        self.assertEqual((section["first_measure"], section["last_measure"]), (3, 6))
        self.assertEqual({n["measure"] for n in page_data(build_page(performance, "m.mp4"))["traces"]["notes"]}, {3, 4, 5, 6})

    def test_wrong_score_for_recording_raises_mismatch(self):
        played = [69, 71, 72, 74, 76, 74, 72, 71]
        score = [60, 62, 64, 65, 55, 57, 59, 60]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "s.musicxml").write_text(musicxml(score))
            write_wav(directory / "a.wav", synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(played)]))
            make_video(directory / "v.mp4", directory / "a.wav", 5)
            with self.assertRaises(MismatchError):
                analyze(directory / "v.mp4", directory / "s.musicxml", pitch_backend="yin")

    def test_midi_score_and_audio_only_upload(self):
        from helpers import write_midi
        midis = [69, 71, 72, 74, 76, 74, 72, 71]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_midi(directory / "s.mid", [(i, i + 1, midi) for i, midi in enumerate(midis)])
            write_wav(directory / "a.wav", synth([(midi, 0.3 + 0.5 * i, 0.4)
                                                   for i, midi in enumerate(midis)]))
            performance = analyze(directory / "a.wav", directory / "s.mid", pitch_backend="yin")
        self.assertEqual(len(performance.assessment["intonation"]["notes"]), len(midis))
        self.assertLess(performance.assessment["intonation"]["mean_abs_cents"], 10)

    def test_bad_inputs_raise_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "s.musicxml").write_text(musicxml([69]))
            with self.assertRaises(InputError):
                analyze(directory / "nope.mp4", directory / "s.musicxml", pitch_backend="yin")


if __name__ == "__main__":
    unittest.main()
