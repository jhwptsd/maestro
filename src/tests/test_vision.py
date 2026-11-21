"""Visual analysis: pose files, the VioPose runner, bow/arm motion, visual pitch, movement metrics and the analyze() hook. The movement metrics use the pose only; VioPose can also use audio and the visual pitch can be calibrated against heard notes. Synthetic violinists and a fake VioPose
checkout come from pose_fixture.py (VioPose itself needs TensorFlow and downloaded weights)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from maestro import analyze_visual
from maestro.assessment.bowing import assess_bowing
from maestro.assessment.posture import assess_posture
from maestro.utils.Performance import Performance
from maestro.utils.errors import InputError, PoseUnavailableError
from maestro.vision.motion_features import extract_motion
from maestro.vision.pose_estimation import estimate_pose, run_viopose
from maestro.vision import viopose_setup
from maestro.vision.pose_io import VIOPOSE_JOINTS, Pose, body_scale, load_pose, save_pose, viopose_result_to_pose
from maestro.vision.visual_pitch import (FIRST_POSITION_MM, OPEN_STRINGS, calibrate_from_audio, calibrate_visual_pitch, distance_from_nut_mm,
                                         estimate_visual_pitch, semitones_above_open)
from pose_fixture import (FAKE_INFERENCE, fingerboard_truth, last_call, minuet_notes, save_viopose_result, steady_mask, synth_pose,
                          write_fake_viopose, write_malicious_npy)

NOTES = minuet_notes()
N_STROKES = sum(1 for k, n in enumerate(NOTES) if k == 0 or not NOTES[k - 1][3])    # slurred notes share a stroke


def motion_of(pose):
    return extract_motion(pose)


class PoseFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.pose = synth_pose(NOTES, noise_mm=0)

    def test_json_roundtrip_keeps_gaps(self):
        self.pose.xyz[3, 2] = np.nan                       # an untracked joint in one frame
        save_pose(self.pose, self.d / "pose.json")
        back = load_pose(self.d / "pose.json")
        self.assertEqual((back.fps, back.joints), (30, self.pose.joints))
        np.testing.assert_allclose(back.xyz, self.pose.xyz, equal_nan=True)

    def test_bad_files_raise_input_error(self):
        (self.d / "x.json").write_text("{not json")
        (self.d / "y.json").write_text('{"fps": 30, "joints": ["a"]}')              # no xyz
        for name in ("x.json", "y.json", "missing.json"):
            with self.subTest(name), self.assertRaises(InputError):
                load_pose(self.d / name)
        with self.assertRaises(InputError):
            Pose(30, ["a", "b"], np.zeros((5, 3, 3)))        # joints/array mismatch

    def test_only_json_is_read_so_nothing_is_ever_unpickled(self):
        marker = self.d / "pwned"
        write_malicious_npy(self.d / "evil.npy", marker)
        with self.assertRaisesRegex(InputError, "convert_viopose"):
            load_pose(self.d / "evil.npy")
        self.assertFalse(marker.exists())                    # the payload never ran

    def test_body_scale_is_the_shoulder_width(self):
        p = Pose(30, ["l_shoulder", "r_shoulder"], np.array([[[0, 0, 0], [2, 0, 0]]] * 4))
        self.assertEqual(body_scale(p), 2.0)
        self.assertEqual(body_scale(Pose(30, ["l_hand"], np.zeros((4, 1, 3)))), 1.0)       # no shoulders: unit scale


class VioPoseResult(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.pose = synth_pose(NOTES, noise_mm=0)

    def test_result_npy_is_converted_with_derived_hands(self):
        save_viopose_result(self.pose, self.d / "result.npy")
        got = viopose_result_to_pose(self.d / "result.npy")
        self.assertEqual(got.joints, VIOPOSE_JOINTS + ["l_hand", "r_hand"])
        self.assertEqual((got.fps, len(got)), (30, len(self.pose)))
        np.testing.assert_allclose(got.joint("l_hand"), self.pose.joint("l_hand"), atol=1e-3)    # mean of index/pinky/thumb
        np.testing.assert_allclose(got.joint("r_wrist"), self.pose.joint("r_wrist"), atol=1e-3)

    def test_wrong_shapes_and_non_results_are_rejected(self):
        np.save(self.d / "a.npy", {"pred": np.zeros((10, 17, 3)), "fps": 30.0})              # 17 joints, not 18
        np.save(self.d / "b.npy", {"pred": np.zeros((10, 18, 3))})                           # no fps
        np.save(self.d / "c.npy", np.zeros((10, 18, 3)))                                    # a plain array, not a dict
        for name in "abc":
            with self.subTest(name), self.assertRaises(InputError):
                viopose_result_to_pose(self.d / f"{name}.npy")


def isolate_from_real_install(test, tmp):
    """Keep a VioPose really installed on this machine (third_party/) out of the test: no saved config, no default checkout."""
    for name, value in (("CONFIG_PATH", tmp / "no_config.json"), ("DEFAULT_DIR", tmp / "no_default_checkout")):
        patcher = mock.patch.object(viopose_setup, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)


class VioPoseRunner(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.pose = synth_pose(NOTES, noise_mm=0)
        self.root = write_fake_viopose(self.d / "viopose", self.pose)
        (self.d / "v.mp4").write_bytes(b"x")
        (self.d / "a.wav").write_bytes(b"x")
        self.env = mock.patch.dict(os.environ, {"VIOPOSE_DIR": str(self.root), "VIOPOSE_PYTHON": "python3"})
        self.env.start()
        self.addCleanup(self.env.stop)
        for k in ("VIOPOSE_AUDIO", "VIOPOSE_FOLDER", "FAKE_VIOPOSE_FAIL", "VIOPOSE_TIMEOUT"):
            os.environ.pop(k, None)
        isolate_from_real_install(self, self.d)

    def test_runs_inference_from_its_own_directory_and_reads_the_result(self):
        pose = run_viopose(self.d / "v.mp4")
        np.testing.assert_allclose(pose.joint("l_hand"), self.pose.joint("l_hand"), atol=1e-3)
        call = last_call(self.root)
        self.assertEqual(Path(call["cwd"]).resolve(), self.root.resolve())                # relative paths need this cwd
        self.assertEqual(call["args"], ["--folder", "Logs/VioPose", "--video_path", str((self.d / "v.mp4").resolve()), "--vis", "False"])
        self.assertEqual(call["weights"], "Logs/VioPose_wo_audio")                       # no audio: the *_wo_audio model
        self.assertFalse((self.root / "output" / "result.npy").exists())                 # result is read, then removed

    def test_audio_is_used_by_default_and_can_be_switched_off(self):
        pose = estimate_pose(self.d / "v.mp4", self.d / "a.wav")
        call = last_call(self.root)
        self.assertEqual(call["args"][-2:], ["--audio_path", str((self.d / "a.wav").resolve())])
        self.assertEqual(call["weights"], "Logs/VioPose")                                # the audio-trained model
        self.assertTrue(pose.audio_used)
        with mock.patch.dict(os.environ, {"VIOPOSE_AUDIO": "0"}):
            pose = estimate_pose(self.d / "v.mp4", self.d / "a.wav")
        self.assertNotIn("--audio_path", last_call(self.root)["args"])
        self.assertFalse(pose.audio_used)
        self.assertFalse(estimate_pose(self.d / "v.mp4").audio_used)                     # no audio given

    def test_missing_audio_weights_fall_back_to_the_video_only_model(self):
        (self.root / "Logs" / "VioPose").rmdir()
        pose = estimate_pose(self.d / "v.mp4", self.d / "a.wav")
        self.assertFalse(pose.audio_used)
        self.assertEqual(last_call(self.root)["weights"], "Logs/VioPose_wo_audio")

    def test_the_audio_flag_survives_saving_the_pose(self):
        pose = estimate_pose(self.d / "v.mp4", self.d / "a.wav")
        save_pose(pose, self.d / "p.json")
        self.assertTrue(load_pose(self.d / "p.json").audio_used)

    def test_setup_problems_are_reported_not_raised_as_crashes(self):
        with mock.patch.dict(os.environ, {"VIOPOSE_DIR": str(self.d / "nowhere")}):
            with self.assertRaisesRegex(PoseUnavailableError, "not installed"):
                run_viopose(self.d / "v.mp4")
        (self.root / "Logs" / "VioPose_wo_audio").rmdir()
        with self.assertRaisesRegex(PoseUnavailableError, "download.sh"):
            run_viopose(self.d / "v.mp4")
        with mock.patch.dict(os.environ, {"VIOPOSE_PYTHON": "/no/such/python"}):
            self.assertRaises(PoseUnavailableError, run_viopose, self.d / "v.mp4", None, self.root)

    def test_a_cancelled_run_is_killed_and_frees_the_lock(self):
        import threading
        import time
        (self.root / "inference.py").write_text("import time\ntime.sleep(60)\n")                 # a run that would take a minute
        cancel = threading.Event()
        threading.Timer(1.0, cancel.set).start()
        started = time.time()
        with self.assertRaisesRegex(PoseUnavailableError, "cancelled"):
            run_viopose(self.d / "v.mp4", cancel=cancel)
        self.assertLess(time.time() - started, 10)
        (self.root / "inference.py").write_text(FAKE_INFERENCE)                                # the next run is not blocked
        self.assertEqual(len(run_viopose(self.d / "v.mp4")), len(self.pose))

    def test_a_failing_run_reports_its_last_line(self):
        with mock.patch.dict(os.environ, {"FAKE_VIOPOSE_FAIL": "CUDA out of memory"}):
            with self.assertRaisesRegex(PoseUnavailableError, "CUDA out of memory"):
                run_viopose(self.d / "v.mp4")

    def test_a_stale_result_is_never_returned(self):
        (self.root / "output").mkdir()
        (self.root / "output" / "result.npy").write_bytes(b"left over from an earlier run")
        with mock.patch.dict(os.environ, {"FAKE_VIOPOSE_FAIL": "boom"}):
            with self.assertRaises(PoseUnavailableError):                                # not the old file
                run_viopose(self.d / "v.mp4")

    def test_the_install_is_found_without_any_environment_variables(self):
        """After setup the pose is generated implicitly: nothing to upload, nothing to export."""
        for k in ("VIOPOSE_DIR", "VIOPOSE_PYTHON"):
            os.environ.pop(k)
        with mock.patch.object(viopose_setup, "CONFIG_PATH", self.d / "viopose.json"):
            viopose_setup.write_config(self.root, "python3")                          # what the setup script records
            pose = estimate_pose(self.d / "v.mp4")
        np.testing.assert_allclose(pose.joint("r_hand"), self.pose.joint("r_hand"), atol=1e-3)
        with mock.patch.object(viopose_setup, "DEFAULT_DIR", self.root):                # or the default checkout location
            self.assertEqual(viopose_setup.load_config()["dir"], str(self.root))
        os.environ["VIOPOSE_DIR"] = "/elsewhere"                                          # environment beats the saved file
        with mock.patch.object(viopose_setup, "CONFIG_PATH", self.d / "viopose.json"):
            self.assertEqual(viopose_setup.load_config()["dir"], "/elsewhere")

    def test_pose_file_wins_and_missing_sources_mean_unavailable(self):
        save_pose(self.pose, self.d / "p.json")
        estimate_pose(self.d / "v.mp4", pose_file=self.d / "p.json")
        self.assertFalse((self.root / "last_call.json").exists())                        # VioPose was not run
        with mock.patch.dict(os.environ):
            os.environ.pop("VIOPOSE_DIR")
            os.environ.pop("VIOPOSE_PYTHON")
            with self.assertRaisesRegex(PoseUnavailableError, "setup"):                     # points at the setup scripts
                estimate_pose(self.d / "v.mp4")
        with self.assertRaisesRegex(PoseUnavailableError, "Audio-only"):
            estimate_pose(None)


class BowAndArmMovement(unittest.TestCase):
    def test_strokes_found_with_and_without_bow_markers(self):
        for markers in (False, True):
            with self.subTest(bow_markers=markers):
                m = motion_of(synth_pose(NOTES, with_bow=markers))
                strokes = m["bow"]["strokes"]
                self.assertEqual(len(strokes), N_STROKES)
                self.assertEqual(m["bow"]["axis_source"], "markers" if markers else "pca")
                self.assertTrue(all(a["dir"] == -b["dir"] for a, b in zip(strokes, strokes[1:])))   # directions alternate
                if markers:
                    self.assertEqual(strokes[0]["dir"], 1)       # the first stroke is a down-bow
                starts = [n[0] for k, n in enumerate(NOTES) if k == 0 or not NOTES[k - 1][3]]
                np.testing.assert_allclose([s["start"] for s in strokes[1:]], starts[1:], atol=0.1)

    def test_elbow_angle_and_shoulder_tilt(self):
        m = motion_of(synth_pose(NOTES, noise_mm=0))
        self.assertEqual(set(m["elbow_angle"]), {"left", "right"})
        self.assertTrue(np.all((m["elbow_angle"]["right"] > 20) & (m["elbow_angle"]["right"] < 180)))
        self.assertLess(abs(np.nanmean(m["shoulder_tilt_deg"])), 1)

    def test_missing_joints_degrade_gracefully(self):
        p = synth_pose(NOTES)
        keep = [i for i, j in enumerate(p.joints) if j.startswith("r_") or j == "head"]
        m = motion_of(Pose(p.fps, [p.joints[i] for i in keep], p.xyz[:, keep]))
        self.assertIsNone(m["shoulder_tilt_deg"])
        self.assertNotIn("left", m["hands"])
        self.assertTrue(m["bow"]["strokes"])

    def test_standing_still_has_no_strokes(self):
        p = synth_pose(NOTES, noise_mm=0)
        p.xyz[:] = p.xyz[0]
        self.assertEqual(motion_of(p)["bow"]["strokes"], [])
        self.assertFalse(assess_bowing(motion_of(p))["available"])


class BowingAndPosture(unittest.TestCase):
    def test_even_bowing_is_recognised(self):
        b = assess_bowing(motion_of(synth_pose(NOTES, with_bow=True)))
        self.assertEqual(b["n_strokes"], N_STROKES)
        self.assertLess(b["stroke_speed_cv"], 0.1)                   # constant bow speed
        self.assertLess(b["speed_cv_within_stroke"], 0.2)
        self.assertGreater(b["strokes_per_second"], 1)

    def test_uneven_bowing_scores_worse(self):
        even = assess_bowing(motion_of(synth_pose(NOTES)))
        uneven = assess_bowing(motion_of(synth_pose(NOTES, stroke_jitter=0.6, seed=3)))
        self.assertGreater(uneven["stroke_length_cv"], even["stroke_length_cv"])

    def test_direction_balance(self):
        m = motion_of(synth_pose(NOTES, noise_mm=0))
        balance = assess_bowing(m)["direction_balance"]
        self.assertTrue(0 < balance <= 1)                            # 1 = both directions use equal bow
        for s in m["bow"]["strokes"]:                                # make every up-stroke half as long
            s["length"] *= 1.0 if s["dir"] > 0 else 0.5
        self.assertLess(assess_bowing(m)["direction_balance"], 0.7)

    def test_posture_metrics_follow_sway(self):
        calm, swaying = synth_pose(NOTES, sway_mm=1), synth_pose(NOTES, sway_mm=40)
        a, b = assess_posture(motion_of(calm), calm), assess_posture(motion_of(swaying), swaying)
        self.assertGreater(b["shoulder_sway"], 3 * a["shoulder_sway"])
        self.assertLess(abs(a["shoulder_tilt_mean_deg"]), 1)

    def test_posture_without_upper_body_joints_is_unavailable(self):
        p = synth_pose(NOTES)
        keep = [i for i, j in enumerate(p.joints) if j in ("r_hand", "l_hand")]
        pose = Pose(p.fps, ["r_hand", "l_hand"], p.xyz[:, keep])
        self.assertFalse(assess_posture(motion_of(pose), pose)["available"])


class VisualPitch(unittest.TestCase):
    """Pitch from the left hand's position, with no audio anywhere."""

    def estimate(self, pose, **kw):
        return estimate_visual_pitch(pose, motion_of(pose)["hands"], **kw)

    def test_string_length_law(self):
        self.assertAlmostEqual(float(semitones_above_open(distance_from_nut_mm(12))), 12, places=9)       # octave = half the string
        self.assertAlmostEqual(float(distance_from_nut_mm(12)), 162.5, places=6)
        self.assertAlmostEqual(float(semitones_above_open(distance_from_nut_mm(7))), 7, places=9)
        self.assertTrue(np.isfinite(semitones_above_open(1e6)))                                            # clipped, not inf

    def test_recovers_semitones_above_the_open_string_from_the_hand_alone(self):
        pose = synth_pose(NOTES)
        vp = self.estimate(pose)
        n_true, midi_true = fingerboard_truth(NOTES, pose.times)
        steady = steady_mask(NOTES, pose.times)                     # the estimate is smoothed over 0.1 s: judge held notes
        self.assertTrue(vp["available"])
        self.assertEqual(vp["method"], "geometry")
        error = np.abs(vp["semitones_above_open"] - n_true)[steady]
        self.assertLess(np.median(error), 0.2)
        self.assertGreater(np.mean(error < 0.5), 0.95)
        # the sounding pitch is one of the four candidates (one per open string): the string itself is not visible
        gap = np.abs(vp["candidates"] - midi_true[:, None]).min(axis=1)[steady]
        self.assertGreater(np.mean(gap < 0.5), 0.95)
        self.assertEqual(vp["candidates"].shape, (len(pose), 4))
        np.testing.assert_allclose(vp["candidates"][:, 0] - vp["semitones_above_open"], OPEN_STRINGS[0])
        self.assertAlmostEqual(vp["range_semitones"], np.ptp(n_true[steady]), delta=1.5)

    def test_hand_toward_the_body_means_higher_pitch(self):
        pose = synth_pose(NOTES, noise_mm=0)
        base = self.estimate(pose)["semitones_above_open"]
        direction = pose.joint("l_hand") - pose.joint("l_shoulder")
        playing = steady_mask(NOTES, pose.times)
        pose.xyz[playing, pose.joints.index("l_hand")] -= 0.04 * direction[playing]     # playing frames ~17 mm closer
        raised = self.estimate(pose)["semitones_above_open"]
        self.assertTrue(np.all(raised[playing] > base[playing] + 0.5))                 # higher on the fingerboard

    def test_without_a_rest_position_shifts_are_still_read_but_absolute_pitch_is_offset(self):
        """The furthest-reach reference assumes the hand returns to 1st position. Cut the rests off the clip and the
        absolute scale is off by a few semitones, but the changes between notes stay close."""
        pose = synth_pose(NOTES, noise_mm=0)
        n_true, _ = fingerboard_truth(NOTES, pose.times)
        first, last = int(NOTES[0][0] * pose.fps), int(NOTES[-1][1] * pose.fps)
        trimmed = Pose(pose.fps, pose.joints, pose.xyz[first:last])
        est = self.estimate(trimmed)["semitones_above_open"]
        truth = n_true[first:last]
        steady = steady_mask(NOTES, trimmed.times + first / pose.fps)
        self.assertTrue(1.0 < np.mean(truth[steady] - est[steady]) < 4.5)             # offset: reads too low
        self.assertLess(np.std(est[steady] - truth[steady]), 0.5)                      # ...but not wobbling: shifts survive

    def test_calibration_gives_absolute_pitch(self):
        a_string = np.array([69 + n for n in range(0, 13)], dtype=float)                       # A4..A5 on one string
        distances = 420.0 - (distance_from_nut_mm(a_string - 69) - FIRST_POSITION_MM)          # as the arm would see them
        cal = calibrate_visual_pitch(distances, a_string)
        self.assertLess(cal["mae_semitones"], 0.2)
        self.assertEqual(cal["n"], 13)
        self.assertLess(cal["r_spearman"], -0.99)                                              # closer to the body = higher
        self.assertEqual(cal["within_1_semitone"], 1.0)
        with self.assertRaises(ValueError):
            calibrate_visual_pitch(distances[:5], a_string[:5])
        vp = self.estimate(synth_pose(NOTES), calibration=calibrate_visual_pitch(distances, a_string))
        self.assertEqual(vp["method"], "calibrated")
        self.assertEqual(vp["midi"].shape, vp["semitones_above_open"].shape)

    def test_calibration_from_the_heard_notes(self):
        pose = synth_pose(NOTES, noise_mm=0)
        hands = extract_motion(pose)["hands"]
        heard = [{"onset": o, "offset": f, "midi": float(m)} for o, f, m, _ in NOTES]
        cal = calibrate_from_audio(pose, hands, heard)
        self.assertGreaterEqual(cal["n"], 6)
        self.assertLess(cal["mae_semitones"], 0.3)                                     # one curve fits all four strings
        vp = estimate_visual_pitch(pose, hands, cal)
        truth, _ = fingerboard_truth(NOTES, pose.times)
        steady = steady_mask(NOTES, pose.times)
        self.assertLess(np.mean(np.abs(vp["semitones_above_open"][steady] - truth[steady])), 0.5)   # the geometry read 1-4 too low
        self.assertEqual(vp["candidates"].shape, (len(pose.times), 4))
        self.assertIsNone(calibrate_from_audio(pose, hands, heard[:3]))                # too few notes to fit a curve
        no_left = synth_pose(NOTES, left_hand=False)
        self.assertIsNone(calibrate_from_audio(no_left, extract_motion(no_left)["hands"], heard))

    def test_unavailable_cases(self):
        no_left = synth_pose(NOTES, left_hand=False)
        self.assertFalse(self.estimate(no_left)["available"])
        short = synth_pose(NOTES)
        short = Pose(short.fps, short.joints, short.xyz[:5])
        self.assertIn("frames", self.estimate(short)["reason"])


class AnalyzeVisualHook(unittest.TestCase):
    """analyze_visual needs only a pose; audio (VioPose's audio model, visual-pitch calibration) is used when present."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        isolate_from_real_install(self, self.d)
        save_pose(synth_pose(NOTES, with_bow=True), self.d / "pose.json")

    def bare_performance(self):
        perf = Performance()
        perf.assessment = {}                                 # no audio_features, no alignment, no score
        return perf

    def test_runs_on_a_performance_without_any_audio_results(self):
        perf = analyze_visual(self.bare_performance(), self.d / "pose.json")
        a = perf.assessment
        self.assertTrue(a["visual_status"]["available"])
        self.assertEqual(a["bowing"]["n_strokes"], N_STROKES)
        self.assertTrue(a["posture"]["available"])
        self.assertTrue(a["visual_pitch"]["available"])
        self.assertIsNotNone(perf.pose)
        self.assertIsNotNone(perf.motion_features)

    def test_without_viopose_or_a_pose_file_it_says_why(self):
        perf = self.bare_performance()
        perf.video = str(self.d / "v.mp4")
        with mock.patch.dict(os.environ):
            os.environ.pop("VIOPOSE_DIR", None)
            analyze_visual(perf)
        self.assertFalse(perf.assessment["visual_status"]["available"])
        self.assertIn("setup", perf.assessment["visual_status"]["reason"])               # tells the user how to install it
        self.assertNotIn("bowing", perf.assessment)

    def test_viopose_generates_the_pose_which_is_used_and_saved(self):
        root = write_fake_viopose(self.d / "viopose", synth_pose(NOTES, noise_mm=0))
        perf = self.bare_performance()
        perf.video = str(self.d / "v.mp4")
        (self.d / "v.mp4").write_bytes(b"x")
        out = self.d / "out"
        out.mkdir()
        with mock.patch.dict(os.environ, {"VIOPOSE_DIR": str(root), "VIOPOSE_PYTHON": "python3"}):
            analyze_visual(perf, save_dir=out)
        self.assertTrue(perf.assessment["visual_status"]["available"])
        self.assertEqual(perf.assessment["bowing"]["n_strokes"], N_STROKES)
        saved = load_pose(out / "pose.json")                                   # the generated pose is kept next to the outputs
        np.testing.assert_allclose(saved.joint("l_hand"), perf.pose.joint("l_hand"))

    def test_the_visual_stage_uses_the_audio_unless_told_not_to(self):
        root = write_fake_viopose(self.d / "viopose", synth_pose(NOTES, noise_mm=0))
        (self.d / "v.mp4").write_bytes(b"x")
        out = self.d / "out"
        out.mkdir()

        def performance():
            perf = self.bare_performance()
            perf.video = str(self.d / "v.mp4")
            perf.audio = {"samples": np.zeros(16000, dtype=np.float32), "sr": 16000}
            perf.audio_features = {"notes": [{"onset": o, "offset": f, "midi": float(m)} for o, f, m, _ in NOTES]}
            return perf

        with mock.patch.dict(os.environ, {"VIOPOSE_DIR": str(root), "VIOPOSE_PYTHON": "python3"}):
            perf = analyze_visual(performance(), save_dir=out)
            status = perf.assessment["visual_status"]
            self.assertEqual((status["viopose_audio"], status["pitch_calibrated"]), (True, True))
            self.assertEqual(perf.assessment["visual_pitch"]["method"], "calibrated")
            self.assertIn("--audio_path", last_call(root)["args"])
            self.assertTrue((out / "viopose_audio.wav").is_file())
            perf = analyze_visual(performance(), save_dir=out, use_audio=False)
            status = perf.assessment["visual_status"]
            self.assertEqual((status["viopose_audio"], status["pitch_calibrated"]), (False, False))
            self.assertEqual(perf.assessment["visual_pitch"]["method"], "geometry")
            self.assertNotIn("--audio_path", last_call(root)["args"])

    def test_a_given_pose_file_is_not_rewritten(self):
        out = self.d / "out"
        out.mkdir()
        analyze_visual(self.bare_performance(), self.d / "pose.json", save_dir=out)
        self.assertFalse((out / "pose.json").exists())


if __name__ == "__main__":
    unittest.main()
