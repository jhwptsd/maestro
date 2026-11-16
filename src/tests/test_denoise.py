"""Denoising (preprocessing/denoise.py): audio keeps vibrato, bow noise and tone while losing background
noise; video noise drops while motion stays; score artefacts are removed but real notes are kept."""
import dataclasses
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import SR, make_video, write_midi, write_wav
from minuet import minuet_musicxml, play_minuet
from noise_fixture import (background_noise, band_energy, bowed_note, pitch_modulation, snr_db, vibrato_note)
from maestro import analyze
from maestro.audio.transcription import transcribe
from maestro.preprocessing.audio import preprocess_audio
from maestro.preprocessing.denoise import denoise_audio, denoise_score, video_filters
from maestro.preprocessing.score import preprocess_score
from maestro.preprocessing.video import preprocess_video
from maestro.score.musicxml import parse_musicxml
from maestro.score.score_representation import Note, Score
from maestro.utils.Performance import Performance
from maestro.utils.convert_file_type import convert_to_musicxml
from maestro.utils.convert_video_type import probe


class AudioDenoising(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.clean = play_minuet(noise=0.02, seed=1)                       # violin-like, with its own bow noise
        cls.noisy = cls.clean + background_noise(len(cls.clean), rng, 0.02).astype(np.float32)   # ~16 dB SNR
        cls.out, cls.info = denoise_audio(cls.noisy, SR)

    def test_background_noise_is_reduced(self):
        gain = snr_db(self.clean, self.out) - snr_db(self.clean, self.noisy)
        self.assertGreater(gain, 6)
        self.assertEqual(self.out.dtype, np.float32)
        self.assertEqual(len(self.out), len(self.noisy))
        gap = slice(0, int(0.3 * SR))                                       # noise-only lead-in
        self.assertLess(np.sqrt((self.out[gap] ** 2).mean()), 0.4 * np.sqrt((self.noisy[gap] ** 2).mean()))

    def test_snr_estimate_is_realistic(self):
        self.assertTrue(self.info["applied"])
        self.assertAlmostEqual(self.info["snr_db"], snr_db(self.clean, self.noisy), delta=3)

    def test_tonal_content_and_pitch_are_preserved(self):
        # YIN on purpose: this checks the denoiser, so it needs a precise tracker that is independent of any neural model
        ref, got = transcribe(self.clean, SR, backend="yin"), transcribe(self.out, SR, backend="yin")
        self.assertEqual(len(got), len(ref))
        cents = [abs(1200 * np.log2(a["hz"] / b["hz"])) for a, b in zip(got, ref)]
        self.assertLess(max(cents), 5)

    def test_vibrato_depth_and_rate_are_preserved(self):
        rng = np.random.default_rng(1)
        for depth in (15, 40, 60):
            with self.subTest(depth_cents=depth):
                note, span = vibrato_note(depth_cents=depth)
                den, _ = denoise_audio(note + background_noise(len(note), rng, 0.03).astype(np.float32), SR)
                (d0, r0), (d1, r1) = pitch_modulation(note, SR, span), pitch_modulation(den, SR, span)
                self.assertGreater(d1 / d0, 0.9)
                self.assertLess(abs(r1 - r0), 0.3)

    def test_bow_noise_is_preserved_but_gap_noise_is_removed(self):
        rng = np.random.default_rng(2)
        bowed, _, span = bowed_note(rng, bow_level=0.02)
        noisy = bowed + background_noise(len(bowed), rng, 0.01).astype(np.float32)   # background 6 dB under the bow noise
        den, _ = denoise_audio(noisy, SR)
        kept = band_energy(den, 4000, 9000, *span) / band_energy(bowed, 4000, 9000, *span)
        self.assertGreater(kept, 0.8)                                      # bow hiss while the note sounds stays
        self.assertLess(kept, 1.25)
        gap = (0.0, 0.5)
        self.assertLess(band_energy(den, 4000, 9000, *gap), 0.2 * band_energy(noisy, 4000, 9000, *gap))

    def test_attenuation_is_limited_so_nothing_is_erased(self):
        silence_plus_noise = background_noise(SR * 3, np.random.default_rng(3), 0.02).astype(np.float32)
        out, _ = denoise_audio(silence_plus_noise, SR, reduction_db=12)
        ratio = np.sqrt((out ** 2).mean()) / np.sqrt((silence_plus_noise ** 2).mean())
        self.assertGreater(ratio, 10 ** (-12 / 20) * 0.7)                  # at most ~12 dB down (plus rumble filter)

    def test_clean_audio_passes_through_unchanged(self):
        clean = play_minuet(noise=0.0, reverb=False)                       # exact digital silence between notes
        out, _ = denoise_audio(clean, SR)
        self.assertGreater(snr_db(clean, out), 40)

    def test_degenerate_input(self):
        for y in (np.zeros(SR, np.float32), np.ones(100, np.float32) * 0.1, np.array([], np.float32)):
            out, info = denoise_audio(y, SR)
            self.assertEqual(len(out), len(y))
            self.assertFalse(info["applied"])
        profile = self.noisy[:int(0.3 * SR)]                               # explicit noise-only stretch
        out, info = denoise_audio(self.noisy, SR, noise_profile=profile)
        self.assertGreater(snr_db(self.clean, out), snr_db(self.clean, self.noisy) + 5)

    def test_preprocess_audio_denoises_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            write_wav(Path(d) / "a.wav", self.noisy)
            on = preprocess_audio(self._perf(Path(d) / "a.wav"))
            off = preprocess_audio(self._perf(Path(d) / "a.wav"), denoise=False)
        self.assertIn("snr_db", on.preprocessing_report["audio"])
        self.assertNotIn("audio", off.preprocessing_report)
        ref = play_minuet(noise=0.02, seed=1)
        self.assertGreater(snr_db(ref, on.audio["samples"]), snr_db(ref, off.audio["samples"]) + 5)

    @staticmethod
    def _perf(path):
        p = Performance()
        p.audio_source = str(path)
        return p

    def test_analysis_with_noisy_recording(self):
        rng = np.random.default_rng(4)
        loud_noise = self.clean + background_noise(len(self.clean), rng, 0.08).astype(np.float32)   # ~4 dB SNR
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "m.musicxml").write_text(minuet_musicxml())
            write_wav(d / "a.wav", loud_noise)
            make_video(d / "v.mp4", d / "a.wav", 18)
            p = analyze(d / "v.mp4", d / "m.musicxml")
        self.assertGreater(min(p.assessment["correspondence"].values()), 0.8)
        self.assertLess(p.assessment["intonation"]["mean_abs_cents"], 15)
        self.assertGreater(p.preprocessing_report["audio"]["snr_db"], 0)


class VideoDenoising(unittest.TestCase):
    """A white square moves across a static background with strong temporal sensor noise."""
    W = 96

    def make(self, path, noisy=True):
        graph = ("color=c=0x303030:s=96x96:r=30:d=2[bg];color=c=white:s=16x16:r=30:d=2[fg];"
                 "[bg][fg]overlay=x='t*30':y=40")
        if noisy:
            graph += ",noise=alls=18:allf=t+u"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", graph,
                        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(path)], check=True)

    def frames(self, path):
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(-1, self.W, self.W).astype(float)

    def test_noise_drops_while_motion_stays(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.make(d / "noisy.mp4")
            write_wav(d / "a.wav", np.zeros(SR * 2, np.float32) + 0.01 * np.sin(np.arange(SR * 2) / 20).astype(np.float32))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(d / "noisy.mp4"), "-i", str(d / "a.wav"),
                            "-shortest", "-c:v", "copy", "-c:a", "aac", str(d / "in.mp4")], check=True)
            on = preprocess_video(Performance(), d / "in.mp4", d / "on", denoise=True)
            off = preprocess_video(Performance(), d / "in.mp4", d / "off", denoise=False)
            f_on, f_off = self.frames(on.video), self.frames(off.video)
        n = min(len(f_on), len(f_off))
        static = (slice(n // 2, n), slice(70, 90), slice(5, 40))              # lower-left corner: square never there
        flicker = lambda f: np.diff(f[:n, 70:90, 5:40], axis=0).std()          # frame-to-frame noise
        self.assertLess(flicker(f_on), 0.7 * flicker(f_off))
        # the moving square is still there, bright, and moves at the same speed
        def centre_x(f):
            return np.array([np.average(np.arange(self.W), weights=np.clip(fr[36:60].mean(0) - 80, 0, None) + 1e-9) for fr in f[:n]])
        cx_on, cx_off = centre_x(f_on), centre_x(f_off)
        self.assertGreater(np.corrcoef(cx_on, np.arange(n))[0, 1], 0.98)
        self.assertLess(np.abs(cx_on - cx_off).mean(), 3)                      # pixels
        self.assertGreater(f_on[10, 40:56, int(cx_on[10]) - 6:int(cx_on[10]) + 6].max(), 180)   # still white

    def test_filters_and_report(self):
        self.assertTrue(video_filters()[0].startswith("hqdn3d"))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_wav(d / "a.wav", np.zeros(SR * 3, np.float32) + 0.01)
            make_video(d / "v.mp4", d / "a.wav", 3)
            perf = preprocess_video(Performance(), d / "v.mp4", d / "out")
            self.assertTrue(perf.preprocessing_report["video"])
            self.assertTrue(probe(perf.video)["has_video"])
            self.assertFalse(preprocess_video(Performance(), d / "v.mp4", d / "out2", denoise=False).preprocessing_report["video"])


def note(midi, onset, offset, measure=1, articulation=None):
    return Note(measure, 1.0, "X", midi, (offset - onset) * 2, articulation, onset, offset)


class ScoreDenoising(unittest.TestCase):
    def test_artefacts_removed_real_notes_kept(self):
        notes = [
            note(69, 0.0, 0.5),                    # A4
            note(69, 0.0, 0.5),                    # exact duplicate (OMR double read)
            note(69, 0.5, 1.0),                    # repeated note, adjacent: REAL, must stay
            note(71, 1.0, 1.0075),                 # 7.5 ms glitch
            note(40, 1.0, 1.5),                    # below the violin's range
            note(120, 1.0, 1.5),                   # above it
            note(72, 1.0, 1.5, articulation="staccato"),
            note(72, 1.2, 1.7),                    # same pitch starting while the previous still sounds
            note(60, 2.0, 2.075),                  # a real 32nd note at 100 bpm: short but valid
        ]
        out, report = denoise_score(Score(notes, 100))
        self.assertEqual([(n.midi_pitch, n.onset) for n in out.notes],
                         [(69, 0.0), (69, 0.5), (72, 1.0), (60, 2.0)])
        self.assertEqual(report, {"out_of_range": 2, "too_short": 1, "duplicates": 1, "overlapping": 1, "snapped": 0})
        self.assertEqual(out.notes[2].articulation, "staccato")
        self.assertAlmostEqual(out.notes[2].offset, 1.7)                      # kept the longer extent
        self.assertEqual(len(notes), 9)                                      # input not modified
        self.assertEqual(out.tempo, 100)

    def test_onset_jitter_is_snapped_so_chords_stay_together(self):
        notes = [note(55, 0.0, 1.0), note(62, 0.004, 1.004), note(69, 0.010, 1.01), note(71, 1.0, 1.5)]
        out, report = denoise_score(Score(notes, 120))
        self.assertEqual({n.onset for n in out.notes[:3]}, {0.0})
        self.assertAlmostEqual(out.notes[1].offset, 1.0)                     # duration preserved
        self.assertEqual(report["snapped"], 2)
        self.assertEqual(out.notes[3].onset, 1.0)

    def test_clean_score_is_untouched_and_idempotent(self):
        from minuet import MIDIS
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "m.musicxml").write_text(minuet_musicxml())
            score = parse_musicxml(Path(d) / "m.musicxml")
        once, report = denoise_score(score)
        self.assertEqual(once, score)
        self.assertEqual(sum(report.values()), 0)
        self.assertEqual([n.midi_pitch for n in once.notes], MIDIS)
        self.assertEqual(denoise_score(once)[0], once)

    def test_midi_import_jitter_and_stray_notes(self):
        """A MIDI file with a jittery double stop and a stray out-of-range blip becomes a clean score."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_midi(d / "s.mid", [(0, 1, 55), (0.003, 1, 62), (1, 2, 69), (1.5, 1.504, 30)])   # G3+D4 chord, A4, blip
            perf = preprocess_score(Performance(), d / "s.mid", d / "out")
            raw = parse_musicxml(convert_to_musicxml(d / "s.mid", d / "out2"))
        self.assertEqual([n.midi_pitch for n in perf.score.notes], [55, 62, 69])
        self.assertEqual(len({n.onset for n in perf.score.notes[:2]}), 1)
        self.assertGreater(len(raw.notes), len(perf.score.notes))
        self.assertEqual(perf.preprocessing_report["score"]["out_of_range"], 1)

    def test_preprocess_score_can_skip_denoising(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_midi(d / "s.mid", [(0, 1, 69), (1, 2, 30)])
            perf = preprocess_score(Performance(), d / "s.mid", d / "out", denoise=False)
        self.assertEqual(len(perf.score.notes), 2)
        self.assertNotIn("score", perf.preprocessing_report)


if __name__ == "__main__":
    unittest.main()
