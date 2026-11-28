"""CREPE backend: deterministic sub-bin decoding, accuracy, automatic model choice, resampling.
Skipped when PyTorch/torchcrepe are not installed (pip install -e '.[crepe]')."""
import importlib.util
import unittest
from unittest import mock

import numpy as np

from helpers import synth
from maestro.audio.backends.base import frame_times
from maestro.audio.pitch import estimate_pitch

HAVE_CREPE = all(importlib.util.find_spec(m) for m in ("torch", "torchcrepe", "librosa", "scipy"))
SR = 22050


@unittest.skipUnless(HAVE_CREPE, "needs torch + torchcrepe")
class Decoder(unittest.TestCase):
    def setUp(self):
        import torch
        from maestro.audio.backends import crepe
        self.torch, self.crepe = torch, crepe

    def probs(self, centres, frames=40, width=1.5):
        """(1, 360, frames) network-like output: a Gaussian bump per frame at a fractional bin."""
        bins = np.arange(360)[:, None]
        centres = np.broadcast_to(np.asarray(centres, dtype=float), (frames,))[None, :]
        return self.torch.from_numpy(np.exp(-0.5 * ((bins - centres) / width) ** 2)[None]).float()

    def test_refines_below_the_20_cent_bin_grid_without_randomness(self):
        import torchcrepe
        for centre in (100.0, 100.37, 100.5, 217.81):
            bins, freq = self.crepe._decode(self.probs(centre))
            cents = torchcrepe.CENTS_PER_BIN * centre + 1997.3794084376191
            expected = 10 * 2 ** (cents / 1200)
            self.assertLess(abs(1200 * np.log2(freq.numpy() / expected)).max(), 1.0, centre)   # < 1 cent
            self.assertEqual(int(bins[0, 0]), int(round(centre)) if centre % 1 != 0.5 else int(bins[0, 0]))
        a, b = self.crepe._decode(self.probs(100.37)), self.crepe._decode(self.probs(100.37))
        self.assertTrue(self.torch.equal(a[1], b[1]))                          # same input, same output, always

    def test_follows_wide_leaps_after_a_gap(self):
        """torchcrepe's softmax-emission Viterbi parks the path after a ~22 semitone leap and loses the next note."""
        bins = np.arange(360)[:, None]
        centres = np.concatenate([np.full(25, 250.0), np.full(9, np.nan), np.full(25, 130.0)])   # note, gap, note
        bump = 0.9 * np.exp(-0.5 * ((bins - np.nan_to_num(centres, nan=-999)[None]) / 1.5) ** 2) + 0.002
        picked, _ = self.crepe._decode(self.torch.from_numpy(bump[None]).float())
        self.assertAlmostEqual(float(picked[0, -10:].float().mean()), 130, delta=1.0)         # the second note is found
        self.assertAlmostEqual(float(picked[0, :20].float().mean()), 250, delta=1.0)

    def test_smooths_with_viterbi_and_survives_masking_and_edges(self):
        centres = np.full(40, 120.0)
        centres[17] = 300.0                                                   # a one-frame octave-error spike
        bins, _ = self.crepe._decode(self.probs(centres))
        self.assertEqual(int(bins[0, 17]), 120)                               # the path ignores the spike
        masked = self.probs(100.0).clone()
        masked[:, :60] = -float("inf")                                        # fmin/fmax masking
        masked[:, 200:] = -float("inf")
        self.assertTrue(np.isfinite(self.crepe._decode(masked)[1].numpy()).all())
        for edge in (0.0, 359.0):                                             # window clipped at the range ends
            self.assertTrue(np.isfinite(self.crepe._decode(self.probs(edge))[1].numpy()).all())


@unittest.skipUnless(HAVE_CREPE, "needs torch + torchcrepe")
class Backend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.notes = [(float(rng.integers(55, 88)) + rng.uniform(-0.45, 0.45), 0.2 + 0.6 * k, 0.5) for k in range(12)]
        cls.audio = synth(cls.notes)

    def track(self, **kw):
        return estimate_pitch(self.audio, SR, backend="crepe", **kw)

    def test_deterministic_and_accurate_per_note(self):
        a, b = self.track(), self.track()
        np.testing.assert_array_equal(a.f0, b.f0)                              # stock torchcrepe dithers: it did not
        errors, jitter = [], []
        for midi, start, dur in self.notes:
            sel = (a.times > start + 0.12) & (a.times < start + dur - 0.08) & (a.confidence > 0.5)
            cents = 1200 * np.log2(a.f0[sel] / (440 * 2 ** ((midi - 69) / 12)))
            errors.append(np.median(cents))
            jitter.append(cents.std())
        self.assertLess(np.max(np.abs(errors)), 25.0)                          # no note lost (stock decoder: 4 of 12 wrong)
        self.assertLess(np.mean(np.abs(errors)), 5.0)                          # was ~7 cents with the stock decoder
        self.assertLess(np.mean(jitter), 2.0)                                  # was ~8 cents of random dither

    def test_auto_model_is_tiny_on_cpu_and_full_on_a_gpu(self):
        import torch
        import torchcrepe
        from maestro.audio.backends import crepe_onnx
        seen = []

        def fake_predict(audio, sr, hop_length=None, fmin=None, fmax=None, model=None, decoder=None, batch_size=None,
                         device=None, return_periodicity=False, **_):
            seen.append(model)
            n = 1 + audio.shape[1] // hop_length
            return torch.full((1, n), 440.0), torch.full((1, n), 0.9)

        def fake_make_run(model, provider):                                      # the ONNX Runtime path: a network that hears a steady A4 (bin 228)
            seen.append(model)
            return lambda frames: np.eye(360, dtype=np.float32)[np.full(len(frames), 228)]

        with mock.patch.object(torchcrepe, "predict", fake_predict), mock.patch.object(crepe_onnx, "make_run", fake_make_run):
            self.track()                                                      # CPU: ONNX Runtime, tiny
            self.track(device="cpu", model="full")                            # explicit choice wins (PyTorch on the CPU)
            with mock.patch.object(torch.cuda, "is_available", return_value=True):
                self.track()                                                  # GPU present -> full
        self.assertEqual(seen, ["tiny", "full", "full"])
        with self.assertRaises(ValueError):
            self.track(model="huge")

    def test_resampling_keeps_pitch_and_timeline_for_any_sample_rate(self):
        from maestro.audio.backends.crepe import _resample
        for sr in (22050, 44100, 48000):
            x = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
            y16 = _resample(x, sr, 16000)
            self.assertAlmostEqual(len(y16), 16000, delta=2)
            freqs = np.fft.rfftfreq(len(y16), 1 / 16000)
            self.assertAlmostEqual(freqs[np.argmax(np.abs(np.fft.rfft(y16)))], 440, delta=1.5)
        for sr in (16000, 44100):
            z = np.interp(np.arange(int(len(self.audio) * sr / SR)) * SR / sr, np.arange(len(self.audio)), self.audio).astype(np.float32)
            tr = estimate_pitch(z, sr, backend="crepe")
            np.testing.assert_allclose(tr.times, frame_times(len(z), sr, 256))   # shared timeline with every backend
            self.assertEqual(len(tr.f0), len(tr.times))


class WithoutPyTorch(unittest.TestCase):
    """crepe_numpy.py: framing, Viterbi and decoding in numpy, the network through ONNX Runtime: the same numbers as torchcrepe, with no PyTorch or librosa."""

    def tone(self, seconds=2.0, sr=16000):
        t = np.arange(int(seconds * sr)) / sr
        return (0.4 * np.sin(2 * np.pi * (440 * t + 3 * np.sin(2 * np.pi * 5 * t) / (2 * np.pi) * 5))).astype(np.float32)

    def test_the_frames_are_torchcrepes(self):
        import torch
        import torchcrepe
        from maestro.audio.backends import crepe_numpy
        audio = self.tone(1.5)
        theirs = next(torchcrepe.core.preprocess(torch.from_numpy(audio)[None], 16000, hop_length=160, batch_size=None, device="cpu")).numpy()
        ours = crepe_numpy.frames(audio, 160)
        self.assertEqual(ours.shape, theirs.shape)
        self.assertLess(np.abs(ours - theirs).max(), 1e-3)
        self.assertEqual(len(crepe_numpy.frames(audio, 160, 10, 20)), 10)                  # a slice of the frames
        self.assertEqual((torchcrepe.SAMPLE_RATE, torchcrepe.WINDOW_SIZE, torchcrepe.PITCH_BINS, torchcrepe.MAX_FMAX, torchcrepe.CENTS_PER_BIN),
                         (crepe_numpy.SAMPLE_RATE, crepe_numpy.WINDOW, crepe_numpy.PITCH_BINS, crepe_numpy.MAX_FMAX, crepe_numpy.CENTS_PER_BIN))

    def test_the_viterbi_path_is_librosas(self):
        import librosa
        from maestro.audio.backends import crepe_numpy
        rng = np.random.default_rng(3)
        emission = rng.random((360, 120)) ** 6 + 1e-6
        emission /= emission.sum(axis=0, keepdims=True)
        self.assertTrue(np.array_equal(crepe_numpy.viterbi(emission, crepe_numpy.transition_matrix()),
                                       librosa.sequence.viterbi(emission, crepe_numpy.transition_matrix())))

    def test_decoding_matches_the_torch_decoder(self):
        import torch
        from maestro.audio.backends import crepe, crepe_numpy
        rng = np.random.default_rng(5)
        probs = (rng.random((90, 360)) ** 8).astype(np.float32)
        for k in range(90):
            probs[k, 200 + k % 40] = 0.9                                                    # a pitch that moves
        low, high = crepe_numpy.bin_range(175.0, 2000.0)
        masked = torch.from_numpy(probs.T[None].copy())
        masked[:, :low], masked[:, high:] = -float("inf"), -float("inf")
        bins, freq = crepe._decode(masked)
        ours_bins, ours_freq, periodicity = crepe_numpy.decode(probs, low, high)
        self.assertTrue(np.array_equal(ours_bins, bins[0].numpy()))
        self.assertLess(np.abs(1200 * np.log2(ours_freq / freq[0].numpy())).max(), 1e-3)     # cents
        self.assertTrue(np.allclose(periodicity, probs[np.arange(90), ours_bins]))

    def test_the_whole_prediction_matches_torchcrepe_with_the_same_network(self):
        import torch
        import torchcrepe
        from maestro.audio.backends import crepe, crepe_numpy, crepe_onnx
        audio = self.tone(2.0)
        run = crepe_onnx.make_run("tiny", "CPUExecutionProvider")
        f0, periodicity = crepe_numpy.predict(audio, 185, 175.0, 2000.0, run)
        with torch.inference_mode():
            pitch, theirs = torchcrepe.predict(torch.from_numpy(audio)[None], 16000, hop_length=185, fmin=175.0, fmax=2000.0, model="tiny", decoder=crepe._decode,
                                               batch_size=512, device="cpu", return_periodicity=True)
        self.assertEqual(len(f0), pitch.shape[1])
        self.assertLess(np.abs(1200 * np.log2(f0 / pitch[0].numpy())).max(), 1.5)            # cents (ONNX Runtime and PyTorch differ in the last digits)
        self.assertLess(np.abs(periodicity - theirs[0].numpy()).max(), 0.05)

    def test_it_runs_with_torch_torchcrepe_and_librosa_unavailable(self):
        from maestro.audio.backends import crepe
        from maestro.utils.phone_sandbox import phone_sandbox
        y = (0.4 * np.sin(2 * np.pi * 440 * np.arange(int(1.5 * 22050)) / 22050)).astype(np.float32)
        with phone_sandbox(("torch", "torchcrepe", "torchaudio", "librosa", "numba")) as box:
            track = crepe.estimate(y, 22050, model="tiny")
        self.assertEqual(box.tried("import"), [])                                            # nothing asked for them
        self.assertTrue(track.runtime.startswith("onnx:"))
        voiced = track.confidence > 0.5
        self.assertGreater(voiced.mean(), 0.8)
        self.assertAlmostEqual(float(np.median(track.f0[voiced])), 440.0, delta=5)

    def test_a_model_shipped_with_the_app_is_found_without_exporting(self):
        import tempfile
        from pathlib import Path
        from maestro.audio.backends import crepe_onnx
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"MAESTRO_MODELS": tmp}):
            shipped = Path(tmp) / "app" / "models"
            shipped.mkdir(parents=True)
            (shipped / "crepe-tiny.onnx").write_bytes(b"x")
            self.assertIsNone(crepe_onnx.find_model("tiny"))                                   # not in the models folder, nothing shipped
            with mock.patch.object(crepe_onnx, "bundled_models", return_value=shipped):
                self.assertEqual(crepe_onnx.find_model("tiny"), shipped / "crepe-tiny.onnx")
                self.assertEqual(crepe_onnx.export("tiny"), shipped / "crepe-tiny.onnx")       # no torch needed to "export" what is already there

    def test_the_installed_app_keeps_its_models_in_its_own_writable_storage(self):
        from maestro.utils.paths import models_root
        with mock.patch.dict("os.environ", {"FLET_APP_STORAGE_DATA": "/data/app", "MAESTRO_MODELS": ""}):
            self.assertEqual(str(models_root()).replace("\\", "/"), "/data/app/models")


if __name__ == "__main__":
    unittest.main()
