"""CREPE backend: deterministic sub-bin decoding, accuracy, automatic model choice, resampling.
Skipped when PyTorch/torchcrepe are not installed (pip install -e '.[crepe]')."""
import importlib.util
import unittest
from unittest import mock

import numpy as np

from helpers import synth

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
        bins, freq = self.crepe._decode(self.probs(centres))
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
        from maestro.audio.pitch import estimate_pitch
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
        seen = []

        def fake_predict(audio, sr, hop_length=None, fmin=None, fmax=None, model=None, decoder=None, batch_size=None,
                         device=None, return_periodicity=False, **_):
            seen.append(model)
            n = 1 + audio.shape[1] // hop_length
            return torch.full((1, n), 440.0), torch.full((1, n), 0.9)

        with mock.patch.object(torchcrepe, "predict", fake_predict):
            self.track()                                                      # CPU
            self.track(device="cpu", model="full")                            # explicit choice wins
            with mock.patch.object(torch.cuda, "is_available", return_value=True):
                self.track()                                                  # GPU present -> full
        self.assertEqual(seen, ["tiny", "full", "full"])
        with self.assertRaises(ValueError):
            self.track(model="huge")

    def test_resampling_keeps_pitch_and_timeline_for_any_sample_rate(self):
        from maestro.audio.backends.crepe import _resample
        t = np.arange(SR) / SR
        tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        for sr in (22050, 44100, 48000):
            x = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
            y16 = _resample(x, sr, 16000)
            self.assertAlmostEqual(len(y16), 16000, delta=2)
            freqs = np.fft.rfftfreq(len(y16), 1 / 16000)
            self.assertAlmostEqual(freqs[np.argmax(np.abs(np.fft.rfft(y16)))], 440, delta=1.5)
        from maestro.audio.backends.base import frame_times
        for sr in (16000, 44100):
            z = np.interp(np.arange(int(len(self.audio) * sr / SR)) * SR / sr, np.arange(len(self.audio)), self.audio).astype(np.float32)
            from maestro.audio.pitch import estimate_pitch
            tr = estimate_pitch(z, sr, backend="crepe")
            np.testing.assert_allclose(tr.times, frame_times(len(z), sr, 256))   # shared timeline with every backend
            self.assertEqual(len(tr.f0), len(tr.times))
        _ = tone


if __name__ == "__main__":
    unittest.main()
