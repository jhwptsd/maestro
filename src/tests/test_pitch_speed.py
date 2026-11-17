"""Guards for the speed optimisations in the pitch path: each fast routine must give the same answer as a
plain reference implementation (no timing assertions: those would be flaky)."""
import unittest

import numpy as np

from helpers import SR, synth
from maestro.audio.backends import yin
from maestro.alignment.dtw import dtw
from maestro.audio.pitch import estimate_pitch
from maestro.audio.transcription import _cut_by_pitch
from maestro.utils.audio_math import frame_rms, frame_signal


def pick_reference(row, tau_min, threshold):
    """The original one-row-at-a-time YIN period choice."""
    seg = row[tau_min:]
    below = np.flatnonzero(seg < threshold)
    if below.size:
        k = below[0]
        while k + 1 < len(seg) and seg[k + 1] < seg[k]:
            k += 1
    else:
        k = int(np.argmin(seg))
    tau = k + tau_min
    confidence = 1.0 - row[tau]
    if 0 < tau < len(row) - 1:
        a, b, c = row[tau - 1], row[tau], row[tau + 1]
        den = a - 2 * b + c
        if den > 0:
            tau += float(np.clip(0.5 * (a - c) / den, -1, 1))
    return tau, float(np.clip(confidence, 0, 1))


def cut_reference(sm, max_dev, persist):
    """The original _cut_by_pitch (re-computes the median of the whole note at every frame)."""
    segs, n, i = [], len(sm), 0
    while i < n:
        if np.isnan(sm[i]):
            i += 1
            continue
        start, j = i, i + 1
        while j < n and not np.isnan(sm[j]):
            centre = np.median(sm[start:j])
            ahead = sm[j:j + persist]
            if len(ahead) == persist and not np.isnan(ahead).any() and np.all(np.abs(ahead - centre) > max_dev):
                break
            j += 1
        segs.append((start, j))
        i = j
    return segs


def dtw_reference(cost, open_begin, open_end):
    """The textbook cell-by-cell DTW that the anti-diagonal version replaced."""
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    if open_begin:
        acc[:, 0] = 1e-6 * np.arange(n + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    i = int(np.flatnonzero(acc[1:, m] <= acc[1:, m].min() + 1e-9)[-1]) + 1 if open_end else n
    j, path, total = m, [], float(acc[i, m])
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    return path[::-1], total


class VectorisedDTW(unittest.TestCase):
    def test_matches_the_cell_by_cell_reference_including_ties_and_open_ends(self):
        rng = np.random.default_rng(0)
        for shape in ((1, 1), (1, 7), (7, 1), (13, 40), (40, 13), (50, 50)):
            for cost in (rng.random(shape), rng.integers(0, 3, shape).astype(float)):   # continuous, and full of ties
                for open_begin in (False, True):
                    for open_end in (False, True):
                        self.assertEqual(dtw(cost, open_begin, open_end), dtw_reference(cost, open_begin, open_end))


class VectorisedPick(unittest.TestCase):
    def test_matches_the_per_row_reference_on_random_curves(self):
        rng = np.random.default_rng(0)
        for width, tau_min in ((124, 7), (60, 2), (300, 10)):
            # smooth-ish curves with several dips (some under the threshold, some not), plus flat and monotone rows
            rows = np.clip(np.abs(np.cumsum(rng.normal(0, 0.12, (400, width)), axis=1)) % 1.6, 0, 1.6)
            rows[0], rows[1], rows[2] = 1.0, np.linspace(1, 0, width), np.linspace(0, 1, width)
            rows[3] = 0.05                                           # below threshold everywhere
            tau, conf = yin._pick(rows, tau_min, 0.15)
            for r, row in enumerate(rows):
                t_ref, c_ref = pick_reference(row, tau_min, 0.15)
                self.assertAlmostEqual(tau[r], t_ref, places=9, msg=f"row {r}")
                self.assertAlmostEqual(conf[r], c_ref, places=12)


class FrameHelpers(unittest.TestCase):
    def test_frame_signal_is_a_view_with_the_old_values(self):
        y = np.random.default_rng(1).standard_normal(5000)
        for w, hop in ((1024, 256), (2048, 512), (1500, 300)):
            padded = np.pad(y, w // 2)
            n = 1 + (len(padded) - w) // hop
            old = padded[np.arange(w)[None, :] + hop * np.arange(n)[:, None]]
            new = frame_signal(y, w, hop)
            np.testing.assert_array_equal(new, old)
            self.assertFalse(new.flags.writeable)                    # read-only strided view, not a copy
            self.assertFalse(new.flags.owndata)

    def test_frame_rms_matches_the_direct_formula(self):
        y = np.random.default_rng(2).standard_normal(20000) * np.linspace(0, 1, 20000)
        direct = np.sqrt(np.mean(np.square(frame_signal(y, 1024, 256)), axis=1))
        np.testing.assert_allclose(frame_rms(y, 1024, 256), direct, rtol=1e-12)


class Segmentation(unittest.TestCase):
    def test_cut_by_pitch_matches_the_reference(self):
        rng = np.random.default_rng(3)
        walk = np.cumsum(rng.choice([0, 0, 0, 1, -1, 2], size=3000) * 1.0) % 30 + 60
        walk[rng.random(3000) < 0.05] = np.nan
        vibrato = 69 + 0.4 * np.sin(np.arange(2000) / 9.0)
        notes = np.repeat([69.0, 71.0, np.nan, 72.0, 72.1], 150)
        for contour in (walk, vibrato, notes, np.full(10, np.nan), np.array([])):
            for max_dev, persist in ((0.6, 3), (0.3, 2), (1.0, 5)):
                self.assertEqual(_cut_by_pitch(contour, max_dev, persist), cut_reference(contour, max_dev, persist))


class EndToEnd(unittest.TestCase):
    def test_yin_finds_the_same_pitch_as_before_on_a_note_sequence(self):
        audio = synth([(69, 0.1, 0.6), (74, 0.8, 0.6), (81, 1.5, 0.6)])
        track = estimate_pitch(audio, SR, backend="yin")
        for midi, t in ((69, 0.4), (74, 1.1), (81, 1.8)):
            i = int(t * SR / 256)
            self.assertAlmostEqual(1200 * np.log2(track.f0[i] / (440 * 2 ** ((midi - 69) / 12))), 0, delta=3)
            self.assertGreater(track.confidence[i], 0.9)

    def test_chunk_boundaries_do_not_change_results(self):
        audio = np.tile(synth([(69, 0.0, 0.5), (76, 0.5, 0.5)]), 12)              # > 2 chunks of frames
        a = yin.estimate(audio, SR)
        old_chunk, yin.CHUNK = yin.CHUNK, 37                                       # odd chunk size
        try:
            b = yin.estimate(audio, SR)
        finally:
            yin.CHUNK = old_chunk
        np.testing.assert_allclose(a.f0, b.f0, atol=1e-9)
        np.testing.assert_allclose(a.confidence, b.confidence, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
