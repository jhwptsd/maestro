"""Fusion encoders: the residual-autoencoder idea itself, the three models, their feature builders, the embedding
pipeline and the training code. Tiny models and synthetic data, so it runs in seconds."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:                                           # the fusion package is optional (pip install -e ".[fusion]")
    raise unittest.SkipTest("needs torch")

from helpers import make_video, musicxml, synth, write_wav
from minuet import minuet_musicxml
from pose_fixture import minuet_notes, synth_pose
from test_assessment_events import scene
from maestro import analyze
from maestro.audio.features import extract_features
from maestro.fusion import audio_encoder, score_encoder, video_encoder
from maestro.fusion.embed import available, embed_performance
from maestro.fusion.residual_autoencoder import (ResidualAutoencoder, cross_covariance_penalty, leakage_r2, load_checkpoint,
                                                 save_checkpoint)
from maestro.score.musicxml import parse_musicxml
from maestro.vision.motion_features import extract_motion
from maestro.vision.pose_io import Pose, save_pose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import fusion_data  # noqa: E402
import train_fusion  # noqa: E402

SR = 22050
BUILDERS = {"audio": audio_encoder.build, "video": video_encoder.build, "score": score_encoder.build}


def score_of(xml):
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "s.musicxml").write_text(xml)
        return parse_musicxml(Path(d) / "s.musicxml")


class TheIdea(unittest.TestCase):
    """The decoder sees the analysis features, so z is left with what they do not explain."""

    def train(self, conditional, leak):
        torch.manual_seed(0)
        n = 2000
        a, h = torch.randn(n, 4), torch.randn(n, 2)              # a: known to the analysis, h: hidden factors
        x = a @ torch.randn(4, 20) + h @ torch.randn(2, 20) + 0.05 * torch.randn(n, 20)

        class Enc(nn.Module):
            def __init__(s):
                super().__init__()
                s.net = nn.Sequential(nn.Linear(20, 64), nn.GELU(), nn.Linear(64, 12))

            def forward(s, x):
                mu, logvar = s.net(x).chunk(2, 1)
                return mu, logvar.clamp(-10, 4)

        class Dec(nn.Module):
            def __init__(s):
                super().__init__()
                s.net = nn.Sequential(nn.Linear(6 + 4, 64), nn.GELU(), nn.Linear(64, 20))

            def forward(s, z, side):
                return s.net(torch.cat([z, side if conditional else torch.zeros_like(side)], 1))

        model = ResidualAutoencoder(Enc(), Dec(), 6, "t", {})
        opt = torch.optim.Adam(model.parameters(), 1e-2)
        for _ in range(600):
            idx = torch.randint(0, n, (256,))
            loss, _ = model.loss(x[idx], a[idx], lambda p, t: ((p - t) ** 2).mean(), leak=leak if conditional else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            z = model.eval().encode(x)
        return leakage_r2(z, a), leakage_r2(z, h)

    def test_conditioning_moves_the_known_information_out_of_z_and_keeps_the_hidden_one(self):
        plain_known, _ = self.train(conditional=False, leak=0.0)
        known, hidden = self.train(conditional=True, leak=1.0)
        self.assertGreater(plain_known, 0.9)                     # an ordinary autoencoder encodes everything, known or not
        self.assertLess(known, 0.3)                              # the residual one barely repeats what the analysis has
        self.assertGreater(hidden, 0.9)                          # ... and still holds the factors the analysis lacks

    def test_the_measures(self):
        torch.manual_seed(1)
        a = torch.randn(400, 3)
        self.assertGreater(leakage_r2(a @ torch.randn(3, 5), a), 0.95)                  # z is a function of a
        self.assertLess(leakage_r2(torch.randn(400, 5), a), 0.1)                        # z is unrelated to a
        self.assertGreater(float(cross_covariance_penalty(a[:, :2], a)), 0.05)
        self.assertLess(float(cross_covariance_penalty(torch.randn(400, 2), a)), 0.01)
        self.assertEqual(float(cross_covariance_penalty(a, torch.ones(400, 2))), 0.0)    # constant side features: nothing to remove


class Audio(unittest.TestCase):
    def test_mel_puts_a_tone_in_the_right_band_and_windows_cover_the_audio(self):
        t = np.arange(SR * 3) / SR
        mel = audio_encoder.log_mel(0.3 * np.sin(2 * np.pi * 880 * t), SR)
        self.assertEqual(mel.shape[1], audio_encoder.N_MELS)
        centres = audio_encoder.mel_filterbank(SR).argmax(1)
        self.assertTrue(abs(np.freqs if False else np.fft.rfftfreq(audio_encoder.N_FFT, 1 / SR)[centres[mel.mean(0).argmax()]] - 880) < 150)
        self.assertTrue(-1.01 <= mel.min() and mel.max() <= 1.01)
        x, side, starts = audio_encoder.make_windows(mel, np.zeros((len(mel), audio_encoder.SIDE_DIM), np.float32))
        self.assertEqual((x.shape[1:], side.shape[1:]), ((audio_encoder.WINDOW, 64), (audio_encoder.WINDOW, audio_encoder.SIDE_DIM)))
        self.assertEqual(starts[1] - starts[0], audio_encoder.STRIDE)
        short, _, s2 = audio_encoder.make_windows(mel[:20], np.zeros((20, 6), np.float32))
        self.assertEqual((len(short), s2.tolist()), (1, [0]))          # a short clip is padded to one window

    def test_side_features_come_from_the_analysis_rows(self):
        perf = scene([{"midi": 69 + i % 3} for i in range(8)])
        y, sr = perf.audio["samples"], perf.audio["sr"]
        side = audio_encoder.side_features(extract_features(y, sr, backend="yin"))
        self.assertEqual(side.shape[1], audio_encoder.SIDE_DIM)
        self.assertTrue(np.isfinite(side).all() and side.min() >= -0.01 and side.max() <= 1.01)
        self.assertGreater(side[:, 0].mean(), 0.5)                      # mostly voiced

    def test_model_shapes_and_it_learns(self):
        torch.manual_seed(0)
        model = audio_encoder.build(z_dim=8, width=32)
        x = torch.randn(6, audio_encoder.WINDOW, 64).clamp(-1, 1) * 0.5
        side = torch.rand(6, audio_encoder.WINDOW, audio_encoder.SIDE_DIM)
        self.assertEqual(model(x, side)[0].shape, x.shape)
        self.assertEqual(model.encode(x).shape, (6, 8))
        opt = torch.optim.Adam(model.parameters(), 3e-3)
        first = last = None
        for _ in range(40):
            loss, parts = model.loss(x, side, audio_encoder.reconstruction_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first, last = first or parts["recon"], parts["recon"]
        self.assertLess(last, 0.6 * first)


class Video(unittest.TestCase):
    pose = synth_pose(minuet_notes(), with_bow=True)

    def windows(self, pose):
        return video_encoder.pose_windows(pose, extract_motion(pose))

    def test_windows_ignore_where_the_player_stands_and_how_big_the_picture_is(self):
        x, side, t = self.windows(self.pose)
        self.assertEqual(x.shape[1:], (video_encoder.WINDOW, video_encoder.N_IN))
        self.assertEqual(side.shape[1:], (video_encoder.WINDOW, video_encoder.SIDE_DIM))
        moved = Pose(self.pose.fps, self.pose.joints, self.pose.xyz * 2.0 + np.array([500.0, -300.0, 900.0]))
        x2, side2, _ = self.windows(moved)                              # twice as large, somewhere else
        np.testing.assert_allclose(x2, x, atol=1e-4)
        np.testing.assert_allclose(side2, side, atol=1e-3)
        self.assertEqual(t[1] - t[0], video_encoder.STRIDE / video_encoder.FPS)

    def test_other_frame_rates_are_resampled(self):
        slow = Pose(15.0, self.pose.joints, self.pose.xyz[::2])
        x, _, _ = self.windows(slow)
        self.assertEqual(x.shape[1], video_encoder.WINDOW)

    def test_model_shapes_and_it_learns(self):
        torch.manual_seed(0)
        x, side, _ = self.windows(self.pose)
        x, side = torch.as_tensor(x[:8]), torch.as_tensor(side[:8])
        model = video_encoder.build(z_dim=8, width=32)
        self.assertEqual(model(x, side)[0].shape, x.shape)
        opt = torch.optim.Adam(model.parameters(), 3e-3)
        first = last = None
        for _ in range(40):
            loss, parts = model.loss(x, side, video_encoder.reconstruction_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first, last = first or parts["recon"], parts["recon"]
        self.assertLess(last, 0.6 * first)


class Score(unittest.TestCase):
    def test_tokens_and_side_summary(self):
        notes = [{"midi_pitch": 74, "duration": 1.0, "beat": 2.0, "articulation": "staccato", "dynamic": "mf", "hairpin": None,
                  "onset": 0.0, "offset": 0.5},
                 {"midi_pitch": 60, "duration": 0.5, "beat": 3.5, "articulation": None, "dynamic": None, "hairpin": "crescendo",
                  "onset": 0.5, "offset": 0.8}]
        tokens = score_encoder.tokenize(notes, (3, 4))
        self.assertEqual(tokens.tolist(), [[31, 6, 8, 1, 5, 0], [17, 4, 20, 0, 0, 1]])
        side = score_encoder.window_side(notes, 1, (3, 4))
        self.assertEqual(side.shape, (score_encoder.SIDE_DIM,))
        self.assertAlmostEqual(side[:12].sum(), 1.0, places=5)           # pitch-class profile
        self.assertAlmostEqual(side[19], 0.5)                            # half the notes articulated

    def test_windows_pad_short_scores_and_the_loss_ignores_padding(self):
        score = score_of(minuet_musicxml())
        notes = [n.to_dict() for n in score.notes]
        x, side, starts = score_encoder.make_windows(notes, score.key_fifths, score.time_signature)
        self.assertEqual(x.shape[1:], (score_encoder.WINDOW, 6))
        self.assertEqual(starts[-1] + score_encoder.WINDOW, len(notes))   # the end of the score is covered
        short, _, _ = score_encoder.make_windows(notes[:5], 0, (3, 4))
        self.assertEqual(len(short), 1)
        self.assertTrue((short[0, 5:] == np.array(score_encoder.SIZES)).all())      # padding after the 5 notes
        tokens = torch.as_tensor(short)
        perfect = tuple(torch.nn.functional.one_hot(tokens[..., k].clamp(max=n - 1), n).float() * 50 for k, n in enumerate(score_encoder.SIZES))
        self.assertLess(float(score_encoder.reconstruction_loss(perfect, tokens)), 1e-3)
        wrong = tuple(l.roll(1, -1) for l in perfect)
        self.assertGreater(float(score_encoder.reconstruction_loss(wrong, tokens)), 10)

    def test_model_shapes_and_it_learns(self):
        torch.manual_seed(0)
        score = score_of(minuet_musicxml())
        x, side, _ = score_encoder.make_windows([n.to_dict() for n in score.notes], score.key_fifths, score.time_signature)
        x, side = torch.as_tensor(x), torch.as_tensor(side)
        model = score_encoder.build(z_dim=8, hidden=48)
        self.assertEqual([l.shape for l in model(x, side)[0]][0], (len(x), score_encoder.WINDOW, score_encoder.N_PITCH))
        opt = torch.optim.Adam(model.parameters(), 5e-3)
        first = last = None
        for _ in range(60):
            loss, parts = model.loss(x, side, score_encoder.reconstruction_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first, last = first or parts["recon"], parts["recon"]
        self.assertLess(last, 0.5 * first)


class Checkpoints(unittest.TestCase):
    def test_round_trip_and_nothing_executable_in_a_checkpoint(self):
        model = score_encoder.build(z_dim=8, hidden=32).eval()
        x = torch.randint(0, 5, (2, score_encoder.WINDOW, 6))
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(model, Path(d) / "score.pt", {"val_recon": 1.5})
            loaded, stats = load_checkpoint(Path(d) / "score.pt", BUILDERS)
            self.assertEqual(stats, {"val_recon": 1.5})
            np.testing.assert_allclose(loaded.encode(x).numpy(), model.encode(x).numpy(), atol=1e-6)
            torch.save({"kind": "score", "config": {}, "state": mock.Mock()}, Path(d) / "evil.pt")      # a pickled object
            with self.assertRaises(Exception):
                load_checkpoint(Path(d) / "evil.pt", BUILDERS)                                         # refused, not loaded


def tiny_checkpoints(directory, kinds=("audio", "video", "score")):
    tiny = {"audio": dict(z_dim=8, width=16), "video": dict(z_dim=8, width=16), "score": dict(z_dim=8, hidden=16)}
    for kind in kinds:
        save_checkpoint(BUILDERS[kind](**tiny[kind]), Path(directory) / f"{kind}.pt")


class Embedding(unittest.TestCase):
    def full_performance(self):
        perf = scene([{"midi": 69 + i % 3, "dur": 0.4} for i in range(60)], spacing=0.5)
        y, sr = perf.audio["samples"], perf.audio["sr"]
        perf.audio_features["frames"] = extract_features(y, sr, backend="yin")
        score = score_of(minuet_musicxml())
        perf.score, perf.score_features = score, [n.to_dict() | {"hz": 440.0} for n in score.notes]
        perf.alignment = [{"score_idx": i, "perf_idx": i} for i in range(min(len(score.notes), 60))]
        perf.audio_features["notes"] = perf.audio_features["notes"][:len(perf.alignment)]
        perf.pose = synth_pose(minuet_notes(), with_bow=True)
        perf.motion_features = extract_motion(perf.pose)
        return perf

    def test_each_modality_gets_one_json_vector_per_window(self):
        perf = self.full_performance()
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(available(d), [])
            tiny_checkpoints(d)
            self.assertEqual(available(d), ["audio", "video", "score"])
            out = embed_performance(perf, d)
        self.assertEqual(set(out), {"audio", "video", "score"})
        self.assertIs(perf.embeddings, out)
        for kind, e in out.items():
            self.assertEqual(len(e["t"]), len(e["z"]), kind)
            self.assertEqual(len(e["recon_error"]), len(e["z"]), kind)
            self.assertTrue(e["z"] and len(e["z"][0]) == 8, kind)
        self.assertEqual(out["score"]["notes"][0][0], 0)
        self.assertEqual(json.loads(json.dumps(out)), out)                           # plain JSON
        self.assertEqual(out["video"]["t"][1] - out["video"]["t"][0], 1.0)

    def test_modalities_without_input_or_checkpoint_are_left_out(self):
        perf = self.full_performance()
        perf.pose = None
        with tempfile.TemporaryDirectory() as d:
            tiny_checkpoints(d, ("audio", "video"))
            self.assertEqual(set(embed_performance(perf, d)), {"audio"})            # no pose; no score checkpoint


class DataAndTraining(unittest.TestCase):
    def test_loaders_read_poses_scores_and_audio_and_split_by_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            pose = synth_pose(minuet_notes(), with_bow=True)
            save_pose(pose, d / "p.json")
            np.save(d / "p.npy", pose.xyz[:, :18])                                  # a plain mm array in VIOPOSE_JOINTS order
            (d / "s.musicxml").write_text(minuet_musicxml())
            write_wav(d / "a.wav", synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate([69, 71, 72, 74, 76, 74, 72, 71])]))
            (d / "bad.musicxml").write_text("not xml")
            self.assertEqual({p.name for p in fusion_data.find(d, "video")}, {"p.json", "p.npy"})
            xj, _ = fusion_data.pose_file_windows(d / "p.json")
            xn, _ = fusion_data.pose_file_windows(d / "p.npy", fps=pose.fps)
            np.testing.assert_allclose(xj, xn, atol=1e-4)
            tokens, side = fusion_data.score_file_windows(d / "s.musicxml")
            self.assertEqual((tokens.shape[1:], side.shape[1]), ((score_encoder.WINDOW, 6), score_encoder.SIDE_DIM))
            x, side = fusion_data.audio_file_windows(d / "a.wav", cache_dir=d / "cache")
            self.assertEqual(len(list((d / "cache").glob("*.npz"))), 1)
            x2, _ = fusion_data.audio_file_windows(d / "a.wav", cache_dir=d / "cache")          # served from the cache
            np.testing.assert_array_equal(x, x2)
            messages = []
            xs, sides = fusion_data.load_windows("score", [d / "s.musicxml", d / "bad.musicxml"], log=messages.append)
            self.assertEqual(len(xs), 1)
            self.assertTrue(any("bad.musicxml" in m for m in messages))               # skipped, not fatal
        files = [np.full((n, 2), i) for i, n in enumerate((5, 5, 5, 5))]
        (xt, _), (xv, _) = fusion_data.split(files, files, 0.25)
        self.assertEqual(len(set(xt[:, 0]) & set(xv[:, 0])), 0)                      # no file on both sides

    def test_training_keeps_the_best_epoch_and_the_cli_writes_a_checkpoint(self):
        score = score_of(minuet_musicxml())
        x, side, _ = score_encoder.make_windows([n.to_dict() for n in score.notes], score.key_fifths, score.time_signature)
        model = score_encoder.build(z_dim=4, hidden=16)
        history = train_fusion.fit(model, (x, side), (x, side), score_encoder.reconstruction_loss, epochs=6, batch=4, lr=5e-3,
                                   log=lambda m: None)
        self.assertEqual(len(history), 6)
        self.assertLess(history[-1]["val_recon"], history[0]["val_recon"])
        self.assertTrue(any(h.get("best") for h in history) and "leakage" in history[0])
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for i in range(3):
                (d / f"s{i}.musicxml").write_text(minuet_musicxml())
            train_fusion.main(["score", str(d), "--epochs", "2", "--z-dim", "4", "--out", str(d / "ck"), "--device", "cpu"])
            model, stats = load_checkpoint(d / "ck" / "score.pt", BUILDERS)
            self.assertEqual(model.z_dim, 4)
            self.assertIn("val_recon", stats)


class InAnalysis(unittest.TestCase):
    def test_analyze_embeds_when_checkpoints_exist_and_can_be_told_not_to(self):
        midis = [69, 71, 72, 74, 76, 74, 72, 71]
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "s.musicxml").write_text(musicxml(midis))
            write_wav(d / "a.wav", synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(midis)]))
            make_video(d / "v.mp4", d / "a.wav", 5)
            (d / "models" / "fusion").mkdir(parents=True)
            tiny_checkpoints(d / "models" / "fusion", ("audio", "score"))
            with mock.patch.dict(os.environ, {"MAESTRO_MODELS": str(d / "models")}):
                perf = analyze(d / "v.mp4", d / "s.musicxml", pitch_backend="yin")
                self.assertEqual(set(perf.embeddings), {"audio", "score"})         # no pose here: no video embedding
                self.assertIsNone(analyze(d / "v.mp4", d / "s.musicxml", pitch_backend="yin", embeddings=False).embeddings)
            with mock.patch.dict(os.environ, {"MAESTRO_MODELS": str(d / "none")}):
                self.assertEqual(analyze(d / "v.mp4", d / "s.musicxml", pitch_backend="yin").embeddings, None)    # nothing trained
                with self.assertRaises(FileNotFoundError):
                    analyze(d / "v.mp4", d / "s.musicxml", pitch_backend="yin", embeddings=True)


if __name__ == "__main__":
    unittest.main()
