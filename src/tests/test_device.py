"""Device choice (any GPU, else CPU) and CREPE through ONNX Runtime."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from maestro.utils import device

try:
    import onnxruntime  # noqa: F401
    import onnx  # noqa: F401
    import torch
    import torchcrepe
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False


class Choose(unittest.TestCase):
    def patched(self, accelerator=None, provider=None, ort=True):
        stack = [mock.patch.object(device, "torch_accelerator", return_value=accelerator),
                 mock.patch.object(device, "onnx_gpu_provider", return_value=provider),
                 mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != device.ENV})]
        for p in stack:
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop(device.ENV, None)
        if not ort:                                          # as if onnxruntime were not installed
            patcher = mock.patch.dict("sys.modules", {"onnxruntime": None})
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_preference_order(self):
        self.patched("cuda:0", "DmlExecutionProvider")
        self.assertEqual(device.choose(), ("torch", "cuda:0"))                    # PyTorch's own accelerator first
        self.patched(None, "DmlExecutionProvider")
        self.assertEqual(device.choose(), ("onnx", "DmlExecutionProvider"))       # an AMD/Intel GPU on Windows
        self.patched(None, None)
        self.assertEqual(device.choose(), ("onnx", "CPUExecutionProvider") if HAVE_ONNX else ("torch", "cpu"))
        self.patched(None, None, ort=False)
        self.assertEqual(device.choose(), ("torch", "cpu"))

    def test_overrides(self):
        self.patched(None, "DmlExecutionProvider")
        self.assertEqual(device.choose("cpu"), ("torch", "cpu"))
        self.assertEqual(device.choose("cuda:1"), ("torch", "cuda:1"))
        self.assertEqual(device.choose("mps"), ("torch", "mps"))
        self.assertEqual(device.choose("onnx"), ("onnx", "DmlExecutionProvider"))
        os.environ[device.ENV] = "cpu"
        self.assertEqual(device.choose(), ("torch", "cpu"))                       # the environment variable
        self.assertEqual(device.choose("mps"), ("torch", "mps"))                  # an argument beats it
        self.patched(None, None)
        self.assertNotEqual(device.choose("directml")[1], "DmlExecutionProvider")  # asked for but absent: something that works

    def test_a_plain_torch_model_ignores_an_onnx_only_gpu(self):
        self.patched(None, "DmlExecutionProvider")
        self.assertEqual(device.torch_device(), "cpu")
        self.patched("cuda:0", None)
        self.assertEqual(device.torch_device(), "cuda:0")
        self.assertEqual(device.torch_device("cpu"), "cpu")
        self.assertTrue(device.is_accelerated("onnx", "DmlExecutionProvider") and not device.is_accelerated("onnx", "CPUExecutionProvider"))
        self.assertIn("auto ->", device.describe())


@unittest.skipUnless(HAVE_ONNX, "needs torch, torchcrepe, onnx and onnxruntime")
class CrepeOnnx(unittest.TestCase):
    def setUp(self):
        from maestro.audio.backends import crepe_onnx
        self.crepe_onnx = crepe_onnx
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.dict(os.environ, {"MAESTRO_MODELS": self._tmp.name})
        patcher.start()
        crepe_onnx._sessions.clear()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(crepe_onnx._sessions.clear)

    def tone(self, hz=440.0, seconds=2.0, sr=22050):
        t = np.arange(int(seconds * sr)) / sr
        return (0.4 * np.sin(2 * np.pi * hz * t) + 0.2 * np.sin(2 * np.pi * 2 * hz * t)).astype(np.float32), sr

    def test_the_exported_network_matches_pytorch_for_any_number_of_frames(self):
        infer = self.crepe_onnx.make_infer("tiny", "CPUExecutionProvider")
        self.assertTrue((Path(self._tmp.name) / "onnx" / "crepe-tiny.onnx").is_file())      # exported once, into the models folder
        torchcrepe.load.model("cpu", "tiny")
        for n in (1, 300, 512, 700):                                                          # padding and several batches
            frames = torch.randn(n, 1024)
            expected = torchcrepe.core.infer(frames, "tiny", "cpu")
            self.assertEqual(tuple(infer(frames).shape), (n, 360))
            self.assertLess(float((infer(frames) - expected).abs().max()), 1e-4)

    def test_crepe_gives_the_same_pitch_through_onnx_and_pytorch(self):
        from maestro.audio.backends import crepe
        y, sr = self.tone()
        onnx_track = crepe.estimate(y, sr, model="tiny")                                       # auto: ONNX Runtime
        torch_track = crepe.estimate(y, sr, model="tiny", device="cpu")
        self.assertTrue(onnx_track.runtime.startswith("onnx:"), onnx_track.runtime)
        self.assertEqual(torch_track.runtime, "torch:cpu")
        voiced = (onnx_track.confidence > 0.5) & (torch_track.confidence > 0.5)
        self.assertGreater(voiced.mean(), 0.8)
        cents = 1200 * np.log2(onnx_track.f0[voiced] / torch_track.f0[voiced])
        self.assertLess(np.abs(cents).max(), 1.0)
        self.assertAlmostEqual(float(np.median(onnx_track.f0[voiced])), 440.0, delta=5)

    def test_an_unavailable_onnx_path_falls_back_to_pytorch_with_a_warning(self):
        from maestro.audio.backends import crepe
        y, sr = self.tone(seconds=1.0)
        with mock.patch.object(self.crepe_onnx, "make_run", side_effect=RuntimeError("no provider")):
            with self.assertWarnsRegex(UserWarning, "using PyTorch on the CPU"):
                track = crepe.estimate(y, sr, model="tiny")
        self.assertEqual(track.runtime, "torch:cpu")
        self.assertAlmostEqual(float(np.median(track.f0[track.confidence > 0.5])), 440.0, delta=5)

    def test_a_provider_that_fails_mid_run_is_redone_on_the_cpu(self):
        from maestro.audio.backends import crepe

        def broken(frames):
            raise RuntimeError("device lost")

        y, sr = self.tone(seconds=1.0)
        with mock.patch.object(self.crepe_onnx, "make_run", return_value=broken):
            with self.assertWarnsRegex(UserWarning, "device lost"):
                track = crepe.estimate(y, sr, model="tiny")
        self.assertEqual(track.runtime, "torch:cpu")                                            # redone on PyTorch's CPU

    def test_a_gpu_provider_means_the_full_model_by_default(self):
        from maestro.audio.backends import crepe
        seen = []
        with mock.patch.object(crepe_choose := __import__("maestro.utils.device", fromlist=["x"]), "choose",
                               return_value=("onnx", "DmlExecutionProvider")), \
                mock.patch.object(self.crepe_onnx, "make_run", side_effect=lambda model, provider: seen.append((model, provider)) or (_ for _ in ()).throw(RuntimeError("stop"))):
            with self.assertWarns(UserWarning):
                crepe.estimate(*self.tone(seconds=0.5))
        self.assertEqual(seen, [("full", "DmlExecutionProvider")])                              # full is cheap on a GPU


if __name__ == "__main__":
    unittest.main()
