"""Run torchcrepe's CREPE network through ONNX Runtime (new file; used by audio/backends/crepe.py).

Why: the network is nearly all of CREPE's run time, and ONNX Runtime runs it (a) on any GPU PyTorch cannot use: DirectML
covers every DirectX 12 GPU on Windows, which is how an AMD or Intel GPU is used there, plus CUDA, ROCm and CoreML providers,
and (b) about twice as fast as PyTorch on the CPU (measured: 512 frames, full model 5.2 s vs 9.6 s, tiny 0.29 s vs 0.59 s;
outputs equal to 2e-7). Only the network is replaced: torchcrepe still does the framing and Maestro's decoder the decoding.

The model is exported once per capacity ("tiny" / "full") to models/onnx/crepe-<capacity>.onnx (needs torch and `onnx`;
the full model takes ~10 s and ~90 MB) with a FIXED batch of BATCH frames, which GPU providers handle best; inputs are padded
to it. If the export, the session or a run fails, crepe.py falls back to PyTorch on the CPU with a warning.
"""
import threading

import numpy as np

from maestro.utils.paths import bundled_models, models_root

BATCH = 512                                  # frames per network call (fixed shape)
FRAME = 1024                                 # samples per CREPE frame
_sessions = {}
_lock = threading.Lock()


def model_path(capacity):
    """Where the exported model is (or will be exported to): the models folder."""
    return models_root() / "onnx" / f"crepe-{capacity}.onnx"


def find_model(capacity):
    """An existing exported model: in the models folder, else shipped with the installed app (src/assets/models), else None."""
    for path in (model_path(capacity), bundled_models() / f"crepe-{capacity}.onnx"):
        if path.is_file():
            return path
    return None


def export(capacity, path=None):
    """Export torchcrepe's `capacity` model to ONNX (once; an existing file is kept) -> path."""
    import torch
    import torchcrepe
    found = None if path else find_model(capacity)
    if found:                                                    # exported already, or shipped with the app (which has no torch to export with)
        return found
    path = path or model_path(capacity)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    torchcrepe.load.model("cpu", capacity)
    model = torchcrepe.core.infer.model.eval()
    partial = path.with_suffix(".onnx.part")
    torch.onnx.export(model, (torch.zeros(BATCH, FRAME),), str(partial), input_names=["frames"], output_names=["probabilities"],
                      opset_version=17, dynamo=False)
    partial.replace(path)
    return path


def session(capacity, provider):
    """An InferenceSession of the exported model on `provider` (CPUExecutionProvider as the fallback), cached."""
    import onnxruntime as ort
    key = (capacity, provider)
    with _lock:
        if key not in _sessions:
            options = ort.SessionOptions()
            if provider != "CPUExecutionProvider":              # recommended for the DirectML provider, harmless for the rest
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [provider] + (["CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [])
            _sessions[key] = ort.InferenceSession(str(export(capacity)), options, providers=providers)
        return _sessions[key]


def make_run(capacity, provider):
    """The network as a numpy function: run(frames (N, 1024) float32) -> probabilities (N, 360). No PyTorch is involved. Builds the session now so that a
    problem shows up before the audio is processed."""
    sess = session(capacity, provider)

    def run(frames):
        x = np.asarray(frames, dtype=np.float32)
        out = np.empty((len(x), 360), dtype=np.float32)
        for start in range(0, len(x), BATCH):
            chunk = x[start:start + BATCH]
            if len(chunk) < BATCH:                                # pad the last batch to the exported shape
                chunk = np.concatenate([chunk, np.zeros((BATCH - len(chunk), FRAME), np.float32)])
            out[start:start + BATCH] = sess.run(None, {"frames": chunk})[0][:len(x) - start]
        return out

    return run


def make_infer(capacity, provider):
    """A drop-in for torchcrepe.core.infer that runs the network on `provider`: infer(frames (N, 1024), model, device,
    embed=False) -> torch tensor (N, 360). Builds the session now so that a problem shows up before the audio is processed."""
    import torch
    run = make_run(capacity, provider)

    def infer(frames, model="full", device="cpu", embed=False):
        return torch.from_numpy(run(frames.detach().cpu().numpy()))

    return infer
