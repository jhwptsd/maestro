# Choose where the neural networks run: any GPU, else the CPU (new file).
#
# Used by the CREPE pitch backend (audio/backends/crepe.py), the CLAP judgement (assessment/neural.py), the fusion encoders
# (fusion/embed.py) and training (src/training/train_fusion.py). Preference order for "auto":
#   1. PyTorch's own accelerator: NVIDIA CUDA, AMD ROCm (its PyTorch builds also answer to "cuda"), Intel XPU or Apple MPS
#   2. ONNX Runtime's GPU provider: DirectML (any DirectX 12 GPU on Windows, which is how an AMD or Intel GPU is used there
#      since ROCm PyTorch is not generally available on Windows), CUDA, ROCm or CoreML. Only the convolutional CREPE model is
#      run this way (exported once to models/onnx/, see audio/backends/crepe_onnx.py).
#   3. ONNX Runtime on the CPU (about twice as fast as PyTorch's CPU for CREPE), if onnxruntime is installed
#   4. PyTorch on the CPU
# `MAESTRO_DEVICE` (or a `device` argument) overrides: "auto", "cpu", a torch device ("cuda", "cuda:1", "mps", "xpu") or
# "onnx" (ONNX Runtime's GPU provider). A GPU path that fails at run time falls back to the CPU, with a warning.
import os

ENV = "MAESTRO_DEVICE"
GPU_PROVIDERS = ("CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider", "DmlExecutionProvider",
                 "CoreMLExecutionProvider")


def torch_accelerator():
    """'cuda:0', 'xpu' or 'mps' if PyTorch can use that accelerator, else None."""
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else None


def onnx_gpu_provider():
    """The first GPU execution provider ONNX Runtime offers (see GPU_PROVIDERS), or None."""
    try:
        import onnxruntime
    except ImportError:
        return None
    available = onnxruntime.get_available_providers()
    return next((p for p in GPU_PROVIDERS if p in available), None)


def choose(preference=None):
    """-> ("torch", device string) or ("onnx", provider name): where a network that supports both should run."""
    pref = str(preference or os.environ.get(ENV) or "auto").lower()
    if pref == "cpu":
        return "torch", "cpu"
    if pref in ("onnx", "directml", "dml"):
        provider = onnx_gpu_provider()
        return ("onnx", provider) if provider else choose("auto")
    if pref != "auto":
        return "torch", pref
    accelerator = torch_accelerator()
    if accelerator:
        return "torch", accelerator
    provider = onnx_gpu_provider()
    if provider:
        return "onnx", provider
    try:
        import onnxruntime  # noqa: F401
        return "onnx", "CPUExecutionProvider"
    except ImportError:
        return "torch", "cpu"


def is_accelerated(kind, target):
    """True if (kind, target) from `choose` is a GPU/accelerator rather than a CPU."""
    return target not in ("cpu", "CPUExecutionProvider")


def torch_device(preference=None):
    """Device string for a plain PyTorch model: the preference, else a PyTorch accelerator, else "cpu" (an ONNX-only GPU does
    not help a PyTorch model)."""
    kind, target = choose(preference)
    return target if kind == "torch" else "cpu"


def describe():
    """One line for logs: what is available and what 'auto' picks."""
    kind, target = choose()
    return f"torch accelerator: {torch_accelerator() or 'none'}; onnx GPU provider: {onnx_gpu_provider() or 'none'}; auto -> {kind}:{target}"
