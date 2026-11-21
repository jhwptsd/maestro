# Optional pretrained neural judgement of tone and expression (new file).
#
# Why this exists: there is no public, pretrained violin tone/expression grader (the closest published work trains
# on private or unreleased recordings: Giraldo et al. 2019 on the TELMI data; Dalmazzo & Ramirez 2019 on motion
# capture; VioTech 2025 on a frozen MERT encoder). What can be run today without labelled violin data is a general
# music-language model: LAION CLAP (`laion/clap-htsat-unfused`, Wu et al. 2023) embeds audio and text in one space, so a
# short window of the performance can be compared with descriptions such as "a warm, smooth violin tone" versus "a
# harsh, scratchy violin tone". The result is a probability per window, used by assessment/tone.py and
# assessment/expression.py as supporting evidence: a window is reported only when the neural score and the measured
# features (loudness, vibrato, harmonic ratio, bow speed ... from audio/ and vision/) can be put side by side, and the
# causes always come from those measurements, never from the network alone.
#
# Input: Performance.audio (samples, sr). Output: assessment["neural"] = {"model", "tone": [window...],
# "expression": [window...]} with windows {"start", "end", "p_good"}. Needs `pip install transformers torch`; weights
# (~600 MB) download on first use. Disabled by default (analyze(..., neural="clap")); a failure to load it raises
# NeuralUnavailable and the analysis carries on without it. Other backends can be added by giving `analyze_windows` any
# object with `p_first(audio_windows, sr, prompts) -> list of floats`.
import numpy as np

MODEL = "laion/clap-htsat-unfused"
CLAP_SR = 48000                                   # what the model was trained on
WINDOW_S, HOP_S = 6.0, 3.0
PROMPTS = {                                       # (good description, bad description)
    "tone": ("a warm, smooth, resonant violin tone", "a harsh, scratchy, thin violin tone"),
    "expression": ("an expressive, musical violin performance with dynamic shaping and vibrato",
                   "a flat, mechanical, lifeless violin performance"),
}


class NeuralUnavailable(RuntimeError):
    """The neural backend cannot be loaded (missing package, no network for the weights, ...)."""


class ClapBackend:
    """LAION CLAP zero-shot scorer, loaded on first use."""

    def __init__(self, model_name=MODEL, device=None):
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
            from maestro.utils.device import torch_device
            device = torch_device(device)
            self.torch = torch
            self.model = ClapModel.from_pretrained(model_name).to(device).eval()
            self.processor = ClapProcessor.from_pretrained(model_name)
        except Exception as e:                              # missing packages or weights: report, do not crash
            raise NeuralUnavailable(f"CLAP could not be loaded ({e}); install transformers and torch") from e
        self.name, self.device = model_name, device

    def p_first(self, windows, sr, prompts):
        """Probability that each audio window matches prompts[0] rather than prompts[1] (softmax over the two)."""
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), CLAP_SR)
        audio = [resample_poly(np.asarray(w, np.float32), CLAP_SR // g, int(sr) // g) for w in windows]
        text = self.processor.tokenizer(list(prompts), padding=True, return_tensors="pt")
        feats = self.processor.feature_extractor(audio, sampling_rate=CLAP_SR, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(**{k: v.to(self.device) for k, v in {**text, **feats}.items()})
            return self.torch.softmax(out.logits_per_audio, dim=1)[:, 0].tolist()


def window_spans(duration, window=WINDOW_S, hop=HOP_S):
    """Overlapping (start, end) windows covering `duration` seconds (one window if the audio is shorter)."""
    if duration <= window:
        return [(0.0, float(duration))]
    starts = np.arange(0, duration - window + 1e-9, hop)
    spans = [(float(s), float(s + window)) for s in starts]
    if spans[-1][1] < duration - 1e-6:                      # make sure the end of the recording is covered
        spans.append((float(duration - window), float(duration)))
    return spans


def analyze_windows(y, sr, backend, window=WINDOW_S, hop=HOP_S):
    """Score every window of the performance for tone and expression -> the assessment["neural"] dict."""
    spans = window_spans(len(y) / sr, window, hop)
    chunks = [y[int(a * sr):int(b * sr)] for a, b in spans]
    out = {"model": getattr(backend, "name", type(backend).__name__)}
    for aspect, prompts in PROMPTS.items():
        out[aspect] = [{"start": a, "end": b, "p_good": float(p)}
                       for (a, b), p in zip(spans, backend.p_first(chunks, sr, prompts))]
    return out


def load_backend(name):
    """`neural` argument of analyze() -> a backend object ("clap"), or None for off."""
    if not name:
        return None
    if name == "clap":
        return ClapBackend()
    raise ValueError(f"unknown neural backend {name!r} (use 'clap')")
