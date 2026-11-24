"""Test fixture: the Minuet in G (Petzold, often credited to Bach) melody, bars 1-8, as MusicXML and as
synthetic *violin-like* audio with the things that broke simple transcribers: vibrato, bow noise,
detached notes with loudness dips, slurs with pitch glides, room reverb, uneven dynamics and timing."""
import numpy as np

from helpers import _NAMES

SR = 22050
Q, E, DH = 1.0, 0.5, 3.0                                   # quarter, eighth, dotted half (in beats)


def _p(name, octave):
    return {"C": 0, "D": 2, "E": 4, "F": 5, "F#": 6, "G": 7, "A": 9, "B": 11}[name] + 12 * (octave + 1)


# bars of (midi, beats, slurred_to_next)
BARS = [
    [(_p("D", 5), Q, 0), (_p("G", 4), E, 0), (_p("A", 4), E, 0), (_p("B", 4), E, 0), (_p("C", 5), E, 0)],
    [(_p("D", 5), Q, 0), (_p("G", 4), Q, 0), (_p("G", 4), Q, 0)],
    [(_p("E", 5), Q, 0), (_p("C", 5), E, 0), (_p("D", 5), E, 0), (_p("E", 5), E, 0), (_p("F#", 5), E, 0)],
    [(_p("G", 5), Q, 0), (_p("G", 4), Q, 0), (_p("G", 4), Q, 0)],
    [(_p("C", 5), Q, 0), (_p("D", 5), E, 1), (_p("C", 5), E, 1), (_p("B", 4), E, 1), (_p("A", 4), E, 0)],
    [(_p("B", 4), Q, 0), (_p("C", 5), E, 1), (_p("B", 4), E, 1), (_p("A", 4), E, 1), (_p("G", 4), E, 0)],
    [(_p("F#", 4), Q, 0), (_p("G", 4), E, 0), (_p("A", 4), E, 0), (_p("B", 4), E, 0), (_p("G", 4), E, 0)],
    [(_p("B", 4), DH, 0)],
]
MIDIS = [m for bar in BARS for m, _, _ in bar]


def minuet_musicxml(bpm=100):
    """The melody as 3/4 MusicXML (divisions=2) at `bpm`."""
    measures = ""
    for k, bar in enumerate(BARS):
        body = ""
        for m, beats, _ in bar:
            step, alter = _NAMES[m % 12]
            body += (f"<note><pitch><step>{step}</step>{'<alter>1</alter>' if alter else ''}"
                     f"<octave>{m // 12 - 1}</octave></pitch><duration>{int(beats * 2)}</duration></note>")
        first = (f'<attributes><divisions>2</divisions><time><beats>3</beats><beat-type>4</beat-type></time></attributes>'
                 f'<direction><sound tempo="{bpm}"/></direction>') if k == 0 else ""
        measures += f'<measure number="{k + 1}">{first}{body}</measure>'
    return ('<score-partwise><part-list><score-part id="P1"/></part-list>'
            f'<part id="P1">{measures}</part></score-partwise>')


def play_minuet(bars=slice(None), bpm=100, vibrato_cents=30, vibrato_hz=5.5, noise=0.01, reverb=True, tuning_cents=0,
                jitter=0.02, seed=0):
    """Synthesise a performance -> float32 audio at SR.

    vibrato_cents: vibrato depth (+-); noise: bow-noise level; tuning_cents: how sharp(+)/flat(-) the
    player is overall; jitter: random note start error (s) per note; seed makes it repeatable;
    bars: which bars of BARS are played, e.g. slice(4, 8) = start halfway down the page (default: all).
    """
    rng = np.random.default_rng(seed)
    spb, t = 60 / bpm, 0.4
    notes = []                                             # [midi, start, dur, slurred]
    for bar in BARS[bars]:
        for m, beats, slur in bar:
            notes.append([m, t + rng.normal(0, jitter), beats * spb, slur])
            t += beats * spb
    n_samples = int((t + 1.0) * SR)
    target, amp = np.zeros(n_samples), np.zeros(n_samples)
    for m, start, dur, slur in notes:
        s, e = int(start * SR), int((start + dur) * SR)
        k = np.arange(e - s)
        a = rng.uniform(0.6, 1.0) * np.minimum(1, k / (0.03 * SR))            # 30 ms bow attack
        if not slur:                                                          # detached: loudness dips
            a *= np.minimum(1, (e - s - k) / (0.06 * SR)) ** 0.7              # before the next note
        a *= 1 + 0.1 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * k / SR)       # bow-pressure wobble
        target[s:e] = m + tuning_cents / 100 + rng.normal(0, 0.05)
        amp[s:e] = np.maximum(amp[s:e], a)
    played = np.flatnonzero(amp > 0)
    pitch = np.interp(np.arange(n_samples), played, target[played])           # slurs: pitch is continuous
    g = int(0.04 * SR)
    pitch = np.convolve(pitch, np.ones(g) / g, mode="same")                   # 40 ms glide between notes
    depth = np.clip(np.convolve(amp, np.ones(int(0.2 * SR)) / int(0.2 * SR), mode="same"), 0, 1)
    pitch = pitch + vibrato_cents / 100 * depth * np.sin(2 * np.pi * vibrato_hz * np.arange(n_samples) / SR + 1.0)
    phase = np.cumsum(440 * 2 ** ((pitch - 69) / 12)) / SR * 2 * np.pi
    y = sum(a * np.sin(h * phase + rng.uniform(0, 6)) for h, a in            # bright, harmonic-rich tone
            enumerate([1, .8, .7, .5, .4, .35, .25, .2, .15, .1, .08, .06], start=1)) * amp
    y = y + noise * rng.standard_normal(n_samples) * np.clip(amp, 0.2, 1)
    if reverb:
        n_ir = int(0.35 * SR)
        ir = np.exp(-np.arange(n_ir) / (0.08 * SR)) * rng.standard_normal(n_ir) * 0.15
        ir[0] = 1
        y = np.convolve(y, ir)[:n_samples]
    return (y / np.abs(y).max() * 0.7).astype(np.float32)
