"""Shared test fixtures: synthetic violin-like audio, matching MusicXML, and a tiny muxed video."""
import subprocess
import wave

import numpy as np

SR = 22050


def synth(notes, sr=SR, total=None):
    """notes: [(midi (float ok), start_s, dur_s)] -> float32 audio of a harmonic tone per note."""
    total = total or max(s + d for _, s, d in notes) + 0.3
    y = np.zeros(int(total * sr))
    for midi, start, dur in notes:
        t = np.arange(int(dur * sr)) / sr
        f = 440.0 * 2 ** ((midi - 69) / 12)
        tone = sum(a * np.sin(2 * np.pi * f * k * t) for k, a in ((1, 1.0), (2, 0.5), (3, 0.25)))
        env = np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.02)   # 10 ms attack, 20 ms release
        i = int(start * sr)
        y[i:i + len(t)] += 0.3 * tone * env
    return y.astype(np.float32)


def write_wav(path, y, sr=SR):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


_NAMES = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0), ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0)]


def musicxml(midis, tempo=120, beats_per_measure=4):
    """Quarter-note MusicXML for the given MIDI pitches (divisions=1)."""
    measures = []
    for k in range(0, len(midis), beats_per_measure):
        body = ""
        for m in midis[k:k + beats_per_measure]:
            step, alter = _NAMES[m % 12]
            body += (f"<note><pitch><step>{step}</step>{f'<alter>{alter}</alter>' if alter else ''}"
                     f"<octave>{m // 12 - 1}</octave></pitch><duration>1</duration><type>quarter</type></note>")
        attrs = ("<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
                 f'<direction><sound tempo="{tempo}"/></direction>') if k == 0 else ""
        measures.append(f'<measure number="{k // beats_per_measure + 1}">{attrs}{body}</measure>')
    return ('<?xml version="1.0"?><score-partwise version="3.1"><part-list><score-part id="P1"/></part-list>'
            f'<part id="P1">{"".join(measures)}</part></score-partwise>')


def make_video(path, wav_path, seconds):
    """Mux a black test video with `wav_path` using ffmpeg."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={seconds}",
                    "-i", str(wav_path), "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(path)], check=True)
