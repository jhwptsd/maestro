"""Shared test fixtures: synthetic violin-like audio, matching MusicXML, and a tiny muxed video."""
import subprocess
import wave
from pathlib import Path

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


def make_video(path, wav_path, seconds, size="64x64"):
    """Mux a black test video of the given size with `wav_path` using ffmpeg."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=black:s={size}:d={seconds}",
                    "-i", str(wav_path), "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(path)], check=True)


def measure(number, midis, left="", right="", direction="", attributes=""):
    """One 4/4 MusicXML measure of quarter notes. A tuple in `midis` is a chord (multiple stop);
    `left`/`right` are barline contents (repeat/ending), `direction` e.g. a <sound .../> element."""
    body = ""
    for item in midis:
        for k, m in enumerate(item if isinstance(item, tuple) else (item,)):
            step, alter = _NAMES[m % 12]
            body += (f"<note>{'<chord/>' if k else ''}<pitch><step>{step}</step>"
                     f"{f'<alter>{alter}</alter>' if alter else ''}<octave>{m // 12 - 1}</octave></pitch>"
                     "<duration>1</duration></note>")
    bar = lambda loc, c: f'<barline location="{loc}">{c}</barline>' if c else ""
    first = '<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>' if number == 1 else ""
    return (f'<measure number="{number}">{first}{attributes}{bar("left", left)}'
            f'{f"<direction><direction-type/>{direction}</direction>" if direction else ""}{body}{bar("right", right)}</measure>')


def score_xml(*measures, part_list='<score-part id="P1"/>'):
    """Wrap measure strings (from `measure`) into a one-part score."""
    return ('<?xml version="1.0"?><score-partwise version="3.1"><part-list>' + part_list +
            '</part-list><part id="P1">' + "".join(measures) + "</part></score-partwise>")


def write_midi(path, notes, ppq=480, bpm=120):
    """Minimal format-0 MIDI writer. notes: [(start_beats, end_beats, pitch)] on channel 0."""
    events = []
    for s, e, p in notes:
        events += [(round(s * ppq), 1, p), (round(e * ppq), 0, p)]   # off (0) sorts before on (1)
    events.sort()
    vlq = lambda v: bytes([(v >> 21) & 0x7F | 0x80, (v >> 14) & 0x7F | 0x80, (v >> 7) & 0x7F | 0x80, v & 0x7F])
    track = b"\x00\xff\x51\x03" + int(60e6 / bpm).to_bytes(3, "big")
    now = 0
    for t, on, p in events:
        track += vlq(t - now) + bytes([0x90 if on else 0x80, p, 90 if on else 0])
        now = t
    track += b"\x00\xff\x2f\x00"
    Path(path).write_bytes(b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big")
                           + ppq.to_bytes(2, "big") + b"MTrk" + len(track).to_bytes(4, "big") + track)
