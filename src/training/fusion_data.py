"""Training data for the fusion encoders (audio, video, score): turn folders of ordinary files into windows.

Every loader runs Maestro's own analysis on a file (the same code the analysis uses, so the side information the decoder
sees in training is exactly what it sees at inference) and returns windows `(x, side)` as numpy arrays, one pair per
file. `load_windows` concatenates files, skipping any file that cannot be read, and `split` divides the windows by FILE
(windows of one recording are strongly correlated, so a random split would leak).

    audio  any audio/video file (wav, mp3, flac, m4a, ogg, mp4, mov ...): extract audio, pitch (yin by default: fast),
           frame features, log-mel; cached per file in `cache_dir` because the analysis is the slow part
    video  Maestro pose JSON (written next to every analysis as pose.json), or a plain .npy array of shape
           (frames, 18, 3) in millimetres in pose_io.VIOPOSE_JOINTS order (motion capture mapped to those joints; pass fps)
    score  MusicXML / .mxl / MIDI (/ PDF with Audiveris) through the normal score preprocessing, or the MusicXML scores
           bundled with `pip install music21` (about 650 scores, many of them string quartets: `music21_scores()`)

Suggested public data (check each licence; none of it is downloaded by this code):
    audio  MusicNet (Zenodo 5120004; the solo-violin and violin-with-ensemble recordings), URMP (violin tracks and
           videos), Good-sounds (violin notes), the audio of VioDat (sj-yoo.info/viopose) and SyncViolinist
           (github.com/Kakanat/SyncViolinist); the recordings behind the MTG violin MIDI dataset (Zenodo 13736820)
    video  VioDat (639 videos, 12 players, Vicon motion capture: non-commercial research licence), SyncViolinist
           (motion capture with audio), MAPdat (AIMusicGuru); or your own recordings run through Maestro/VioPose
    score  PDMX (public-domain MusicXML from MuseScore, Zenodo 15571083: keep the no-licence-conflict subset and the
           violin parts), the Violin MIDI dataset (Zenodo 13736820), OpenScore, the music21 corpus
"""
import hashlib
from pathlib import Path

import numpy as np

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".mp4", ".mov", ".webm", ".mkv", ".avi"}
SCORE_EXT = {".musicxml", ".xml", ".mxl", ".mid", ".midi", ".pdf"}
POSE_EXT = {".json", ".npy"}
EXTENSIONS = {"audio": AUDIO_EXT, "video": POSE_EXT, "score": SCORE_EXT}


def find(root, kind):
    """All files below `root` (or `root` itself) that `kind` can read, sorted."""
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS[kind])


def music21_scores(limit=None):
    """MusicXML paths of the corpus shipped with music21 (`pip install music21`); [] if it is not installed."""
    try:
        from music21 import corpus
    except ImportError:
        return []
    paths = sorted(p for p in corpus.getCorePaths() if p.suffix in (".mxl", ".xml", ".musicxml"))
    return paths[:limit] if limit else paths


def audio_file_windows(path, cache_dir=None, pitch_backend="yin"):
    """(x, side) windows of one audio/video file. Cached as .npz in `cache_dir` (keyed on path, size, time, backend)."""
    from maestro.audio.features import extract_features
    from maestro.audio.pitch import estimate_pitch
    from maestro.fusion import audio_encoder
    from maestro.preprocessing.audio import DEFAULT_SR, extract_audio
    path = Path(path)
    cache = None
    if cache_dir:
        key = hashlib.sha1(f"{path.resolve()}|{path.stat().st_size}|{path.stat().st_mtime_ns}|{pitch_backend}".encode()).hexdigest()
        cache = Path(cache_dir) / f"{key}.npz"
        if cache.is_file():
            with np.load(cache) as d:
                return d["x"], d["side"]
    y, sr = extract_audio(path, DEFAULT_SR)
    track = estimate_pitch(y, sr, backend=pitch_backend)
    frames = extract_features(y, sr, pitch_track=track)
    x, side, _ = audio_encoder.make_windows(audio_encoder.log_mel(y, sr), audio_encoder.side_features(frames))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, x=x, side=side)
    return x, side


def pose_file_windows(path, fps=30.0):
    """(x, side) windows of one pose file: Maestro pose JSON, or a plain .npy (frames, 18, 3) mm array at `fps`."""
    from maestro.fusion import video_encoder
    from maestro.vision.motion_features import extract_motion
    from maestro.vision.pose_io import load_pose, pose_from_array
    path = Path(path)
    pose = load_pose(path) if path.suffix.lower() == ".json" else pose_from_array(np.load(path, allow_pickle=False), fps)
    x, side, _ = video_encoder.pose_windows(pose, extract_motion(pose))
    return x, side


def score_file_windows(path):
    """(tokens, side) windows of one score file (its violin part, else its first part)."""
    from maestro.fusion import score_encoder
    from maestro.score.musicxml import parse_musicxml
    from maestro.utils.convert_file_type import convert_to_musicxml
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        score = parse_musicxml(convert_to_musicxml(path, tmp))
    if not score.notes:
        raise ValueError("no notes")
    tokens, side, _ = score_encoder.make_windows([n.to_dict() for n in score.notes], score.key_fifths, score.time_signature)
    return tokens, side


LOADERS = {"audio": audio_file_windows, "video": pose_file_windows, "score": score_file_windows}


def load_windows(kind, paths, log=print, **options):
    """Windows of every readable file -> (list of x arrays, list of side arrays), one entry per file. Unreadable files are
    skipped with a message; `options` go to the loader (cache_dir, pitch_backend, fps)."""
    xs, sides = [], []
    for i, path in enumerate(paths, 1):
        try:
            x, side = LOADERS[kind](path, **{k: v for k, v in options.items() if k in LOADER_OPTIONS[kind]})
        except Exception as e:                                       # one bad file must not end a long run
            log(f"  skipped {Path(path).name}: {e}")
            continue
        xs.append(x), sides.append(side)
        if i % 25 == 0 or i == len(paths):
            log(f"  read {i}/{len(paths)} files, {sum(len(v) for v in xs)} windows")
    return xs, sides


LOADER_OPTIONS = {"audio": {"cache_dir", "pitch_backend"}, "video": {"fps"}, "score": set()}


def split(xs, sides, val_fraction=0.15, seed=0):
    """Per-file window lists -> ((x_train, side_train), (x_val, side_val)) split by file (at least one file each way
    when there are two or more; with one file the last windows are validation)."""
    n = len(xs)
    if n == 0:
        raise ValueError("no usable files")
    order = np.random.default_rng(seed).permutation(n)
    n_val = int(round(n * val_fraction)) if n > 1 else 0
    n_val = max(1, min(n_val, n - 1)) if n > 1 else 0
    val_files, train_files = order[:n_val], order[n_val:]
    cat = lambda arrays, idx: np.concatenate([arrays[i] for i in idx])
    if n == 1:
        cut = max(1, int(len(xs[0]) * (1 - val_fraction)))
        return (xs[0][:cut], sides[0][:cut]), (xs[0][cut:], sides[0][cut:])
    return (cat(xs, train_files), cat(sides, train_files)), (cat(xs, val_files), cat(sides, val_files))

