"""Dataset installation and layout for the fusion encoders (audio and video).

Everything lives in the repository's `data/` folder (move it with the MAESTRO_DATA environment variable):

    data/downloads/<dataset>/            archives and videos as downloaded (safe to delete once installed)
    data/<kind>/<dataset>/train|val|test/   the files the models train, validate and test on, by recording
    data/<kind>/<dataset>/manifest.json     one entry per file: id, split, source, licence, attribution
    data/cache/                          analysed windows (audio features are slow to compute)

Partitions are decided per RECORDING (windows of one recording are strongly correlated) by a hash of the recording id, so
re-running an install never moves a file between splits and a growing dataset keeps its old splits. Trained models are
written to `models/fusion/` (see maestro.fusion.embed.models_dir).

Datasets (python src/training/install_data.py list):
    musicnet     AUDIO. The violin recordings of MusicNet (Thickstun et al.; Zenodo 5120004, CC BY 4.0): solo violin, violin with
                 piano and string quartets, chosen by the `ensemble` column of the metadata. Downloads musicnet.tar.gz (large,
                 resumable), keeps only the violin .wav files and honours MusicNet's own train/test folders.
    youtube-cc   VIDEO (and audio). Violin performance videos that YouTube lists under the Creative Commons licence (reuse
                 allowed), found with yt-dlp; each one is run through VioPose (which Maestro's setup installs) to get a pose.json
                 and its audio is saved too. The uploader, title and URL of every video are kept in the manifest for attribution.
    manual       AUDIO or VIDEO from sets that need a request form (VioDat, SyncViolinist, MOSA: all research-only, not
                 redistributable). Request access, convert to wav / (frames, 18, 3) mm .npy arrays, and run
                 `install_data.py ingest NAME KIND FOLDER`: the files are checked, copied and partitioned like the others.
"""
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
MUSICNET_RECORD = "https://zenodo.org/records/5120004"
MUSICNET_URLS = {"metadata": MUSICNET_RECORD + "/files/musicnet_metadata.csv?download=1",
                 "archive": MUSICNET_RECORD + "/files/musicnet.tar.gz?download=1",
                 "api": "https://zenodo.org/api/records/5120004"}          # lists each file's md5, used to verify the download
CC_LICENSE_FILTER = "license *= 'Creative Commons' & duration >= 30 & duration <= 900"
YOUTUBE_QUERIES = ("solo violin performance", "violin etude", "violin sonata", "violin concerto solo violinist",
                   "violin practice full body", "violin recital")
AUDIO_EXT, POSE_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}, {".json", ".npy"}


# ---- paths and layout -------------------------------------------------------------------------------------------------
def data_dir():
    """The data folder: $MAESTRO_DATA, else <repo>/data."""
    return Path(os.environ.get("MAESTRO_DATA") or Path(__file__).resolve().parents[2] / "data")


def dataset_dir(kind, name):
    return data_dir() / kind / name


def split_of(key, fractions=(0.8, 0.1, 0.1), seed="maestro"):
    """'train' / 'val' / 'test' for a recording id, from a hash (stable across runs and machines)."""
    h = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 16 ** 8
    return SPLITS[0] if h < fractions[0] else SPLITS[1] if h < fractions[0] + fractions[1] else SPLITS[2]


def files(kind, name, split):
    """Files of one partition of an installed dataset (sorted)."""
    exts = AUDIO_EXT if kind == "audio" else POSE_EXT
    return sorted(p for p in (dataset_dir(kind, name) / split).glob("*") if p.suffix.lower() in exts)


def write_manifest(kind, name, source, license_, items):
    path = dataset_dir(kind, name) / "manifest.json"
    path.write_text(json.dumps({"dataset": name, "kind": kind, "source": source, "license": license_, "items": items}, indent=1))
    return path


def read_manifest(kind, name):
    path = dataset_dir(kind, name) / "manifest.json"
    return json.loads(path.read_text()) if path.is_file() else None


def installed():
    """{kind: {dataset: {split: file count}}} for everything under data/audio and data/video."""
    out = {}
    for kind in ("audio", "video"):
        for d in sorted(p for p in (data_dir() / kind).glob("*") if (p / "manifest.json").is_file()):
            out.setdefault(kind, {})[d.name] = {s: len(files(kind, d.name, s)) for s in SPLITS}
    return out


# ---- downloading ------------------------------------------------------------------------------------------------------
def download(url, dest, md5=None, log=print, chunk=1 << 20):
    """Download `url` to `dest`, resuming a partial `dest.part` (HTTP Range) and verifying `md5` if given. Returns dest.
    An existing `dest` is kept (delete it to force a new download)."""
    dest = Path(dest)
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        r = urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        if e.code != 416:                                             # 416: the partial file is already the whole file
            raise
        r = None
    if r is not None:
        with r:
            resumed = r.status == 206
            total = int(r.headers.get("Content-Length", 0)) + (have if resumed else 0)
            if have and not resumed:                                  # the server ignored Range: start over
                have = 0
            done, t0, last = have, time.time(), 0.0
            with open(part, "ab" if resumed else "wb") as f:
                while block := r.read(chunk):
                    f.write(block)
                    done += len(block)
                    if time.time() - last > 5:
                        last = time.time()
                        log(f"  {dest.name}: {done / 1e6:,.0f} MB" + (f" of {total / 1e6:,.0f} MB ({done / total:.0%})" if total else "")
                            + f", {(done - have) / 1e6 / max(time.time() - t0, 1e-3):.1f} MB/s")
    if md5:
        digest = hashlib.md5()
        with open(part, "rb") as f:
            while block := f.read(chunk):
                digest.update(block)
        if digest.hexdigest() != md5:
            part.unlink()
            raise IOError(f"{dest.name}: checksum mismatch (got {digest.hexdigest()}, expected {md5}); the partial file was removed, run again")
    part.rename(dest)
    return dest


def zenodo_md5(api_url, filename, log=print):
    """md5 of `filename` from a Zenodo record's API, or None if it cannot be read (the download then goes unverified)."""
    try:
        with urllib.request.urlopen(api_url) as r:
            record = json.load(r)
        entry = next(f for f in record["files"] if f["key"] == filename)
        return entry["checksum"].split(":")[-1]
    except Exception as e:
        log(f"  could not read the checksum of {filename} ({e}); not verifying")
        return None


# ---- audio: MusicNet --------------------------------------------------------------------------------------------------
def musicnet_violin_ids(metadata_csv, ensembles=None):
    """MusicNet metadata -> {id: row} of the recordings whose `ensemble` contains 'violin' (or is in `ensembles`).
    Raises with the ensembles that exist if none match."""
    import csv
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pick = (lambda e: e in ensembles) if ensembles else (lambda e: "violin" in e.lower())
    chosen = {r["id"]: r for r in rows if pick(r["ensemble"])}
    if not chosen:
        raise ValueError("no MusicNet ensemble matched; the metadata has: " + ", ".join(sorted({r["ensemble"] for r in rows})))
    return chosen


def install_musicnet(urls=None, ensembles=None, keep_archive=False, log=print):
    """Download and partition the MusicNet violin recordings into data/audio/musicnet/{train,val,test}/<id>.wav.

    One streaming pass over musicnet.tar.gz extracts only the chosen .wav files (the rest of the archive is read and
    discarded). The archive stays in data/downloads/musicnet/ unless it is deleted afterwards (default: deleted once
    everything was extracted; keep_archive=True keeps it). MusicNet's test_data folder is the test split; the rest is split
    train/val by hash."""
    urls = {**MUSICNET_URLS, **(urls or {})}
    downloads = data_dir() / "downloads" / "musicnet"
    log("MusicNet: metadata")
    meta = download(urls["metadata"], downloads / "musicnet_metadata.csv", log=log)
    wanted = musicnet_violin_ids(meta, ensembles)
    log(f"MusicNet: {len(wanted)} violin recordings; downloading the archive (large; resumes if interrupted)")
    md5 = zenodo_md5(urls["api"], "musicnet.tar.gz", log) if urls.get("api") else None
    archive = download(urls["archive"], downloads / "musicnet.tar.gz", md5, log)
    root, items, seen = dataset_dir("audio", "musicnet"), [], set()
    for s in SPLITS:
        (root / s).mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r|gz") as tar:                      # streaming: members are read in order, once
        for member in tar:
            path = Path(member.name)
            if not member.isfile() or path.suffix.lower() != ".wav" or path.stem not in wanted or path.stem in seen:
                continue
            official_test = "test" in path.parent.name.lower()
            split = "test" if official_test else ("val" if split_of(path.stem, (0.85, 0.15, 0.0)) == "val" else "train")
            with tar.extractfile(member) as src, open(root / split / f"{path.stem}.wav", "wb") as dst:
                shutil.copyfileobj(src, dst)
            seen.add(path.stem)
            row = wanted[path.stem]
            items.append({"id": path.stem, "file": f"{split}/{path.stem}.wav", "split": split, "composer": row.get("composer"),
                          "composition": row.get("composition"), "ensemble": row.get("ensemble"), "seconds": row.get("seconds")})
    missing = set(wanted) - seen
    if missing:
        log(f"  warning: {len(missing)} chosen recordings were not in the archive (ids {sorted(missing)[:5]} ...)")
    write_manifest("audio", "musicnet", MUSICNET_RECORD, "CC BY 4.0 (Thickstun, Harchaoui, Kakade; check the Zenodo record)", items)
    if not keep_archive and items:
        archive.unlink()
    log(f"MusicNet: {len(items)} recordings in " + ", ".join(f"{s} {sum(i['split'] == s for i in items)}" for s in SPLITS))
    return items


# ---- video: Creative Commons videos -> VioPose poses ------------------------------------------------------------------
def default_downloader(queries, count, folder, log=print):
    """Download Creative Commons violin videos with yt-dlp -> list of info dicts (id, title, uploader, webpage_url, license,
    duration, path). Needs `pip install yt-dlp` and ffmpeg."""
    try:
        import yt_dlp
        from yt_dlp.utils import match_filter_func
    except ImportError:
        raise SystemExit("youtube-cc needs yt-dlp: pip install yt-dlp")
    folder.mkdir(parents=True, exist_ok=True)
    options = {"format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]", "merge_output_format": "mp4",
               "outtmpl": str(folder / "%(id)s.%(ext)s"), "match_filter": match_filter_func(CC_LICENSE_FILTER),
               "download_archive": str(folder / "archive.txt"), "ignoreerrors": True, "quiet": True, "noprogress": True,
               "max_downloads": count, "writeinfojson": True, "noplaylist": True}
    per_query = max(count * 3 // len(queries), count)                # search results that pass the filter are a fraction
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            for q in queries:
                try:
                    ydl.download([f"ytsearch{per_query}:{q}"])
                except yt_dlp.utils.MaxDownloadsReached:
                    break
    except yt_dlp.utils.MaxDownloadsReached:
        pass
    out = []
    for info_file in sorted(folder.glob("*.info.json")):
        info = json.loads(info_file.read_text())
        video = next(iter(folder.glob(f"{info['id']}.mp4")), None)
        if video:
            out.append({k: info.get(k) for k in ("id", "title", "uploader", "webpage_url", "license", "duration")} | {"path": video})
    return out[:count]


def default_pose_maker(video, audio_wav):
    """Pose of a video from the installed VioPose (with its audio model when `audio_wav` is given)."""
    from maestro.vision.pose_estimation import estimate_pose
    return estimate_pose(str(video), str(audio_wav))


def install_youtube_cc(count=60, queries=YOUTUBE_QUERIES, downloader=default_downloader, pose_maker=default_pose_maker, log=print):
    """Download up to `count` Creative Commons violin videos, estimate their poses with VioPose and save
    data/video/youtube-cc/{train,val,test}/<id>.json (Maestro pose JSON) and data/audio/youtube-cc/.../<id>.wav.
    Videos already processed are skipped, so an interrupted run continues; a video VioPose cannot handle is skipped."""
    from maestro.preprocessing.audio import DEFAULT_SR, extract_audio
    from maestro.vision.pose_io import save_pose
    from scipy.io import wavfile
    infos = downloader(list(queries), count, data_dir() / "downloads" / "youtube-cc", log)
    log(f"youtube-cc: {len(infos)} videos downloaded; estimating poses (VioPose, several minutes per video)")
    items = {i["id"]: i for i in (read_manifest("video", "youtube-cc") or {"items": []})["items"]}
    for k, info in enumerate(infos, 1):
        split = split_of(info["id"])
        pose_path = dataset_dir("video", "youtube-cc") / split / f"{info['id']}.json"
        wav_path = dataset_dir("audio", "youtube-cc") / split / f"{info['id']}.wav"
        if pose_path.is_file():
            continue
        try:
            y, sr = extract_audio(info["path"], DEFAULT_SR)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(wav_path, sr, np.asarray(y, dtype=np.float32))
            pose = pose_maker(info["path"], wav_path)
        except Exception as e:                                       # one unusable video must not end the run
            log(f"  skipped {info['id']}: {e}")
            wav_path.unlink(missing_ok=True)
            continue
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        save_pose(pose, pose_path)
        items[info["id"]] = {"id": info["id"], "file": f"{split}/{info['id']}.json", "split": split, "title": info["title"],
                             "uploader": info["uploader"], "url": info["webpage_url"], "license": info["license"],
                             "seconds": info["duration"], "viopose_audio": bool(pose.audio_used)}
        log(f"  [{k}/{len(infos)}] {info['id']} -> {split}")
    done = list(items.values())
    source = "YouTube videos under the Creative Commons licence (yt-dlp); poses by VioPose"
    write_manifest("video", "youtube-cc", source, "CC BY (per video, see items); attribute the uploaders", done)
    write_manifest("audio", "youtube-cc", source, "CC BY (per video, see items); attribute the uploaders",
                   [{**i, "file": i["file"].replace(".json", ".wav")} for i in done])
    return done


# ---- manual (request-only) datasets ------------------------------------------------------------------------------------
def ingest(name, kind, source, log=print):
    """Check, copy and partition files you obtained yourself into data/<kind>/<name>/. `kind` is 'audio' (wav, flac, mp3 ...)
    or 'video' (Maestro pose .json, or plain .npy arrays of shape (frames, 18, 3) in millimetres in pose_io.VIOPOSE_JOINTS
    order: motion capture mapped to those joints). Files that do not pass the check are reported and skipped."""
    from maestro.vision.pose_io import load_pose, pose_from_array
    source = Path(source)
    exts = AUDIO_EXT if kind == "audio" else POSE_EXT
    items = []
    for path in sorted(p for p in (source.rglob("*") if source.is_dir() else [source]) if p.suffix.lower() in exts):
        try:
            if kind == "video":
                (load_pose(path) if path.suffix.lower() == ".json" else pose_from_array(np.load(path, allow_pickle=False), 30.0))
        except Exception as e:
            log(f"  skipped {path.name}: {e}")
            continue
        key = hashlib.sha1(str(path.relative_to(source) if source.is_dir() else path.name).encode()).hexdigest()[:10]
        stem = f"{path.stem}-{key}"                                  # unique even when different folders reuse a file name
        split = split_of(stem)
        target = dataset_dir(kind, name) / split / (stem + path.suffix.lower())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        items.append({"id": stem, "file": f"{split}/{target.name}", "split": split, "original": str(path.name)})
    write_manifest(kind, name, f"ingested from {source}", "as per the original dataset (check its licence)", items)
    log(f"{name}: {len(items)} {kind} files ingested")
    return items
