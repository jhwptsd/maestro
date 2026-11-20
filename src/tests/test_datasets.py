"""Dataset installation (src/training/datasets.py, install_data.py) and training from the data/ partitions. No network: a local
HTTP server with range support stands in for Zenodo, and fake downloaders/pose makers stand in for yt-dlp and VioPose."""
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    raise unittest.SkipTest("needs torch")

from helpers import make_video, synth, write_wav
from pose_fixture import minuet_notes, synth_pose
from maestro.fusion.residual_autoencoder import load_checkpoint
from maestro.fusion.embed import BUILDERS
from maestro.vision.pose_io import VIOPOSE_JOINTS, load_pose, save_pose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import datasets  # noqa: E402
import install_data  # noqa: E402
import train_fusion  # noqa: E402


class Server:
    """Serves {path: bytes} on localhost with HTTP Range support (like Zenodo); counts requests that carried a Range."""

    def __init__(self, routes):
        outer = self
        self.routes, self.ranged = routes, 0

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = outer.routes.get(self.path.split("?")[0])
                if body is None:
                    return self.send_error(404)
                start, status = 0, 200
                if "Range" in self.headers:
                    start, status = int(self.headers["Range"].split("=")[1].split("-")[0]), 206
                    outer.ranged += 1
                    if start >= len(body):
                        return self.send_error(416)
                self.send_response(status)
                self.send_header("Content-Length", str(len(body) - start))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                self.end_headers()
                self.wfile.write(body[start:])

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def wav_bytes(seed, seconds=1.0):
    with tempfile.TemporaryDirectory() as d:
        write_wav(Path(d) / "a.wav", synth([(60 + seed % 12, 0.1, seconds - 0.2)]))
        return (Path(d) / "a.wav").read_bytes()


def fake_musicnet():
    """(metadata csv, tar.gz bytes, ids): ids 1, 2 violin (train_data), 3 violin (test_data), 4 piano, 5 violin+piano."""
    csv = ("id,composer,composition,movement,ensemble,source,transcriber,catalog_name,seconds\n"
           "1,Bach,Partita,1,Solo Violin,x,y,z,60\n2,Bach,Partita,2,Solo Violin,x,y,z,61\n3,Beethoven,Sonata,1,Solo Violin,x,y,z,62\n"
           "4,Chopin,Etude,1,Solo Piano,x,y,z,63\n5,Mozart,Sonata,1,Accompanied Violin,x,y,z,64\n")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for i, folder in ((1, "train_data"), (2, "train_data"), (3, "test_data"), (4, "train_data"), (5, "train_data")):
            data = wav_bytes(i)
            info = tarfile.TarInfo(f"musicnet/{folder}/{i}.wav")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        label = b"start_time,end_time\n"
        info = tarfile.TarInfo("musicnet/train_labels/1.csv")
        info.size = len(label)
        tar.addfile(info, io.BytesIO(label))
    return csv.encode(), buf.getvalue()


class WithDataDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        patcher = mock.patch.dict(os.environ, {"MAESTRO_DATA": str(self.tmp / "data"), "MAESTRO_MODELS": str(self.tmp / "models")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self.log = []


class Layout(WithDataDir):
    def test_partitions_are_by_recording_stable_and_roughly_80_10_10(self):
        ids = [f"rec{i}" for i in range(3000)]
        counts = {s: sum(datasets.split_of(i) == s for i in ids) for s in datasets.SPLITS}
        self.assertTrue(2250 < counts["train"] < 2550 and 200 < counts["val"] < 400 and 200 < counts["test"] < 400, counts)
        self.assertEqual([datasets.split_of(i) for i in ids[:50]], [datasets.split_of(i) for i in ids[:50]])   # no randomness
        self.assertEqual(datasets.split_of("abc", seed="other") in datasets.SPLITS, True)

    def test_everything_goes_under_the_data_folder(self):
        self.assertEqual(datasets.data_dir(), self.tmp / "data")
        self.assertEqual(datasets.dataset_dir("audio", "musicnet"), self.tmp / "data" / "audio" / "musicnet")


class Downloading(WithDataDir):
    def test_resume_verify_and_keep(self):
        body = os.urandom(100_000)
        srv = Server({"/f.bin": body})
        try:
            dest = self.tmp / "f.bin"
            (self.tmp / "f.bin.part").write_bytes(body[:40_000])                     # an interrupted download
            datasets.download(srv.url + "/f.bin", dest, hashlib.md5(body).hexdigest(), self.log.append)
            self.assertEqual(dest.read_bytes(), body)
            self.assertEqual(srv.ranged, 1)                                             # it resumed, not restarted
            self.assertFalse((self.tmp / "f.bin.part").exists())
            self.assertEqual(datasets.download(srv.url + "/f.bin", dest, log=self.log.append), dest)   # existing file: nothing to do
            (self.tmp / "g.bin.part").write_bytes(body)                                 # the partial file is already complete (416)
            datasets.download(srv.url + "/f.bin", self.tmp / "g.bin", log=self.log.append)
            self.assertEqual((self.tmp / "g.bin").read_bytes(), body)
            with self.assertRaisesRegex(IOError, "checksum mismatch"):
                datasets.download(srv.url + "/f.bin", self.tmp / "h.bin", "0" * 32, self.log.append)
            self.assertFalse((self.tmp / "h.bin").exists() or (self.tmp / "h.bin.part").exists())   # the bad file is not kept
        finally:
            srv.close()


class MusicNet(WithDataDir):
    def install(self, **kw):
        csv, archive = fake_musicnet()
        api = json.dumps({"files": [{"key": "musicnet.tar.gz", "checksum": "md5:" + hashlib.md5(archive).hexdigest()}]}).encode()
        srv = Server({"/meta.csv": csv, "/musicnet.tar.gz": archive, "/api": api})
        try:
            urls = {"metadata": srv.url + "/meta.csv", "archive": srv.url + "/musicnet.tar.gz", "api": srv.url + "/api"}
            return datasets.install_musicnet(urls, log=self.log.append, **kw)
        finally:
            srv.close()

    def test_only_violin_recordings_are_kept_and_partitioned_with_a_manifest(self):
        items = self.install()
        self.assertEqual({i["id"] for i in items}, {"1", "2", "3", "5"})                  # no piano
        by_id = {i["id"]: i for i in items}
        self.assertEqual(by_id["3"]["split"], "test")                                      # MusicNet's own test folder
        self.assertNotEqual(by_id["1"]["split"], "test")
        root = datasets.dataset_dir("audio", "musicnet")
        for i in items:
            self.assertTrue((root / i["file"]).is_file())
            self.assertGreater((root / i["file"]).stat().st_size, 1000)
        manifest = datasets.read_manifest("audio", "musicnet")
        self.assertEqual(manifest["items"][0]["composer"] in ("Bach", "Beethoven", "Mozart"), True)
        self.assertIn("CC BY", manifest["license"])
        self.assertFalse((self.tmp / "data" / "downloads" / "musicnet" / "musicnet.tar.gz").exists())   # archive removed after use
        self.assertEqual(sum(datasets.installed()["audio"]["musicnet"].values()), 4)
        self.assertEqual([p.name for p in datasets.files("audio", "musicnet", "test")], ["3.wav"])

    def test_keep_archive_and_unknown_ensembles(self):
        self.install(keep_archive=True)
        self.assertTrue((self.tmp / "data" / "downloads" / "musicnet" / "musicnet.tar.gz").is_file())
        with self.assertRaisesRegex(ValueError, "Solo Piano"):                             # lists what exists
            datasets.musicnet_violin_ids(self.tmp / "data" / "downloads" / "musicnet" / "musicnet_metadata.csv", ["Harp"])


class YouTube(WithDataDir):
    def test_videos_become_poses_and_audio_in_partitions_with_attribution(self):
        folder = self.tmp / "dl"
        folder.mkdir()
        wav = self.tmp / "a.wav"
        write_wav(wav, synth([(69, 0.1, 1.2)]))
        for vid in ("aaa", "bbb", "ccc"):
            make_video(folder / f"{vid}.mp4", wav, 2)

        def downloader(queries, count, dest, log):
            self.assertTrue(queries and count == 3)
            return [{"id": v, "title": f"Violin {v}", "uploader": "someone", "webpage_url": f"https://youtu.be/{v}",
                     "license": "Creative Commons Attribution license (reuse allowed)", "duration": 2, "path": folder / f"{v}.mp4"}
                    for v in ("aaa", "bbb", "ccc")]

        calls = []

        def pose_maker(video, audio_wav):
            calls.append(Path(video).stem)
            if Path(video).stem == "bbb":
                raise RuntimeError("VioPose failed: out of memory")
            pose = synth_pose(minuet_notes()[:6])
            pose.audio_used = True
            return pose

        done = datasets.install_youtube_cc(3, downloader=downloader, pose_maker=pose_maker, log=self.log.append)
        self.assertEqual({i["id"] for i in done}, {"aaa", "ccc"})                           # bbb was skipped, not fatal
        self.assertTrue(any("skipped bbb" in m for m in self.log))
        for i in done:
            pose = load_pose(datasets.dataset_dir("video", "youtube-cc") / i["file"])
            self.assertTrue(pose.audio_used)
            self.assertTrue((datasets.dataset_dir("audio", "youtube-cc") / i["file"].replace(".json", ".wav")).is_file())
            self.assertEqual((i["uploader"], i["url"].startswith("https://youtu.be/")), ("someone", True))
        self.assertEqual(datasets.read_manifest("video", "youtube-cc")["items"][0]["license"][:8], "Creative")
        calls.clear()
        datasets.install_youtube_cc(3, downloader=downloader, pose_maker=pose_maker, log=self.log.append)
        self.assertEqual(calls, ["bbb"])                                                    # finished videos are not redone


class Ingest(WithDataDir):
    def test_your_own_files_are_checked_copied_and_partitioned(self):
        src = self.tmp / "src"
        (src / "sub").mkdir(parents=True)
        pose = synth_pose(minuet_notes()[:6])
        full = np.zeros((len(pose), 18, 3))
        for j, name in enumerate(pose.joints):
            if name in VIOPOSE_JOINTS:
                full[:, VIOPOSE_JOINTS.index(name)] = pose.xyz[:, j]
        np.save(src / "take1.npy", full)
        np.save(src / "sub" / "take1.npy", full)                                            # same name in another folder
        np.save(src / "bad.npy", np.zeros((10, 5, 3)))                                       # wrong joint count
        items = datasets.ingest("viodat", "video", src, log=self.log.append)
        self.assertEqual(len(items), 2)
        self.assertEqual(len({i["id"] for i in items}), 2)                                  # no collision
        self.assertTrue(any("bad.npy" in m for m in self.log))
        self.assertEqual(sum(len(datasets.files("video", "viodat", s)) for s in datasets.SPLITS), 2)
        write_wav(src / "a.wav", synth([(69, 0.1, 1.0)]))
        self.assertEqual(len(datasets.ingest("mosa", "audio", src / "a.wav", log=self.log.append)), 1)


class TrainingFromTheDataFolder(WithDataDir):
    def put(self, kind, name, split, filename, writer):
        path = datasets.dataset_dir(kind, name) / split / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path)

    def test_video_model_trains_on_the_partitions_and_saves_into_models(self):
        for split, n in (("train", 3), ("val", 1), ("test", 1)):
            for k in range(n):
                self.put("video", "toy", split, f"{split}{k}.json", lambda p, k=k: save_pose(synth_pose(minuet_notes(), seed=k), p))
        train_fusion.main(["video", "--dataset", "toy", "--epochs", "1", "--z-dim", "4", "--batch", "8", "--device", "cpu"])
        ckpt = self.tmp / "models" / "fusion" / "video.pt"                                  # no --out: the models folder
        model, stats = load_checkpoint(ckpt, BUILDERS)
        self.assertEqual((model.z_dim, stats["dataset"]), (4, "toy"))
        self.assertTrue({"test_recon", "test_leakage", "val_recon"} <= set(stats))          # the test split was scored
        self.assertGreater(stats["test_windows"], 0)

    def test_audio_model_reads_the_cache_inside_the_data_folder(self):
        for split, n in (("train", 2), ("val", 1)):
            for k in range(n):
                self.put("audio", "toy", split, f"{split}{k}.wav",
                         lambda p, k=k: write_wav(p, synth([(60 + 2 * i + k, 0.1 + 0.4 * i, 0.4) for i in range(8)])))
        train_fusion.main(["audio", "--dataset", "toy", "--epochs", "1", "--z-dim", "4", "--batch", "8", "--device", "cpu"])
        self.assertTrue(list((self.tmp / "data" / "cache" / "audio").glob("*.npz")))
        model, stats = load_checkpoint(self.tmp / "models" / "fusion" / "audio.pt", BUILDERS)
        self.assertNotIn("test_recon", stats)                                               # no test split installed

    def test_missing_dataset_says_how_to_install_it(self):
        with self.assertRaises(SystemExit) as cm:
            train_fusion.main(["audio", "--dataset", "musicnet"])
        self.assertIn("install_data.py audio", str(cm.exception))

    def test_cli_lists_what_is_installed(self):
        self.put("audio", "toy", "train", "a.wav", lambda p: write_wav(p, synth([(69, 0.1, 1.0)])))
        datasets.write_manifest("audio", "toy", "test", "x", [])
        with mock.patch("builtins.print") as printed:
            install_data.main(["list"])
        text = "\n".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("audio/toy: train 1, val 0, test 0", text)
        self.assertIn(str(self.tmp / "data"), text)


if __name__ == "__main__":
    unittest.main()
