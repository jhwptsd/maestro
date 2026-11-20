"""Install and inspect the training data: python src/training/install_data.py COMMAND

    list                                  what is installed (data/), per dataset and split
    audio [--dataset musicnet]            MusicNet violin recordings -> data/audio/musicnet/{train,val,test}/
    video [--dataset youtube-cc] [--count 60] [--query "..."]...
                                          Creative Commons violin videos -> VioPose poses -> data/video/youtube-cc/{train,val,test}/
                                          (also data/audio/youtube-cc/: usable by the audio model)
    ingest NAME {audio,video} FOLDER      add files you obtained yourself (VioDat, SyncViolinist, MOSA ...), see datasets.py

then train: python src/training/train_fusion.py audio --dataset musicnet
            python src/training/train_fusion.py video --dataset youtube-cc
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import datasets  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    a = sub.add_parser("audio")
    a.add_argument("--dataset", default="musicnet", choices=["musicnet"])
    a.add_argument("--keep-archive", action="store_true", help="keep musicnet.tar.gz after extracting the violin recordings")
    a.add_argument("--ensemble", action="append", help="MusicNet ensemble to include (default: every ensemble with 'violin' in it)")
    v = sub.add_parser("video")
    v.add_argument("--dataset", default="youtube-cc", choices=["youtube-cc"])
    v.add_argument("--count", type=int, default=60, help="how many videos to download and process")
    v.add_argument("--query", action="append", help="YouTube search (repeatable; default: several violin searches)")
    i = sub.add_parser("ingest")
    i.add_argument("name")
    i.add_argument("kind", choices=["audio", "video"])
    i.add_argument("folder")
    args = p.parse_args(argv)

    if args.command == "list":
        print(f"data folder: {datasets.data_dir()}")
        for kind, sets in datasets.installed().items():
            for name, counts in sets.items():
                print(f"  {kind}/{name}: " + ", ".join(f"{s} {n}" for s, n in counts.items()))
        if not datasets.installed():
            print("  nothing installed yet")
    elif args.command == "audio":
        datasets.install_musicnet(ensembles=args.ensemble, keep_archive=args.keep_archive)
    elif args.command == "video":
        datasets.install_youtube_cc(args.count, tuple(args.query) if args.query else datasets.YOUTUBE_QUERIES)
    else:
        datasets.ingest(args.name, args.kind, args.folder)


if __name__ == "__main__":
    main()
