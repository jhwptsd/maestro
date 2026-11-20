"""Train one fusion encoder: python src/training/train_fusion.py {audio,video,score} PATH [PATH ...] [options]

PATH is a file or a folder (searched recursively) of audio/video recordings (audio), pose files (video) or scores (score);
see fusion_data.py for the formats and for suggested public datasets. `--music21` adds the scores bundled with music21.
The model is a residual autoencoder (maestro/fusion/residual_autoencoder.py): the decoder sees the analysis features
as well, so the small latent z is pushed to hold what the analysis does not. Each epoch prints the validation
reconstruction loss and `leakage` (R^2 of reading the analysis features back from z: lower = z repeats less of it). The best
epoch (lowest validation loss) is saved to <out>/<kind>.pt, which maestro.analyze() picks up from models/fusion/.

    python src/training/train_fusion.py score --music21 --epochs 20
    python src/training/train_fusion.py audio recordings/ --cache cache/ --epochs 30
    python src/training/train_fusion.py video poses/ --epochs 40
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import fusion_data  # noqa: E402
from maestro.fusion import audio_encoder, score_encoder, video_encoder  # noqa: E402
from maestro.fusion.embed import models_dir  # noqa: E402
from maestro.fusion.residual_autoencoder import leakage_r2, save_checkpoint  # noqa: E402

MODULES = {"audio": audio_encoder, "video": video_encoder, "score": score_encoder}


def batches(x, side, size, rng):
    """Shuffled minibatches of tensors."""
    order = rng.permutation(len(x))
    for i in range(0, len(x), size):
        idx = order[i:i + size]
        yield x[idx], side[idx]


@torch.no_grad()
def evaluate(model, x, side, loss_fn, beta, leak, batch=256):
    """Mean validation loss parts, and the leakage R^2 of z (posterior mean) against the side features."""
    model.eval()
    parts, zs, n = {"recon": 0.0, "kl": 0.0, "xcov": 0.0}, [], 0
    for i in range(0, len(x), batch):
        xb, sb = x[i:i + batch], side[i:i + batch]
        _, p = model.loss(xb, sb, loss_fn, beta, leak)
        for k in parts:
            parts[k] += p[k] * len(xb)
        zs.append(model.encode(xb))
        n += len(xb)
    parts = {k: v / n for k, v in parts.items()}
    return parts, leakage_r2(torch.cat(zs), side)


def fit(model, train, val, loss_fn, epochs=30, batch=64, lr=1e-3, beta=1e-3, leak=1.0, device="cpu", log=print, seed=0):
    """Train `model` on `train` = (x, side) numpy arrays; keep the weights of the best validation epoch (validation =
    (x, side), or None to keep the last). -> history list of per-epoch dicts."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    as_tensor = lambda a: torch.as_tensor(a, device=device)
    xt, st = as_tensor(train[0]), as_tensor(train[1])
    xv, sv = (as_tensor(val[0]), as_tensor(val[1])) if val is not None and len(val[0]) else (None, None)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, history = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, run, count = time.time(), 0.0, 0
        for xb, sb in batches(xt, st, batch, rng):
            loss, _ = model.loss(xb, sb, loss_fn, beta, leak)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run, count = run + float(loss.detach()) * len(xb), count + len(xb)
        row = {"epoch": epoch, "train": run / count}
        if xv is not None:
            parts, leakage = evaluate(model, xv, sv, loss_fn, beta, leak)
            row.update(val_recon=parts["recon"], val_kl=parts["kl"], val_xcov=parts["xcov"], leakage=leakage)
            score = parts["recon"] + beta * parts["kl"] + leak * parts["xcov"]
        else:
            score = row["train"]
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
            row["best"] = True
        history.append(row)
        log(f"epoch {epoch:3d}  train {row['train']:.4f}" + (f"  val recon {row['val_recon']:.4f}  xcov {row['val_xcov']:.4f}  leakage "
            f"{row['leakage']:.3f}" if xv is not None else "") + f"  ({time.time() - t0:.1f}s){'  *' if row.get('best') else ''}")
    model.load_state_dict(best_state)
    return history


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=sorted(MODULES))
    p.add_argument("paths", nargs="*", help="files or folders (recursively) of the right kind")
    p.add_argument("--music21", action="store_true", help="(score) also use the scores bundled with music21")
    p.add_argument("--limit", type=int, help="use at most this many files")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--beta", type=float, default=1e-3, help="weight of the KL term")
    p.add_argument("--leak", type=float, default=1.0, help="weight of the z/side cross-covariance penalty")
    p.add_argument("--val", type=float, default=0.15, help="fraction of files held out for validation")
    p.add_argument("--cache", help="(audio) folder to cache the analysed windows in")
    p.add_argument("--pitch-backend", default="yin", help="(audio) pitch backend for the side features (default yin: fast)")
    p.add_argument("--fps", type=float, default=30.0, help="(video) frame rate of .npy motion-capture arrays")
    p.add_argument("--out", help="checkpoint folder (default: models/fusion)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    files = [f for path in args.paths for f in fusion_data.find(path, args.kind)]
    if args.music21 and args.kind == "score":
        files += fusion_data.music21_scores()
    files = files[:args.limit] if args.limit else files
    if not files:
        p.error(f"no {args.kind} files found (give PATHs{', or --music21' if args.kind == 'score' else ''})")
    print(f"reading {len(files)} {args.kind} files ...")
    xs, sides = fusion_data.load_windows(args.kind, files, cache_dir=args.cache, pitch_backend=args.pitch_backend, fps=args.fps)
    train, val = fusion_data.split(xs, sides, args.val)
    print(f"{len(train[0])} training windows, {len(val[0])} validation windows from {len(xs)} files")

    module = MODULES[args.kind]
    model = module.build(z_dim=args.z_dim)
    history = fit(model, train, val, module.reconstruction_loss, args.epochs, args.batch, args.lr, args.beta, args.leak, args.device)
    out = Path(args.out) if args.out else models_dir()
    out.mkdir(parents=True, exist_ok=True)
    best = min((h for h in history if "val_recon" in h), key=lambda h: h["val_recon"] + args.beta * h["val_kl"] + args.leak * h["val_xcov"],
               default=history[-1])
    save_checkpoint(model, out / f"{args.kind}.pt", {"files": len(xs), "train_windows": len(train[0]), "epochs": args.epochs,
                                                       "best_epoch": best["epoch"], **{k: v for k, v in best.items() if k.startswith(("val_", "leak"))}})
    print(f"saved {out / (args.kind + '.pt')}")


if __name__ == "__main__":
    main()
