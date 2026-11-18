"""
train_cyclegan.py -- trains ONE CycleGAN on the pooled artifact domain (A) and clean domain (B) of the 676 images.

Reads   clusters\\clustered_676.csv  (needs the final `domain` column; run label_grid.py --score first)
        Setup 3's 256 x 256 images (2A-input_for_stylegan2ada\\{training,validation,testing}), reused as they are
Writes  cyclegan\\cyclegan_latest.pt        every epoch (atomic overwrite)
        cyclegan\\cyclegan_epoch025.pt ...  every 25th epoch and the last one (the epoch-200 file is the one used)
        cyclegan\\training_history.csv, run_config.json, samples\\epoch_XXX.png

Recipe (Zhu et al. defaults): 200 epochs, lr 2e-4 constant for 100 then linear decay, Adam(0.5, 0.999), batch 1,
LSGAN, lambda_cycle 10, lambda_identity 5, image pool 50, ResNet-9 generators, 70 x 70 PatchGAN.
Augmentation: horizontal flip only, on the whole 256 x 256 image. One epoch = max(|A|, |B|) iterations; A is walked in a
shuffled order, B is drawn at random for each step (the reference recipe's unaligned sampling).
All 676 images are used, test images included, following Chehade et al. (stated limitation).

The 676 images live in memory (44 MB), so there are no DataLoader workers: a single seeded generator makes every
choice, which makes the run reproducible and removes the repeated-sample problem of the friend's per-worker RNG.

Run (project root, GPU free):
    venv\\Scripts\\python.exe scripts\\2b-cyclegan\\train_cyclegan.py --smoke        (8 + 8 images, 2 epochs, own folder)
    venv\\Scripts\\python.exe scripts\\2b-cyclegan\\train_cyclegan.py                (the real run)
    venv\\Scripts\\python.exe scripts\\2b-cyclegan\\train_cyclegan.py --resume auto   (continue after an interruption)
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cyclegan_model import CycleGAN
from setup4_config import (GAN_BATCH, GAN_BETAS, GAN_BLOCKS, GAN_DECAY_START, GAN_DIR, GAN_EPOCHS, GAN_LAMBDA_CYCLE,
                           GAN_LAMBDA_IDENTITY, GAN_LR, GAN_NDF_LAYERS, GAN_POOL, GAN_SAVE_EVERY, GAN_SEED, GAN_SIZE)
from setup4_images import load_256, read_domains, save_png

LATEST = "cyclegan_latest.pt"


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _sample_sheet(model, fixed_a, path):
    """Rows of [artifact | cleaned | |difference| x4] for a fixed set of artifact images."""
    x = fixed_a.float().div(127.5).sub(1).unsqueeze(1)
    y = model.clean(x).cpu()
    x = x.cpu()
    rows = []
    for a, b in zip(x, y):
        a8 = ((a[0] + 1) * 127.5).round().clamp(0, 255)
        b8 = ((b[0] + 1) * 127.5).round().clamp(0, 255)
        d8 = ((a8 - b8).abs() * 4).clamp(0, 255)
        rows.append(torch.cat([a8, b8, d8], 1))
    save_png(torch.cat(rows, 0).numpy().astype(np.uint8), path)


def run_training(imgs_a, imgs_b, out_dir, *, epochs=GAN_EPOCHS, decay_start=GAN_DECAY_START, seed=GAN_SEED,
                 device="cuda", resume=None, force=False, save_every=GAN_SAVE_EVERY, sample_every=25, extra_meta=None,
                 stop_after=None):
    """imgs_a / imgs_b: uint8 arrays (N, H, W). Returns the path of the last checkpoint.
    stop_after is for tests only: stop cleanly after that many epochs, as if the run had been interrupted."""
    assert imgs_a.dtype == np.uint8 and imgs_b.dtype == np.uint8 and imgs_a.shape[1:] == imgs_b.shape[1:]
    os.makedirs(out_dir, exist_ok=True)
    latest = os.path.join(out_dir, LATEST)
    if os.path.exists(latest) and not resume and not force:
        raise SystemExit(f"{latest} exists. Use --resume auto to continue it, or --force to start over (it is overwritten).")

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    model = CycleGAN(device, epochs=epochs, decay_start=decay_start, lr=GAN_LR, betas=GAN_BETAS,
                     lambda_cycle=GAN_LAMBDA_CYCLE, lambda_identity=GAN_LAMBDA_IDENTITY, pool_size=GAN_POOL,
                     n_blocks=GAN_BLOCKS, n_layers_d=GAN_NDF_LAYERS, seed=seed)
    assert GAN_BATCH == 1, "the reference recipe (and this loop) uses a batch of 1"

    start, history = 0, []
    hist_csv = os.path.join(out_dir, "training_history.csv")
    if resume:
        ck_path = latest if resume == "auto" else resume
        if not os.path.exists(ck_path):
            raise SystemExit(f"--resume: {ck_path} not found")
        ck = torch.load(ck_path, map_location=device, weights_only=True)
        model.load_state_dict(ck)
        rng.bit_generator.state = ck["sampler_state"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        start = int(ck["epoch"])
        if os.path.exists(hist_csv):
            history = [r for r in pd.read_csv(hist_csv).to_dict("records") if r["epoch"] <= start]
        print(f"resumed from {ck_path} at epoch {start}", flush=True)
        if start >= epochs:
            print("already finished")
            return latest

    A = torch.from_numpy(imgs_a).to(device)           # (nA, H, W) uint8, kept on the device
    B = torch.from_numpy(imgs_b).to(device)
    nA, nB, H, W = len(A), len(B), A.shape[1], A.shape[2]
    iters = max(nA, nB)
    fixed_a = A[:min(6, nA)].clone()
    meta = dict(epochs=epochs, decay_start=decay_start, seed=seed, n_A=nA, n_B=nB, iters_per_epoch=iters, size=[H, W],
                device=str(device), torch=str(torch.__version__), **(extra_meta or {}))
    json.dump(meta, open(os.path.join(out_dir, "run_config.json"), "w"), indent=2)
    print(f"A (artifact) {nA}   B (clean) {nB}   {iters} iterations/epoch   {epochs} epochs   image {H}x{W}   device {device}",
          flush=True)

    t_start = time.time()
    for ep in range(start, epochs):
        t0 = time.time()
        order = rng.permutation(iters)
        ia, ib = order % nA, rng.integers(0, nB, iters)
        fa, fb = rng.random(iters) < 0.5, rng.random(iters) < 0.5      # horizontal flips, the only augmentation
        sums = {}
        for k in range(iters):
            a = A[int(ia[k])].float().div(127.5).sub(1).view(1, 1, H, W)
            b = B[int(ib[k])].float().div(127.5).sub(1).view(1, 1, H, W)
            if fa[k]:
                a = a.flip(-1)
            if fb[k]:
                b = b.flip(-1)
            for name, v in model.optimize(a, b).items():
                sums[name] = sums[name] + v if name in sums else v.clone()
        means = {n: float(v) / iters for n, v in sums.items()}
        if not all(np.isfinite(list(means.values()))):
            raise SystemExit(f"epoch {ep + 1}: a loss is not finite {means}. Training stopped; the last good checkpoint is {latest}")
        row = dict(epoch=ep + 1, lr=model.lr, seconds=round(time.time() - t0, 1), **{n: round(v, 5) for n, v in means.items()})
        history.append(row)
        model.step_schedulers()
        pd.DataFrame(history).to_csv(hist_csv, index=False)

        done = ep + 1
        elapsed, per = time.time() - t_start, (time.time() - t_start) / (done - start)
        print(f"epoch {done:>3}/{epochs}  G {means['G_total']:.3f}  D_A {means['D_A']:.3f}  D_B {means['D_B']:.3f}  "
              f"cyc {means['cycle_A'] + means['cycle_B']:.3f}  idt {means['idt_A'] + means['idt_B']:.3f}  "
              f"lr {row['lr']:.2e}  {row['seconds']:.0f}s  ({iters / row['seconds']:.1f} it/s)  "
              f"ETA {(epochs - done) * per / 60:.0f} min", flush=True)

        if done == 1 or done % sample_every == 0 or done == epochs:
            _sample_sheet(model, fixed_a, os.path.join(out_dir, "samples", f"epoch_{done:03d}.png"))
        ck = model.state_dict()
        ck.update(epoch=done, sampler_state=rng.bit_generator.state, torch_rng=torch.get_rng_state(), meta=meta)
        _atomic_save(ck, latest)
        if done % save_every == 0 or done == epochs:
            shutil.copyfile(latest, os.path.join(out_dir, f"cyclegan_epoch{done:03d}.pt"))
        if stop_after and done >= stop_after:
            break
    print(f"finished {epochs} epochs in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return latest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=GAN_EPOCHS)
    ap.add_argument("--decay_start", type=int, default=None, help="default: half of --epochs (100 for 200)")
    ap.add_argument("--seed", type=int, default=GAN_SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=GAN_DIR)
    ap.add_argument("--resume", default=None, help="'auto' = continue cyclegan_latest.pt, or a checkpoint path")
    ap.add_argument("--force", action="store_true", help="start over even if a checkpoint exists")
    ap.add_argument("--smoke", action="store_true", help="8 + 8 images, 2 epochs, writes to <out>\\_smoke; checks speed")
    ap.add_argument("--allow_resize_from_1024", action="store_true")
    args = ap.parse_args()

    df = read_domains()
    if args.smoke:
        df = pd.concat([df[df.domain == "artifact"].head(8), df[df.domain == "clean"].head(8)])
        args.epochs, args.out, args.force = 2, os.path.join(args.out, "_smoke"), True
    decay = args.decay_start if args.decay_start is not None else (GAN_DECAY_START if args.epochs == GAN_EPOCHS else args.epochs // 2)

    sets = {}
    for dom in ("artifact", "clean"):
        sub = df[df.domain == dom].sort_values("Image Index")
        sets[dom] = (sub["Image Index"].tolist(),
                     np.stack([load_256(r, args.allow_resize_from_1024) for _, r in sub.iterrows()]))
        print(f"{dom}: {len(sub)} images, shape {sets[dom][1].shape}")
    assert sets["artifact"][1].shape[1:] == (GAN_SIZE, GAN_SIZE)
    run_training(sets["artifact"][1], sets["clean"][1], args.out, epochs=args.epochs, decay_start=decay, seed=args.seed,
                 device=args.device, resume=args.resume, force=args.force,
                 extra_meta=dict(names_A=sets["artifact"][0], names_B=sets["clean"][0]))


if __name__ == "__main__":
    main()
