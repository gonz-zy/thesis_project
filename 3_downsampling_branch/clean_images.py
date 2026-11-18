"""
Stage 3, step 7: writes the 224 x 224 images the Setup 4 classifier reads.
Artifact images go through G_AB (epoch-100 checkpoint), clean images are only resized.

Input:  a manifest with Image Index, split, domain (the paper used output/clusters/clustered_all.csv),
        output/cyclegan/cyclegan_epoch100.pt, 2_upsampling_branch/output/{training,validation,testing}/
Output: 3_downsampling_branch/output/cleaned/224/, translated_256/, cleaned_manifest.csv, clean_report.json
Run:    python 3_downsampling_branch/clean_images.py --manifest 3_downsampling_branch/output/clusters/clustered_all.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cyclegan_model import load_generator
from setup4_config import CLEAN_DIR, GAN_DIR, GAN_EPOCH_USED, IMG
from setup4_images import from_model, load_256, read_domains, save_png, to_224


@torch.no_grad()
def translate(G, arrs, device, batch=8):
    """List of uint8 (256, 256) arrays -> list of uint8 arrays after G_AB, rounded to 8 bit."""
    out = []
    for i in range(0, len(arrs), batch):
        x = torch.from_numpy(np.stack(arrs[i:i + batch])).to(device).float().div(127.5).sub(1).unsqueeze(1)
        y = from_model(G(x))
        out += [y[j, 0] for j in range(len(y))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="CSV with Image Index, split, domain (default: clustered_676.csv)")
    ap.add_argument("--checkpoint", default=os.path.join(GAN_DIR, f"cyclegan_epoch{GAN_EPOCH_USED:03d}.pt"))
    ap.add_argument("--out", default=CLEAN_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_resize_from_1024", action="store_true")
    ap.add_argument("--verify_setup3_dir", default=None,
                    help="folder holding Setup 3's cxr_data.py; compares every clean image with its make_transform output")
    args = ap.parse_args()

    df = read_domains(args.manifest) if args.manifest else read_domains()
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"{args.checkpoint} not found. Train first (train_cyclegan.py), or pass --checkpoint.")
    G, epoch = load_generator(args.checkpoint, args.device)
    if epoch != GAN_EPOCH_USED:
        print(f"WARNING: this checkpoint is epoch {epoch}, not epoch {GAN_EPOCH_USED} used in the paper")
    d224, d256 = os.path.join(args.out, "224"), os.path.join(args.out, "translated_256")
    os.makedirs(d224, exist_ok=True)
    os.makedirs(d256, exist_ok=True)
    print(f"{len(df)} images ({(df.domain == 'artifact').sum()} artifact, {(df.domain == 'clean').sum()} clean), "
          f"generator epoch {epoch}, device {args.device}")

    rows = df.to_dict("records")
    arr = [load_256(r, args.allow_resize_from_1024) for r in rows]
    is_art = np.array([r["domain"] == "artifact" for r in rows])
    final256 = list(arr)
    idx = np.where(is_art)[0]
    changed = {}
    if len(idx):
        gen = translate(G, [arr[i] for i in idx], args.device)
        for i, g in zip(idx, gen):
            final256[i] = g
            diff = np.abs(g.astype(np.int16) - arr[i].astype(np.int16))
            changed[rows[i]["Image Index"]] = (float(diff.mean()), float((diff > 0).mean()))
            save_png(g, os.path.join(d256, rows[i]["Image Index"]))

    paths = []
    for r, a in zip(rows, final256):
        p = os.path.join(d224, r["Image Index"])
        a224 = to_224(a)
        assert a224.shape == (IMG, IMG) and a224.dtype == np.uint8
        save_png(a224, p)
        paths.append(p)

    out = df.copy()
    out["translated"] = is_art
    out["path"] = paths
    out.to_csv(os.path.join(args.out, "cleaned_manifest.csv"), index=False)

    rep = dict(n=len(out), n_translated=int(is_art.sum()), n_passthrough=int((~is_art).sum()), checkpoint=args.checkpoint,
               checkpoint_epoch=epoch, by_split=out.groupby(["split", "domain"]).size().unstack(fill_value=0).to_dict())
    if changed:
        m = np.array([v[0] for v in changed.values()])
        f = np.array([v[1] for v in changed.values()])
        rep["translated_mean_abs_grey_change"] = dict(mean=float(m.mean()), min=float(m.min()), max=float(m.max()))
        rep["translated_fraction_pixels_changed"] = dict(mean=float(f.mean()), min=float(f.min()), max=float(f.max()))
        print(f"G_AB changed the artifact images by {m.mean():.2f} grey levels on average (min {m.min():.2f}, max {m.max():.2f}); "
              f"{100 * f.mean():.0f}% of their pixels differ")
        if m.mean() < 0.5:
            print("WARNING: the generator hardly changes anything; look at the samples before using these images")
        if m.max() > 40:
            print("WARNING: at least one image changed by more than 40 grey levels on average; look at it")
    json.dump(rep, open(os.path.join(args.out, "clean_report.json"), "w"), indent=2)

    if args.verify_setup3_dir:
        sys.path.insert(0, args.verify_setup3_dir)
        import cxr_data
        from PIL import Image
        bad = 0
        tf = cxr_data.make_transform(False)
        clean_rows = [(r, p) for r, p, a in zip(rows, paths, is_art) if not a]
        for r, p in clean_rows:
            src = os.path.join(cxr_data.ROOT, {"train": "training", "validation": "validation", "test": "testing"}[r["split"]],
                               r["Image Index"])
            with Image.open(src) as im:
                ref = tf(im.convert("L"))
            with Image.open(p) as im:
                mine = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).float().div(255).unsqueeze(0)
            if not torch.equal(ref, mine):
                bad += 1
        print(f"verify: {len(clean_rows) - bad}/{len(clean_rows)} clean images are identical to Setup 3's make_transform output")
        if bad:
            raise SystemExit(f"{bad} clean images differ from Setup 3's pipeline; do not train on these")
    print(f"wrote {len(out)} images to {d224}\n      {os.path.join(args.out, 'cleaned_manifest.csv')}")


if __name__ == "__main__":
    main()
