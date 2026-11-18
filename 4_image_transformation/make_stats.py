"""
Stage 4: pixel mean and standard deviation of a training set (224 x 224, [0, 1]) for standardization.

Input:  a CSV with a `path` column (or --img_root)
Output: a JSON with mean and std
Run:    Setups 1 and 3 (gives the same mean and SD as the saved file, to about 1e-14):
          python 4_image_transformation/make_stats.py --csv 2_upsampling_branch/output/labels/combined_training_labels.csv
                 --out 4_image_transformation/output/train_stats_setup1_setup3.json
        Setup 4:
          python 4_image_transformation/make_stats.py --csv 3_downsampling_branch/output/cleaned/cleaned_manifest.csv
                 --split training --out 4_image_transformation/output/train_stats_setup4.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image

IMG = 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default=None, help="keep only rows whose 'split' column equals this (e.g. training)")
    ap.add_argument("--img_root", default=None, help="used only if the CSV has no 'path' column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="existing stats json to compare with (read only)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    if a.split:
        df = df[df["split"].astype(str).str.lower().str.startswith(a.split.lower()[:5])]
    if a.limit:
        df = df.head(a.limit)
    if "path" in df.columns:
        paths = df["path"].tolist()
    elif a.img_root:
        paths = [os.path.join(a.img_root, f) for f in df["Image Index"]]
    else:
        raise SystemExit("CSV has no 'path' column: give --img_root")
    if not len(paths):
        raise SystemExit("no images selected")

    tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor()])
    s = s2 = 0.0
    n_pix = 0
    img_means = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            x = tf(im.convert("L")).double().numpy()
        s += x.sum()
        s2 += (x ** 2).sum()
        n_pix += x.size
        img_means.append(x.mean())
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)
    mean = s / n_pix
    std = float(np.sqrt(s2 / n_pix - mean ** 2))
    out = dict(mean=float(mean), std=std, n=len(paths), n_pixels=int(n_pix),
               std_of_image_means=float(np.std(img_means)), source_csv=a.csv, split=a.split)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    if a.compare:
        old = json.load(open(a.compare))
        print(f"\nOLD  mean {old['mean']:.4f}  std {old['std']:.4f}  n {old.get('n')}")
        print(f"NEW  mean {mean:.4f}  std {std:.4f}  n {len(paths)}")
        ok = abs(old["mean"] - mean) < 5e-4 and abs(old["std"] - std) < 5e-4
        print("MATCH (to 3 decimals)" if ok else "DIFFERENT: the old file was made some other way; do not reuse it for Setup 4")


if __name__ == "__main__":
    main()
