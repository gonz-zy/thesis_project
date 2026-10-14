"""
make_stats.py -- pixel mean / std of a TRAINING set, as used for T.Normalize in the classifier.

Pipeline per image = exactly cxr_data.make_transform(train=False): grayscale -> Resize((224,224)) -> ToTensor [0,1].
Mean and std are pooled over every pixel of every image (population std, float64 sums).

Two uses
  1) CHECK on Setup 3 (does this script reproduce the old file? expected mean 0.4929, std 0.2934, n 84408):
       python scripts\\2b-cyclegan\\make_stats.py --csv data\\2-processed\\2A-input_for_stylegan2ada\\labels\\combined_training_labels.csv ^
              --out results\\stats_check_setup3.json --compare data\\2-processed\\train_stats.json
     (the CSV has a 'path' column; the old file is only READ, never written)
  2) Setup 4 (487 cleaned training images):
       python scripts\\2b-cyclegan\\make_stats.py --csv data\\2-processed\\2B-input_for_cyclegan\\cleaned_manifest.csv ^
              --split training --out data\\2-processed\\2B-input_for_cyclegan\\train_stats.json
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
