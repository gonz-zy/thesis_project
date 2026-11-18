"""
2_generate_synthetic_metadata.py

Creates synthetic_labels.csv for the 41,972 StyleGAN2-ADA pneumonia images,
matching the NIH ChestX-ray14 CSV column format exactly.

- Patient Age / Sex (and optionally View Position) are sampled from the REAL
  pneumonia rows of the TRAINING split only (232 rows). Each synthetic row copies
  the values from ONE randomly chosen real row, so age/Sex correlations are kept.
- Patient IDs are SYNTH_00001 ... SYNTH_41972 and are checked against real IDs.
- Random seed = 42.

Usage (from the project root):
    python 2_generate_synthetic_metadata.py
    python 2_generate_synthetic_metadata.py --rename          # rename files to SYNTH_4000_XXXXX.png
    python 2_generate_synthetic_metadata.py --view_mode sample
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

SEED = 42
EXPECTED_N = 41972
COLUMNS = [
    "Image Index",
    "Finding Labels",
    "Patient ID",
    "Patient Age",
    "Patient Sex",
    "View Position",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--synth_dir",
        default=os.path.join("data", "2-processed", "2A-input_for_stylegan2ada", "generated-pneumonia"),
    )
    p.add_argument(
        "--train_csv",
        default=os.path.join(
            "data", "2-processed", "2A-input_for_stylegan2ada", "labels", "training_labels.csv"
        ),
    )
    p.add_argument(
        "--out_csv", default=os.path.join("data", "2-processed", "synthetic_labels.csv")
    )
    p.add_argument(
        "--view_mode",
        choices=["PA", "sample"],
        default="PA",
        help="PA = all rows 'PA' (per handoff); sample = copy View Position from the sampled real row",
    )
    p.add_argument(
        "--rename",
        action="store_true",
        help="Rename image files to SYNTH_4000_00001.png ... (sorted order) before writing the CSV",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(SEED)
    
    real = pd.read_csv(args.train_csv)

    # ---- real pneumonia rows (training split only) ----
    missing = [c for c in COLUMNS if c not in real.columns]
    if missing:
        sys.exit(f"Missing columns in {args.train_csv}: {missing}")
    real_pneu = real[real["Finding Labels"] == "Pneumonia"].reset_index(drop=True)
    print(f"Real training pneumonia rows: {len(real_pneu)} (expected 232)")
    if len(real_pneu) == 0:
        sys.exit("No real pneumonia rows found.")

    # ---- synthetic image files ----
    # Numeric sort by seed: names are seed0000..seed9999, seed10000..seed41971
    # (plain alphabetical sort would scramble the order).
    def seed_of(fname):
        m = re.search(r"(\d+)", fname)
        if not m:
            sys.exit(f"No seed number in filename: {fname}")
        return int(m.group(1))

    files = sorted(
        (f for f in os.listdir(args.synth_dir) if f.lower().endswith(".png")), key=seed_of
    )
    print(f"Synthetic PNGs found: {len(files)} (expected {EXPECTED_N})")
    if len(files) != EXPECTED_N:
        sys.exit("File count mismatch; check --synth_dir.")
    if [seed_of(f) for f in files] != list(range(EXPECTED_N)):
        sys.exit("Seeds are not exactly 0..41971; some images are missing or duplicated.")

    n = len(files)
    if args.rename:
        new_names = [f"SYNTH_4000_{i:05d}.png" for i in range(1, n + 1)]
        for old, new in tqdm(list(zip(files, new_names)), desc="Renaming"):
            if old == new:
                continue
            dst = os.path.join(args.synth_dir, new)
            if os.path.exists(dst):
                sys.exit(f"Refusing to overwrite existing file: {dst}")
            os.rename(os.path.join(args.synth_dir, old), dst)
        files = new_names

    # ---- sample metadata jointly from real pneumonia rows ----
    idx = rng.integers(0, len(real_pneu), size=n)
    sampled = real_pneu.iloc[idx].reset_index(drop=True)

    synth = pd.DataFrame(
        {
            "Image Index": files,
            "Finding Labels": "Pneumonia",
            "Patient ID": [f"SYNTH_{i:05d}" for i in range(1, n + 1)],
            "Patient Age": sampled["Patient Age"].to_numpy(),
            "Patient Sex": sampled["Patient Sex"].to_numpy(),
            "View Position": "PA" if args.view_mode == "PA" else sampled["View Position"].to_numpy(),
        }
    )[COLUMNS]

    # ---- safety checks ----
    assert synth["Image Index"].is_unique, "Duplicate image names"
    assert synth["Patient ID"].is_unique, "Duplicate synthetic patient IDs"
    overlap = set(synth["Patient ID"].astype(str)) & set(real["Patient ID"].astype(str))
    assert not overlap, f"Synthetic IDs overlap real IDs: {list(overlap)[:5]}"

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    synth.to_csv(args.out_csv, index=False)

    # ---- summary: real vs synthetic distributions ----
    print(f"\nSaved {len(synth)} rows -> {args.out_csv}")
    print("\nAge (real pneumonia vs synthetic):")
    print(pd.DataFrame({"real": real_pneu["Patient Age"].describe(),
                        "synth": synth["Patient Age"].describe()}).round(2))
    print("\Sex share (real | synthetic):")
    print(pd.concat([real_pneu["Patient Sex"].value_counts(normalize=True),
                     synth["Patient Sex"].value_counts(normalize=True)], axis=1).round(3))
    print("\nView Position share (real | synthetic):")
    print(pd.concat([real_pneu["View Position"].value_counts(normalize=True),
                     synth["View Position"].value_counts(normalize=True)], axis=1).round(3))


if __name__ == "__main__":
    main()