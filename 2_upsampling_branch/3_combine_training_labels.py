"""
Stage 2: training set of Setups 1 and 3 = 42,436 real training images + 41,972 synthetic Pneumonia images.
Rebuilt after the fact: it reproduces the saved combined_training_labels.csv row for row.

Input:  2_upsampling_branch/output/labels/training_labels.csv, synthetic_labels.csv
Output: 2_upsampling_branch/output/labels/combined_training_labels.csv  (84,408 rows, columns below)
Run:    python 2_upsampling_branch/3_combine_training_labels.py
"""
import os

import pandas as pd

OUT_ROOT = os.path.join("2_upsampling_branch", "output")
LABELS = os.path.join(OUT_ROOT, "labels")
COLUMNS = ["Image Index", "Finding Labels", "Patient ID", "Patient Age", "Patient Sex", "View Position"]


def main():
    real = pd.read_csv(os.path.join(LABELS, "training_labels.csv"))[COLUMNS]
    real["is_synthetic"] = 0
    real["path"] = [os.path.join(OUT_ROOT, "training", f) for f in real["Image Index"]]

    synth = pd.read_csv(os.path.join(LABELS, "synthetic_labels.csv"))[COLUMNS]
    synth["is_synthetic"] = 1
    synth["path"] = [os.path.join(OUT_ROOT, "generated-pneumonia", f) for f in synth["Image Index"]]

    combined = pd.concat([real, synth], ignore_index=True)  # real rows first, then synthetic, both in file order
    assert combined["Image Index"].is_unique
    out = os.path.join(LABELS, "combined_training_labels.csv")
    combined.to_csv(out, index=False)
    print(f"{len(real)} real + {len(synth)} synthetic = {len(combined)} rows -> {out}")
    print(combined["Finding Labels"].value_counts().to_string())


if __name__ == "__main__":
    main()
