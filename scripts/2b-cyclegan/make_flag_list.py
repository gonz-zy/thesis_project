"""
make_flag_list.py -- list the images whose CycleGAN translation FAILED, and write the test list without them.

Rule (fixed before any Setup 4 result was seen): an artifact-domain image is "failed" if the mean absolute difference
between its before and after versions, measured in the CENTRE of the image (rows/cols 15 %-85 %), is above 20 grey levels
(translation_change.py, column mae_centre). Clean-domain images are never translated, so they are never flagged.
The rule uses only the images, never labels or classifier results.

Reads   <cleaned folder>\\translation_change.csv   (written by translation_change.py)
Writes  <labels folder>\\flagged_failed.csv        every flagged image (split, class, mae_centre, ssim_centre)
        <labels folder>\\test_not_failed.csv       testing_labels.csv without the flagged images, same columns
Use test_not_failed.csv with  evaluate.py --subset  for the sensitivity result. The main results keep all images.

Run (project root):
    python scripts\\2b-cyclegan\\make_flag_list.py
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import setup4_config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--change_csv", default=os.path.join(C.CLEAN_DIR, "translation_change.csv"))
    ap.add_argument("--threshold", type=float, default=20.0, help="grey levels, centre mean absolute change")
    ap.add_argument("--out_dir", default=C.LABELS)
    a = ap.parse_args()

    ch = pd.read_csv(a.change_csv)
    flagged = ch[ch["mae_centre"] > a.threshold].copy()
    os.makedirs(a.out_dir, exist_ok=True)
    flagged[["Image Index", "split", "class", "mae_centre", "ssim_centre"]].sort_values(
        "mae_centre", ascending=False).to_csv(os.path.join(a.out_dir, "flagged_failed.csv"), index=False)

    te = pd.read_csv(C.SRC_SPLIT_CSV["test"])
    keep = te[~te["Image Index"].isin(set(flagged["Image Index"]))]
    keep.to_csv(os.path.join(a.out_dir, "test_not_failed.csv"), index=False)

    print(f"rule: centre change > {a.threshold:g} grey levels; {len(ch)} artifact-domain images examined")
    print("\nflagged images by split and class:")
    print(pd.crosstab(flagged["split"], flagged["class"], margins=True).to_string() if len(flagged) else "  none")
    npos, nneg = int((keep["Finding Labels"] == "Pneumonia").sum()), int((keep["Finding Labels"] != "Pneumonia").sum())
    print(f"\ntest set without flagged images: {len(keep)} of {len(te)} images "
          f"({npos} pneumonia of {int((te['Finding Labels'] == 'Pneumonia').sum())}, {nneg} No Finding)")
    print(f"written: {os.path.join(a.out_dir, 'flagged_failed.csv')}\n         {os.path.join(a.out_dir, 'test_not_failed.csv')}")


if __name__ == "__main__":
    main()
