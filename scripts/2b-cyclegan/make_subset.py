"""
make_subset.py -- builds the 676-image Setup 4 subset from the fixed patient-level split. Step 1 of Setup 4.

CPU only, a few seconds, writes only CSVs. Safe to run while a GPU job is training.

What it does
  * keeps EVERY pneumonia image of each split (232 / 31 / 59 = 322, Chehade's total)
  * draws No Finding images per split to reach the 354 : 322 ratio (255 / 34 / 65 = 354)
  * draws them from THAT SPLIT'S OWN patients, so the patient-level split is preserved
  * draws at most one image per patient, so no patient contributes several near-identical radiographs
  * writes the shared 676 CSV, the balanced 124-image test list and the 65-image small validation list
  * prints a patient summary and the checks the later scripts rely on

Outputs (all under data\\2-processed\\2B-input_for_cyclegan\\labels\\):
    subset_676.csv              the 676 images: Image Index, Finding Labels, Patient ID, clinical values,
                                split, src_path
    test_balanced_124.csv       the 59 + 65 test images (secondary table, comparable to Chehade)
    validation_small_65.csv     the 31 + 34 validation images (run B)

Run (project root):
    python scripts\\2b-cyclegan\\make_subset.py
    python scripts\\2b-cyclegan\\make_subset.py --seed 7 --dry_run

The draw is reproducible: the same --seed always selects the same images.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from setup4_config import (BALANCED_TEST_CSV, COLUMNS, LABELS, NOFINDING_PER_SPLIT, PNEUMONIA_PER_SPLIT,
                           SMALL_VAL_CSV, SRC_IMG_DIR, SRC_SPLIT_CSV, SUBSET_CSV, SUBSET_SEED, TOTAL_676,
                           make_dirs)

SPLITS = ["train", "validation", "test"]


def draw_nofinding(df, n, rng):
    """Pick n No Finding images, at most one per patient, from this split's own patients.

    Patients are shuffled and one image is taken from each (the first by Image Index, so the choice does not
    depend on CSV row order). One image per patient avoids several near-identical radiographs of the same
    person entering the 676 and dominating the clustering."""
    nf = df[df["Finding Labels"] == "No Finding"]
    patients = np.array(sorted(nf["Patient ID"].unique()))
    if len(patients) < n:
        raise SystemExit(f"only {len(patients)} No Finding patients available, need {n}")
    chosen = rng.choice(patients, size=n, replace=False)
    rows = [nf[nf["Patient ID"] == p].sort_values("Image Index").iloc[0] for p in chosen]
    return pd.DataFrame(rows)


def patient_summary(name, df):
    pn = df[df["Finding Labels"] == "Pneumonia"]
    nf = df[df["Finding Labels"] == "No Finding"]
    mixed = df.groupby("Patient ID")["Finding Labels"].nunique()
    print(f"  {name:<11} {len(df):>4} images | {df['Patient ID'].nunique():>4} patients | "
          f"pneumonia {len(pn):>3} from {pn['Patient ID'].nunique():>3} patients | "
          f"No Finding {len(nf):>3} from {nf['Patient ID'].nunique():>3} patients | "
          f"mixed-label patients {int((mixed > 1).sum()):>2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SUBSET_SEED, help="seed for the No Finding draw")
    ap.add_argument("--dry_run", action="store_true", help="print everything, write nothing")
    ap.add_argument("--check_images", action="store_true", help="also verify every source image exists on disk")
    args = ap.parse_args()

    make_dirs()
    rng = np.random.default_rng(args.seed)
    print(f"Setup 4 subset builder | No Finding draw seed {args.seed} | one image per patient\n")

    parts = []
    for split in SPLITS:
        src = pd.read_csv(SRC_SPLIT_CSV[split])
        src["Patient ID"] = src["Patient ID"].astype(str)  # string everywhere: the combined CSV uses strings

        pn = src[src["Finding Labels"] == "Pneumonia"].copy()
        want_pn = PNEUMONIA_PER_SPLIT[split]
        if len(pn) != want_pn:
            raise SystemExit(f"{split}: expected {want_pn} pneumonia images, found {len(pn)}")

        nf = draw_nofinding(src, NOFINDING_PER_SPLIT[split], rng).copy()

        part = pd.concat([pn, nf], ignore_index=True)
        part["split"] = split
        part["src_path"] = [os.path.join(SRC_IMG_DIR[split], f) for f in part["Image Index"]]
        parts.append(part[COLUMNS + ["split", "src_path"]])

    subset = pd.concat(parts, ignore_index=True).sort_values(["split", "Finding Labels", "Image Index"])
    subset = subset.reset_index(drop=True)

    # ---------------------------------------------------------------- checks
    print("PER SPLIT")
    for split in SPLITS:
        patient_summary(split, subset[subset["split"] == split])
    patient_summary("TOTAL", subset)

    n_pn = int((subset["Finding Labels"] == "Pneumonia").sum())
    n_nf = int((subset["Finding Labels"] == "No Finding").sum())
    print(f"\nCLASS TOTALS  pneumonia {n_pn} (Chehade 322) | No Finding {n_nf} (Chehade 354) | "
          f"total {len(subset)} (Chehade {TOTAL_676})")
    problems = []
    if n_pn != 322 or n_nf != 354 or len(subset) != TOTAL_676:
        problems.append("class totals do not match 322 / 354 / 676")
    if subset["Image Index"].duplicated().any():
        problems.append("duplicate Image Index values")

    sets = {s: set(subset.loc[subset["split"] == s, "Patient ID"]) for s in SPLITS}
    print("\nPATIENT OVERLAP BETWEEN SPLITS (must all be 0)")
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            n = len(sets[a] & sets[b])
            print(f"  {a} / {b}: {n}" + ("   <-- PROBLEM" if n else "   ok"))
            if n:
                problems.append(f"{a} and {b} share {n} patients")

    nf_per_patient = subset[subset["Finding Labels"] == "No Finding"].groupby("Patient ID").size()
    if (nf_per_patient > 1).any():
        problems.append("a No Finding patient contributed more than one image")
    print(f"\nNo Finding images per patient: max {int(nf_per_patient.max())} (1 by design)")

    pn_per_patient = subset[subset["Finding Labels"] == "Pneumonia"].groupby("Patient ID").size()
    print(f"Pneumonia images per patient: max {int(pn_per_patient.max())} "
          f"({int((pn_per_patient > 1).sum())} patients contribute more than one; all pneumonia images are kept)")

    print("\nCLINICAL VALUES (fed to the classifier head)")
    print(f"  Patient Sex   {subset['Patient Sex'].value_counts().to_dict()}")
    print(f"  View Position {subset['View Position'].value_counts().to_dict()}")
    age = subset["Patient Age"]
    print(f"  Patient Age   min {age.min()} median {age.median():.0f} max {age.max()}"
          + ("   <-- ages above 100 are clipped by CXRDataset" if age.max() > 100 else ""))

    if args.check_images:
        missing = [p for p in subset["src_path"] if not os.path.isfile(p)]
        print(f"\nSOURCE IMAGES  {len(subset) - len(missing)} of {len(subset)} found")
        if missing:
            problems.append(f"{len(missing)} source images missing, e.g. {missing[:3]}")

    if problems:
        print("\n" + "=" * 70)
        for p in problems:
            print("PROBLEM: " + p)
        raise SystemExit("not written")

    # ---------------------------------------------------------------- write
    bal_test = subset[subset["split"] == "test"]
    small_val = subset[subset["split"] == "validation"]
    if args.dry_run:
        print("\n--dry_run: nothing written")
        return
    os.makedirs(LABELS, exist_ok=True)
    subset.to_csv(SUBSET_CSV, index=False)
    bal_test.to_csv(BALANCED_TEST_CSV, index=False)
    small_val.to_csv(SMALL_VAL_CSV, index=False)
    print(f"\nWROTE  {SUBSET_CSV}  ({len(subset)} rows)")
    print(f"       {BALANCED_TEST_CSV}  ({len(bal_test)} rows: the Chehade-comparable test list)")
    print(f"       {SMALL_VAL_CSV}  ({len(small_val)} rows: run B's validation)")
    print("\nNext: features.py (HOG + 4 scalars at 1024), then cluster_kmeans.py.")


if __name__ == "__main__":
    main()
