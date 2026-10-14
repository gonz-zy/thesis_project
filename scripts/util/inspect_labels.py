"""
inspect_labels.py -- prints the structure of the label CSVs so the Setup 4 subset builder can be written correctly.

Reads nothing but CSVs, uses no GPU and takes seconds, so it is safe to run while Setup 1 is training.
Prints, for every label file found: its columns and dtypes, row and class counts, patient counts, a few example
rows (with the Patient ID column blanked), and the checks the Setup 4 subset builder depends on:
  * is there a Patient ID column, and is it the same type everywhere
  * do train / validation / test share any patient
  * how many No Finding images and patients each split has (enough to draw 255 / 34 / 65 from)
  * how many patients carry BOTH Pneumonia and No Finding images (mixed labels)
  * whether Image Index values are unique and carry a .png extension

Run from the project root:
    python scripts\\util\\inspect_labels.py
Paste the whole output into the chat. No file is written and nothing is modified.
"""
import os

import pandas as pd

ROOT = os.path.join("data", "2-processed", "2A-input_for_stylegan2ada")
LABELS = os.path.join(ROOT, "labels")
PROC = os.path.join("data", "2-processed")

CANDIDATES = [
    os.path.join(LABELS, "training_labels.csv"),
    os.path.join(LABELS, "validation_labels.csv"),
    os.path.join(LABELS, "testing_labels.csv"),
    os.path.join(LABELS, "combined_training_labels.csv"),
    os.path.join(PROC, "filtered_labels.csv"),
    os.path.join(PROC, "synthetic_labels.csv"),
]

PID = "Patient ID"
IDX = "Image Index"
LAB = "Finding Labels"


def describe(path):
    print("\n" + "=" * 78)
    print(path)
    print("=" * 78)
    df = pd.read_csv(path)
    print(f"rows: {len(df):,}")
    print("\ncolumns and dtypes:")
    for c in df.columns:
        n_na = int(df[c].isna().sum())
        print(f"    {c:<24} {str(df[c].dtype):<10} {n_na:>6} empty")

    if LAB in df.columns:
        print(f"\n{LAB} value counts:")
        for v, n in df[LAB].value_counts().items():
            print(f"    {str(v):<24} {n:>8,}")

    if PID in df.columns:
        print(f"\n{PID}: {df[PID].nunique():,} unique, dtype {df[PID].dtype}, "
              f"example values {df[PID].dropna().unique()[:3].tolist()}")
        if LAB in df.columns:
            per = df.groupby(PID)[LAB].nunique()
            print(f"    patients with BOTH labels (mixed): {int((per > 1).sum()):,}")
            ipp = df.groupby(PID).size()
            print(f"    images per patient: min {ipp.min()}, median {ipp.median():.0f}, max {ipp.max()}")
            nf = df[df[LAB] == "No Finding"]
            if len(nf):
                print(f"    No Finding: {len(nf):,} images from {nf[PID].nunique():,} patients")
    else:
        print(f"\n*** no '{PID}' column ***")

    if IDX in df.columns:
        print(f"\n{IDX}: {df[IDX].nunique():,} unique of {len(df):,} rows"
              f"{'  (DUPLICATES)' if df[IDX].nunique() != len(df) else ''}")
        ex = df[IDX].iloc[0]
        print(f"    example: {ex!r}   ends with .png: {str(ex).lower().endswith('.png')}")
    if "path" in df.columns:
        print(f"    path example: {df['path'].iloc[0]!r}")
    if "is_synthetic" in df.columns:
        print(f"    is_synthetic counts: {df['is_synthetic'].value_counts().to_dict()}")

    show = [c for c in [IDX, LAB, "Patient Age", "Patient Sex", "View Position"] if c in df.columns]
    if show:
        print("\nfirst 3 rows (Patient ID not shown):")
        print(df[show].head(3).to_string(index=False))
    return df


def main():
    found = {}
    for p in CANDIDATES:
        if os.path.exists(p):
            try:
                found[p] = describe(p)
            except Exception as e:  # noqa: BLE001
                print(f"\n{p}: could not read ({e})")
        else:
            print(f"[not found] {p}")

    others = []
    for d in (LABELS, PROC):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if f.lower().endswith(".csv") and p not in CANDIDATES:
                    others.append(p)
    if others:
        print("\n" + "=" * 78)
        print("other CSV files present (not read):")
        for p in others:
            print("    " + p)

    # ---- the checks the Setup 4 subset builder depends on
    print("\n" + "=" * 78)
    print("SPLIT CHECKS")
    print("=" * 78)
    splits = {}
    for name, fn in [("train", "training_labels.csv"), ("validation", "validation_labels.csv"),
                     ("test", "testing_labels.csv")]:
        p = os.path.join(LABELS, fn)
        if p in found and PID in found[p].columns:
            splits[name] = found[p]
    if len(splits) < 2:
        print("not enough split files with a Patient ID column to check overlap")
    else:
        names = list(splits)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                shared = set(splits[a][PID]) & set(splits[b][PID])
                print(f"    {a} / {b}: {len(shared)} shared patients"
                      + ("  <-- PROBLEM" if shared else "  ok"))
        print("\n    No Finding available per split (Setup 4 needs 255 train / 34 validation / 65 test):")
        for name, df in splits.items():
            nf = df[df[LAB] == "No Finding"] if LAB in df.columns else df.iloc[0:0]
            pn = df[df[LAB] == "Pneumonia"] if LAB in df.columns else df.iloc[0:0]
            print(f"      {name:<11} No Finding {len(nf):>7,} images / {nf[PID].nunique():>6,} patients"
                  f"   |   Pneumonia {len(pn):>5,} images / {pn[PID].nunique():>5,} patients")

    print("\nDone. Paste this whole output into the chat.")


if __name__ == "__main__":
    main()
