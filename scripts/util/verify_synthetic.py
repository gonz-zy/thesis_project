"""
verify_synthetic.py  --  READ-ONLY checks before Step 2 (combine real + synthetic).
Run from the project root (thesis_project):
    python scripts\2-stylegan2-ada\verify_synthetic.py
Changes nothing on disk.
"""
import os
import sys
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm

BASE = os.path.join("data", "2-processed")
SYN_DIR = os.path.join(BASE, "2A-input_for_stylegan2ada", "generated-pneumonia")
SYN_CSV = os.path.join(BASE, "synthetic_labels.csv")
LABEL_DIR = os.path.join(BASE, "2A-input_for_stylegan2ada", "labels")
REAL_CSVS = {
    "train": os.path.join(LABEL_DIR, "training_labels.csv"),
    "test": os.path.join(LABEL_DIR, "testing_labels.csv"),
    "val": os.path.join(LABEL_DIR, "validation_labels.csv"),
}
EXPECTED_N = 41972
failures = []


def check(ok, msg):
    print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not ok:
        failures.append(msg)


def sex_col(df):
    for c in ("Patient Gender", "Patient Sex"):
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------- 1. files
print("\n1) Files on disk vs synthetic_labels.csv")
syn = pd.read_csv(SYN_CSV)
files = sorted(f for f in os.listdir(SYN_DIR) if f.lower().endswith(".png"))
check(len(files) == EXPECTED_N, f"PNG count on disk = {len(files)} (expected {EXPECTED_N})")
check(len(syn) == EXPECTED_N, f"rows in synthetic_labels.csv = {len(syn)} (expected {EXPECTED_N})")
check(syn["Image Index"].is_unique, "Image Index unique in CSV")
disk, csv = set(files), set(syn["Image Index"])
check(disk == csv, f"disk and CSV names identical (only on disk: {len(disk - csv)}, only in CSV: {len(csv - disk)})")
if disk != csv:
    print("     e.g. only on disk:", sorted(disk - csv)[:3], "| only in CSV:", sorted(csv - disk)[:3])
check(all(f.startswith("SYNTH_4000_") for f in files), "all files named SYNTH_4000_XXXXX.png (renamed, no seedXXXX left)")

# numeric order check: SYNTH_4000_N should map to row N
nums = [int(os.path.splitext(f)[0].split("_")[-1]) for f in files]
check(sorted(nums) == list(range(1, EXPECTED_N + 1)), "file numbers are contiguous 1..41972")

# ---------------------------------------------------------------- 2. image mode
print("\n2) Image mode / size (all files)")
bad = []
modes = {}
for f in tqdm(files, desc="  reading headers"):
    with Image.open(os.path.join(SYN_DIR, f)) as im:
        key = (im.mode, im.size)
        modes[key] = modes.get(key, 0) + 1
        if key != ("L", (256, 256)):
            bad.append((f, key))
print("     mode/size counts:", modes)
check(len(bad) == 0, f"all images are mode L, 256x256 (offenders: {len(bad)})")
if bad:
    print("     first offenders:", bad[:5])

# ---------------------------------------------------------------- 3. metadata
print("\n3) Synthetic metadata sanity")
sc = sex_col(syn)
print(f"     columns in synthetic_labels.csv: {list(syn.columns)}")
check(sc is not None, "has a sex/gender column")
required = ["Image Index", "Finding Labels", "Patient ID", "Patient Age", "View Position"]
check(all(c in syn.columns for c in required), "has all required non-sex columns")
check(not syn.isna().any().any(), "no NaNs in synthetic CSV")
check((syn["Finding Labels"] == "Pneumonia").all(), "all synthetic labels == 'Pneumonia'")
check(syn["Patient ID"].astype(str).str.startswith("SYNTH_").all(), "all Patient IDs start with SYNTH_")

# ---------------------------------------------------------------- 4. distributions
print("\n4) Real (232 train pneumonia) vs synthetic distributions")
train = pd.read_csv(REAL_CSVS["train"])
tc = sex_col(train)
real_pn = train[train["Finding Labels"] == "Pneumonia"]
check(len(real_pn) == 232, f"real training pneumonia rows = {len(real_pn)} (expected 232)")
if sc and tc:
    print("     age (mean / std / min / max)")
    for name, df in (("real", real_pn), ("synth", syn)):
        a = df["Patient Age"]
        print(f"       {name:6s} {a.mean():6.2f} / {a.std():5.2f} / {a.min()} / {a.max()}")
    print("     sex / gender (proportion)")
    print("       real :", real_pn[tc].value_counts(normalize=True).round(3).to_dict())
    print("       synth:", syn[sc].value_counts(normalize=True).round(3).to_dict())
    print("     view position (proportion)")
    print("       real :", real_pn["View Position"].value_counts(normalize=True).round(3).to_dict())
    print("       synth:", syn["View Position"].value_counts(normalize=True).round(3).to_dict())
    # joint age+sex pairs must exist in real set (they were copied jointly)
    real_pairs = set(zip(real_pn["Patient Age"], real_pn[tc]))
    syn_pairs = set(zip(syn["Patient Age"], syn[sc]))
    check(syn_pairs <= real_pairs, "every synthetic (age, sex) pair exists among the real rows")

# ---------------------------------------------------------------- 5. ID overlap and column agreement
print("\n5) ID overlap and column agreement across CSVs")
real_ids = set()
cols = {"synthetic": list(syn.columns)}
for name, p in REAL_CSVS.items():
    d = pd.read_csv(p)
    real_ids |= set(d["Patient ID"].astype(str))
    cols[name] = list(d.columns)
check(len(real_ids & set(syn["Patient ID"].astype(str))) == 0, "no Patient ID overlap between real and synthetic")
print("     column sets per CSV:")
for k, v in cols.items():
    print(f"       {k:9s} sex-col = {sex_col(pd.DataFrame(columns=v))}")
same = len({tuple(sorted(v)) for v in cols.values()}) == 1
check(same, "all four CSVs have identical column names (rerun after your Patient Sex rename)")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for m in failures:
        print("  -", m)
    sys.exit(1)
print("ALL CHECKS PASSED")