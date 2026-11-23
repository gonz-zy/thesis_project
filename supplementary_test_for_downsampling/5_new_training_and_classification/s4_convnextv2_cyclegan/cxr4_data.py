"""
SUPPLEMENTARY TEST (new K-Means + new CycleGAN): data tables read the new cleaned images;
test = the 124-image test set only. Copied from supplementary_test_for_downsampling/5_new_training_and_classification/s4_convnextv2_cyclegan.
Setup 4: builds the train, validation and test tables. Reuses CXRDataset from s3_convnextv2_stylegan2-ada
and the paths in 3_downsampling_branch/setup4_config.py.
  train       487 cleaned images (232 Pneumonia + 255 No Finding)
  validation  run B (Setup 4B): validation_small_65.csv    run A (Setup 4A): all 5,985 images
  test        all 12,262 images, in the row order of testing_labels.csv

Input:  3_downsampling_branch/output/cleaned/cleaned_manifest.csv, cleaned/224/, labels/,
        2_upsampling_branch/output/labels/, 4_image_transformation/output/train_stats_setup4.json
Run:    python supplementary_test_for_downsampling/5_new_training_and_classification/s4_convnextv2_cyclegan/cxr4_data.py --run B     (self-check)
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SUPP = os.path.dirname(os.path.dirname(HERE))          # supplementary_test_for_downsampling
STAGE5 = os.path.join(os.path.dirname(SUPP), "5_training_and_classification")   # original model code (read only)
for _p in (os.path.join(STAGE5, "s3_convnextv2_stylegan2-ada"), os.path.join(SUPP, "3_new_downsampling_branch")):
    if _p not in sys.path:
        sys.path.append(_p)          # append, so files in this folder are found first

import setup4_config as C
from cxr_data import CXRDataset, run_paths

RESULTS_DIR = C.RESULTS_DIR
STATS_JSON = C.STATS_JSON
CLUSTERED_ALL = C.CLUSTERED_CSV                       # the new clustering of the 676 images
NEED = ["Image Index", "Finding Labels", "Patient ID", "Patient Age", "Patient Sex", "View Position"]


def run_dir(run):
    return C.run_dir(run)


def _read_manifest():
    """cleaned_manifest.csv (written by clean_images.py) -> one row per image with 'path' and 'domain'."""
    if not os.path.exists(C.CLEANED_CSV):
        raise SystemExit(f"{C.CLEANED_CSV} not found: run clean_images.py first")
    man = pd.read_csv(C.CLEANED_CSV)
    assert "Image Index" in man.columns, f"cleaned manifest has no 'Image Index' column: {list(man.columns)}"
    assert man["Image Index"].is_unique, "duplicate Image Index in the cleaned manifest"
    if "path" not in man.columns:                       # fall back to the documented folder layout
        man["path"] = [os.path.join(C.CLEAN_224_DIR, i) for i in man["Image Index"]]
    if "domain" not in man.columns:                     # take it from the clustering file
        cl = pd.read_csv(CLUSTERED_ALL)[["Image Index", "domain"]]
        man = man.merge(cl, on="Image Index", how="left")
    return man[["Image Index", "path", "domain"]]


def _attach(official, man, what):
    miss = set(official["Image Index"]) - set(man["Image Index"])
    if miss:
        raise SystemExit(f"{what}: {len(miss)} images have no cleaned file in the manifest "
                         f"(e.g. {sorted(miss)[:3]}). Run clean_images.py on clustered_all.csv.")
    out = official.merge(man, on="Image Index", how="left", validate="one_to_one")
    assert (out["Image Index"].values == official["Image Index"].values).all(), f"{what}: row order changed"
    return out


def get_tables(run, strict=True):
    """Returns (train, val, test) DataFrames ready for CXRDataset (they carry a 'path' column)."""
    assert run == "B", "the supplementary test runs the 4B design only"
    man = _read_manifest()
    tr_o = pd.read_csv(C.SRC_SPLIT_CSV["train"])
    va_o = pd.read_csv(C.SRC_SPLIT_CSV["validation"])
    te_o = pd.read_csv(C.SRC_SPLIT_CSV["test"])
    for name, d in (("train", tr_o), ("validation", va_o), ("test", te_o)):
        assert all(c in d.columns for c in NEED), f"{name} CSV lacks {[c for c in NEED if c not in d.columns]}"

    # train: the 487 training images of the 676-image set
    cl = pd.read_csv(CLUSTERED_ALL)
    tr_ids = set(cl.loc[cl["split"].astype(str).str.startswith("train"), "Image Index"])
    tr = _attach(tr_o[tr_o["Image Index"].isin(tr_ids)].reset_index(drop=True), man, "train")
    tr["is_synthetic"] = (tr["domain"] == "artifact").astype(int)     # is_synthetic carries the artifact-domain flag in Setup 4

    if run == "B":
        small = pd.read_csv(C.SMALL_VAL_CSV)
        ids = set(small["Image Index"] if "Image Index" in small.columns else small.iloc[:, 0])
        va_o = va_o[va_o["Image Index"].isin(ids)].reset_index(drop=True)
    va = _attach(va_o, man, "validation")
    t124 = set(pd.read_csv(C.BALANCED_TEST_CSV)["Image Index"])
    te = _attach(te_o[te_o["Image Index"].isin(t124)].reset_index(drop=True), man, "test")  # the 124 images, official order

    if strict:
        ntr_pn, ntr_nf = int((tr["Finding Labels"] == "Pneumonia").sum()), int((tr["Finding Labels"] != "Pneumonia").sum())
        assert (ntr_pn, ntr_nf) == (C.PNEUMONIA_PER_SPLIT["train"], C.NOFINDING_PER_SPLIT["train"]), (ntr_pn, ntr_nf)
        npn = int((va["Finding Labels"] == "Pneumonia").sum())
        if run == "B":
            assert (npn, len(va) - npn) == (31, 34), (npn, len(va) - npn)
        else:
            assert (len(va), npn) == (5985, 31), (len(va), npn)
        npn = int((te["Finding Labels"] == "Pneumonia").sum())
        assert (len(te), npn) == (124, 59), (len(te), npn)
    assert not (set(tr["Image Index"]) & set(va["Image Index"])), "train/validation overlap"
    assert not (set(tr["Image Index"]) & set(te["Image Index"])), "train/test overlap"
    assert not (set(va["Image Index"]) & set(te["Image Index"])), "validation/test overlap"
    return tr, va, te


def load_stats():
    if not os.path.exists(STATS_JSON):
        raise SystemExit(f"{STATS_JSON} not found: run make_stats.py --split training on the cleaned manifest first")
    st = json.load(open(STATS_JSON))
    return st["mean"], st["std"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["B"], required=True)
    ap.add_argument("--no_file_check", action="store_true", help="skip the existence check of every PNG")
    ap.add_argument("--relax", action="store_true", help="synthetic smoke test: do not assert the published counts")
    a = ap.parse_args()
    tr, va, te = get_tables(a.run, strict=not a.relax)
    print(f"run {a.run}: train {len(tr)} | validation {len(va)} | test {len(te)}")
    for name, d in (("train", tr), ("validation", va), ("test", te)):
        y = d["Finding Labels"].map(lambda s: "Pneumonia" if s == "Pneumonia" else "No Finding")
        print(f"\n{name}: class x domain")
        print(pd.crosstab(y, d["domain"].fillna("?"), margins=True).to_string())
        if not a.no_file_check:
            missing = [p for p in d["path"] if not os.path.exists(p)]
            print(f"  files missing: {len(missing)}" + (f"  e.g. {missing[:2]}" if missing else ""))
    m, s = load_stats()
    print(f"\nSetup 4 normalisation: mean {m:.4f} std {s:.4f} (Setup 3 was 0.4929 / 0.2934)")
    print("test order identical to testing_labels.csv: OK" if not a.relax else "")
    print("results go to:", run_dir(a.run))


if __name__ == "__main__":
    main()
