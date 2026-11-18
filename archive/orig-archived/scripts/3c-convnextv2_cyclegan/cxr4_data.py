"""
cxr4_data.py -- Setup 4 classifier: where the data is, how the three tables are built, and a self-check.

Setup 4 = the Setup 3 classifier (ConvNeXt V2-Tiny + CBAM, same recipe) trained on the 487 CYCLEGAN-CLEANED training
images instead of the StyleGAN2-ADA set. Everything that must be identical between train.py, bo_search.py,
run_experiments.py and run_all.py lives here, so a path or a split rule is defined exactly once.

It reuses, unchanged, from Setup 3 (scripts\\3b-convnextv2_stylegan2-ada): CXRDataset, ChehadeAugment, run_paths and the
model (convnext_cbam.py). Paths and counts come from scripts\\2b-cyclegan\\setup4_config.py.
Do NOT add a file named cxr_data.py or convnext_cbam.py to this folder: it would shadow the 3b modules.

Tables
  train       487 cleaned images (232 pneumonia + 255 No Finding), from training_labels.csv
  validation  run B: the 65-image list validation_small_65.csv (31 + 34)      -> Chehade-style small validation
              run A: all 5,985 validation images
  test        always all 12,262 images, in EXACTLY the row order of testing_labels.csv (evaluate.py pairs the
              predictions of the three setups on 'Image Index')
Clinical values (sex, view, age) and Patient ID come from the official split CSVs; the pixels come from the cleaned
224 x 224 PNGs written by clean_images.py. 'is_synthetic' is re-used to carry the ARTIFACT-DOMAIN flag of the
training images (1 = the image went through G_AB), so train.py can log recall per domain.

Self-check (project root):
    python scripts\\3c-convnextv2_cyclegan\\cxr4_data.py --run B
    python scripts\\3c-convnextv2_cyclegan\\cxr4_data.py --run A
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
for _p in (os.path.join(SCRIPTS, "3b-convnextv2_stylegan2-ada"), os.path.join(SCRIPTS, "2b-cyclegan")):
    if _p not in sys.path:
        sys.path.append(_p)          # append, never insert(0): files in this folder must win

import setup4_config as C                         # noqa: E402  (scripts\2b-cyclegan)
from cxr_data import CXRDataset, run_paths        # noqa: E402,F401  (scripts\3b-...; re-exported for the other scripts)

RESULTS_DIR = C.RESULTS_DIR
STATS_JSON = C.STATS_JSON
CLUSTERED_ALL = os.path.join(C.CLUSTERS_DIR, "clustered_all.csv")
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
    """Returns (train, val, test) DataFrames ready for CXRDataset (they carry a 'path' column).
    strict=True asserts the published counts; strict=False is for the synthetic CPU smoke test only."""
    assert run in ("A", "B"), run
    man = _read_manifest()
    tr_o = pd.read_csv(C.SRC_SPLIT_CSV["train"])
    va_o = pd.read_csv(C.SRC_SPLIT_CSV["validation"])
    te_o = pd.read_csv(C.SRC_SPLIT_CSV["test"])
    for name, d in (("train", tr_o), ("validation", va_o), ("test", te_o)):
        assert all(c in d.columns for c in NEED), f"{name} CSV lacks {[c for c in NEED if c not in d.columns]}"

    # train: the 487 training images of the 676 subset = official train rows that exist in the manifest
    cl = pd.read_csv(CLUSTERED_ALL)
    tr_ids = set(cl.loc[cl["split"].astype(str).str.startswith("train"), "Image Index"])
    tr = _attach(tr_o[tr_o["Image Index"].isin(tr_ids)].reset_index(drop=True), man, "train")
    tr["is_synthetic"] = (tr["domain"] == "artifact").astype(int)     # artifact-domain flag (see docstring)

    if run == "B":
        small = pd.read_csv(C.SMALL_VAL_CSV)
        ids = set(small["Image Index"] if "Image Index" in small.columns else small.iloc[:, 0])
        va_o = va_o[va_o["Image Index"].isin(ids)].reset_index(drop=True)
    va = _attach(va_o, man, "validation")
    te = _attach(te_o, man, "test")                                   # official order, all rows

    if strict:
        ntr_pn, ntr_nf = int((tr["Finding Labels"] == "Pneumonia").sum()), int((tr["Finding Labels"] != "Pneumonia").sum())
        assert (ntr_pn, ntr_nf) == (C.PNEUMONIA_PER_SPLIT["train"], C.NOFINDING_PER_SPLIT["train"]), (ntr_pn, ntr_nf)
        npn = int((va["Finding Labels"] == "Pneumonia").sum())
        if run == "B":
            assert (npn, len(va) - npn) == (31, 34), (npn, len(va) - npn)
        else:
            assert (len(va), npn) == (5985, 31), (len(va), npn)
        npn = int((te["Finding Labels"] == "Pneumonia").sum())
        assert (len(te), npn) == (12262, 59), (len(te), npn)
        t0 = pd.read_csv(C.SRC_SPLIT_CSV["test"])["Image Index"].values
        assert (te["Image Index"].values == t0).all(), "test order differs from testing_labels.csv"
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
    ap.add_argument("--run", choices=["A", "B"], required=True)
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
