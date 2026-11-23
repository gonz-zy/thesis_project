"""
SUPPLEMENTARY TEST (no CycleGAN): train / validation / test tables of the 676-image set, read from the UNTRANSLATED
224 images of this variant folder (images/manifest.csv, written by make_images_no_cyclegan.py).
Everything else is the main study's Setup 2 / Setup 4B design: 487 training images, the 65-image validation list,
the 124-image test list (in the row order of testing_labels.csv). `domain` = the ORIGINAL clustering, used only for
the descriptive domain check (is_synthetic carries the artifact flag, as in Setup 4).

Input:  ../images/manifest.csv, ../images/train_stats.json, 3_downsampling_branch/output/labels/ (read only),
        2_upsampling_branch/output/labels/ (read only)
Run:    python supplementary_test_for_downsampling/<variant>/<this folder>/cxr4_data.py --run B     (self-check)
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANT = os.path.dirname(HERE)                         # 5_no_cyclegan_1024_to_224 or 5_no_cyclegan_1024_to_256_to_224
ROOT = os.path.dirname(os.path.dirname(VARIANT))        # project root
STAGE5 = os.path.join(ROOT, "5_training_and_classification")   # original model code (read only)
for _p in (os.path.join(STAGE5, "s3_convnextv2_stylegan2-ada"), os.path.join(ROOT, "3_downsampling_branch")):
    if _p not in sys.path:
        sys.path.append(_p)

import setup4_config as C                               # original config: split lists, counts, epoch cap, seeds (read only)
from cxr_data import CXRDataset, run_paths

MANIFEST = os.path.join(VARIANT, "images", "manifest.csv")
STATS_JSON = os.path.join(VARIANT, "images", "train_stats.json")
RESULTS_DIR = os.path.join(HERE, "output")
NEED = ["Image Index", "Finding Labels", "Patient ID", "Patient Age", "Patient Sex", "View Position"]


def run_dir(run):
    assert run == "B", "the no-CycleGAN test runs the 4B design only"
    return os.path.join(RESULTS_DIR, f"run_{run}")


def _attach(official, man, what):
    miss = set(official["Image Index"]) - set(man["Image Index"])
    if miss:
        raise SystemExit(f"{what}: {len(miss)} images missing from {MANIFEST} (e.g. {sorted(miss)[:3]})")
    out = official.merge(man[["Image Index", "path", "domain"]], on="Image Index", how="left", validate="one_to_one")
    assert (out["Image Index"].values == official["Image Index"].values).all(), f"{what}: row order changed"
    return out


def get_tables(run, strict=True):
    assert run == "B"
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"{MANIFEST} not found: run make_images_no_cyclegan.py first")
    man = pd.read_csv(MANIFEST)
    assert man["Image Index"].is_unique and len(man) == C.TOTAL_676
    assert not man["translated"].any(), "no image may be translated in this test"
    tr_o, va_o, te_o = (pd.read_csv(C.SRC_SPLIT_CSV[s]) for s in ("train", "validation", "test"))
    for name, d in (("train", tr_o), ("validation", va_o), ("test", te_o)):
        assert all(c in d.columns for c in NEED), f"{name} CSV lacks {[c for c in NEED if c not in d.columns]}"

    tr_ids = set(man.loc[man["split"].astype(str).str.startswith("train"), "Image Index"])
    tr = _attach(tr_o[tr_o["Image Index"].isin(tr_ids)].reset_index(drop=True), man, "train")
    tr["is_synthetic"] = (tr["domain"] == "artifact").astype(int)
    small = pd.read_csv(C.SMALL_VAL_CSV)
    ids = set(small["Image Index"] if "Image Index" in small.columns else small.iloc[:, 0])
    va = _attach(va_o[va_o["Image Index"].isin(ids)].reset_index(drop=True), man, "validation")
    t124 = set(pd.read_csv(C.BALANCED_TEST_CSV)["Image Index"])
    te = _attach(te_o[te_o["Image Index"].isin(t124)].reset_index(drop=True), man, "test")

    if strict:
        ntr_pn = int((tr["Finding Labels"] == "Pneumonia").sum())
        assert (ntr_pn, len(tr) - ntr_pn) == (C.PNEUMONIA_PER_SPLIT["train"], C.NOFINDING_PER_SPLIT["train"])
        npn = int((va["Finding Labels"] == "Pneumonia").sum())
        assert (npn, len(va) - npn) == (31, 34), (npn, len(va) - npn)
        npn = int((te["Finding Labels"] == "Pneumonia").sum())
        assert (len(te), npn) == (124, 59), (len(te), npn)
    for a_, b_, n_ in ((tr, va, "train/validation"), (tr, te, "train/test"), (va, te, "validation/test")):
        assert not (set(a_["Image Index"]) & set(b_["Image Index"])), f"{n_} overlap"
    return tr, va, te


def load_stats():
    if not os.path.exists(STATS_JSON):
        raise SystemExit(f"{STATS_JSON} not found: run make_stats.py on {MANIFEST} first")
    st = json.load(open(STATS_JSON))
    return st["mean"], st["std"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["B"], required=True)
    ap.add_argument("--relax", action="store_true")
    a = ap.parse_args()
    tr, va, te = get_tables(a.run, strict=not a.relax)
    print(f"variant {os.path.basename(VARIANT)} | run {a.run}: train {len(tr)} | validation {len(va)} | test {len(te)}")
    for name, d in (("train", tr), ("validation", va), ("test", te)):
        y = d["Finding Labels"].map(lambda s: "Pneumonia" if s == "Pneumonia" else "No Finding")
        print(f"\n{name}: class x domain (original clustering)")
        print(pd.crosstab(y, d["domain"], margins=True).to_string())
        missing = [p for p in d["path"] if not os.path.exists(p)]
        print(f"  files missing: {len(missing)}")
        assert not missing
    m, s = load_stats()
    print(f"\nnormalisation: mean {m:.4f} std {s:.4f}")
    print("results go to:", run_dir(a.run))


if __name__ == "__main__":
    main()
