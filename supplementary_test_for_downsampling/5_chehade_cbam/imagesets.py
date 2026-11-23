"""
SUPPLEMENTARY TEST (Chehade-style CBAM): train / validation / test tables of the 676-image set for one of three image
sets, chosen with the environment variable CXR_IMAGESET (inherited by every subprocess of the search and seed runs):

  cyclegan                        main-study images: original K-Means + epoch-100 CycleGAN (= Setups 2 / 4B)
  nocyclegan_1024_to_256_to_224   untranslated, equalized 1024 -> Setup 3's 256 file -> 224 (clean images = cyclegan's)
  nocyclegan_1024_to_224          untranslated, equalized 1024 -> 224 in one resize

Design as in Setups 2 / 4B: 487 training images, the 65-image validation list, the 124-image test list (row order of
testing_labels.csv). Every input is read only. Imported through the small cxr2_data.py / cxr4_data.py of each folder,
which put the matching model code on sys.path first.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # 5_chehade_cbam
SUPP = os.path.dirname(HERE)                                # supplementary_test_for_downsampling

import setup4_config as C                                   # original 3_downsampling_branch config (read only)
from cxr_data import CXRDataset, run_paths                  # noqa: F401  (re-exported for train.py)

IMAGESETS = {
    "cyclegan": dict(manifest=C.CLEANED_CSV, stats=C.STATS_JSON, untranslated=False),
    "nocyclegan_1024_to_256_to_224": dict(
        manifest=os.path.join(SUPP, "5_no_cyclegan_1024_to_256_to_224", "images", "manifest.csv"),
        stats=os.path.join(SUPP, "5_no_cyclegan_1024_to_256_to_224", "images", "train_stats.json"), untranslated=True),
    "nocyclegan_1024_to_224": dict(
        manifest=os.path.join(SUPP, "5_no_cyclegan_1024_to_224", "images", "manifest.csv"),
        stats=os.path.join(SUPP, "5_no_cyclegan_1024_to_224", "images", "train_stats.json"), untranslated=True),
}
IMAGESET = os.environ.get("CXR_IMAGESET", "")
if IMAGESET not in IMAGESETS:
    raise SystemExit(f"set CXR_IMAGESET to one of {list(IMAGESETS)} (got '{IMAGESET}')")
SPEC = IMAGESETS[IMAGESET]
NEED = ["Image Index", "Finding Labels", "Patient ID", "Patient Age", "Patient Sex", "View Position"]


def _manifest():
    man = pd.read_csv(SPEC["manifest"])
    ids676 = set(pd.read_csv(C.CLUSTERED_CSV)["Image Index"])          # the 676 images (original clustering file)
    man = man[man["Image Index"].isin(ids676)].reset_index(drop=True)
    assert len(man) == C.TOTAL_676 and man["Image Index"].is_unique, (IMAGESET, len(man))
    if SPEC["untranslated"]:
        assert not man["translated"].astype(bool).any(), f"{IMAGESET}: an image is translated"
    return man


def _attach(official, man, what):
    miss = set(official["Image Index"]) - set(man["Image Index"])
    assert not miss, f"{what}: {len(miss)} images missing from {SPEC['manifest']}"
    out = official.merge(man[["Image Index", "path", "domain"]], on="Image Index", how="left", validate="one_to_one")
    assert (out["Image Index"].values == official["Image Index"].values).all(), f"{what}: row order changed"
    return out


def get_tables(run, strict=True):
    assert run == "B", "the 4B design only"
    man = _manifest()
    tr_o, va_o, te_o = (pd.read_csv(C.SRC_SPLIT_CSV[s]) for s in ("train", "validation", "test"))
    for name, d in (("train", tr_o), ("validation", va_o), ("test", te_o)):
        assert all(c in d.columns for c in NEED), f"{name} CSV lacks columns"
    tr_ids = set(man.loc[man["split"].astype(str).str.startswith("train"), "Image Index"])
    tr = _attach(tr_o[tr_o["Image Index"].isin(tr_ids)].reset_index(drop=True), man, "train")
    tr["is_synthetic"] = (tr["domain"] == "artifact").astype(int)        # artifact flag, for the domain check
    small = pd.read_csv(C.SMALL_VAL_CSV)
    ids = set(small["Image Index"] if "Image Index" in small.columns else small.iloc[:, 0])
    va = _attach(va_o[va_o["Image Index"].isin(ids)].reset_index(drop=True), man, "validation")
    t124 = set(pd.read_csv(C.BALANCED_TEST_CSV)["Image Index"])
    te = _attach(te_o[te_o["Image Index"].isin(t124)].reset_index(drop=True), man, "test")
    if strict:
        npn = int((tr["Finding Labels"] == "Pneumonia").sum())
        assert (npn, len(tr) - npn) == (C.PNEUMONIA_PER_SPLIT["train"], C.NOFINDING_PER_SPLIT["train"])
        npn = int((va["Finding Labels"] == "Pneumonia").sum())
        assert (npn, len(va) - npn) == (31, 34)
        npn = int((te["Finding Labels"] == "Pneumonia").sum())
        assert (len(te), npn) == (124, 59)
    for a_, b_ in ((tr, va), (tr, te), (va, te)):
        assert not (set(a_["Image Index"]) & set(b_["Image Index"])), "split overlap"
    return tr, va, te


def load_stats():
    st = json.load(open(SPEC["stats"]))
    return st["mean"], st["std"]


def self_check():
    tr, va, te = get_tables("B")
    for name, d in (("train", tr), ("validation", va), ("test", te)):
        missing = [p for p in d["path"] if not os.path.exists(p)]
        assert not missing, f"{IMAGESET} {name}: {len(missing)} files missing"
    m, s = load_stats()
    print(f"{IMAGESET}: train {len(tr)} | validation {len(va)} | test {len(te)} | files ok | mean {m:.4f} std {s:.4f}")
