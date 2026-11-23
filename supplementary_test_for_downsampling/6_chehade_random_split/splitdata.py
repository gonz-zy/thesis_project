"""
SUPPLEMENTARY TEST (Chehade-style random split): training / validation / test tables of one random image-level split
of the 676 images (make_split.py). Chosen with two environment variables, inherited by every search / seed subprocess:

  CXR_SPLIT     split number k (1 ... 10)
  CXR_IMAGESET  cyclegan                 original K-Means + epoch-100 CycleGAN images (main study, 256 -> 224)
                nocyclegan_1024_to_224   untranslated equalized 1024 images, one resize to 224

Clinical values (sex, view, age) come from the official label CSVs (read only). Imported through the small
cxr2_data.py / cxr4_data.py of each recipe folder, which put the matching model code on sys.path first.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))           # 6_chehade_random_split

import setup4_config as C                                   # original 3_downsampling_branch config (read only)
from cxr_data import CXRDataset, run_paths                  # noqa: F401  (re-exported for train.py)

IMAGESETS = ("cyclegan", "nocyclegan_1024_to_224")
IMAGESET = os.environ.get("CXR_IMAGESET", "")
if IMAGESET not in IMAGESETS:
    raise SystemExit(f"set CXR_IMAGESET to one of {list(IMAGESETS)} (got '{IMAGESET}')")
try:
    SPLIT = int(os.environ.get("CXR_SPLIT", ""))
except ValueError:
    raise SystemExit("set CXR_SPLIT to the split number (1 ... 10)")
SPLIT_TAG = f"split_{SPLIT:02d}"
SPLIT_DIR = os.path.join(HERE, "splits", SPLIT_TAG)
MANIFEST = os.path.join(SPLIT_DIR, f"manifest_{IMAGESET}.csv")
STATS_JSON = os.path.join(SPLIT_DIR, f"train_stats_{IMAGESET}.json")
NEED = ["Image Index", "Finding Labels", "Patient ID", "Patient Age", "Patient Sex", "View Position"]


def get_tables(run="B", strict=True):
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"{MANIFEST} not found: run make_split.py --split {SPLIT} first")
    man = pd.read_csv(MANIFEST)
    assert len(man) == C.TOTAL_676 and man["Image Index"].is_unique
    if IMAGESET.startswith("nocyclegan"):
        assert not man["translated"].astype(bool).any(), "an image is translated"
    meta = pd.concat([pd.read_csv(C.SRC_SPLIT_CSV[s])[NEED] for s in ("train", "validation", "test")])
    meta = meta.drop_duplicates("Image Index")
    df = man[["Image Index", "path", "domain", "split"]].merge(meta, on="Image Index", how="left", validate="one_to_one")
    assert df[NEED].notna().all().all(), "clinical values missing"
    df["is_synthetic"] = (df["domain"] == "artifact").astype(int)  # artifact flag, for the domain check only
    tr, va, te = (df[df["split"] == s].sort_values("Image Index").reset_index(drop=True)
                  for s in ("train", "validation", "test"))
    assert len(tr) + len(va) + len(te) == C.TOTAL_676
    for a_, b_ in ((tr, va), (tr, te), (va, te)):
        assert not (set(a_["Image Index"]) & set(b_["Image Index"])), "split overlap"
    if strict:
        for d, lo, hi in ((tr, 0.69, 0.71), (va, 0.09, 0.11), (te, 0.19, 0.21)):
            assert lo <= len(d) / C.TOTAL_676 <= hi, len(d)
    return tr, va, te


def test_csv():
    """The test images of this split (for evaluate.py --subset)."""
    return os.path.join(SPLIT_DIR, "test_images.csv")


def load_stats():
    st = json.load(open(STATS_JSON))
    return st["mean"], st["std"]


def self_check():
    tr, va, te = get_tables()
    for name, d in (("train", tr), ("validation", va), ("test", te)):
        missing = [p for p in d["path"] if not os.path.exists(p)]
        assert not missing, f"{IMAGESET} {name}: {len(missing)} files missing"
    te[["Image Index"]].to_csv(test_csv(), index=False)
    m, s = load_stats()
    print(f"{SPLIT_TAG} {IMAGESET}: train {len(tr)} | validation {len(va)} | test {len(te)} | files ok | "
          f"mean {m:.4f} std {s:.4f}")
