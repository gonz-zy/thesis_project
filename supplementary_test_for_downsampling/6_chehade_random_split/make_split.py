"""
SUPPLEMENTARY TEST (Chehade-style random split): one random IMAGE-LEVEL split of the 676 images, 70% training /
20% test / 10% validation, stratified by class, patients ignored. Chehade et al. (2025) split their 676 images at
random 70/10/20 and mention no patient-level grouping, so a patient's images can fall in more than one part.
Split k uses random seed k (k = 1 ... 10).

The original K-Means and the epoch-100 CycleGAN were fitted on all 676 images (Chehade's procedure), so they do not
depend on the split: the image files are reused as they are (read only).

Per image set it writes, in splits/split_<kk>/:
  manifest_<image set>.csv     the 676 rows of that image set, `split` replaced by the new split
  train_stats_<image set>.json mean / std of the new training images (4_image_transformation/make_stats.py)
and split_<kk>.csv (Image Index, Finding Labels, Patient ID, split) and leakage.txt (patients in more than one part).

Run from the project root:  python supplementary_test_for_downsampling/6_chehade_random_split/make_split.py --split 1
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SUPP = os.path.dirname(HERE)
ROOT = os.path.dirname(SUPP)
sys.path.append(os.path.join(ROOT, "3_downsampling_branch"))
import setup4_config as C  # noqa: E402  (read only)

FRACTIONS = {"test": 0.20, "validation": 0.10}           # training = the rest (70%)
IMAGESETS = {
    "cyclegan": C.CLEANED_CSV,                                                               # original K-Means + CycleGAN
    "nocyclegan_1024_to_224": os.path.join(SUPP, "5_no_cyclegan_1024_to_224", "images", "manifest.csv"),
}
NEED = ["Image Index", "Finding Labels", "Patient ID", "Patient Age", "Patient Sex", "View Position"]


def metadata_676():
    ids = set(pd.read_csv(C.CLUSTERED_CSV)["Image Index"])
    meta = pd.concat([pd.read_csv(C.SRC_SPLIT_CSV[s])[NEED] for s in ("train", "validation", "test")])
    meta = meta[meta["Image Index"].isin(ids)].drop_duplicates("Image Index").sort_values("Image Index")
    assert len(meta) == C.TOTAL_676, len(meta)
    return meta.reset_index(drop=True)


def make_split(meta, k):
    rng = np.random.default_rng(k)
    part = pd.Series("train", index=meta.index)
    for cls in sorted(meta["Finding Labels"].unique()):
        idx = meta.index[meta["Finding Labels"] == cls].to_numpy()
        idx = idx[rng.permutation(len(idx))]
        n_te = int(round(FRACTIONS["test"] * len(idx)))
        n_va = int(round(FRACTIONS["validation"] * len(idx)))
        part[idx[:n_te]] = "test"
        part[idx[n_te:n_te + n_va]] = "validation"
    out = meta[["Image Index", "Finding Labels", "Patient ID"]].copy()
    out["split"] = part.values
    return out


def leakage(sp):
    lines = []
    g = sp.groupby("Patient ID")["split"].agg(lambda s: tuple(sorted(set(s))))
    multi = g[g.map(len) > 1]
    lines.append(f"patients: {sp['Patient ID'].nunique()} | with images in more than one part: {len(multi)}")
    for a, b in (("train", "test"), ("train", "validation"), ("validation", "test")):
        pa = set(sp.loc[sp["split"] == a, "Patient ID"])
        rows = sp[(sp["split"] == b) & sp["Patient ID"].isin(pa)]
        lines.append(f"  {b} images whose patient also has a {a} image: {len(rows)} ({rows['Patient ID'].nunique()} patients)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True, help="split number k = random seed (1 ... 10)")
    a = ap.parse_args()
    out_dir = os.path.join(HERE, "splits", f"split_{a.split:02d}")
    os.makedirs(out_dir, exist_ok=True)

    meta = metadata_676()
    sp = make_split(meta, a.split)
    sp.to_csv(os.path.join(out_dir, f"split_{a.split:02d}.csv"), index=False)
    print(pd.crosstab(sp["Finding Labels"], sp["split"], margins=True).to_string())
    leak = leakage(sp)
    print(leak)
    open(os.path.join(out_dir, "leakage.txt"), "w").write(leak + "\n")

    for name, src in IMAGESETS.items():
        man = pd.read_csv(src)
        man = man[man["Image Index"].isin(set(meta["Image Index"]))].copy()
        assert len(man) == C.TOTAL_676 and man["Image Index"].is_unique, (name, len(man))
        man = man.drop(columns=["split"]).merge(sp[["Image Index", "split"]], on="Image Index", validate="one_to_one")
        assert len(man) == C.TOTAL_676
        mpath = os.path.join(out_dir, f"manifest_{name}.csv")
        man.to_csv(mpath, index=False)
        spath = os.path.join(out_dir, f"train_stats_{name}.json")
        if not os.path.exists(spath):
            subprocess.run([sys.executable, os.path.join(ROOT, "4_image_transformation", "make_stats.py"),
                            "--csv", mpath, "--split", "train", "--out", spath], check=True, stdout=subprocess.DEVNULL)
        print(f"{name}: {mpath} | {spath}")


if __name__ == "__main__":
    main()
