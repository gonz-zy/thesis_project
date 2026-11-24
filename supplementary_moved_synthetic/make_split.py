"""
Supplementary (moved synthetic): Step 0 checks and Step 1 split.

Real images keep their original partition. Only the 41,972 existing synthetic Pneumonia images are reassigned
at random (without replacement): 29,422 train / 4,281 val / 8,269 test. No new images, no real image dropped.

Input:  2_upsampling_branch/output/labels/{training,validation,testing,synthetic,combined_training}_labels.csv (read only)
Output: supplementary_moved_synthetic/output/manifests/combined_training_labels.csv, validation_labels.csv,
        testing_labels.csv (the file names train.py expects), slice_*.csv, split_log.json, overlap_check.csv
Run:    python supplementary_moved_synthetic/make_split.py
"""
import json
import os

import numpy as np
import pandas as pd

SEED = 20261124  # split seed (logged in split_log.json)
N_SYN = dict(train=29422, val=4281, test=8269)
EXPECTED = dict(train=(232, 42204), val=(31, 5954), test=(59, 12203))
N_SYN_TOTAL = 41972

SRC = os.path.join("2_upsampling_branch", "output")
LAB = os.path.join(SRC, "labels")
OUT = os.path.join("supplementary_moved_synthetic", "output", "manifests")
COLS = ["Image Index", "Finding Labels", "Patient ID", "Patient Age", "Patient Sex", "View Position",
        "is_synthetic", "source", "path"]


def real(csv, folder):
    d = pd.read_csv(os.path.join(LAB, csv))
    d["is_synthetic"], d["source"] = 0, "real"
    d["path"] = [os.path.join(SRC, folder, f) for f in d["Image Index"]]
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = dict(train=real("training_labels.csv", "training"), val=real("validation_labels.csv", "validation"),
                 test=real("testing_labels.csv", "testing"))
    # Step 0: counts must match the stated assumptions
    for k, (npn, nnf) in EXPECTED.items():
        vc = parts[k]["Finding Labels"].value_counts()
        got = (int(vc.get("Pneumonia", 0)), int(vc.get("No Finding", 0)))
        assert got == (npn, nnf), f"{k}: expected {npn} P / {nnf} NF, found {got}"
    comb = pd.read_csv(os.path.join(LAB, "combined_training_labels.csv"))
    syn = comb[comb["is_synthetic"] == 1].copy()
    syn_meta = pd.read_csv(os.path.join(LAB, "synthetic_labels.csv"))
    assert len(syn) == N_SYN_TOTAL == len(syn_meta), (len(syn), len(syn_meta))
    assert set(syn["Image Index"]) == set(syn_meta["Image Index"])
    assert (syn["Finding Labels"] == "Pneumonia").all()
    for p in syn["path"]:
        assert os.path.exists(p), p
    syn["source"] = "synthetic"

    # Step 1: random assignment of synthetic images (each used exactly once)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(syn))
    syn = syn.iloc[perm].reset_index(drop=True)
    a, b = N_SYN["train"], N_SYN["train"] + N_SYN["val"]
    syn_parts = dict(train=syn.iloc[:a], val=syn.iloc[a:b], test=syn.iloc[b:])
    assert sum(len(v) for v in syn_parts.values()) == N_SYN_TOTAL

    M = {}
    for k in parts:
        m = pd.concat([parts[k], syn_parts[k]], ignore_index=True)[COLS]
        m = m.sort_values(["is_synthetic", "Image Index"]).reset_index(drop=True)
        M[k] = m
    names = dict(train="combined_training_labels.csv", val="validation_labels.csv", test="testing_labels.csv")
    for k, f in names.items():
        M[k].to_csv(os.path.join(OUT, f), index=False)

    # evaluation slices of the new test set
    te = M["test"]
    nf = te["Finding Labels"] == "No Finding"
    te[["Image Index"]].to_csv(os.path.join(OUT, "slice_a_mixed.csv"), index=False)
    te.loc[te["source"].eq("real"), ["Image Index"]].to_csv(os.path.join(OUT, "slice_b_real_only.csv"), index=False)
    te.loc[te["source"].eq("synthetic") | nf, ["Image Index"]].to_csv(os.path.join(OUT, "slice_c_synthetic_only.csv"), index=False)

    # overlap check (as in Table IV2): real patients and images, synthetic images
    rows = []
    for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
        rx, ry = (M[k][M[k]["source"] == "real"] for k in (x, y))
        sx, sy = (M[k][M[k]["source"] == "synthetic"] for k in (x, y))
        rows.append(dict(pair=f"{x}-{y}",
                         real_patient_overlap=len(set(rx["Patient ID"]) & set(ry["Patient ID"])),
                         real_image_overlap=len(set(rx["Image Index"]) & set(ry["Image Index"])),
                         synthetic_image_overlap=len(set(sx["Image Index"]) & set(sy["Image Index"])),
                         any_image_overlap=len(set(M[x]["Image Index"]) & set(M[y]["Image Index"]))))
    ov = pd.DataFrame(rows)
    ov.to_csv(os.path.join(OUT, "overlap_check.csv"), index=False)
    assert (ov.drop(columns="pair").values == 0).all(), ov
    all_syn = pd.concat([M[k].loc[M[k]["source"] == "synthetic", "Image Index"] for k in M])
    assert all_syn.is_unique and len(all_syn) == N_SYN_TOTAL

    counts = {}
    for k, m in M.items():
        counts[k] = dict(total=len(m),
                         real_pneumonia=int(((m.source == "real") & (m["Finding Labels"] == "Pneumonia")).sum()),
                         synthetic_pneumonia=int((m.source == "synthetic").sum()),
                         no_finding=int((m["Finding Labels"] == "No Finding").sum()))
    assert [counts[k]["total"] for k in ("train", "val", "test")] == [71858, 10266, 20531], counts
    log = dict(split_seed=SEED, rng="numpy.random.default_rng(seed).permutation over synthetic rows in "
               "combined_training_labels.csv order", synthetic_assignment=N_SYN, counts=counts,
               slices=dict(a_mixed=len(te), b_real_only=int(te.source.eq("real").sum()),
                           c_synthetic_only=int((te.source.eq("synthetic") | nf).sum())),
               training_seeds=[42, 43, 44, 45, 46], evaluation_bootstrap_seed=42)
    json.dump(log, open(os.path.join(OUT, "split_log.json"), "w"), indent=2)
    print(json.dumps(log, indent=2))
    print(ov.to_string(index=False))


if __name__ == "__main__":
    main()
