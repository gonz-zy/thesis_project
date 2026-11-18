"""
translation_change.py -- how much did G_AB change each artifact-domain image, and WHERE?

For every artifact-domain image in one or more cleaned folders (output of clean_images.py --out <folder>) it compares
    before  Setup 3's 224 image (the 256 image -> torchvision Resize((224, 224)), no G_AB)
    after   <folder>\\224\\<Image Index>   (what Setup 4 would feed the classifier)
and measures, in grey levels (0-255) unless stated:
    mae_all      mean absolute difference over the whole image
    mean_shift   signed mean difference (after - before): a global brightness shift, not a structural change
    mae_centre   mean absolute difference in the CENTRE (rows/cols 15 %-85 %): lungs, heart, spine = the anatomy
    mae_border   mean absolute difference in the outer 15 % frame: where markers, tubes and edges usually sit
    ssim_centre  structural similarity (1 = identical structure) of the centre
Two readings matter:
  * mae_centre should be SMALL: the cleaning should not touch the anatomy. mae_border may be larger: that is where
    the artifacts are. (The centre/border split is a crude stand-in for a lung mask; markers can sit inside the centre.)
  * pneumonia images must not be changed MORE than No Finding images. A Mann-Whitney test on mae_centre is printed;
    a small p means the generator treats the classes differently, which would be a new shortcut.
Thresholds (10 and 20 grey levels in the centre) are rules of thumb, not standards: use them to compare candidates.

Run (project root), several candidates at once:
    python scripts\\2b-cyclegan\\translation_change.py --cleaned_dir data\\2-processed\\2B-input_for_cyclegan\\cand_e075 ^
        data\\2-processed\\2B-input_for_cyclegan\\cand_e100
Output per folder: <folder>\\translation_change.csv, most_changed.png, pneumonia_most_changed.png (before | after | |difference| x4);
and results\\cyclegan_translation_change\\summary.csv for the comparison table.
Needs: pip install scikit-image scipy
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from scipy.stats import mannwhitneyu
from skimage.metrics import structural_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import setup4_config as C  # noqa: E402

CLUSTERED_ALL = os.path.join(C.CLUSTERS_DIR, "clustered_all.csv")
OUT_DIR = os.path.join("results", "cyclegan_translation_change")
LO, HI = int(0.15 * C.IMG), int(0.85 * C.IMG)          # centre box: 33 .. 190
RESIZE = T.Resize((C.IMG, C.IMG))


def load_before(split, idx):
    with Image.open(os.path.join(C.SRC_IMG_256[split], idx)) as im:
        return np.asarray(RESIZE(im.convert("L")), dtype=np.float32)


def load_after(folder, idx):
    with Image.open(os.path.join(folder, "224", idx)) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def measure(b, a):
    d = a - b
    cb, ca = b[LO:HI, LO:HI], a[LO:HI, LO:HI]
    border = np.ones_like(b, dtype=bool)
    border[LO:HI, LO:HI] = False
    return dict(mae_all=float(np.abs(d).mean()), mean_shift=float(d.mean()),
                mae_centre=float(np.abs(ca - cb).mean()), mae_border=float(np.abs(d[border]).mean()),
                ssim_centre=float(structural_similarity(cb, ca, data_range=255)))


def sheet(df, folder, path, n=8):
    """before | after | |difference| x4, one row per image, most changed first."""
    rows = []
    for _, r in df.head(n).iterrows():
        b, a = load_before(r["split"], r["Image Index"]), load_after(folder, r["Image Index"])
        diff = np.clip(np.abs(a - b) * 4.0, 0, 255)
        rows.append(np.concatenate([b, a, diff], axis=1))
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned_dir", nargs="+", required=True, help="folder(s) written by clean_images.py --out")
    ap.add_argument("--manifest", default=CLUSTERED_ALL)
    ap.add_argument("--limit", type=int, default=0, help="only the first N artifact images (quick test)")
    ap.add_argument("--n_sheet", type=int, default=8)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cl = pd.read_csv(a.manifest)
    cl["class"] = np.where(cl["Finding Labels"] == "Pneumonia", "Pneumonia", "No Finding")
    summ = []
    for folder in a.cleaned_dir:
        have = set(os.listdir(os.path.join(folder, "224")))
        d = cl[(cl["domain"] == "artifact") & cl["Image Index"].isin(have)]
        if a.limit:
            d = d.head(a.limit)
        if d.empty:
            print(f"[skip] {folder}: no artifact-domain images found in {folder}\\224")
            continue
        rows = []
        for i, (_, r) in enumerate(d.iterrows()):
            m = measure(load_before(r["split"], r["Image Index"]), load_after(folder, r["Image Index"]))
            rows.append(dict(**{k: r[k] for k in ("Image Index", "split", "class")}, **m))
            if (i + 1) % 2000 == 0:
                print(f"  {folder}: {i + 1}/{len(d)}", flush=True)
        res = pd.DataFrame(rows).sort_values("mae_centre", ascending=False)
        res.to_csv(os.path.join(folder, "translation_change.csv"), index=False)
        sheet(res, folder, os.path.join(folder, "most_changed.png"), a.n_sheet)
        sheet(res[res["class"] == "Pneumonia"], folder, os.path.join(folder, "pneumonia_most_changed.png"), a.n_sheet)

        pn, nf = res.loc[res["class"] == "Pneumonia", "mae_centre"], res.loc[res["class"] == "No Finding", "mae_centre"]
        p = mannwhitneyu(pn, nf, alternative="two-sided").pvalue if len(pn) > 2 and len(nf) > 2 else float("nan")
        s = dict(candidate=os.path.basename(os.path.normpath(folder)), n_artifact_images=len(res),
                 mae_all_mean=res["mae_all"].mean(), mean_shift_mean=res["mean_shift"].mean(),
                 mae_centre_mean=res["mae_centre"].mean(), mae_centre_median=res["mae_centre"].median(),
                 mae_centre_p90=res["mae_centre"].quantile(0.9), mae_centre_max=res["mae_centre"].max(),
                 mae_border_mean=res["mae_border"].mean(), ssim_centre_mean=res["ssim_centre"].mean(),
                 ssim_centre_min=res["ssim_centre"].min(), n_centre_gt10=int((res["mae_centre"] > 10).sum()),
                 n_centre_gt20=int((res["mae_centre"] > 20).sum()),
                 mae_centre_pneumonia=pn.mean() if len(pn) else float("nan"),
                 mae_centre_nofinding=nf.mean() if len(nf) else float("nan"), p_pneumonia_vs_nofinding=p)
        summ.append(s)

    if not summ:
        raise SystemExit("nothing to report")
    df = pd.DataFrame(summ)
    df.round(4).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print("\n=== CHANGE BETWEEN BEFORE (Setup 3 image) AND AFTER (cleaned), ARTIFACT-DOMAIN IMAGES ===")
    for _, s in df.iterrows():
        print(f"\n{s['candidate']}  ({int(s['n_artifact_images'])} images)")
        print(f"  whole image   mean abs change {s['mae_all_mean']:.2f}   mean brightness shift {s['mean_shift_mean']:+.2f}")
        print(f"  centre (anatomy)  mean {s['mae_centre_mean']:.2f}  median {s['mae_centre_median']:.2f}  "
              f"90th pct {s['mae_centre_p90']:.2f}  max {s['mae_centre_max']:.2f}   "
              f"SSIM mean {s['ssim_centre_mean']:.3f}  min {s['ssim_centre_min']:.3f}")
        print(f"  border (markers)  mean {s['mae_border_mean']:.2f}")
        print(f"  images with centre change > 10: {int(s['n_centre_gt10'])}   > 20: {int(s['n_centre_gt20'])}")
        print(f"  centre change, pneumonia {s['mae_centre_pneumonia']:.2f} vs No Finding {s['mae_centre_nofinding']:.2f}   "
              f"Mann-Whitney p = {s['p_pneumonia_vs_nofinding']:.3f}")
    print(f"\nwritten: {os.path.join(a.out, 'summary.csv')}; per folder: translation_change.csv, most_changed.png, "
          f"pneumonia_most_changed.png")


if __name__ == "__main__":
    main()
