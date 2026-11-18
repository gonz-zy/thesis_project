"""
features.py -- Chehade's contour/texture feature vector for the 676 subset. Step 2 of Setup 4.

CPU only, no GPU, but it is the long job: HOG at 1024 x 1024 takes roughly 0.5-1.5 s per image per core.
Safe to run while a GPU job trains, but use --workers 4 then so it does not starve the trainer's dataloader.

The vector (580,648 values per image), our reading of Chehade et al. Sec. 3.4.1:
    HOG at 1024 x 1024, scikit-image defaults   127 x 127 blocks x (2 x 2 cells x 9 orientations) = 580,644
    variance of gradients                       variance of the Sobel gradient magnitude over the whole image
    lung sharpness                              mean gradient magnitude inside a lung mask
    black ratio                                 share of pixels below BLACK_THRESHOLD
    white ratio                                 share of pixels above WHITE_THRESHOLD
                                                                                        total = 580,648
580,648 is the number the paper reports, which is how we inferred that HOG was computed at full resolution
and dominates the vector. The paper does not publish the Sobel kernel, the lung-mask method or the
black/white cut-offs, so those are OUR assumptions (setup4_config.py) and the thesis must say so.

Lung mask (our assumption): the darkest 25% of the pixels inside a central window (rows 15-88%, columns
12-88%), small holes closed, thin bridges opened, then the two largest components, which are the two lung
fields. Otsu cannot be used because histogram equalisation makes it split every image at about grey 128.
If the mask collapses (< 1% of the image) the whole image is used instead and the fallback is recorded in
the manifest, so these images can be checked.

Note on scale: HOG is 580,644 of the 580,648 values, so after per-feature standardisation the four scalars
contribute about 0.0007% of the Euclidean distance K-Means works with. The clustering is therefore a
HOG clustering in practice. This follows the paper's described vector and is reported as such; the four
scalars are still saved separately so cluster_kmeans.py can report them per cluster.

Outputs (data\\2-processed\\2B-input_for_cyclegan\\features\\):
    features_676.npy       float32 memmap, 676 x 580,648 (about 1.6 GB), row order = subset_676.csv
    scalars_676.csv        the 4 scalars per image, readable, plus the mask fallback flag
    feature_manifest.csv   Image Index, class, split, row index, timing
    feature_report.json    settings, dimensions and timing, for the thesis

Run (project root):
    python scripts\\2b-cyclegan\\features.py --workers 4        (while the GPU is busy)
    python scripts\\2b-cyclegan\\features.py --workers 8        (when nothing else is running)
    python scripts\\2b-cyclegan\\features.py --probe            (one image: check the dimension first)
Resume-safe: rows already written are skipped unless --restart is given.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog as skimage_hog

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from setup4_config import (BLACK_THRESHOLD, FEATURE_DIM, FEATURE_SIZE, FEATURES_DIR, HOG_BLOCK_NORM,
                           HOG_CELLS_PER_BLOCK, HOG_DIM, HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL,
                           SOBEL_KSIZE, SUBSET_CSV, WHITE_THRESHOLD, make_dirs)

FEATURES_NPY = os.path.join(FEATURES_DIR, "features_676.npy")
SCALARS_CSV = os.path.join(FEATURES_DIR, "scalars_676.csv")
MANIFEST_CSV = os.path.join(FEATURES_DIR, "feature_manifest.csv")
REPORT_JSON = os.path.join(FEATURES_DIR, "feature_report.json")
SCALAR_NAMES = ["variance_of_gradients", "lung_sharpness", "black_ratio", "white_ratio"]


def load_1024(path):
    """8-bit grayscale PNG -> float32 in [0, 1] at FEATURE_SIZE. The images are already equalised; they are
    NOT equalised again here."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape != (FEATURE_SIZE, FEATURE_SIZE):
        img = cv2.resize(img, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


# Lung mask settings. OUR assumptions: Chehade does not publish how the lung region was found.
# Why not Otsu: the images are histogram-equalised, so their grey levels are spread almost evenly and Otsu
# lands at about 128 on every image. "Darker than Otsu" then selects half the picture (lungs, neck, arms and
# the black frame all joined together). A fixed percentile inside a central window works instead, because
# equalisation makes a percentile mean the same grey level on every image.
MASK_ROI = (0.15, 0.88, 0.12, 0.88)      # rows from/to, columns from/to, as fractions of the image
MASK_PERCENTILE = 25                      # lung = darkest 25% of the pixels inside that window


def lung_mask(img):
    """Darkest MASK_PERCENTILE % of the pixels inside the central window MASK_ROI, close small holes, open to
    break thin bridges, keep the two largest components (the two lung fields).
    Returns (mask, used_fallback). Fallback = whole image, recorded in the manifest."""
    u8 = (img * 255).astype(np.uint8)
    h, w = u8.shape
    r0, r1, c0, c1 = MASK_ROI
    roi = np.zeros((h, w), dtype=bool)
    roi[int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)] = True
    thr = np.percentile(u8[roi], MASK_PERCENTILE)
    dark = ((u8 < thr) & roi).astype(np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.0146 * h) * 2 + 1,) * 2)   # 31 px at 1024
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.03 * h) | 1,) * 2)           # 31 px at 1024
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_close)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_open)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return np.ones_like(dark, dtype=bool), True
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.argsort(areas)[::-1][:2] + 1          # the two largest components
    mask = np.isin(lab, keep)
    if mask.mean() < 0.01:                           # mask collapsed: use the whole image instead
        return np.ones_like(dark, dtype=bool), True
    return mask, False


def four_scalars(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=SOBEL_KSIZE)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=SOBEL_KSIZE)
    mag = cv2.magnitude(gx, gy)
    mask, fallback = lung_mask(img)
    return (float(mag.var()),                        # variance of gradients
            float(mag[mask].mean()),                 # lung sharpness
            float((img < BLACK_THRESHOLD).mean()),   # black ratio
            float((img > WHITE_THRESHOLD).mean())),  fallback


def hog_vector(img):
    return skimage_hog(img, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK, block_norm=HOG_BLOCK_NORM,
                       feature_vector=True).astype(np.float32)


def extract(args):
    """One image. Returns (row_index, full feature vector, 4 scalars, fallback flag, seconds)."""
    i, path = args
    t0 = time.time()
    img = load_1024(path)
    scal, fallback = four_scalars(img)
    h = hog_vector(img)
    vec = np.empty(len(h) + 4, dtype=np.float32)
    vec[:len(h)] = h
    vec[len(h):] = scal
    return i, vec, scal, fallback, time.time() - t0


def probe(subset):
    """Extract one image and print the real dimensions, so a mismatch is caught before the long run."""
    path = subset["src_path"].iloc[0]
    print(f"probe: {path}")
    img = load_1024(path)
    print(f"  image      {img.shape} {img.dtype} range [{img.min():.3f}, {img.max():.3f}]")
    t0 = time.time()
    h = hog_vector(img)
    t_hog = time.time() - t0
    cells = FEATURE_SIZE // HOG_PIXELS_PER_CELL[0]
    blocks = cells - HOG_CELLS_PER_BLOCK[0] + 1
    per_block = HOG_CELLS_PER_BLOCK[0] * HOG_CELLS_PER_BLOCK[1] * HOG_ORIENTATIONS
    print(f"  HOG        {len(h):,} values  ({blocks} x {blocks} blocks x {per_block})  in {t_hog:.2f} s")
    print(f"  expected   {HOG_DIM:,} (Chehade 580,644 + 4 scalars = 580,648)")
    t0 = time.time()
    scal, fallback = four_scalars(img)
    print(f"  scalars    {dict(zip(SCALAR_NAMES, [round(s, 5) for s in scal]))} in {time.time() - t0:.2f} s"
          + ("   (lung mask fell back to whole image)" if fallback else ""))
    total = len(h) + 4
    print(f"  TOTAL      {total:,}" + ("  OK" if total == FEATURE_DIM else f"  MISMATCH, config says {FEATURE_DIM:,}"))
    print(f"\n  676 images at {t_hog:.2f} s each: about {676 * t_hog / 60:.0f} core-minutes "
          f"({676 * t_hog / 60 / 4:.0f} min on 4 workers, {676 * t_hog / 60 / 8:.0f} min on 8)")
    print(f"  matrix on disk: {676 * total * 4 / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4, help="4 while the GPU is busy, 8 when it is free")
    ap.add_argument("--probe", action="store_true", help="one image only: check dimensions and timing")
    ap.add_argument("--restart", action="store_true", help="ignore existing rows and extract everything again")
    args = ap.parse_args()

    make_dirs()
    if not os.path.exists(SUBSET_CSV):
        raise SystemExit(f"{SUBSET_CSV} not found: run make_subset.py first")
    subset = pd.read_csv(SUBSET_CSV)
    n = len(subset)

    if args.probe:
        probe(subset)
        return

    missing = [p for p in subset["src_path"] if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"{len(missing)} source images missing, e.g. {missing[:3]}")

    done = np.zeros(n, dtype=bool)
    if os.path.exists(FEATURES_NPY) and not args.restart:
        if os.path.exists(MANIFEST_CSV):
            prev = pd.read_csv(MANIFEST_CSV)
            if len(prev) == n and prev["Image Index"].tolist() == subset["Image Index"].tolist():
                done = prev["extracted"].to_numpy().astype(bool)
                print(f"resuming: {int(done.sum())} of {n} rows already extracted")
        mm = np.lib.format.open_memmap(FEATURES_NPY, mode="r+")
        if mm.shape != (n, FEATURE_DIM):
            raise SystemExit(f"{FEATURES_NPY} has shape {mm.shape}, expected {(n, FEATURE_DIM)}; use --restart")
    else:
        mm = np.lib.format.open_memmap(FEATURES_NPY, mode="w+", dtype=np.float32, shape=(n, FEATURE_DIM))
        print(f"created {FEATURES_NPY}  ({n} x {FEATURE_DIM:,} float32 = {n * FEATURE_DIM * 4 / 1e9:.2f} GB)")

    todo = [(i, subset["src_path"].iloc[i]) for i in range(n) if not done[i]]
    if not todo:
        print("every row is already extracted; nothing to do")
        return
    print(f"extracting {len(todo)} images with {args.workers} workers\n")

    scalars = np.full((n, 4), np.nan, dtype=np.float64)
    fallbacks = np.zeros(n, dtype=bool)
    if os.path.exists(SCALARS_CSV) and not args.restart:
        prev = pd.read_csv(SCALARS_CSV)
        if len(prev) == n:
            scalars = prev[SCALAR_NAMES].to_numpy()
            fallbacks = prev["mask_fallback"].to_numpy().astype(bool)

    t_all, times, n_done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, vec, scal, fallback, secs in pool.map(extract, todo, chunksize=1):
            mm[i] = vec
            scalars[i] = scal
            fallbacks[i] = fallback
            done[i] = True
            times.append(secs)
            n_done += 1
            if n_done % 25 == 0 or n_done == len(todo):
                rate = np.mean(times[-50:])
                left = (len(todo) - n_done) * rate / args.workers / 60
                print(f"  {n_done}/{len(todo)}  {rate:.2f} s/image/core  about {left:.0f} min left", flush=True)
    mm.flush()

    out = subset[["Image Index", "Finding Labels", "split"]].copy()
    out["row"] = np.arange(n)
    out["extracted"] = done
    out.to_csv(MANIFEST_CSV, index=False)
    sc = subset[["Image Index", "Finding Labels", "split"]].copy()
    for j, name in enumerate(SCALAR_NAMES):
        sc[name] = scalars[:, j]
    sc["mask_fallback"] = fallbacks
    sc.to_csv(SCALARS_CSV, index=False)

    elapsed = (time.time() - t_all) / 60
    print(f"\nextracted {int(done.sum())} of {n} rows in {elapsed:.1f} min "
          f"({np.mean(times):.2f} s per image per core)")
    print(f"lung mask fell back to the whole image for {int(fallbacks.sum())} images"
          + ("  <-- check these in the label grid" if fallbacks.sum() else ""))
    print("\nthe four scalars, per class (HOG is not summarised: 580,644 values):")
    print(sc.groupby("Finding Labels")[SCALAR_NAMES].mean().round(5).to_string())

    json.dump(dict(
        n_images=n, feature_dim=int(FEATURE_DIM), hog_dim=int(HOG_DIM), n_scalars=4,
        feature_size=FEATURE_SIZE, hog_orientations=HOG_ORIENTATIONS,
        hog_pixels_per_cell=list(HOG_PIXELS_PER_CELL), hog_cells_per_block=list(HOG_CELLS_PER_BLOCK),
        hog_block_norm=HOG_BLOCK_NORM, sobel_ksize=SOBEL_KSIZE,
        black_threshold=BLACK_THRESHOLD, white_threshold=WHITE_THRESHOLD,
        lung_mask=f"darkest {MASK_PERCENTILE}% inside window {MASK_ROI}, close+open, two largest components",
        mask_roi=list(MASK_ROI), mask_percentile=MASK_PERCENTILE,
        mask_fallbacks=int(fallbacks.sum()), seconds_per_image_per_core=float(np.mean(times)),
        minutes_total=float(elapsed), workers=args.workers,
        chehade_reported_dim=580648, dims_match=bool(FEATURE_DIM == 580648),
    ), open(REPORT_JSON, "w"), indent=2)
    print(f"\nwrote {FEATURES_NPY}\n      {SCALARS_CSV}\n      {MANIFEST_CSV}\n      {REPORT_JSON}")
    print("\nNext: cluster_kmeans.py (per-class K-Means, k=2 on each class).")


if __name__ == "__main__":
    main()
