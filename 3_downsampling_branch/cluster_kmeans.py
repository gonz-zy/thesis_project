"""
Stage 3, step 3: K-Means (k = 2) fitted separately on the Pneumonia and the No Finding images.

Input:  output/labels/subset_676.csv, output/features/features_676.npy, scalars_676.csv
Output: 3_downsampling_branch/output/clusters/clustered_676.csv, cluster_report.json,
        kmeans_pneumonia.joblib, kmeans_no_finding.joblib
Run:    python 3_downsampling_branch/cluster_kmeans.py
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from setup4_config import (CLUSTERS_DIR, FEATURES_DIR, KMEANS_MAX_ITER, KMEANS_N_INIT, KMEANS_SEED,
                           N_CLUSTERS, SUBSET_CSV, make_dirs)

FEATURES_NPY = os.path.join(FEATURES_DIR, "features_676.npy")
SCALARS_CSV = os.path.join(FEATURES_DIR, "scalars_676.csv")
CLUSTERED_CSV = os.path.join(CLUSTERS_DIR, "clustered_676.csv")
REPORT_JSON = os.path.join(CLUSTERS_DIR, "cluster_report.json")
SCALAR_NAMES = ["variance_of_gradients", "lung_sharpness", "black_ratio", "white_ratio"]
CLASSES = ["Pneumonia", "No Finding"]
CHEHADE = {"Pneumonia": (152, 170), "No Finding": (133, 221)}  # (artifact, clean) from their Table 1


def fit_one_class(X, seed):
    """Standardise, then K-Means with k=2. Returns labels, the scaler and the fitted K-Means."""
    scaler = StandardScaler(copy=False)
    Xs = scaler.fit_transform(X)                      # in place: X is a throwaway copy already
    km = KMeans(n_clusters=N_CLUSTERS, n_init=KMEANS_N_INIT, max_iter=KMEANS_MAX_ITER, random_state=seed)
    labels = km.fit_predict(Xs)
    return labels, scaler, km, Xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=KMEANS_SEED)
    ap.add_argument("--quality", action="store_true", help="silhouette / Calinski-Harabasz / Davies-Bouldin")
    args = ap.parse_args()

    make_dirs()
    for p in (SUBSET_CSV, FEATURES_NPY, SCALARS_CSV):
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found: run make_subset.py and features.py first")

    subset = pd.read_csv(SUBSET_CSV)
    scalars = pd.read_csv(SCALARS_CSV)
    assert subset["Image Index"].tolist() == scalars["Image Index"].tolist(), \
        "scalars_676.csv is out of step with subset_676.csv: rerun features.py"
    F = np.load(FEATURES_NPY, mmap_mode="r")
    n = len(subset)
    assert F.shape[0] == n, f"features_676.npy has {F.shape[0]} rows, subset has {n}"
    print(f"features {F.shape[0]} x {F.shape[1]:,} float32 ({F.nbytes / 1e9:.2f} GB on disk)")
    print(f"per-class K-Means, k={N_CLUSTERS}, n_init={KMEANS_N_INIT}, seed {args.seed}\n")

    subset["cluster"] = -1
    subset["cluster_key"] = ""
    report = {"seed": args.seed, "per_class": {}, "standardisation": "per class, StandardScaler, no PCA"}

    for cls in CLASSES:
        rows = np.where(subset["Finding Labels"].to_numpy() == cls)[0]
        t0 = time.time()
        X = np.array(F[rows], dtype=np.float32)       # 322 or 354 rows: about 0.75 GB
        print(f"{cls}: {len(rows)} images, matrix {X.shape[0]} x {X.shape[1]:,}", flush=True)
        labels, scaler, km, Xs = fit_one_class(X, args.seed)
        del X

        sizes = {int(c): int((labels == c).sum()) for c in range(N_CLUSTERS)}
        subset.loc[rows, "cluster"] = labels
        subset.loc[rows, "cluster_key"] = [f"{cls}|{c}" for c in labels]

        # per-cluster means of the four scalars: the only interpretable summary of a 580k-dim cluster
        sc = scalars.iloc[rows][SCALAR_NAMES].to_numpy()
        means = {int(c): {name: float(sc[labels == c, j].mean()) for j, name in enumerate(SCALAR_NAMES)}
                 for c in range(N_CLUSTERS)}

        # provisional artifact cluster = lower mean lung sharpness (confirmed later by label_grid.py)
        prov = int(min(means, key=lambda c: means[c]["lung_sharpness"]))

        q = {}
        if args.quality:
            q = dict(silhouette=float(silhouette_score(Xs, labels)),
                     calinski_harabasz=float(calinski_harabasz_score(Xs, labels)),
                     davies_bouldin=float(davies_bouldin_score(Xs, labels)))
        del Xs

        # save the fitted scaler and K-Means model (reused by assign_domains.py)
        joblib.dump({"scaler": scaler, "kmeans": km, "class": cls, "provisional_artifact_cluster": prov},
                    os.path.join(CLUSTERS_DIR, f"kmeans_{cls.replace(' ', '_').lower()}.joblib"))

        print(f"  cluster sizes {sizes}   (Chehade: {CHEHADE[cls][0]} artifact / {CHEHADE[cls][1]} clean)")
        for c in range(N_CLUSTERS):
            m = means[c]
            print(f"    cluster {c} (n={sizes[c]:>3})  sharpness {m['lung_sharpness']:.5f}  "
                  f"grad var {m['variance_of_gradients']:.5f}  black {m['black_ratio']:.4f}  "
                  f"white {m['white_ratio']:.4f}" + ("   <- provisionally ARTIFACT" if c == prov else ""))
        if q:
            print(f"    quality {({k: round(v, 4) for k, v in q.items()})}")
        print(f"    fitted in {time.time() - t0:.0f} s\n", flush=True)

        report["per_class"][cls] = dict(n=len(rows), sizes=sizes, scalar_means=means,
                                        provisional_artifact_cluster=prov, quality=q,
                                        chehade_artifact=CHEHADE[cls][0], chehade_clean=CHEHADE[cls][1],
                                        inertia=float(km.inertia_))

    subset["provisional_domain"] = [
        "artifact" if r["cluster"] == report["per_class"][r["Finding Labels"]]["provisional_artifact_cluster"]
        else "clean" for _, r in subset.iterrows()]

    # diagnostic: the 4 scalars alone
    print("DIAGNOSTIC: K-Means on the four scalars alone (HOG is 99.9993% of the main vector)")
    subset["scalar_cluster"] = -1
    agree = {}
    for cls in CLASSES:
        rows = np.where(subset["Finding Labels"].to_numpy() == cls)[0]
        S = StandardScaler().fit_transform(scalars.iloc[rows][SCALAR_NAMES].to_numpy())
        sl = KMeans(n_clusters=N_CLUSTERS, n_init=KMEANS_N_INIT, random_state=args.seed).fit_predict(S)
        subset.loc[rows, "scalar_cluster"] = sl
        main = subset.loc[rows, "cluster"].to_numpy()
        a = max((main == sl).mean(), (main != sl).mean())   # cluster numbering is arbitrary
        agree[cls] = float(a)
        print(f"  {cls:<11} agreement with the HOG clustering: {100 * a:.1f}%")
    report["scalar_only_agreement"] = agree
    print("  (high agreement: the scalars and HOG see the same split. low: the partition is HOG-driven,\n"
          "   which the thesis should state, since the four physical measures then did not drive it.)\n")

    # summary
    ct = pd.crosstab(subset["Finding Labels"], subset["provisional_domain"])
    print("PROVISIONAL DOMAIN BY CLASS (Chehade Table 1: 152/133 artifact, 170/221 clean)")
    print(ct.to_string())
    share = subset.groupby("Finding Labels")["provisional_domain"].apply(lambda s: (s == "artifact").mean())
    print("\nartifact share per class: " + ", ".join(f"{c} {100 * v:.0f}%" for c, v in share.items())
          + f"   (Chehade: Pneumonia 47%, No Finding 38%)")
    print("\nby split (so you can see how many test images get cleaned):")
    print(pd.crosstab(subset["split"], subset["provisional_domain"]).to_string())

    report["crosstab"] = ct.to_dict()
    subset.to_csv(CLUSTERED_CSV, index=False)
    json.dump(report, open(REPORT_JSON, "w"), indent=2)
    print(f"\nwrote {CLUSTERED_CSV}\n      {REPORT_JSON}\n      kmeans_*.joblib in {CLUSTERS_DIR}")
    print("\nThe artifact direction above is PROVISIONAL. Next: label_grid.py, then your manual check\n"
          "decides which cluster CycleGAN cleans and gives the agreement % for the thesis.")


if __name__ == "__main__":
    main()
