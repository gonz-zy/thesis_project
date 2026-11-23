"""
SUPPLEMENTARY TEST, step 1: a corrected K-Means artifact clustering of the same 676 images.

Why: in the original clustering (3_downsampling_branch/cluster_kmeans.py) the 580,644 HOG values outweighed the
4 scalar features, and the clusters followed view position (AP vs PA) more than artifacts (Table IV20); agreement
with the manual marks was 65% (Pneumonia) and 56% (No Finding).

PROTOCOL (fixed before any candidate was scored; nothing downstream is looked at to choose)
  Data      the same 676 images and the same saved features (features_676.npy, scalars_676.csv), read only.
            K-Means is fitted separately per class on all 676 images, as before (Chehade's procedure).
  Labels    the 200 existing manual marks (50 per original cluster per class) are split once, stratified by
            class and original cluster, 25/25 per stratum, seed 42:
              "select" half (100 images): used to choose the candidate and the artifact direction
              "report" half (100 images): used only to report the agreement of the chosen candidate
  Candidates (k = 2, n_init 50, max_iter 500, seed 42, per class)
     C0 hog_full            the original: all 580,648 values, each standardized (reproduces the old clusters)
     C1 scalars             the 4 scalars only, standardized
     C2 scalars_vr          C1, view position removed (per-view mean subtracted within the class)
     C3 pca20_scalars       HOG standardized -> 20 principal components; HOG block and scalar block each
                            scaled to total variance 1, so the two blocks weigh the same
     C4 pca20_scalars_vr    C3, view position removed from both blocks
     C5 pca50_scalars       as C3 with 50 components
     C6 pca50_scalars_vr    C5, view position removed
  Direction  per class, the artifact cluster is the one with the higher share of "a" marks in the select half.
  Criterion  balanced accuracy on the select half (mean of: share of "a" marks placed in the artifact cluster,
             share of "c" marks placed in the clean cluster), averaged over the two classes. Balanced accuracy
             is used rather than raw agreement because only about 12% of No Finding marks are artifacts:
             raw agreement would reward a near-empty artifact cluster.
  Guard      a candidate is eligible only if every cluster holds at least 10% of its class (CycleGAN needs
             images in both domains).
  Ties       the highest criterion wins; an exact tie goes to the earlier candidate in the list.
  Report     for the chosen candidate on the report half: agreement (same definition as the thesis, Table IV7),
             balanced accuracy, Wilson 95% CI, Fisher exact p; also the AP share per cluster (cf. Table IV20).
             Because the marks were drawn 50 per ORIGINAL cluster, a stratum-weighted agreement (weights = original
             cluster size / marks) is reported as a check.

AMENDMENT 1 (declared after round 1, before any amendment candidate was scored)
  Round 1 chose C0 (= the original clustering): every rebalanced candidate (C1-C6) isolated 15-18 extreme Pneumonia
  images in one cluster and failed the 10% guard; for No Finding the C0 clusters tied on the select half, so the
  direction was arbitrary (and wrong on the report half). Round-1 files: output/clusters/round1/.
  Cause: the 4 scalars have extreme outliers, and once they carry equal weight K-Means splits off the outliers.
  Round 2 candidates replace standardization by a rank-based (quantile -> normal) transform per class:
     C0 hog_full            unchanged, kept as the comparator
     Q1 scalars_q           the 4 scalars, quantile-transformed
     Q2 scalars_q_vr        Q1, view position removed
     Q3 pca20_scalars_q     20 HOG components and the 4 scalars, each quantile-transformed, equal block weights
     Q4 pca20_scalars_q_vr  Q3, view position removed
  Same label halves, criterion, guard, tie rule and report. If C0 wins again, no CycleGAN is trained.
  Run: ... cluster_kmeans_v2.py --round 2

Input:  3_downsampling_branch/output/labels/subset_676.csv, features/features_676.npy, scalars_676.csv,
        label_grids/manual_labels.csv, clusters/clustered_676.csv (original clusters, for the strata) -- all read only
Output: supplementary_test_for_downsampling/3_new_downsampling_branch/output/clusters/
        clustered_676.csv (with `domain`, `origin`), label_split.csv, candidates.csv, candidate_labels.csv,
        view_position_per_cluster.csv, cluster_report.json, kmeans_<class>.joblib
Run:    python supplementary_test_for_downsampling/3_new_downsampling_branch/cluster_kmeans_v2.py
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import setup4_config as C

FEATURES_NPY = os.path.join(C.FEATURES_DIR, "features_676.npy")
SCALARS_CSV = os.path.join(C.FEATURES_DIR, "scalars_676.csv")
ORIG_CLUSTERED = os.path.join(C.ORIG_ROOT, "clusters", "clustered_676.csv")
OUT = C.CLUSTERS_DIR
SCALAR_NAMES = ["variance_of_gradients", "lung_sharpness", "black_ratio", "white_ratio"]
CLASSES = ["Pneumonia", "No Finding"]
SEED = 42
MIN_SHARE = 0.10
CANDIDATES_ROUND1 = [  # name, hog ("full" | n_components | None), remove view position, scaling
    ("C0_hog_full", "full", False, "standard"),
    ("C1_scalars", None, False, "standard"),
    ("C2_scalars_vr", None, True, "standard"),
    ("C3_pca20_scalars", 20, False, "standard"),
    ("C4_pca20_scalars_vr", 20, True, "standard"),
    ("C5_pca50_scalars", 50, False, "standard"),
    ("C6_pca50_scalars_vr", 50, True, "standard"),
]
CANDIDATES_ROUND2 = [  # amendment 1: rank-based scaling
    ("C0_hog_full", "full", False, "standard"),
    ("Q1_scalars_q", None, False, "quantile"),
    ("Q2_scalars_q_vr", None, True, "quantile"),
    ("Q3_pca20_scalars_q", 20, False, "quantile"),
    ("Q4_pca20_scalars_q_vr", 20, True, "quantile"),
]


def remove_view(X, view):
    """Subtract the per-view (AP / PA) mean of every column: removes the view-position shift within a class."""
    X = X.copy()
    for v in np.unique(view):
        m = view == v
        X[m] -= X[m].mean(0)
    return X


def unit_block(X):
    """Scale a block so that its total variance is 1 (each block then weighs the same in the distance)."""
    X = X - X.mean(0)
    tv = float(X.var(0).sum())
    return X / np.sqrt(tv) if tv > 0 else X


def features_for(cand, F, rows, scal, view, pca_cache):
    name, hog, vr, scaling = cand

    def scale(X):
        if scaling == "quantile":
            return QuantileTransformer(n_quantiles=len(X), output_distribution="normal", random_state=SEED).fit_transform(X)
        return StandardScaler().fit_transform(X)

    S = scale(scal)
    if hog == "full":                                   # the original vector, standardized as in cluster_kmeans.py
        X = np.array(F[rows], dtype=np.float32)
        return StandardScaler(copy=False).fit_transform(X)
    if hog is None:
        X = remove_view(S, view) if vr else S
        return X
    if hog not in pca_cache:
        H = StandardScaler(copy=False).fit_transform(np.array(F[rows], dtype=np.float32)[:, :C.HOG_DIM])
        pca_cache[hog] = PCA(n_components=hog, svd_solver="randomized", random_state=SEED).fit_transform(H)
        del H
    P = pca_cache[hog]
    if scaling == "quantile":
        P = scale(P)
    if vr:
        P, S = remove_view(P, view), remove_view(S, view)
    return np.hstack([unit_block(P), unit_block(S)])


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def score(marks, artifact_cluster):
    """marks: DataFrame with columns cluster, manual (a|c). Returns agreement, balanced accuracy, counts."""
    pred = np.where(marks["cluster"].to_numpy() == artifact_cluster, "a", "c")
    truth = marks["manual"].to_numpy()
    agree = float((pred == truth).mean()) if len(truth) else float("nan")
    sens = float((pred[truth == "a"] == "a").mean()) if (truth == "a").any() else float("nan")
    spec = float((pred[truth == "c"] == "c").mean()) if (truth == "c").any() else float("nan")
    return dict(agreement=agree, sensitivity=sens, specificity=spec, balanced_accuracy=np.nanmean([sens, spec]),
                n=int(len(truth)), n_artifact_marks=int((truth == "a").sum()), n_agree=int((pred == truth).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=[1, 2], default=1)
    a = ap.parse_args()
    CANDIDATES = CANDIDATES_ROUND1 if a.round == 1 else CANDIDATES_ROUND2
    os.makedirs(OUT, exist_ok=True)
    subset = pd.read_csv(C.SUBSET_CSV)
    scalars = pd.read_csv(SCALARS_CSV)
    assert subset["Image Index"].tolist() == scalars["Image Index"].tolist()
    F = np.load(FEATURES_NPY, mmap_mode="r")
    assert F.shape == (len(subset), C.FEATURE_DIM), F.shape
    orig = pd.read_csv(ORIG_CLUSTERED).set_index("Image Index")
    orig_cluster = subset["Image Index"].map(orig["cluster"]).astype(int).to_numpy()

    # 1. split the 200 marks once (stratified by class and original cluster)
    lab = pd.read_csv(C.ORIG_MANUAL_LABELS)
    lab["manual"] = lab["manual"].astype(str).str.strip().str.lower()
    assert set(lab["manual"]) <= {"a", "c"} and len(lab) == 200, lab["manual"].value_counts()
    lab["orig_cluster"] = lab["image"].map(orig["cluster"]).astype(int)
    rng = np.random.default_rng(SEED)
    lab["half"] = ""
    for _, g in lab.groupby(["cls", "orig_cluster"]):
        idx = g.index.to_numpy()[rng.permutation(len(g))]
        lab.loc[idx[: len(idx) // 2], "half"] = "select"
        lab.loc[idx[len(idx) // 2:], "half"] = "report"
    lab[["image", "cls", "orig_cluster", "manual", "half"]].to_csv(os.path.join(OUT, "label_split.csv"), index=False)
    print("label split:\n" + pd.crosstab([lab["cls"], lab["half"]], lab["manual"]).to_string() + "\n")

    # 2. every candidate, per class
    pos = {n: i for i, n in enumerate(subset["Image Index"])}
    rows_cand, all_labels, fitted = [], {}, {}
    for cls in CLASSES:
        rows = np.where(subset["Finding Labels"].to_numpy() == cls)[0]
        scal = scalars.iloc[rows][SCALAR_NAMES].to_numpy(dtype=np.float64)
        view = subset["View Position"].to_numpy()[rows]
        pca_cache = {}
        marks = lab[lab["cls"] == cls].copy()
        marks["row"] = marks["image"].map(lambda n: int(np.where(rows == pos[n])[0][0]))
        for cand in CANDIDATES:
            t0 = time.time()
            X = features_for(cand, F, rows, scal, view, pca_cache)
            km = KMeans(n_clusters=2, n_init=C.KMEANS_N_INIT, max_iter=C.KMEANS_MAX_ITER, random_state=SEED)
            labels = km.fit_predict(X)
            del X
            all_labels[(cand[0], cls)] = labels
            fitted[(cand[0], cls)] = km
            marks["cluster"] = labels[marks["row"].to_numpy()]
            sel = marks[marks["half"] == "select"]
            share = {k: float((sel.loc[sel["cluster"] == k, "manual"] == "a").mean()) if (sel["cluster"] == k).any()
                     else -1.0 for k in (0, 1)}
            art = int(max(share, key=share.get))
            s_sel = score(sel, art)
            sizes = np.bincount(labels, minlength=2)
            ap_share = [float((view[labels == k] == "AP").mean()) if sizes[k] else float("nan") for k in (0, 1)]
            same_as_orig = float(max((labels == orig_cluster[rows]).mean(), (labels != orig_cluster[rows]).mean()))
            rows_cand.append(dict(candidate=cand[0], cls=cls, artifact_cluster=art,
                                  n_artifact=int(sizes[art]), n_clean=int(sizes[1 - art]),
                                  min_share=float(sizes.min() / sizes.sum()),
                                  ap_share_artifact=ap_share[art], ap_share_clean=ap_share[1 - art],
                                  agreement_with_original_clusters=same_as_orig,
                                  select_balanced_accuracy=s_sel["balanced_accuracy"],
                                  select_agreement=s_sel["agreement"], select_sensitivity=s_sel["sensitivity"],
                                  select_specificity=s_sel["specificity"], seconds=round(time.time() - t0, 1)))
            r = rows_cand[-1]
            print(f"{cls:<10} {cand[0]:<20} sizes {r['n_artifact']:>3}/{r['n_clean']:<3} AP {ap_share[art]:.2f}/{ap_share[1 - art]:.2f}  "
                  f"select BA {r['select_balanced_accuracy']:.3f} agr {r['select_agreement']:.3f}  "
                  f"same-as-old {same_as_orig:.2f}  {r['seconds']}s", flush=True)
    cands = pd.DataFrame(rows_cand)
    cands.to_csv(os.path.join(OUT, "candidates.csv"), index=False)

    # 3. choose (rule fixed in the docstring)
    summ = cands.groupby("candidate", sort=False).agg(select_balanced_accuracy=("select_balanced_accuracy", "mean"),
                                                      min_share=("min_share", "min")).reset_index()
    summ["eligible"] = summ["min_share"] >= MIN_SHARE
    print("\n" + summ.round(4).to_string(index=False))
    elig = summ[summ["eligible"]]
    if elig.empty:
        raise SystemExit("no eligible candidate")
    chosen = elig.loc[elig["select_balanced_accuracy"].idxmax(), "candidate"]  # idxmax keeps the first on ties
    print(f"\nCHOSEN (round {a.round}): {chosen}\n")
    if a.round == 2 and chosen == "C0_hog_full":
        json.dump(dict(round=2, chosen=chosen, candidates=summ.to_dict("records")),
                  open(os.path.join(OUT, "cluster_report.json"), "w"), indent=2, default=float)
        raise SystemExit("STOP: the original clustering won again; per the amendment, no new CycleGAN is trained.")

    # 4. write the chosen clustering
    subset["cluster"] = -1
    subset["domain"] = ""
    art_of = {}
    for cls in CLASSES:
        rows = np.where(subset["Finding Labels"].to_numpy() == cls)[0]
        labels = all_labels[(chosen, cls)]
        art = int(cands[(cands.candidate == chosen) & (cands.cls == cls)]["artifact_cluster"].iloc[0])
        art_of[cls] = art
        subset.loc[rows, "cluster"] = labels
        subset.loc[rows, "domain"] = np.where(labels == art, "artifact", "clean")
        joblib.dump({"kmeans": fitted[(chosen, cls)], "class": cls, "candidate": chosen, "artifact_cluster": art},
                    os.path.join(OUT, f"kmeans_{cls.replace(' ', '_').lower()}.joblib"))
    subset["origin"] = "subset_676"
    subset["candidate"] = chosen
    subset.to_csv(C.CLUSTERED_CSV, index=False)
    cl = pd.DataFrame({"Image Index": subset["Image Index"]})
    for (cname, cls), labels in all_labels.items():
        rows = np.where(subset["Finding Labels"].to_numpy() == cls)[0]
        cl.loc[rows, cname] = labels
    cl.to_csv(os.path.join(OUT, "candidate_labels.csv"), index=False)

    # 5. report on the held-out half
    rep = {}
    cur = subset.set_index("Image Index")["cluster"]
    orig_size = orig.groupby(["Finding Labels", "cluster"]).size()
    for cls in CLASSES:
        m = lab[(lab["cls"] == cls) & (lab["half"] == "report")].copy()
        m["cluster"] = m["image"].map(cur).astype(int)
        s = score(m, art_of[cls])
        lo, hi = wilson(s["n_agree"], s["n"])
        table = [[int(((m["cluster"] == k) & (m["manual"] == "a")).sum()), int(((m["cluster"] == k) & (m["manual"] == "c")).sum())]
                 for k in (art_of[cls], 1 - art_of[cls])]
        p = float(ss.fisher_exact(table)[1])
        # stratum-weighted agreement: each mark stands for (original cluster size / marks drawn from it)
        w = m["orig_cluster"].map(lambda k: orig_size[(cls, k)] / (m["orig_cluster"] == k).sum())
        correct = (np.where(m["cluster"] == art_of[cls], "a", "c") == m["manual"]).astype(float)
        wagree = float((w * correct).sum() / w.sum())
        mall = lab[lab["cls"] == cls].copy()
        mall["cluster"] = mall["image"].map(cur).astype(int)
        s_all = score(mall, art_of[cls])
        rep[cls] = dict(report_half=s, agreement_ci95=[lo, hi], fisher_p=p, contingency_artifact_then_clean=table,
                        weighted_agreement=wagree, all_200_marks_optimistic=s_all,
                        original_agreement_reported_in_thesis={"Pneumonia": 0.65, "No Finding": 0.56}[cls])
        print(f"{cls} REPORT half (n={s['n']}, {s['n_artifact_marks']} artifact marks): agreement {100 * s['agreement']:.0f}% "
              f"[95% CI {100 * lo:.0f}-{100 * hi:.0f}], weighted {100 * wagree:.0f}%, balanced acc {s['balanced_accuracy']:.3f}, "
              f"Fisher p {p:.4f}   (original clustering: {rep[cls]['original_agreement_reported_in_thesis']:.0%})")

    vp = pd.crosstab([subset["Finding Labels"], subset["domain"]], subset["View Position"])
    vp["AP_share"] = (vp["AP"] / vp.sum(1)).round(3)
    vp.to_csv(os.path.join(OUT, "view_position_per_cluster.csv"))
    print("\nview position per cluster:\n" + vp.to_string())
    print("\nby split:\n" + pd.crosstab(subset["split"], subset["domain"]).to_string())
    json.dump(dict(round=a.round, chosen=chosen, protocol="see docstring of cluster_kmeans_v2.py", seed=SEED, min_share=MIN_SHARE,
                   candidates=summ.to_dict("records"), artifact_cluster=art_of, report=rep,
                   sizes=pd.crosstab(subset["Finding Labels"], subset["domain"]).to_dict()),
              open(os.path.join(OUT, "cluster_report.json"), "w"), indent=2, default=float)
    print(f"\nwrote {C.CLUSTERED_CSV}")


if __name__ == "__main__":
    main()
