"""
Utility: Chapter 4 Tables IV5, IV9, IV10, IV11 to IV18 and IV20, named by their manuscript number.
Tables IV1 to IV4 and IV6 to IV8 come from make_chapter4_tables.py. Table IV19 has no script.

Input:  2_upsampling_branch/output/labels/combined_training_labels.csv
        3_downsampling_branch/output/labels/subset_676.csv, clusters/clustered_676.csv,
        translation_change/summary.csv, fid_kid/fid_kid_fid.csv
        6_testing_and_validation/output/test_12262/, test_124/, test_12203_without_failed/
Output: 7_util/output/chapter4/tables/tableIV<N>_<name>.csv
Run:    python 7_util/make_chapter4_result_tables.py
"""
import argparse
import os

import pandas as pd
from scipy.stats import chi2_contingency

EVAL = os.path.join("6_testing_and_validation", "output")
FULL, SMALL, NOFAIL = "test_12262", "test_124", "test_12203_without_failed"
MAIN_TEST = {"S1": FULL, "S3": FULL, "S4A": FULL, "S4B": SMALL}
OTHER_TEST = {"S1": SMALL, "S3": SMALL, "S4A": SMALL, "S4B": FULL}
N_TEST = {FULL: 12262, SMALL: 124, NOFAIL: 12203}
NAME = {"S1": "Setup 1", "S3": "Setup 3", "S4A": "Setup 4A", "S4B": "Setup 4B"}
METRICS = ["accuracy", "precision", "recall", "specificity", "f1"]
DECLARED = [("S1", "S3", FULL), ("S3", "S4A", FULL), ("S3", "S4B", SMALL)]  # Setup 4B is compared on its main test set


def ensemble(test):
    return pd.read_csv(os.path.join(EVAL, test, "ensemble_metrics.csv"))


def delong(test):
    return pd.read_csv(os.path.join(EVAL, test, "delong.csv"))


def holm(p):
    order = sorted(range(len(p)), key=lambda i: p[i])
    adj, running = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def metrics_row(setup, test):
    e = ensemble(test)
    e = e[e.setup == setup].set_index("metric")
    row = {"setup": NAME[setup], "test_set": N_TEST[test], "auc": e.loc["auc", "estimate"],
           "auc_ci_lower": e.loc["auc", "ci_lower"], "auc_ci_upper": e.loc["auc", "ci_upper"],
           "youden_threshold": e.loc["auc", "threshold"]}
    row.update({m: e.loc[m, "estimate"] for m in METRICS})
    return row


def delong_row(a, b, test):
    d = delong(test)
    r = d[(d.setup_a == a) & (d.setup_b == b)].iloc[0]
    return {"comparison": f"{NAME[a]} vs {NAME[b]}", "test_set": N_TEST[test], "auc_a": r.auc_a, "auc_b": r.auc_b,
            "difference": r["diff"], "ci_lower": r.ci_lower, "ci_upper": r.ci_upper, "z": r.z, "p": r.p_value}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("7_util", "output", "chapter4", "tables"))
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    tables = {}

    # IV5: image count of each setup's training set
    comb = pd.read_csv(os.path.join("2_upsampling_branch", "output", "labels", "combined_training_labels.csv"))
    sub = pd.read_csv(os.path.join("3_downsampling_branch", "output", "labels", "subset_676.csv"))
    tr4 = sub[sub.split == "train"]
    tables["tableIV5_training_set_per_setup"] = pd.DataFrame([
        {"setup": "Setup 1 & 3", "pneumonia": int((comb["Finding Labels"] == "Pneumonia").sum()),
         "pneumonia_real": int(((comb["Finding Labels"] == "Pneumonia") & (comb.is_synthetic == 0)).sum()),
         "pneumonia_synthetic": int(comb.is_synthetic.sum()),
         "no_finding": int((comb["Finding Labels"] == "No Finding").sum()), "total": len(comb)},
        {"setup": "Setup 4", "pneumonia": int((tr4["Finding Labels"] == "Pneumonia").sum()),
         "pneumonia_real": int((tr4["Finding Labels"] == "Pneumonia").sum()), "pneumonia_synthetic": 0,
         "no_finding": int((tr4["Finding Labels"] == "No Finding").sum()), "total": len(tr4)}])

    # IV9: change made by the CycleGAN translation
    s = pd.read_csv(os.path.join("3_downsampling_branch", "output", "translation_change", "summary.csv")).iloc[0]
    tables["tableIV9_cyclegan_change"] = pd.DataFrame(
        [("translated images", s.n_artifact_images), ("mean center change (grey levels)", s.mae_centre_mean),
         ("median center change", s.mae_centre_median), ("90th percentile of center change", s.mae_centre_p90),
         ("maximum center change", s.mae_centre_max), ("images with center change > 10", s.n_centre_gt10),
         ("images with center change > 20 (flagged as failed)", s.n_centre_gt20),
         ("mean center change, Pneumonia", s.mae_centre_pneumonia),
         ("mean center change, No Finding", s.mae_centre_nofinding),
         ("Mann-Whitney p, Pneumonia vs No Finding", s.p_pneumonia_vs_nofinding)], columns=["item", "value"])

    # IV10: KID (x 1000) and FID before and after CycleGAN translation
    f = pd.read_csv(os.path.join("3_downsampling_branch", "output", "fid_kid", "fid_kid_fid.csv"))
    tables["tableIV10_kid_fid"] = f[["group", "n_artifact_total", "n_clean_total", "kid1000_before", "kid1000_after",
                                     "kid_improved_in", "fid_before", "fid_after", "kid1000_floor", "kid1000_floor_sd"]]

    # IV11 and IV12: main test set of each setup
    main_rows = pd.DataFrame([metrics_row(k, MAIN_TEST[k]) for k in MAIN_TEST])
    tables["tableIV11_auc_threshold"] = main_rows[["setup", "test_set", "auc", "auc_ci_lower", "auc_ci_upper",
                                                   "youden_threshold"]]
    tables["tableIV12_classification_metrics"] = main_rows[["setup", "test_set"] + METRICS]

    # IV13: DeLong tests of the declared comparisons, Holm over these three
    t13 = pd.DataFrame([delong_row(a, b, t) for a, b, t in DECLARED])
    t13["p_holm"] = holm(t13.p.tolist())
    tables["tableIV13_delong_declared"] = t13

    # IV14: seed ANOVA of AUC on the 12,262 images
    an = pd.read_csv(os.path.join(EVAL, FULL, "anova_seeds.csv"))
    an = an[an.metric == "auc"].copy()
    an["comparison"] = [f"{NAME[a]} vs {NAME[b]}" for a, b in zip(an.setup_a, an.setup_b)]
    tables["tableIV14_seed_anova_auc"] = an[["comparison", "mean_a", "mean_b", "F", "p_value", "eta_squared", "p_holm"]]

    # IV15: each setup on its other test set
    other = pd.DataFrame([metrics_row(k, OTHER_TEST[k]) for k in ("S4B", "S1", "S3", "S4A")])
    tables["tableIV15_other_test_set"] = other[["setup", "test_set", "auc", "auc_ci_lower", "auc_ci_upper"] + METRICS]

    # IV16: supplementary DeLong tests (not declared, p not adjusted)
    sup = [("S3", "S4B", FULL), ("S1", "S3", SMALL), ("S1", "S4A", SMALL), ("S1", "S4B", SMALL),
           ("S3", "S4A", SMALL), ("S4A", "S4B", SMALL)]
    tables["tableIV16_delong_supplementary"] = pd.DataFrame([delong_row(a, b, t) for a, b, t in sup])

    # IV17 and IV18: without failed translations (12,203 images)
    t17 = pd.DataFrame([metrics_row(k, NOFAIL) for k in MAIN_TEST])
    tables["tableIV17_without_failed"] = t17[["setup", "test_set", "auc", "auc_ci_lower", "auc_ci_upper"] + METRICS]
    d18 = delong(NOFAIL)
    d18["comparison"] = [f"{NAME[a]} vs {NAME[b]}" for a, b in zip(d18.setup_a, d18.setup_b)]
    tables["tableIV18_delong_without_failed"] = d18[["comparison", "auc_a", "auc_b", "diff", "ci_lower", "ci_upper",
                                                     "z", "p_value", "p_holm"]]

    # IV20: view position of the images in each cluster, chi-square of AP share artifact vs clean
    cl = pd.read_csv(os.path.join("3_downsampling_branch", "output", "clusters", "clustered_676.csv"))
    rows = []
    for cls in ("Pneumonia", "No Finding"):
        d = cl[cl["Finding Labels"] == cls]
        ct = pd.crosstab(d.domain, d["View Position"]).reindex(index=["artifact", "clean"], columns=["AP", "PA"],
                                                               fill_value=0)
        p = chi2_contingency(ct.values)[1]
        for dom in ("artifact", "clean"):
            ap_n, pa_n = int(ct.loc[dom, "AP"]), int(ct.loc[dom, "PA"])
            rows.append({"class": cls, "cluster": dom, "AP": ap_n, "PA": pa_n,
                         "AP_share_pct": round(100 * ap_n / (ap_n + pa_n), 1),
                         "chi_square_p": p if dom == "artifact" else None})
    tables["tableIV20_view_position_per_cluster"] = pd.DataFrame(rows)

    for name, t in tables.items():
        t.to_csv(os.path.join(out, name + ".csv"), index=False)
        print(f"{name}: {len(t)} rows")


if __name__ == "__main__":
    main()
