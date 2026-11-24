"""
Supplementary (moved synthetic): Step 2 evaluation.

Runs the ORIGINAL 6_testing_and_validation/evaluate.py (unchanged) once per test slice, then adds:
Holm correction of the Setup 1 vs 3 DeLong p-values over the three slices, PR-AUC with stratified-bootstrap CI,
ROC / PR curves and confusion-matrix plots per slice, the shortcut check (real vs synthetic Pneumonia recall) on
train / val / test, and a side-by-side with the main-study 12,262-image results.

Slices of the new 20,531-image test set (manifests/slice_*.csv):
  a  mixed           all 20,531 (59 real P + 8,269 synthetic P + 12,203 NF)
  b  real only       59 real P + 12,203 NF = the ORIGINAL 12,262-image test set (headline)
  c  synthetic only  8,269 synthetic P + 12,203 NF
Threshold: Youden on the five-seed mean VALIDATION probability (mixed validation set), applied unchanged.

Run:    python supplementary_moved_synthetic/evaluate_slices.py
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

sys.path.insert(0, "6_testing_and_validation")
from evaluate import holm, load_setup, stratified_indices, Z95  # noqa: E402  (original code)

H = "supplementary_moved_synthetic"
MAN = os.path.join(H, "output", "manifests")
SLICES = dict(a_mixed="slice_a_mixed.csv", b_real_only="slice_b_real_only.csv",
              c_synthetic_only="slice_c_synthetic_only.csv")
NAMES = dict(S1="Setup 1 (DenseNet-121)", S3="Setup 3 (ConvNeXt V2)")


def ensemble(folder):
    seeds, preds, thr, vals = load_setup(folder)
    p = np.mean([d["prob_pneumonia"].to_numpy() for d in preds], axis=0)
    vy = vals[0]["label"].to_numpy()
    vp = np.mean([v["prob_pneumonia"].to_numpy() for v in vals], axis=0)
    fpr, tpr, th = roc_curve(vy, vp)
    t = float(th[np.argmax(tpr - fpr)])
    test = preds[0][["Image Index", "label"]].copy()
    test["prob"] = p
    val = vals[0][["Image Index", "label"]].copy()
    val["prob"] = vp
    return dict(seeds=seeds, preds=preds, thr=thr, vals=vals, test=test, val=val, t=t)


def rec(df, mask, thr):
    m = df[mask]
    return float((m["prob"] >= thr).mean()) if len(m) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(H, "output"), help="folder holding s1/ and s3/ run outputs")
    ap.add_argument("--out", default=os.path.join(H, "output", "evaluation"))
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    folders = dict(S1=os.path.join(a.root, "s1", "final", "cbam_all"), S3=os.path.join(a.root, "s3", "final", "cbam_all"))

    # 1. original evaluate.py per slice (AUC + CI, threshold metrics, confusion, DeLong, ANOVA)
    for sl, csv in SLICES.items():
        cmd = [sys.executable, os.path.join("6_testing_and_validation", "evaluate.py"),
               "--setup", f"S1={folders['S1']}", "--setup", f"S3={folders['S3']}", "--pair", "S1=S3",
               "--boot", str(a.boot), "--out", os.path.join(a.out, sl)]
        if sl != "a_mixed":
            cmd += ["--subset", os.path.join(MAN, csv)]
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        open(os.path.join(a.out, sl, "evaluate_stdout.txt"), "w").write(r.stdout + r.stderr)
        assert r.returncode == 0, r.stderr

    # 2. Holm over the three slices
    dl = pd.concat([pd.read_csv(os.path.join(a.out, sl, "delong.csv")).assign(slice=sl) for sl in SLICES],
                   ignore_index=True)
    dl["p_holm_over_slices"] = holm(dl["p_value"].to_numpy())
    dl["significant_holm_0.05"] = dl["p_holm_over_slices"] < 0.05
    dl.to_csv(os.path.join(a.out, "delong_holm_over_slices.csv"), index=False)
    print(dl.round(4).to_string(index=False))

    # 3. PR-AUC + curves + confusion plots
    E = {k: ensemble(f) for k, f in folders.items()}
    man_te = pd.read_csv(os.path.join(MAN, "testing_labels.csv"))[["Image Index", "source"]]
    man_va = pd.read_csv(os.path.join(MAN, "validation_labels.csv"))[["Image Index", "source"]]
    rows = []
    fig_roc, ax_roc = plt.subplots(1, 3, figsize=(15, 4.6))
    fig_pr, ax_pr = plt.subplots(1, 3, figsize=(15, 4.6))
    for j, (sl, csv) in enumerate(SLICES.items()):
        ids = set(pd.read_csv(os.path.join(MAN, csv))["Image Index"])
        for k, e in E.items():
            d = e["test"][e["test"]["Image Index"].isin(ids)].reset_index(drop=True)
            y, p = d["label"].to_numpy().astype(int), d["prob"].to_numpy()
            ap_est = float(average_precision_score(y, p))
            res = stratified_indices(y, a.boot, 42)  # same resamples as evaluate.py
            b = np.array([average_precision_score(y[np.r_[pi, ni]], p[np.r_[pi, ni]]) for pi, ni in res])
            sd = float(b.std(ddof=1))
            rows.append(dict(slice=sl, setup=k, n=len(y), n_pos=int(y.sum()), prevalence=float(y.mean()),
                             pr_auc=ap_est, sd=sd, ci_lower=max(0.0, ap_est - Z95 * sd),
                             ci_upper=min(1.0, ap_est + Z95 * sd)))
            fpr, tpr, _ = roc_curve(y, p)
            ens = pd.read_csv(os.path.join(a.out, sl, "ensemble_metrics.csv"))
            auc = float(ens[(ens.setup == k) & (ens.metric == "auc")]["estimate"].iloc[0])
            ax_roc[j].plot(fpr, tpr, label=f"{NAMES[k]}  AUC {auc:.3f}")
            pr, rc, _ = precision_recall_curve(y, p)
            ax_pr[j].plot(rc, pr, label=f"{NAMES[k]}  PR-AUC {ap_est:.3f}")
            # confusion matrix plot
            t = e["t"]
            pred = p >= t
            cm = np.array([[int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())],
                           [int((~pred & (y == 1)).sum()), int((pred & (y == 1)).sum())]])
            f, ax = plt.subplots(figsize=(3.8, 3.4))
            ax.imshow(cm, cmap="Blues")
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, f"{v:,}", ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black")
            ax.set_xticks([0, 1], ["No Finding", "Pneumonia"])
            ax.set_yticks([0, 1], ["No Finding", "Pneumonia"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{NAMES[k]}\nslice {sl}, threshold {t:.4f}", fontsize=9)
            f.tight_layout()
            f.savefig(os.path.join(a.out, sl, f"confusion_{k}.png"), dpi=150)
            plt.close(f)
        ax_roc[j].plot([0, 1], [0, 1], "k:", lw=0.8)
        ax_roc[j].set(title=f"ROC, slice {sl} (n={len(y):,})", xlabel="1 - specificity", ylabel="sensitivity")
        ax_roc[j].legend(fontsize=8, loc="lower right")
        ax_pr[j].axhline(y.mean(), color="k", ls=":", lw=0.8, label=f"prevalence {y.mean():.3f}")
        ax_pr[j].set(title=f"Precision-recall, slice {sl}", xlabel="recall", ylabel="precision")
        ax_pr[j].legend(fontsize=8, loc="upper right")
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(a.out, "roc_curves.png"), dpi=150)
    fig_pr.tight_layout()
    fig_pr.savefig(os.path.join(a.out, "pr_curves.png"), dpi=150)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "pr_auc.csv"), index=False)

    # 4. shortcut check: real vs synthetic Pneumonia recall (train at the best epoch, val and test)
    sc = []
    for k, f in folders.items():
        e = E[k]
        for mpath in sorted(glob.glob(os.path.join(f, "**", "metrics_*.json"), recursive=True)):
            m = json.load(open(mpath))
            h = pd.read_csv(os.path.join(os.path.dirname(mpath), f"history_{m['run_id']}.csv"))
            best = h[h["epoch"] == m["best_epoch"]].iloc[0]
            row = dict(setup=k, seed=m["seed"], best_epoch=int(m["best_epoch"]), epochs_run=len(h),
                       best_val_auc=m["best_val_auc"],
                       train_recall_real_pn_t05=float(best["train_recall_real_pn"]),
                       train_recall_synth_pn_t05=float(best["train_recall_synth_pn"]),
                       train_acc_nf_t05=float(best["train_acc_nf"]))
            i = e["seeds"].index(m["seed"])
            for part, d0, man in (("val", e["vals"][i], man_va), ("test", e["preds"][i], man_te)):
                d = d0.rename(columns={"prob_pneumonia": "prob"}).merge(man, on="Image Index", how="left")
                assert d["source"].notna().all()
                for tname, t in (("t05", 0.5), ("tseed", m["threshold_from_val"])):
                    row[f"{part}_recall_real_pn_{tname}"] = rec(d, (d.label == 1) & (d.source == "real"), t)
                    row[f"{part}_recall_synth_pn_{tname}"] = rec(d, (d.label == 1) & (d.source == "synthetic"), t)
                    row[f"{part}_spec_nf_{tname}"] = 1 - rec(d, d.label == 0, t)
            sc.append(row)
        # ensemble at the ensemble threshold
        row = dict(setup=k, seed="ensemble")
        for part, d0, man in (("val", e["val"], man_va), ("test", e["test"], man_te)):
            d = d0.merge(man, on="Image Index", how="left")
            row[f"{part}_recall_real_pn_tens"] = rec(d, (d.label == 1) & (d.source == "real"), e["t"])
            row[f"{part}_recall_synth_pn_tens"] = rec(d, (d.label == 1) & (d.source == "synthetic"), e["t"])
            row[f"{part}_spec_nf_tens"] = 1 - rec(d, d.label == 0, e["t"])
            from sklearn.metrics import roc_auc_score
            for src in ("real", "synthetic"):
                mm = (d.label == 0) | (d.source == src)
                row[f"{part}_auc_{src}_pn_vs_nf"] = float(roc_auc_score(d[mm].label, d[mm].prob))
        row["ensemble_threshold"] = e["t"]
        sc.append(row)
    pd.DataFrame(sc).to_csv(os.path.join(a.out, "shortcut_check.csv"), index=False)
    print(pd.DataFrame(sc).round(3).T.to_string())

    # 5. side by side with the main study (12,262-image test set = slice b)
    main = pd.read_csv(os.path.join("6_testing_and_validation", "output", "test_12262", "ensemble_metrics.csv"))
    sup = pd.read_csv(os.path.join(a.out, "b_real_only", "ensemble_metrics.csv"))
    cmp_ = main[main.setup.isin(["S1", "S3"])].merge(sup, on=["setup", "metric"], suffixes=("_main", "_moved"))
    cmp_ = cmp_[["setup", "metric", "estimate_main", "ci_lower_main", "ci_upper_main", "threshold_main",
                 "estimate_moved", "ci_lower_moved", "ci_upper_moved", "threshold_moved"]]
    cmp_["diff_moved_minus_main"] = cmp_["estimate_moved"] - cmp_["estimate_main"]
    cmp_.to_csv(os.path.join(a.out, "compare_with_main_slice_b.csv"), index=False)
    print(cmp_.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
