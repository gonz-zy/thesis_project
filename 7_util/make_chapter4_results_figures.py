"""
Utility: Chapter 4 result figures IV21 to IV25 and the data behind them.

Input:  2_upsampling_branch/output/stylegan2-ada/metric-fid50k_full.jsonl,
        5_training_and_classification/*/output/**/test_predictions_*.csv, shortcut_check.csv,
        6_testing_and_validation/output/test_12262/, test_124/,
        3_downsampling_branch/output/labels/test_balanced_124.csv
Output: 7_util/output/chapter4/figures/fig_IV21 ... fig_IV25 (.png), 7_util/output/chapter4/figure_data/*.csv
Run:    python 7_util/make_chapter4_results_figures.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

RESULTS = "."
ROOT = "."
SEEDS = [42, 43, 44, 45, 46]
SETUPS = {  # key -> (prediction folder, label, colour, line style, marker)
    "S1": (os.path.join("5_training_and_classification", "s1_densenet121_stylegan2-ada", "output", "final", "cbam_all"),
           "Setup 1 (DenseNet-121 + StyleGAN2-ADA)", "#2a78d6", "-", "o"),
    "S3": (os.path.join("5_training_and_classification", "s3_convnextv2_stylegan2-ada", "output", "final", "cbam_all"),
           "Setup 3 (ConvNeXt V2 + StyleGAN2-ADA)", "#eb6834", "--", "s"),
    "S4A": (os.path.join("5_training_and_classification", "s4_convnextv2_cyclegan", "output", "run_A", "final", "cbam_all"),
            "Setup 4 run A (ConvNeXt V2 + CycleGAN)", "#1baf7a", "-.", "^"),
    "S4B": (os.path.join("5_training_and_classification", "s4_convnextv2_cyclegan", "output", "run_B", "final", "cbam_all"),
            "Setup 4 run B (ConvNeXt V2 + CycleGAN)", "#eda100", ":", "D"),
}
SHORT = {"S1": "Setup 1", "S3": "Setup 3", "S4A": "Setup 4 run A", "S4B": "Setup 4 run B"}
MAIN_TEST = {"S1": "full", "S3": "full", "S4A": "full", "S4B": "124"}  # main test set of each setup
TEST_NAME = {"full": "12,262 test images", "124": "124 test images"}
EVAL_DIR = {"full": os.path.join("6_testing_and_validation", "output", "test_12262"),
            "124": os.path.join("6_testing_and_validation", "output", "test_124")}
SUBSET_124 = os.path.join("3_downsampling_branch", "output", "labels", "test_balanced_124.csv")
SHORTCUT_DIR = {"S1": os.path.join("5_training_and_classification", "s1_densenet121_stylegan2-ada", "output"), "S3": os.path.join("5_training_and_classification", "s3_convnextv2_stylegan2-ada", "output")}
CHEHADE_AUC = 0.8387
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def clean_axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def load_ensemble(setup):
    """(mean probability, per-seed probabilities, labels), aligned by Image Index."""
    cols, label = [], None
    for s in SEEDS:
        d = pd.read_csv(os.path.join(RESULTS, SETUPS[setup][0], f"seed{s}", f"test_predictions_cbam-all_seed{s}.csv"))
        d = d.set_index("Image Index").sort_index()
        cols.append(d["prob_pneumonia"].rename(s))
        label = d["label"].astype(int) if label is None else label
    P = pd.concat(cols, axis=1)
    return P.mean(axis=1), P, label.reindex(P.index)


def ensemble_table():
    """AUC, CI and exact threshold of each setup on its MAIN test set (MAIN_TEST)."""
    rows = {}
    for s, t in MAIN_TEST.items():
        em = pd.read_csv(os.path.join(RESULTS, EVAL_DIR[t], "ensemble_metrics.csv"))
        rows[s] = em[(em.metric == "auc") & (em.setup == s)].set_index("setup").loc[s, ["threshold", "estimate", "ci_lower", "ci_upper"]]
    return pd.DataFrame(rows).T.astype(float)


def main_view(ens, s, root):
    """(ensemble probability, labels) of setup s restricted to its MAIN test set."""
    p, _, y = ens[s]
    if MAIN_TEST[s] == "124":
        ids = pd.read_csv(os.path.join(root, SUBSET_124))["Image Index"]
        p, y = p.loc[ids], y.loc[ids]
    return p, y


def counts(y, p, t):
    pred = (p >= t).astype(int)
    return dict(tn=int(((y == 0) & (pred == 0)).sum()), fp=int(((y == 0) & (pred == 1)).sum()),
                fn=int(((y == 1) & (pred == 0)).sum()), tp=int(((y == 1) & (pred == 1)).sum()))


# Figure IV21
def fig_fid(out):
    rows = [json.loads(l) for l in open(os.path.join(RESULTS, "2_upsampling_branch", "output", "stylegan2-ada", "metric-fid50k_full.jsonl"))]
    d = pd.DataFrame({"kimg": [int(r["snapshot_pkl"].split("-")[-1][:6]) for r in rows],
                      "fid": [r["results"]["fid50k_full"] for r in rows]}).sort_values("kimg")
    b = d.loc[d.fid.idxmin()]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(d.kimg, d.fid, color="#2a78d6", lw=2, marker="o", ms=5)
    ax.scatter([b.kimg], [b.fid], s=90, facecolor="none", edgecolor=INK, lw=1.4, zorder=4)
    ax.annotate(f"lowest FID {b.fid:.2f} at {int(b.kimg):,} kimg\n(snapshot used for generation)",
                (b.kimg, b.fid), xytext=(3600, 76), textcoords="data", fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("Training progress (kimg)")
    ax.set_ylabel("FID (50,000 generated images)")
    ax.set_xlim(800, 5200)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_IV21_fid_by_snapshot.png"), dpi=300)
    plt.close(fig)
    return d


# Figure IV22
def fig_roc(out, ens, et):
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (2, 2)), label="Chance")
    for s, (folder, lab, col, ls, mk) in SETUPS.items():
        p, y = main_view(ens, s, ROOT)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=col, ls=ls, lw=1.8, label=f"{lab}\n{TEST_NAME[MAIN_TEST[s]]}, AUC = {auc:.3f}")
        c = counts(y, p, et.loc[s, "threshold"])
        ax.scatter(c["fp"] / (c["fp"] + c["tn"]), c["tp"] / (c["tp"] + c["fn"]), marker=mk, s=55,
                   facecolor=col, edgecolor="white", lw=1.2, zorder=4)
    ax.scatter([], [], marker="o", s=40, facecolor="none", edgecolor=INK, label="Operating point (Youden threshold)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    clean_axes(ax)
    ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.95, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_IV22_roc_curves.png"), dpi=300)
    plt.close(fig)


# Figure IV23
def fig_auc_ci(out, et):
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    x = np.arange(len(SETUPS))
    for i, s in enumerate(SETUPS):
        r = et.loc[s]
        ax.errorbar(i, r.estimate, yerr=[[r.estimate - r.ci_lower], [r.ci_upper - r.estimate]], fmt="o",
                    color=INK, ms=6, capsize=5, lw=1.4)
        ax.text(i + 0.08, r.estimate, f"{r.estimate:.3f}\n({r.ci_lower:.3f} to {r.ci_upper:.3f})",
                fontsize=7.5, va="center", ha="left", color=INK)
    ax.axhline(CHEHADE_AUC, color=MUTED, ls="--", lw=1)
    ax.text(len(SETUPS) - 0.55, CHEHADE_AUC + 0.008, "0.8387 reported by Chehade et al. (2025), different test sets",
            ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.axhline(0.5, color=MUTED, ls=":", lw=0.8)
    ax.text(-0.4, 0.505, "chance (0.5)", fontsize=7.5, color=MUTED, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([SETUPS[s][1].replace(" (", "\n").replace(" + ", "\n+ ").rstrip(")")
                        + "\n[" + TEST_NAME[MAIN_TEST[s]] + "]" for s in SETUPS], fontsize=7.5)
    ax.set_xlim(-0.5, len(SETUPS) - 0.3)
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("AUC (95% CI)")
    clean_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_IV23_auc_ci.png"), dpi=300)
    plt.close(fig)


# Figure IV24
def fig_confusion(out, ens, et):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.3))
    found = {}
    for ax, s in zip(axes.flat, SETUPS):
        ref = pd.read_csv(os.path.join(RESULTS, EVAL_DIR[MAIN_TEST[s]], "confusion_matrices.csv"))
        ref = ref[ref.level == "ensemble"].set_index("setup")
        p, y = main_view(ens, s, ROOT)
        t = et.loc[s, "threshold"]
        c = counts(y, p, t)
        for k in ("tp", "fn", "fp", "tn"):  # guard: must equal evaluate.py exactly
            if c[k] != int(ref.loc[s, k]):
                raise SystemExit(f"{s} {k}: figure {c[k]} != {EVAL_DIR[MAIN_TEST[s]]}/confusion_matrices.csv {int(ref.loc[s, k])}")
        found[s] = c
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        share = m / m.sum(axis=1, keepdims=True)  # row share = specificity / recall view
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]:,}\n({100 * share[i, j]:.1f}%)", ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Finding", "Pneumonia"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No Finding", "Pneumonia"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{SHORT[s]} ({TEST_NAME[MAIN_TEST[s]]})\nthreshold = {t:.6f}", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.5, 0.005, "Percentages are of the true class (row). No Finding row = specificity; Pneumonia row = recall.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out, "fig_IV24_confusion_matrices.png"), dpi=300)
    plt.close(fig)
    return found


# Figure IV25
def fig_shortcut(out):
    frames = []
    for s, folder in (("S1", SHORTCUT_DIR["S1"]), ("S3", SHORTCUT_DIR["S3"])):
        d = pd.read_csv(os.path.join(RESULTS, folder, "final", "shortcut_check.csv"))
        d["setup"] = s
        frames.append(d)
    d = pd.concat(frames)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True)
    w = 0.38
    for ax, s in zip(axes, ("S1", "S3")):
        r = d[d.setup == s].sort_values("seed")
        x = np.arange(len(r))
        ax.bar(x - w / 2 - 0.01, r.real_recall_best, w, color="#2a78d6", label="Real Pneumonia images")
        ax.bar(x + w / 2 + 0.01, r.synth_recall_best, w, color="#eb6834", label="Synthetic Pneumonia images")
        for xi, v in zip(x, r.real_recall_best):
            ax.text(xi - w / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in r.seed])
        ax.set_xlabel("Seed")
        ax.set_title(SETUPS[s][1], fontsize=8.5)
        ax.set_ylim(0, 1.12)
        clean_axes(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Training recall at best epoch")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, "fig_IV25_shortcut_check.png"), dpi=300)
    plt.close(fig)
    return d


# data behind each figure
def export_data(dout, ens, et, fid):
    """One CSV per figure with exactly the numbers that are plotted, for manual checking."""
    os.makedirs(dout, exist_ok=True)
    # IV19 training curves (data only; the figure itself is not drawn by this script)
    rows = []
    for s in SETUPS:
        for seed in SEEDS:
            h = pd.read_csv(os.path.join(RESULTS, SETUPS[s][0], f"seed{seed}", f"history_cbam-all_seed{seed}.csv"))
            h.insert(0, "seed", seed)
            h.insert(0, "setup", SHORT[s])
            rows.append(h[["setup", "seed", "epoch", "train_loss", "val_auc"]])
    pd.concat(rows).to_csv(os.path.join(dout, "fig_IV19_training_curves_data.csv"), index=False)
    # IV20 sample probabilities (copy of figures/table_IV_sample_predictions.csv)
    sp = os.path.join(RESULTS, "7_util", "output", "chapter4", "figures", "table_IV_sample_predictions.csv")
    if os.path.exists(sp):
        pd.read_csv(sp).to_csv(os.path.join(dout, "fig_IV20_sample_probabilities_data.csv"), index=False)
    # IV21 FID
    fid.assign(lowest=fid.fid == fid.fid.min()).to_csv(os.path.join(dout, "fig_IV21_fid_data.csv"), index=False)
    # IV22 and IV24: every test image, per-seed and ensemble probability of every setup
    first = next(iter(SETUPS))
    tab = pd.DataFrame({"label": ens[first][2]})
    for s in SETUPS:
        p, P, y = ens[s]
        assert (y == tab.label).all()
        for seed in SEEDS:
            tab[f"{s}_seed{seed}"] = P[seed]
        tab[f"{s}_ensemble"] = p
    tab.index.name = "Image Index"
    ids124 = set(pd.read_csv(os.path.join(ROOT, SUBSET_124))["Image Index"])
    tab["in_124_test_set"] = tab.index.isin(ids124).astype(int)
    tab.to_csv(os.path.join(dout, "fig_IV22_IV24_test_probabilities.csv"), float_format="%.10g")
    # IV22 ROC points and operating points
    roc = []
    for s in SETUPS:
        p, y = main_view(ens, s, ROOT)
        fpr, tpr, thr = roc_curve(y, p)
        roc.append(pd.DataFrame({"setup": SHORT[s], "test_set": TEST_NAME[MAIN_TEST[s]], "threshold": thr,
                                 "fpr": fpr, "tpr": tpr}))
    pd.concat(roc).to_csv(os.path.join(dout, "fig_IV22_roc_points.csv"), index=False)
    # IV23 AUC and CI;  IV24 counts at the exact thresholds
    out = []
    for s in SETUPS:
        p, y = main_view(ens, s, ROOT)
        c = counts(y, p, et.loc[s, "threshold"])
        out.append(dict(setup=SHORT[s], test_set=TEST_NAME[MAIN_TEST[s]], threshold=et.loc[s, "threshold"], auc=et.loc[s, "estimate"],
                        ci_lower=et.loc[s, "ci_lower"], ci_upper=et.loc[s, "ci_upper"], TN=c["tn"], FP=c["fp"],
                        FN=c["fn"], TP=c["tp"], recall=c["tp"] / (c["tp"] + c["fn"]),
                        specificity=c["tn"] / (c["tn"] + c["fp"])))
    pd.DataFrame(out).to_csv(os.path.join(dout, "fig_IV23_IV24_auc_thresholds_counts.csv"), index=False,
                             float_format="%.10g")
    # IV25 shortcut check
    frames = []
    for s, folder in (("S1", SHORTCUT_DIR["S1"]), ("S3", SHORTCUT_DIR["S3"])):
        d = pd.read_csv(os.path.join(RESULTS, folder, "final", "shortcut_check.csv"))
        d.insert(0, "setup", SHORT[s])
        frames.append(d[["setup", "seed", "best_epoch", "real_recall_best", "synth_recall_best",
                         "real_recall_last", "synth_recall_last"]])
    pd.concat(frames).to_csv(os.path.join(dout, "fig_IV25_shortcut_check_data.csv"), index=False)


def main():
    global RESULTS, ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS, help="project root")
    ap.add_argument("--root", default=".", help="project root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data_out", default=None, help="folder for the per-figure data CSVs (default <out>/../figure_data)")
    a = ap.parse_args()
    RESULTS = a.results
    ROOT = a.root
    out = a.out or os.path.join(RESULTS, "7_util", "output", "chapter4", "figures")
    os.makedirs(out, exist_ok=True)

    et = ensemble_table()
    ens = {s: load_ensemble(s) for s in SETUPS}
    fid = fig_fid(out)
    fig_roc(out, ens, et)
    fig_auc_ci(out, et)
    found = fig_confusion(out, ens, et)
    fig_shortcut(out)
    dout = a.data_out or os.path.join(os.path.dirname(out.rstrip("/\\")), "figure_data")
    export_data(dout, ens, et, fid)

    print("FID by snapshot:\n" + fid.to_string(index=False))
    print("\nMain-experiment confusion counts at the exact Youden thresholds (all equal confusion_matrices.csv):")
    for s, c in found.items():
        print(f"  {SHORT[s]:14s} [{TEST_NAME[MAIN_TEST[s]]:18s}] thr {et.loc[s, 'threshold']:.9f}  TN {c['tn']:,}  FP {c['fp']:,}"
              f"  FN {c['fn']}  TP {c['tp']}  recall {c['tp'] / (c['tp'] + c['fn']):.3f}")
    print("figures written to", out)
    print("figure data written to", dout)


if __name__ == "__main__":
    main()
