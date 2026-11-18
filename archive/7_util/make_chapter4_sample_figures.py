"""
make_chapter4_results_figures.py -- figures for Chapter IV, Section D (Figures IV21 to IV25).

Replaces the earlier, unsaved figure code. Reads only files that already exist in results\\ (nothing is retrained):
    stylegan2-ada\\metric-fid50k_full.jsonl                       FID of every saved StyleGAN2-ADA snapshot
    <setup>\\final\\cbam_all\\seed42..46\\test_predictions_*.csv     Image Index, label, prob_pneumonia
    evaluation_final\\ensemble_metrics.csv                          AUC, 95% CI and the EXACT Youden threshold per setup
    evaluation_final\\confusion_matrices.csv                        ensemble TP / FN / FP / TN (used as a check)
    densenet121_stylegan2-ada\\final\\shortcut_check.csv            training recall on real vs synthetic Pneumonia
    convnextv2_stylegan2-ada\\final\\shortcut_check.csv

Writes into --out (default results\\chapter4\\figures):
    fig_IV21_fid_by_snapshot.png
    fig_IV22_roc_curves.png
    fig_IV23_auc_ci.png
    fig_IV24_confusion_matrices.png
    fig_IV25_shortcut_check.png

The ensemble probability is the mean of the five seeds' test probabilities, the same rule as evaluate.py.
The thresholds are read at full precision from ensemble_metrics.csv and are NEVER rounded before use; rounding
only happens in the printed labels. The script stops if its confusion counts differ from confusion_matrices.csv.

Run (project root):
    venv\\Scripts\\python.exe scripts\\util\\make_chapter4_results_figures.py
Needs: pandas, numpy, matplotlib, scikit-learn.
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

RESULTS = "results"
SEEDS = [42, 43, 44, 45, 46]
SETUPS = {  # key -> (folder under results, label, colour, line style, marker)
    "S1": (os.path.join("densenet121_stylegan2-ada", "final", "cbam_all"),
           "Setup 1 (DenseNet-121 + StyleGAN2-ADA)", "#2a78d6", "-", "o"),
    "S3": (os.path.join("convnextv2_stylegan2-ada", "final", "cbam_all"),
           "Setup 3 (ConvNeXt V2 + StyleGAN2-ADA)", "#eb6834", "--", "s"),
    "S4A": (os.path.join("convnextv2_cyclegan", "run_A", "final", "cbam_all"),
            "Setup 4 run A (ConvNeXt V2 + CycleGAN)", "#1baf7a", "-.", "^"),
    "S4B": (os.path.join("convnextv2_cyclegan", "run_B", "final", "cbam_all"),
            "Setup 4 run B (ConvNeXt V2 + CycleGAN)", "#eda100", ":", "D"),
}
SHORT = {"S1": "Setup 1", "S3": "Setup 3", "S4A": "Setup 4 run A", "S4B": "Setup 4 run B"}
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
    em = pd.read_csv(os.path.join(RESULTS, "evaluation_final", "ensemble_metrics.csv"))
    return em[em.metric == "auc"].set_index("setup")[["threshold", "estimate", "ci_lower", "ci_upper"]]


def counts(y, p, t):
    pred = (p >= t).astype(int)
    return dict(tn=int(((y == 0) & (pred == 0)).sum()), fp=int(((y == 0) & (pred == 1)).sum()),
                fn=int(((y == 1) & (pred == 0)).sum()), tp=int(((y == 1) & (pred == 1)).sum()))


# ------------------------------------------------------------------ Figure IV21
def fig_fid(out):
    rows = [json.loads(l) for l in open(os.path.join(RESULTS, "stylegan2-ada", "metric-fid50k_full.jsonl"))]
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


# ------------------------------------------------------------------ Figure IV22
def fig_roc(out, ens, et):
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (2, 2)), label="Chance")
    for s, (folder, lab, col, ls, mk) in SETUPS.items():
        p, _, y = ens[s]
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=col, ls=ls, lw=1.8, label=f"{lab}, AUC = {auc:.3f}")
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


# ------------------------------------------------------------------ Figure IV23
def fig_auc_ci(out, et):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(SETUPS))
    for i, s in enumerate(SETUPS):
        r = et.loc[s]
        ax.errorbar(i, r.estimate, yerr=[[r.estimate - r.ci_lower], [r.ci_upper - r.estimate]], fmt="o",
                    color=INK, ms=6, capsize=5, lw=1.4)
        ax.text(i + 0.08, r.estimate, f"{r.estimate:.3f}\n({r.ci_lower:.3f} to {r.ci_upper:.3f})",
                fontsize=7.5, va="center", ha="left", color=INK)
    ax.axhline(CHEHADE_AUC, color=MUTED, ls="--", lw=1)
    ax.text(len(SETUPS) - 0.55, CHEHADE_AUC + 0.008, "0.8387 reported by Chehade et al. (2025), different test set",
            ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.axhline(0.5, color=MUTED, ls=":", lw=0.8)
    ax.text(-0.4, 0.505, "chance (0.5)", fontsize=7.5, color=MUTED, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([SETUPS[s][1].replace(" (", "\n").replace(" + ", "\n+ ").rstrip(")") for s in SETUPS],
                       fontsize=7.5)
    ax.set_xlim(-0.5, len(SETUPS) - 0.3)
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("AUC (95% CI)")
    clean_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_IV23_auc_ci.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ Figure IV24
def fig_confusion(out, ens, et):
    ref = pd.read_csv(os.path.join(RESULTS, "evaluation_final", "confusion_matrices.csv"))
    ref = ref[ref.level == "ensemble"].set_index("setup")
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.0))
    found = {}
    for ax, s in zip(axes.flat, SETUPS):
        p, _, y = ens[s]
        t = et.loc[s, "threshold"]
        c = counts(y, p, t)
        for k in ("tp", "fn", "fp", "tn"):  # guard: must equal evaluate.py exactly
            if c[k] != int(ref.loc[s, k]):
                raise SystemExit(f"{s} {k}: figure {c[k]} != confusion_matrices.csv {int(ref.loc[s, k])}")
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
        ax.set_title(f"{SHORT[s]}\nthreshold = {t:.6f}", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.5, 0.005, "Percentages are of the true class (row). No Finding row = specificity; Pneumonia row = recall.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out, "fig_IV24_confusion_matrices.png"), dpi=300)
    plt.close(fig)
    return found


# ------------------------------------------------------------------ Figure IV25
def fig_shortcut(out):
    frames = []
    for s, folder in (("S1", "densenet121_stylegan2-ada"), ("S3", "convnextv2_stylegan2-ada")):
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


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    RESULTS = a.results
    out = a.out or os.path.join(RESULTS, "chapter4", "figures")
    os.makedirs(out, exist_ok=True)

    et = ensemble_table()
    ens = {s: load_ensemble(s) for s in SETUPS}
    fid = fig_fid(out)
    fig_roc(out, ens, et)
    fig_auc_ci(out, et)
    found = fig_confusion(out, ens, et)
    fig_shortcut(out)

    print("FID by snapshot:\n" + fid.to_string(index=False))
    print("\nEnsemble confusion counts at the exact Youden thresholds (all equal confusion_matrices.csv):")
    for s, c in found.items():
        print(f"  {SHORT[s]:14s} thr {et.loc[s, 'threshold']:.9f}  TN {c['tn']:,}  FP {c['fp']:,}  FN {c['fn']}  TP {c['tp']}"
              f"  recall {c['tp'] / 59:.3f}")
    print("written to", out)


if __name__ == "__main__":
    main()