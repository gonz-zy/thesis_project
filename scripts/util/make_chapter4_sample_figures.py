"""
make_chapter4_sample_figures.py -- figures and table for Chapter IV, Section C (sample simulation), modules B5 and B6.

Reads only files that already exist in results\\ (nothing is retrained):
    <setup>\\final\\cbam_all\\seed42..46\\history_cbam-all_seed##.csv         epoch, train_loss, val_auc, ...
    <setup>\\final\\cbam_all\\seed42..46\\test_predictions_cbam-all_seed##.csv Image Index, label, prob_pneumonia
    evaluation_final\\ensemble_metrics.csv                                      Youden threshold of each setup (validation)

Writes into --out (default results\\chapter4\\figures):
    fig_B5_training_curves_all_setups.png   validation AUC and training loss per epoch, all five seeds of every setup
    fig_B6_sample_probabilities.png         ensemble probability of the sample test images, one panel per setup
    table_IV_sample_predictions.csv         the numbers behind the B6 figure (ensemble probability, threshold, prediction,
                                            outcome, per-seed probabilities)
    table_IV_sample_counts.csv              TP / FP / TN / FN of the sample images per setup

The ensemble probability is the mean of the five seeds' test probabilities, exactly the rule used by
evaluation_final (--check recomputes each setup's full-test AUC from the same files and compares it with
evaluation_final\\ensemble_metrics.csv, as a guard that the averaging is the same).

Run (project root):
    venv\\Scripts\\python.exe scripts\\util\\make_chapter4_sample_figures.py
    venv\\Scripts\\python.exe scripts\\util\\make_chapter4_sample_figures.py --images 00013721_000.png 00001403_000.png 00013272_005.png 00017659_000.png --check
Needs: pandas, numpy, matplotlib (scikit-learn only for --check).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

RESULTS = "results"
SEEDS = [42, 43, 44, 45, 46]
SETUPS = {  # key -> (folder under results, panel title)
    "S1": (os.path.join("densenet121_stylegan2-ada", "final", "cbam_all"), "Setup 1: DenseNet-121\n+ StyleGAN2-ADA"),
    "S3": (os.path.join("convnextv2_stylegan2-ada", "final", "cbam_all"), "Setup 3: ConvNeXt V2\n+ StyleGAN2-ADA"),
    "S4A": (os.path.join("convnextv2_cyclegan", "run_A", "final", "cbam_all"), "Setup 4 run A: ConvNeXt V2\n+ CycleGAN"),
    "S4B": (os.path.join("convnextv2_cyclegan", "run_B", "final", "cbam_all"), "Setup 4 run B: ConvNeXt V2\n+ CycleGAN"),
}
# default sample: N1 and P1 are in the artifact group, N2 and P2 in the clean group (reviewer and K-Means agree)
DEFAULT_IMAGES = ["00013721_000.png", "00001403_000.png", "00013272_005.png", "00017659_000.png"]
IMAGE_NAMES = ["N1", "N2", "P1", "P2"]
BLUE, RED, GREY = "#1f5fa8", "#b3261e", "#444444"

plt.rcParams.update({"font.family": "serif", "font.size": 9})


def seed_dir(setup, seed):
    return os.path.join(RESULTS, SETUPS[setup][0], f"seed{seed}")


def load_history(setup, seed):
    return pd.read_csv(os.path.join(seed_dir(setup, seed), f"history_cbam-all_seed{seed}.csv"))


def load_test(setup):
    """Return (probabilities DataFrame: rows = Image Index, columns = seeds, true label Series)."""
    cols, label = [], None
    for s in SEEDS:
        d = pd.read_csv(os.path.join(seed_dir(setup, s), f"test_predictions_cbam-all_seed{s}.csv")).set_index("Image Index")
        cols.append(d["prob_pneumonia"].rename(s))
        label = d["label"] if label is None else label
    return pd.concat(cols, axis=1), label


def thresholds():
    em = pd.read_csv(os.path.join(RESULTS, "evaluation_final", "ensemble_metrics.csv"))
    return em.groupby("setup")["threshold"].first().to_dict()


def fig_training_curves(out):
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.6), sharex="col")
    for j, s in enumerate(SETUPS):
        for k, seed in enumerate(SEEDS):
            h = load_history(s, seed)
            bold = seed == 42
            ax[0, j].plot(h["epoch"], h["val_auc"], color=BLUE, lw=1.8 if bold else 0.8, alpha=1 if bold else 0.55)
            b = h["val_auc"].idxmax()
            ax[0, j].scatter(h["epoch"][b], h["val_auc"][b], s=18, color=RED, zorder=3)
            ax[1, j].plot(h["epoch"], h["train_loss"], color=GREY, lw=1.8 if bold else 0.8, alpha=1 if bold else 0.55)
        ax[0, j].set_title(SETUPS[s][1], fontsize=9)
        ax[0, j].grid(alpha=0.25)
        ax[1, j].grid(alpha=0.25)
        ax[1, j].set_yscale("log")
        ax[1, j].set_xlabel("Epoch")
        ax[0, j].xaxis.set_major_locator(MaxNLocator(integer=True))
        for r in (0, 1):
            for sp in ("top", "right"):
                ax[r, j].spines[sp].set_visible(False)
    ax[0, 0].set_ylabel("Validation AUC")
    ax[1, 0].set_ylabel("Training loss (log scale)")
    fig.legend(handles=[plt.Line2D([], [], color=BLUE, lw=1.8, label="seed 42 (bold) and seeds 43 to 46"),
                        plt.Line2D([], [], marker="o", ls="", color=RED, label="best validation epoch of each seed")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "fig_B5_training_curves_all_setups.png"), dpi=200)
    plt.close(fig)


def sample_table(images):
    thr = thresholds()
    rows = []
    for s in SETUPS:
        P, label = load_test(s)
        ens = P.mean(axis=1)
        for name, im in zip(IMAGE_NAMES, images):
            y, p = int(label[im]), float(ens[im])
            pred = int(p >= thr[s])
            outcome = {(1, 1): "TP", (0, 1): "FP", (0, 0): "TN", (1, 0): "FN"}[(y, pred)]
            rows.append(dict(setup=s, sample=name, image=im, label=y, ensemble_prob=p, threshold=thr[s],
                             pred=pred, outcome=outcome,
                             rank_percentile=100 * float((ens < p).mean()),
                             **{f"seed{k}": float(P.loc[im, k]) for k in SEEDS}))
    return pd.DataFrame(rows)


def fig_sample_probabilities(df, out):
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    names = {n: f"{n}  {'Pneumonia' if n.startswith('P') else 'No Finding'}\n{df[df['sample'] == n].image.iloc[0][:-4]}"
             for n in IMAGE_NAMES}
    for a, s in zip(ax, SETUPS):
        d = df[df.setup == s].set_index("sample")
        thr = d.threshold.iloc[0]
        vals = list(d.ensemble_prob) + [thr]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.25
        a.set_xlim(lo - pad, hi + pad)
        small = s in ("S1", "S3")
        for i, n in enumerate(IMAGE_NAMES):
            y, p = 3 - i, d.loc[n, "ensemble_prob"]
            pos = d.loc[n, "label"] == 1
            a.scatter(p, y, s=70, marker="s" if pos else "o", color=RED if pos else BLUE, zorder=3)
            a.annotate(f"{p:.4f}" if small else f"{p:.3f}", (p, y), textcoords="offset points", xytext=(0, 9),
                       ha="center", fontsize=8)
        a.axvline(thr, color=GREY, ls="--", lw=1)
        a.text(thr, 3.55, f"threshold {thr:.4f}" if small else f"threshold {thr:.3f}", fontsize=7.5, color=GREY,
               ha="center", va="bottom")
        a.xaxis.set_major_locator(MaxNLocator(3))
        a.tick_params(axis="x", labelsize=8)
        a.set_title(SETUPS[s][1], fontsize=9, pad=16)
        a.set_yticks(range(4))
        a.set_yticklabels([names[n] for n in IMAGE_NAMES][::-1], fontsize=8)
        a.set_ylim(-0.5, 3.5)
        a.grid(axis="x", alpha=0.25)
        a.set_xlabel("Ensemble probability of Pneumonia")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, label="true label: No Finding"),
                        plt.Line2D([], [], marker="s", ls="", color=RED, label="true label: Pneumonia"),
                        plt.Line2D([], [], ls="--", color=GREY, label="Youden threshold from validation")],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, "fig_B6_sample_probabilities.png"), dpi=200)
    plt.close(fig)


def check_against_evaluation_final():
    from sklearn.metrics import roc_auc_score
    em = pd.read_csv(os.path.join(RESULTS, "evaluation_final", "ensemble_metrics.csv"))
    ref = em[em.metric == "auc"].set_index("setup")["estimate"]
    print("full-test ensemble AUC recomputed here vs evaluation_final:")
    for s in SETUPS:
        P, label = load_test(s)
        auc = roc_auc_score(label.reindex(P.index), P.mean(axis=1))
        print(f"  {s:4s} {auc:.4f}  vs  {ref[s]:.4f}   {'OK' if abs(auc - ref[s]) < 1e-3 else 'DIFFERENT'}")


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--out", default=os.path.join(RESULTS, "chapter4", "figures"))
    ap.add_argument("--images", nargs=4, default=DEFAULT_IMAGES, help="Image Index of N1 N2 P1 P2, in this order")
    ap.add_argument("--check", action="store_true", help="recompute full-test ensemble AUCs and compare with evaluation_final")
    a = ap.parse_args()
    RESULTS = a.results
    os.makedirs(a.out, exist_ok=True)

    fig_training_curves(a.out)
    df = sample_table(a.images)
    df.to_csv(os.path.join(a.out, "table_IV_sample_predictions.csv"), index=False)
    counts = df.pivot_table(index="setup", columns="outcome", values="image", aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=["TP", "FP", "TN", "FN"], fill_value=0).reindex(list(SETUPS))
    counts.to_csv(os.path.join(a.out, "table_IV_sample_counts.csv"))
    fig_sample_probabilities(df, a.out)
    print(df[["setup", "sample", "image", "label", "ensemble_prob", "threshold", "pred", "outcome"]].to_string(index=False))
    print(counts)
    if a.check:
        check_against_evaluation_final()
    print("written to", a.out)


if __name__ == "__main__":
    main()
