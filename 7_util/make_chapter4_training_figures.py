"""
Utility: Chapter 4 Figure IV19 (training curves) and Figure IV20 (sample probabilities), plus the sample tables.

Input:  5_training_and_classification/*/output/**/final/cbam_all/seed42..46/history_*.csv, metrics_*.json,
        test_predictions_*.csv
        6_testing_and_validation/output/test_12262/ and test_124/ ensemble_metrics.csv (Youden thresholds)
Output: 7_util/output/chapter4/figures/fig_IV19_training_curves.png, fig_IV20_sample_probabilities.png,
        table_IV_sample_predictions.csv, table_IV_sample_counts.csv
Run:    python 7_util/make_chapter4_training_figures.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import percentileofscore

T5 = "5_training_and_classification"
SEEDS = [42, 43, 44, 45, 46]
SETUPS = {  # key -> (final run folder, title)
    "S1": (os.path.join(T5, "s1_densenet121_stylegan2-ada", "output", "final", "cbam_all"),
           "Setup 1: DenseNet-121\n+ StyleGAN2-ADA"),
    "S3": (os.path.join(T5, "s3_convnextv2_stylegan2-ada", "output", "final", "cbam_all"),
           "Setup 3: ConvNeXt V2\n+ StyleGAN2-ADA"),
    "S4A": (os.path.join(T5, "s4_convnextv2_cyclegan", "output", "run_A", "final", "cbam_all"),
            "Setup 4 run A: ConvNeXt V2\n+ CycleGAN"),
    "S4B": (os.path.join(T5, "s4_convnextv2_cyclegan", "output", "run_B", "final", "cbam_all"),
            "Setup 4 run B: ConvNeXt V2\n+ CycleGAN"),
}
EVAL_DIR = {"S1": "test_12262", "S3": "test_12262", "S4A": "test_12262", "S4B": "test_124"}  # main test set
# the four sample images shown in Figures IV11 to IV18
SAMPLES = [("N1", "00013721_000.png"), ("N2", "00001403_000.png"),
           ("P1", "00013272_005.png"), ("P2", "00017659_000.png")]

BLUE, LIGHT_BLUE, RED = "#2a62b0", "#7fa3d4", "#b8251f"
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def run_files(setup, seed, kind):
    folder = os.path.join(SETUPS[setup][0], f"seed{seed}")
    ext = "json" if kind == "metrics" else "csv"
    return os.path.join(folder, f"{kind}_cbam-all_seed{seed}.{ext}")


# Figure IV19: validation AUC (top) and training loss (bottom) of every seed
def fig_training_curves(out):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex="col")
    for c, s in enumerate(SETUPS):
        top, bot = axes[0, c], axes[1, c]
        for seed in SEEDS:
            h = pd.read_csv(run_files(s, seed, "history"))
            best = json.load(open(run_files(s, seed, "metrics")))["best_epoch"]
            main = seed == SEEDS[0]
            top.plot(h.epoch, h.val_auc, color=BLUE if main else LIGHT_BLUE,
                     lw=1.8 if main else 0.9, zorder=3 if main else 2)
            b = h[h.epoch == best].iloc[0]
            top.scatter([b.epoch], [b.val_auc], color=RED, s=18, zorder=4)
            bot.plot(h.epoch, h.train_loss, color="#333333" if main else "#9a9a9a",
                     lw=1.8 if main else 0.9, zorder=3 if main else 2)
        bot.set_yscale("log")
        top.set_title(SETUPS[s][1], fontsize=9)
        bot.set_xlabel("Epoch")
        bot.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        for ax in (top, bot):
            clean_axes(ax)
    axes[0, 0].set_ylabel("Validation AUC")
    axes[1, 0].set_ylabel("Training loss (log scale)")
    handles = [Line2D([], [], color=BLUE, lw=1.8, label="seed 42 (bold) and seeds 43 to 46"),
               Line2D([], [], color=RED, marker="o", ls="", ms=5, label="best validation epoch of each seed")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "fig_IV19_training_curves.png"), dpi=300)
    plt.close(fig)


def sample_table():
    """Ensemble probability, threshold and outcome of the four sample images for every setup."""
    rows = []
    for s in SETUPS:
        probs = pd.concat([pd.read_csv(run_files(s, k, "test_predictions")).set_index("Image Index")
                           ["prob_pneumonia"].rename(f"seed{k}") for k in SEEDS], axis=1)
        labels = pd.read_csv(run_files(s, SEEDS[0], "test_predictions")).set_index("Image Index")["label"]
        ens = probs.mean(axis=1)
        em = pd.read_csv(os.path.join("6_testing_and_validation", "output", EVAL_DIR[s], "ensemble_metrics.csv"))
        thr = float(em[(em.setup == s) & (em.metric == "auc")].threshold.iloc[0])
        for name, img in SAMPLES:
            y, p = int(labels[img]), float(ens[img])
            pred = int(p >= thr)
            outcome = {(1, 1): "TP", (0, 1): "FP", (0, 0): "TN", (1, 0): "FN"}[(y, pred)]
            rows.append(dict(setup=s, sample=name, image=img, label=y, ensemble_prob=p, threshold=thr,
                             pred=pred, outcome=outcome,
                             rank_percentile=percentileofscore(ens, p, kind="strict"),  # among all 12,262 test images
                             **probs.loc[img].to_dict()))
    tab = pd.DataFrame(rows)
    counts = (tab.groupby("setup", sort=False).outcome.value_counts().unstack(fill_value=0)
              .reindex(columns=["TP", "FP", "TN", "FN"], fill_value=0).reset_index())
    return tab, counts


# Figure IV20: ensemble probability of the four sample images, with the Youden threshold
def fig_sample_probabilities(out, tab):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    ypos = {name: len(SAMPLES) - 1 - i for i, (name, _) in enumerate(SAMPLES)}
    for c, s in enumerate(SETUPS):
        ax = axes[c]
        d = tab[tab.setup == s]
        thr = d.threshold.iloc[0]
        small = d.ensemble_prob.max() < 0.05
        for r in d.itertuples():
            y = ypos[r.sample]
            ax.scatter([r.ensemble_prob], [y], s=55, zorder=3,
                       marker="s" if r.label == 1 else "o", color=RED if r.label == 1 else BLUE)
            ax.annotate(f"{r.ensemble_prob:.4f}" if small else f"{r.ensemble_prob:.3f}", (r.ensemble_prob, y),
                        xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
        ax.axvline(thr, color="#444444", ls="--", lw=1)
        ax.text(thr, len(SAMPLES) - 0.45, f"threshold {thr:.4f}" if small else f"threshold {thr:.3f}",
                ha="center", va="bottom", fontsize=7, color=MUTED)
        lo, hi = min(d.ensemble_prob.min(), thr), max(d.ensemble_prob.max(), thr)
        pad = (hi - lo) * 0.18
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.6, len(SAMPLES) - 0.4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_yticks(list(ypos.values()))
        if c == 0:
            names = dict(SAMPLES)
            ax.set_yticklabels([f"{n}  {'Pneumonia' if n.startswith('P') else 'No Finding'}\n"
                                f"{names[n].replace('.png', '')}" for n in ypos], fontsize=7.5)
        else:
            ax.set_yticklabels([])
        ax.set_title(SETUPS[s][1], fontsize=9, pad=14)
        ax.set_xlabel("Ensemble probability of Pneumonia")
        clean_axes(ax)
        ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], color=BLUE, marker="o", ls="", ms=6, label="true label: No Finding"),
               Line2D([], [], color=RED, marker="s", ls="", ms=6, label="true label: Pneumonia"),
               Line2D([], [], color="#444444", ls="--", lw=1, label="Youden threshold from validation")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, "fig_IV20_sample_probabilities.png"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("7_util", "output", "chapter4", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_training_curves(a.out)
    tab, counts = sample_table()
    tab.to_csv(os.path.join(a.out, "table_IV_sample_predictions.csv"), index=False)
    counts.to_csv(os.path.join(a.out, "table_IV_sample_counts.csv"), index=False)
    fig_sample_probabilities(a.out, tab)
    print(tab[["setup", "sample", "ensemble_prob", "threshold", "outcome"]].to_string(index=False))
    print("written to", a.out)


if __name__ == "__main__":
    main()
