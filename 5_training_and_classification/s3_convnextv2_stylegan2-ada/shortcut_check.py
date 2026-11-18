"""
Setup 3: training recall on real vs synthetic Pneumonia of every final run.

Input:  5_training_and_classification/s3_convnextv2_stylegan2-ada/output/final/**/history_*.csv, metrics_*.json
Output: 5_training_and_classification/s3_convnextv2_stylegan2-ada/output/final/shortcut_check.csv
Run:    python 5_training_and_classification/s3_convnextv2_stylegan2-ada/shortcut_check.py
"""
import argparse
import glob
import json
import os

import pandas as pd

from cxr_data import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--phase", choices=["final", "search"], default="final")
    args = ap.parse_args()
    rows = []
    for mpath in sorted(glob.glob(os.path.join(args.out, args.phase, "**", "metrics_*.json"), recursive=True)):
        m = json.load(open(mpath))
        hpath = os.path.join(os.path.dirname(mpath), f"history_{m['run_id']}.csv")
        if not os.path.exists(hpath) or m.get("diverged"):
            print(f"[skipped] {m['run_id']} (no history or diverged)")
            continue
        h = pd.read_csv(hpath)
        best, last = h[h["epoch"] == m["best_epoch"]].iloc[0], h.iloc[-1]
        rows.append(dict(
            run_id=m["run_id"], cbam=m["cbam"], seed=m["seed"], epochs_run=len(h), best_epoch=int(m["best_epoch"]),
            best_val_auc=round(float(best["val_auc"]), 4),
            real_recall_best=round(float(best["train_recall_real_pn"]), 3),
            synth_recall_best=round(float(best["train_recall_synth_pn"]), 3),
            real_recall_last=round(float(last["train_recall_real_pn"]), 3),
            synth_recall_last=round(float(last["train_recall_synth_pn"]), 3),
            nofinding_acc_best=round(float(best["train_acc_nf"]), 3)))
    if not rows:
        print("no finished runs found")
        return
    df = pd.DataFrame(rows).sort_values(["cbam", "seed"])
    df.to_csv(os.path.join(args.out, args.phase, "shortcut_check.csv"), index=False)
    print("\n=== PER RUN ===")
    print(df.to_string(index=False))
    print("\n=== MEAN PER CBAM PLACEMENT ===")
    cols = ["best_val_auc", "real_recall_best", "synth_recall_best", "real_recall_last", "synth_recall_last"]
    print(df.groupby("cbam")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
