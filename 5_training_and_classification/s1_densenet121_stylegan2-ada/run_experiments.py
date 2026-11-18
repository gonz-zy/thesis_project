"""
Setup 1: final runs with the tuned values, seeds 42-46.

Input:  5_training_and_classification/s1_densenet121_stylegan2-ada/output/search/best_params_cbam-all.json
Output: 5_training_and_classification/s1_densenet121_stylegan2-ada/output/final/cbam_all/seed42 ... seed46, runs.csv, summary.csv, run_log.txt
Run:    python 5_training_and_classification/s1_densenet121_stylegan2-ada/run_experiments.py --cbam all --eval_test
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import pandas as pd

from cxr_data import RESULTS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train.py")


def summarize(out):
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(out, "final", "**", "metrics_*.json"), recursive=True)]
    if not rows:
        print("no finished final runs found")
        return
    df = pd.DataFrame(rows).sort_values(["cbam", "seed"])
    keep = [c for c in ["run_id", "cbam", "seed", "lr", "min_lr_ratio", "epochs_run", "best_epoch", "diverged",
                        "best_val_auc", "test_auc", "accuracy", "precision", "recall", "f1"] if c in df.columns]
    df[keep].to_csv(os.path.join(out, "final", "runs.csv"), index=False)
    agg = df.groupby("cbam")["best_val_auc"].agg(["count", "mean", "std", "min", "max"]).round(4).reset_index()
    agg.to_csv(os.path.join(out, "final", "summary.csv"), index=False)
    print("\n=== VALIDATION SUMMARY (best validation AUC per seed) ===")
    print(agg.to_string(index=False))
    if "test_auc" in df.columns:
        print("\n=== PER-SEED TEST AUC (details, CIs, DeLong and ANOVA: evaluate.py) ===")
        print(df[[c for c in ["cbam", "seed", "test_auc"] if c in df.columns]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbam", choices=["all", "last", "none"], default="all")
    ap.add_argument("--params", default=None, help="best_params json (default: the search result for this --cbam)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--eval_test", action="store_true")
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("extra", nargs="*", help="extra args passed to train.py")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "final"), exist_ok=True)
    if args.summarize_only:
        summarize(args.out)
        return
    # TUNING: the final runs use the values found by bo_search.py
    params = args.params or os.path.join(args.out, "search", f"best_params_cbam-{args.cbam}.json")
    if not os.path.exists(params):
        raise SystemExit(f"{params} not found: run bo_search.py --cbam {args.cbam} first (or pass --params)")
    p = json.load(open(params))
    print(f"Using lr={p['lr']:.3e}, min_lr_ratio={p['min_lr_ratio']:.3f} from {params}")

    log_path = os.path.join(args.out, "final", "run_log.txt")
    failed, t_all = [], time.time()
    for n, seed in enumerate(args.seeds, 1):
        cmd = [sys.executable, TRAIN, "--cbam", args.cbam, "--lr", repr(p["lr"]), "--min_lr_ratio",
               repr(p["min_lr_ratio"]), "--seed", str(seed), "--out", args.out, "--skip_if_done", *args.extra]
        if args.eval_test:
            cmd.append("--eval_test")
        print(f"\n##### [{n}/{len(args.seeds)}] cbam={args.cbam} seed={seed} #####", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        with open(log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | cbam={args.cbam} | seed={seed} | "
                    f"{'ok' if rc == 0 else f'FAILED (exit {rc})'} | {(time.time() - t0) / 60:.1f} min\n")
        if rc != 0:
            failed.append(seed)
    print(f"\nAll done in {(time.time() - t_all) / 3600:.2f} h. Failed seeds: {failed if failed else 'none'}")
    summarize(args.out)


if __name__ == "__main__":
    main()
