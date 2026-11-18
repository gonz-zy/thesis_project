"""
run_experiments.py -- final runs of Setup 4 with the TUNED hyper-parameters: five seeds, one after another.

Reads lr and min_lr_ratio from this run's search result (run_<A|B>\\search\\best_params_cbam-all.json), then runs train.py
for seeds 42-46 with the same values, so the five runs differ only by their random seed. Finished runs are skipped on
restart; a crashed run does not stop the rest. --eval_test scores the full 12,262-image test set (once per run).

Run (project root):
    python scripts\\3c-convnextv2_cyclegan\\run_experiments.py --run B --eval_test
    python scripts\\3c-convnextv2_cyclegan\\run_experiments.py --run B --summarize_only
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import pandas as pd

import cxr4_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train.py")


def summarize(out):
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(out, "final", "**", "metrics_*.json"), recursive=True)]
    if not rows:
        print("no finished final runs found")
        return
    df = pd.DataFrame(rows).sort_values(["cbam", "seed"])
    keep = [c for c in ["run_id", "run", "cbam", "seed", "lr", "min_lr_ratio", "epochs_run", "best_epoch", "diverged",
                        "best_val_auc", "test_auc", "accuracy", "precision", "recall", "f1"] if c in df.columns]
    df[keep].to_csv(os.path.join(out, "final", "runs.csv"), index=False)
    agg = df.groupby("cbam")["best_val_auc"].agg(["count", "mean", "std", "min", "max"]).round(4).reset_index()
    agg.to_csv(os.path.join(out, "final", "summary.csv"), index=False)
    print("\n=== VALIDATION SUMMARY (best validation AUC per seed) ===")
    print(agg.to_string(index=False))
    if "test_auc" in df.columns:
        print("\n=== PER-SEED TEST AUC (details, CIs, DeLong and ANOVA: evaluate.py) ===")
        print(df[[c for c in ["cbam", "seed", "test_auc"] if c in df.columns]].to_string(index=False))
    # domain check at the best epoch: training recall on artifact (translated) vs clean images
    dom = []
    for m in rows:
        h = os.path.join(out, "final", f"cbam_{m['cbam']}", f"seed{m['seed']}", f"history_{m['run_id']}.csv")
        if os.path.exists(h) and not m.get("diverged"):
            r = pd.read_csv(h)
            r = r[r["epoch"] == m["best_epoch"]].iloc[0]
            dom.append(dict(seed=m["seed"], best_epoch=int(m["best_epoch"]),
                            pn_recall_artifact=round(float(r["train_recall_pn_artifact"]), 3),
                            pn_recall_clean=round(float(r["train_recall_pn_clean"]), 3),
                            nf_acc_artifact=round(float(r["train_acc_nf_artifact"]), 3),
                            nf_acc_clean=round(float(r["train_acc_nf_clean"]), 3)))
    if dom:
        d = pd.DataFrame(dom).sort_values("seed")
        d.to_csv(os.path.join(out, "final", "domain_check.csv"), index=False)
        print("\n=== DOMAIN CHECK (training images, threshold 0.5, at the best-validation epoch) ===")
        print(d.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B"], required=True)
    ap.add_argument("--cbam", choices=["all", "last", "none"], default="all")
    ap.add_argument("--params", default=None, help="best_params json (default: this run's search result)")
    ap.add_argument("--seeds", type=int, nargs="+", default=D.C.SEEDS)
    ap.add_argument("--epochs", type=int, default=None, help="epoch cap (default: train.py's 50)")
    ap.add_argument("--eval_test", action="store_true")
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="CPU test on synthetic data only")
    ap.add_argument("extra", nargs="*", help="extra args passed to train.py")
    args = ap.parse_args()
    out = args.out or D.run_dir(args.run)

    os.makedirs(os.path.join(out, "final"), exist_ok=True)
    if args.summarize_only:
        summarize(out)
        return
    params = args.params or os.path.join(out, "search", f"best_params_cbam-{args.cbam}.json")
    if not os.path.exists(params):
        raise SystemExit(f"{params} not found: run bo_search.py --run {args.run} first (or pass --params)")
    p = json.load(open(params))
    print(f"run {args.run}: using lr={p['lr']:.3e}, min_lr_ratio={p['min_lr_ratio']:.3f} from {params}")

    log_path = os.path.join(out, "final", "run_log.txt")
    failed, t_all = [], time.time()
    for n, seed in enumerate(args.seeds, 1):
        cmd = [sys.executable, TRAIN, "--run", args.run, "--cbam", args.cbam, "--lr", repr(p["lr"]),
               "--min_lr_ratio", repr(p["min_lr_ratio"]), "--seed", str(seed), "--out", out, "--skip_if_done",
               *args.extra]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.eval_test:
            cmd.append("--eval_test")
        if args.smoke:
            cmd.append("--smoke")
        print(f"\n##### [{n}/{len(args.seeds)}] run {args.run} cbam={args.cbam} seed={seed} #####", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        with open(log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | run={args.run} | cbam={args.cbam} | seed={seed} | "
                    f"{'ok' if rc == 0 else f'FAILED (exit {rc})'} | {(time.time() - t0) / 60:.1f} min\n")
        if rc != 0:
            failed.append(seed)
    print(f"\nAll done in {(time.time() - t_all) / 3600:.2f} h. Failed seeds: {failed if failed else 'none'}")
    summarize(out)


if __name__ == "__main__":
    main()
