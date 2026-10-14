"""
bo_search.py -- Bayesian Optimization of the Adam learning rate and the decay schedule (one search per setup).

Manuscript: Optuna Bayesian Optimization supplies Adam's initial learning rate and decay scheduling, chosen on the
VALIDATION set only. This script tunes exactly two values:
    lr            1e-5 .. 3e-4   (log scale)      initial Adam learning rate
    min_lr_ratio  0 .. 0.5                         cosine-decay floor, as a fraction of lr
Everything else is fixed (batch 16, weight decay 1e-4, head, 3 warmup epochs) and reported as fixed.

Search design
  sampler     Optuna TPE (a Bayesian-style sampler); the first 5 trials are random "startup" trials, the other 15
              are guided by what the earlier trials showed
  trials      20, each a real training run of up to 15 epochs with patience 5 (train.py, --trial N)
  objective   mean of the 3 highest validation AUCs of the trial (validation has only 31 pneumonia cases, so a
              single best epoch is too noisy); a diverged trial (NaN loss) scores 0.5 so the search keeps going
  seed        100 -- deliberately NOT one of the final seeds 42-46, so no final run is favoured by the tuning
  test set    never opened during the search
  resume      the study lives in an SQLite file; rerun the same command to continue after an interruption

If more than 5 of the 20 trials diverge, lower --lr_max to 1e-4 and rerun (the script prints this advice).
Note: the best validation AUC found by a search is optimistic (it is the best of many noisy scores). Only the
5-seed test results of the final runs are reported as performance.

Run (project root):
    python scripts\\3b-convnextv2_stylegan2-ada\\bo_search.py --cbam all
Output: results\\convnextv2_stylegan2-ada\\search\\best_params_cbam-all.json, trials_cbam-all.csv, optuna_cbam-all.db
"""
import argparse
import json
import os
import subprocess
import sys

import optuna

from cxr_data import RESULTS_DIR, run_paths

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbam", choices=["all", "last", "none"], default="all")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--startup", type=int, default=5, help="random startup trials before TPE guides the search")
    ap.add_argument("--epochs", type=int, default=15, help="epoch cap per trial")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--search_seed", type=int, default=100, help="outside the final seeds 42-46")
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--lr_max", type=float, default=3e-4)
    ap.add_argument("--floor_min", type=float, default=0.0)
    ap.add_argument("--floor_max", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args()

    tag = f"cbam-{args.cbam}"
    search_dir = os.path.join(args.out, "search")
    os.makedirs(search_dir, exist_ok=True)
    db = os.path.abspath(os.path.join(search_dir, f"optuna_{tag}.db")).replace("\\", "/")
    study = optuna.create_study(
        study_name=f"convnextv2_{tag}", storage=f"sqlite:///{db}", load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.search_seed, n_startup_trials=args.startup))

    def objective(trial):
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        floor = trial.suggest_float("min_lr_ratio", args.floor_min, args.floor_max)
        cmd = [sys.executable, TRAIN, "--cbam", args.cbam, "--lr", repr(lr), "--min_lr_ratio", repr(floor),
               "--seed", str(args.search_seed), "--epochs", str(args.epochs), "--patience", str(args.patience),
               "--trial", str(trial.number), "--workers", str(args.workers), "--out", args.out]
        print(f"\n##### trial {trial.number}: lr={lr:.3e} min_lr_ratio={floor:.3f} #####", flush=True)
        if subprocess.run(cmd).returncode != 0:
            raise RuntimeError(f"train.py failed for trial {trial.number}")  # trial is recorded as FAIL
        P = run_paths(args.out, args.cbam, args.search_seed, trial.number)
        m = json.load(open(P["metrics"]))
        trial.set_user_attr("diverged", bool(m["diverged"]))
        trial.set_user_attr("best_val_auc", m["best_val_auc"])
        trial.set_user_attr("epochs_run", m["epochs_run"])
        if os.path.exists(P["best"]):
            os.remove(P["best"])  # checkpoints of search trials are not needed
        return m["top3_val_auc"]  # 0.5 when diverged

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = args.trials - done
    print(f"{tag}: {done} trials already complete, running {max(0, remaining)} more")
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, catch=(RuntimeError,))

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise SystemExit("no completed trials")
    study.trials_dataframe().to_csv(os.path.join(search_dir, f"trials_{tag}.csv"), index=False)
    n_div = sum(1 for t in complete if t.user_attrs.get("diverged"))
    best = study.best_trial
    out = dict(cbam=args.cbam, lr=best.params["lr"], min_lr_ratio=best.params["min_lr_ratio"],
               objective_top3_val_auc=best.value, best_trial=best.number, n_complete=len(complete),
               n_diverged=n_div, search_seed=args.search_seed, sampler="TPE", n_startup_trials=args.startup,
               epoch_cap=args.epochs, patience=args.patience,
               search_space=dict(lr=[args.lr_min, args.lr_max, "log"], min_lr_ratio=[args.floor_min, args.floor_max]))
    json.dump(out, open(os.path.join(search_dir, f"best_params_{tag}.json"), "w"), indent=2)
    print("\nBEST:", json.dumps(out, indent=2))
    if n_div > 5:
        print(f"ADVICE: {n_div} of {len(complete)} trials diverged (> 5). Lower --lr_max to 1e-4 and rerun the search "
              f"(delete {db} first so the new range starts a fresh study).")
    if len(complete) < args.trials:
        print(f"NOTE: only {len(complete)} of {args.trials} trials completed; rerun the same command to finish.")


if __name__ == "__main__":
    main()
