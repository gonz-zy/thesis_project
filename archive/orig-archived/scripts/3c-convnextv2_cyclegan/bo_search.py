"""
bo_search.py -- Bayesian Optimization of the Adam learning rate and the decay floor for Setup 4 (one search per run).

Identical to Setup 3's search except for the data: each trial is a real training run of train.py on the 487 cleaned
images and is scored on THIS run's validation set (run B: 65 images, run A: 5,985 images). The test set is never opened.
    lr            1e-5 .. 3e-4   (log)    min_lr_ratio  0 .. 0.5
    sampler       Optuna TPE, 5 random startup trials, 20 trials, seed 100 (not a final seed)
    trial         up to 50 epochs, patience 5 (train.py defaults)
    objective     mean of the 3 highest validation AUCs of the trial (a diverged trial scores 0.5)
    resume        the study lives in an SQLite file; rerun the same command to continue
Run B has only 31 + 34 validation images, so its objective is noisy; that is a property of the design, not a bug.

Run (project root):
    python scripts\\3c-convnextv2_cyclegan\\bo_search.py --run B
Output: results\\convnextv2_cyclegan\\run_B\\search\\best_params_cbam-all.json, trials_cbam-all.csv, optuna_cbam-all.db
"""
import argparse
import json
import os
import subprocess
import sys

import optuna

import cxr4_data as D
from cxr4_data import run_paths

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B"], required=True)
    ap.add_argument("--cbam", choices=["all", "last", "none"], default="all")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--startup", type=int, default=5, help="random startup trials before TPE guides the search")
    ap.add_argument("--epochs", type=int, default=D.C.EPOCH_CAP, help="epoch cap per trial")
    ap.add_argument("--patience", type=int, default=D.C.PATIENCE)
    ap.add_argument("--search_seed", type=int, default=D.C.BO_SEED, help="outside the final seeds 42-46")
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--lr_max", type=float, default=3e-4)
    ap.add_argument("--floor_min", type=float, default=0.0)
    ap.add_argument("--floor_max", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="CPU test on synthetic data only")
    args = ap.parse_args()
    out = args.out or D.run_dir(args.run)

    tag = f"cbam-{args.cbam}"
    search_dir = os.path.join(out, "search")
    os.makedirs(search_dir, exist_ok=True)
    db = os.path.abspath(os.path.join(search_dir, f"optuna_{tag}.db")).replace("\\", "/")
    study = optuna.create_study(
        study_name=f"setup4_run{args.run}_{tag}", storage=f"sqlite:///{db}", load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.search_seed, n_startup_trials=args.startup))

    def objective(trial):
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        floor = trial.suggest_float("min_lr_ratio", args.floor_min, args.floor_max)
        cmd = [sys.executable, TRAIN, "--run", args.run, "--cbam", args.cbam, "--lr", repr(lr),
               "--min_lr_ratio", repr(floor), "--seed", str(args.search_seed), "--epochs", str(args.epochs),
               "--patience", str(args.patience), "--trial", str(trial.number), "--workers", str(args.workers),
               "--out", out]
        if args.smoke:
            cmd.append("--smoke")
        print(f"\n##### run {args.run} trial {trial.number}: lr={lr:.3e} min_lr_ratio={floor:.3f} #####", flush=True)
        if subprocess.run(cmd).returncode != 0:
            raise RuntimeError(f"train.py failed for trial {trial.number}")  # trial is recorded as FAIL
        P = run_paths(out, args.cbam, args.search_seed, trial.number)
        m = json.load(open(P["metrics"]))
        trial.set_user_attr("diverged", bool(m["diverged"]))
        trial.set_user_attr("best_val_auc", m["best_val_auc"])
        trial.set_user_attr("epochs_run", m["epochs_run"])
        if os.path.exists(P["best"]):
            os.remove(P["best"])  # checkpoints of search trials are not needed
        return m["top3_val_auc"]  # 0.5 when diverged

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = args.trials - done
    print(f"run {args.run} {tag}: {done} trials already complete, running {max(0, remaining)} more")
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, catch=(RuntimeError,))

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise SystemExit("no completed trials")
    study.trials_dataframe().to_csv(os.path.join(search_dir, f"trials_{tag}.csv"), index=False)
    n_div = sum(1 for t in complete if t.user_attrs.get("diverged"))
    best = study.best_trial
    res = dict(run=args.run, cbam=args.cbam, lr=best.params["lr"], min_lr_ratio=best.params["min_lr_ratio"],
               objective_top3_val_auc=best.value, best_trial=best.number, n_complete=len(complete),
               n_diverged=n_div, search_seed=args.search_seed, sampler="TPE", n_startup_trials=args.startup,
               epoch_cap=args.epochs, patience=args.patience,
               val_set="validation_small_65" if args.run == "B" else "validation_full",
               search_space=dict(lr=[args.lr_min, args.lr_max, "log"], min_lr_ratio=[args.floor_min, args.floor_max]))
    json.dump(res, open(os.path.join(search_dir, f"best_params_{tag}.json"), "w"), indent=2)
    print("\nBEST:", json.dumps(res, indent=2))
    if n_div > 5:
        print(f"ADVICE: {n_div} of {len(complete)} trials diverged (> 5). Lower --lr_max to 1e-4 and rerun the search "
              f"(delete {db} first so the new range starts a fresh study).")
    if len(complete) < args.trials:
        print(f"NOTE: only {len(complete)} of {args.trials} trials completed; rerun the same command to finish.")


if __name__ == "__main__":
    main()
