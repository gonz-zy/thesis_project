"""
Setup 1: Optuna search of the learning rate and the decay floor (20 trials, seed 100, validation set only).

Input:  train.py
Output: 5_training_and_classification/s1_densenet121_stylegan2-ada/output/search/best_params_cbam-all.json, trials_cbam-all.csv, optuna_cbam-all.db
Run:    python 5_training_and_classification/s1_densenet121_stylegan2-ada/bo_search.py --cbam all
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
        study_name=f"densenet121_{tag}", storage=f"sqlite:///{db}", load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.search_seed, n_startup_trials=args.startup))

    def objective(trial):
        # TUNING: the only two tuned values (learning rate and decay floor)
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
