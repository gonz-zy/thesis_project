"""
run_all.py -- runs the whole Setup 4 classifier pipeline unattended, one step after the other, for run B, run A or both:
    1. bo_search.py        Bayesian Optimization (lr + decay floor) on THIS run's validation set
    2. run_experiments.py  five final seeds (42-46) with the tuned values, test set scored once (--eval_test)
The chain stops at the first step that fails, so the final runs never start from a broken search. Every step is
resume-safe: if the PC restarts, run the same command again and finished work is skipped.
Run it in a standalone PowerShell window (not inside VS Code).

Run (project root), B first, then A:
    python scripts\\3c-convnextv2_cyclegan\\run_all.py --run B
    python scripts\\3c-convnextv2_cyclegan\\run_all.py --run A
    (or  --run both  = B then A in one go)
Extra options go to bo_search.py, e.g.  --lr_max 1e-4
Results: results\\convnextv2_cyclegan\\run_B\\   and   run_A\\
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def step(name, script, *args):
    print(f"\n{'=' * 70}\n{time.strftime('%Y-%m-%d %H:%M:%S')} | START {name}\n{'=' * 70}", flush=True)
    t0 = time.time()
    rc = subprocess.run([sys.executable, os.path.join(HERE, script), *args]).returncode
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} | {name}: {'ok' if rc == 0 else f'FAILED (exit {rc})'} "
          f"after {(time.time() - t0) / 3600:.2f} h", flush=True)
    if rc != 0:
        sys.exit(f"Stopped: {name} failed. Fix the problem and rerun this command; finished work is kept.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B", "both"], required=True)
    ap.add_argument("--smoke", action="store_true", help="CPU test on synthetic data only")
    ap.add_argument("--final_epochs", type=int, default=None, help="epoch cap of the final runs (smoke test only)")
    a, rest = ap.parse_known_args()
    smoke = ["--smoke"] if a.smoke else []
    fin = ["--epochs", str(a.final_epochs)] if a.final_epochs else []
    for r in (["B", "A"] if a.run == "both" else [a.run]):
        step(f"run {r}: 1/2 Bayesian Optimization (CBAM all)", "bo_search.py", "--run", r, "--cbam", "all", *smoke, *rest)
        step(f"run {r}: 2/2 Final runs, seeds 42-46 (CBAM all)", "run_experiments.py", "--run", r, "--cbam", "all",
             "--eval_test", *smoke, *fin)
        print(f"\nRun {r} finished. Results: results\\convnextv2_cyclegan\\run_{r}\\")
