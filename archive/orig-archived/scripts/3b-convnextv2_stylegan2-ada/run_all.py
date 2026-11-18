"""
run_all.py -- runs the whole Setup 3 pipeline unattended, one step after the other:
    1. bo_search.py        Bayesian Optimization (lr + decay floor) on the validation set
    2. run_experiments.py  five final seeds (42-46) with the tuned values, test set scored once (--eval_test)
    3. shortcut_check.py   real vs synthetic pneumonia recall of every final run
The chain stops at the first step that fails, so the final runs never start from a broken search.
Every step is resume-safe: if the PC restarts, run the same command again and finished work is skipped.

Run (project root):
    python scripts\\3b-convnextv2_stylegan2-ada\\run_all.py
Extra options go to bo_search.py, e.g. a lower learning-rate ceiling:
    python scripts\\3b-convnextv2_stylegan2-ada\\run_all.py --lr_max 1e-4
"""
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
    step("1/3 Bayesian Optimization (CBAM all)", "bo_search.py", "--cbam", "all", *sys.argv[1:])
    step("2/3 Final runs, seeds 42-46 (CBAM all)", "run_experiments.py", "--cbam", "all", "--eval_test")
    step("3/3 Shortcut check", "shortcut_check.py")
    print("\nPipeline finished. Results: results\\convnextv2_stylegan2-ada\\")
