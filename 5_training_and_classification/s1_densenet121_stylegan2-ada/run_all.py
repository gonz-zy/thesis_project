"""
Setup 1: runs bo_search.py, run_experiments.py (--eval_test) and shortcut_check.py in order.

Output: 5_training_and_classification/s1_densenet121_stylegan2-ada/output/
Run:    python 5_training_and_classification/s1_densenet121_stylegan2-ada/run_all.py
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
    print("\nPipeline finished. Results: 5_training_and_classification/s1_densenet121_stylegan2-ada/output/")
