"""
Setup 2: runs bo_search.py and run_experiments.py (--eval_test) in order, for run B (the Chehade-style design; run A is not used for Setup 2).

Output: 5_training_and_classification/s2_densenet121_cyclegan/output/run_A/, run_B/
Run:    python 5_training_and_classification/s2_densenet121_cyclegan/run_all.py --run both
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
    ap.add_argument("--run", choices=["B"], required=True)
    ap.add_argument("--smoke", action="store_true", help="CPU test on synthetic data only")
    ap.add_argument("--final_epochs", type=int, default=None, help="epoch cap of the final runs (smoke test only)")
    a, rest = ap.parse_known_args()
    smoke = ["--smoke"] if a.smoke else []
    fin = ["--epochs", str(a.final_epochs)] if a.final_epochs else []
    for r in [a.run]:
        step(f"run {r}: 1/2 Bayesian Optimization (CBAM all)", "bo_search.py", "--run", r, "--cbam", "all", *smoke, *rest)
        step(f"run {r}: 2/2 Final runs, seeds 42-46 (CBAM all)", "run_experiments.py", "--run", r, "--cbam", "all",
             "--eval_test", *smoke, *fin)
        print(f"\nRun {r} finished. Results: 5_training_and_classification/s2_densenet121_cyclegan/output/run_{r}/")
