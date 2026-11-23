"""
SUPPLEMENTARY TEST (Chehade-style CBAM, our recipe): data tables for the image set in CXR_IMAGESET (see ../../imagesets.py).
Results: <this folder>/output/<image set>/run_B/.  Self-check:  CXR_IMAGESET=cyclegan python <this file> --run B
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.dirname(os.path.dirname(HERE))                    # 5_chehade_cbam
ROOT = os.path.dirname(os.path.dirname(CH))                    # project root
for _p in (os.path.join(ROOT, "5_training_and_classification", "s1_densenet121_stylegan2-ada"), os.path.join(ROOT, "3_downsampling_branch"), CH):
    if _p not in sys.path:
        sys.path.append(_p)

from imagesets import C, CXRDataset, IMAGESET, get_tables, load_stats, run_paths, self_check  # noqa: F401

RESULTS_DIR = os.path.join(HERE, "output", IMAGESET)


def run_dir(run):
    assert run == "B"
    return os.path.join(RESULTS_DIR, f"run_{run}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["B"], required=True)
    ap.parse_args()
    self_check()
    print("results go to:", run_dir("B"))
