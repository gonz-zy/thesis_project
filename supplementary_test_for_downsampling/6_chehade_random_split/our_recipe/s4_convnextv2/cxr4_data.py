"""
SUPPLEMENTARY TEST (Chehade-style random split, our original recipe = CBAM after every block): data tables of split
CXR_SPLIT for the image set CXR_IMAGESET (see ../../splitdata.py).
Results: <this folder>/output/split_<kk>/<image set>/run_B/.
Self-check:  CXR_SPLIT=1 CXR_IMAGESET=cyclegan python <this file> --run B
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.dirname(os.path.dirname(HERE))                    # 6_chehade_random_split
ROOT = os.path.dirname(os.path.dirname(RS))                    # project root
for _p in (os.path.join(ROOT, "5_training_and_classification", "s3_convnextv2_stylegan2-ada"), os.path.join(ROOT, "3_downsampling_branch"), RS):
    if _p not in sys.path:
        sys.path.append(_p)

from splitdata import C, CXRDataset, IMAGESET, SPLIT_TAG, get_tables, load_stats, run_paths, self_check  # noqa: F401

RESULTS_DIR = os.path.join(HERE, "output", SPLIT_TAG, IMAGESET)


def run_dir(run):
    assert run == "B"
    return os.path.join(RESULTS_DIR, f"run_{run}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["B"], required=True)
    ap.parse_args()
    self_check()
    print("results go to:", run_dir("B"))
