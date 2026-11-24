"""
Supplementary (moved synthetic): runs the ORIGINAL Setup 1 or Setup 3 train.py unchanged, only pointed at the new
manifests, the new training-set mean/std and a new output folder. Nothing in the main experiment is read for
writing or overwritten.

Every train.py flag passes through (same defaults as the main runs: batch 16, 15 epochs, 3 warmup, patience 5,
weight decay 1e-4, samples_per_epoch 84,384, pos_weight from the training counts = 42,204 / 29,654 = 1.423).

Run:    python supplementary_moved_synthetic/train_moved.py --setup s1 --cbam all --seed 42 --eval_test
            --params 5_training_and_classification/s1_densenet121_stylegan2-ada/output/search/best_params_cbam-all.json
"""
import os
import sys

SETUPS = dict(s1="s1_densenet121_stylegan2-ada", s3="s3_convnextv2_stylegan2-ada")
HERE = os.path.join("supplementary_moved_synthetic")
MANIFESTS = os.path.join(HERE, "output", "manifests")
STATS = os.path.join(HERE, "output", "train_stats_moved.json")


def main():
    argv = sys.argv[1:]
    i = argv.index("--setup")
    setup = argv[i + 1]
    del argv[i:i + 2]
    code_dir = os.path.join("5_training_and_classification", SETUPS[setup])
    sys.path.insert(0, code_dir)

    import cxr_data  # original module: patch only its path constants before train.py imports them
    cxr_data.LABELS = MANIFESTS
    cxr_data.COMBINED_CSV = os.path.join(MANIFESTS, "combined_training_labels.csv")
    cxr_data.STATS_JSON = STATS
    cxr_data.RESULTS_DIR = os.path.join(HERE, "output", setup)
    import train  # original training script, unchanged
    assert train.COMBINED_CSV == cxr_data.COMBINED_CSV and train.LABELS == MANIFESTS and train.STATS_JSON == STATS
    sys.argv = [os.path.join(code_dir, "train.py")] + argv
    if "--out" not in argv:
        sys.argv += ["--out", cxr_data.RESULTS_DIR]
    train.main()


if __name__ == "__main__":
    main()
