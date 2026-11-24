#!/usr/bin/env bash
# Supplementary (moved synthetic): full pipeline. Run from the repository root:
#   bash supplementary_moved_synthetic/run_all.sh
# Safe to re-run: finished seeds are skipped (--skip_if_done).
set -euo pipefail
cd "$(dirname "$0")/.."
H=supplementary_moved_synthetic
LOG=$H/output/run_log.txt
mkdir -p $H/output/logs
export TQDM_MININTERVAL=120 PYTHONUNBUFFERED=1
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a $LOG; }

say "step 1: split"
python $H/make_split.py > $H/output/logs/make_split.txt
say "step 1b: training-set mean/std (new training set)"
[ -f $H/output/train_stats_moved.json ] || python 4_image_transformation/make_stats.py \
    --csv $H/output/manifests/combined_training_labels.csv --out $H/output/train_stats_moved.json \
    --compare 4_image_transformation/output/train_stats_setup1_setup3.json > $H/output/logs/make_stats.txt

for s in s1 s3; do
  case $s in s1) d=s1_densenet121_stylegan2-ada;; s3) d=s3_convnextv2_stylegan2-ada;; esac
  for seed in 42 43 44 45 46; do
    t0=$(date +%s)
    python $H/train_moved.py --setup $s --cbam all --seed $seed --eval_test --skip_if_done \
        --params 5_training_and_classification/$d/output/search/best_params_cbam-all.json \
        > $H/output/logs/train_${s}_seed${seed}.txt 2>&1
    say "train $s seed $seed ok | $(( ($(date +%s) - t0) / 60 )) min"
  done
done

say "step 3: leakage diagnostics"
python $H/leakage.py > $H/output/logs/leakage.txt 2>&1
say "step 2b: evaluation on slices a/b/c"
python $H/evaluate_slices.py > $H/output/logs/evaluate_slices.txt 2>&1 || say "evaluate_slices failed (see log)"
say "done"
