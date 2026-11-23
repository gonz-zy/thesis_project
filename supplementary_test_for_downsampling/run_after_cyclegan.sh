#!/usr/bin/env bash
# Supplementary test: every step after the CycleGAN training, in order. Stops at the first failure.
# Run from the project root in Git Bash:  bash supplementary_test_for_downsampling/run_after_cyclegan.sh
set -euo pipefail
source venv/Scripts/activate
S=supplementary_test_for_downsampling
N3=$S/3_new_downsampling_branch
N5=$S/5_new_training_and_classification
LOG=$S/run_after_cyclegan_log.txt
step() { echo -e "\n===== $(date '+%F %T') | $1" | tee -a "$LOG"; }

step "1/8 clean the 676 images with the new epoch-100 generator"
python -u $N3/clean_images.py >> "$LOG" 2>&1

step "2/8 normalization statistics of the 487 newly cleaned training images"
python -u 4_image_transformation/make_stats.py --csv $N3/output/cleaned/cleaned_manifest.csv --split train \
       --out $N3/output/train_stats_new_cleaning.json >> "$LOG" 2>&1

step "3/8 self-check of the Setup 2 and Setup 4B data tables"
python -u $N5/s2_densenet121_cyclegan/cxr2_data.py --run B >> "$LOG" 2>&1
python -u $N5/s4_convnextv2_cyclegan/cxr4_data.py --run B >> "$LOG" 2>&1

step "4/8 how much the translation changed each image"
python -u $N3/translation_change.py --cleaned_dir $N3/output/cleaned --manifest $N3/output/clusters/clustered_676.csv >> "$LOG" 2>&1

step "5/8 FID / KID of artifact vs clean images, before and after"
python -u $N3/fid_kid.py --manifest $N3/output/clusters/clustered_676.csv --backbone fid >> "$LOG" 2>&1

step "6/8 Setup 2 (DenseNet-121): search + seeds 42-46"
python -u $N5/s2_densenet121_cyclegan/run_all.py --run B > $N5/s2_densenet121_cyclegan/run_all_log.txt 2>&1

step "7/8 Setup 4B (ConvNeXt V2): search + seeds 42-46"
python -u $N5/s4_convnextv2_cyclegan/run_all.py --run B > $N5/s4_convnextv2_cyclegan/run_all_log.txt 2>&1

step "8/8 evaluation on the 124 test images"
# declared comparison of the supplementary test (as in the main study): new Setup 2 vs new Setup 4B
python -u 6_testing_and_validation/evaluate.py \
  --setup S2new=$N5/s2_densenet121_cyclegan/output/run_B/final/cbam_all \
  --setup S4Bnew=$N5/s4_convnextv2_cyclegan/output/run_B/final/cbam_all \
  --pair S2new=S4Bnew --subset 3_downsampling_branch/output/labels/test_balanced_124.csv \
  --out $N5/evaluation_124 >> "$LOG" 2>&1
# descriptive: did the new cleaning change each network? (same 124 images; not declared, Holm over these two only)
python -u 6_testing_and_validation/evaluate.py \
  --setup S2=5_training_and_classification/s2_densenet121_cyclegan/output/run_B/final/cbam_all \
  --setup S2new=$N5/s2_densenet121_cyclegan/output/run_B/final/cbam_all \
  --setup S4B=5_training_and_classification/s4_convnextv2_cyclegan/output/run_B/final/cbam_all \
  --setup S4Bnew=$N5/s4_convnextv2_cyclegan/output/run_B/final/cbam_all \
  --pair S2=S2new --pair S4B=S4Bnew --subset 3_downsampling_branch/output/labels/test_balanced_124.csv \
  --out $N5/evaluation_124_old_vs_new >> "$LOG" 2>&1

step "DONE"
