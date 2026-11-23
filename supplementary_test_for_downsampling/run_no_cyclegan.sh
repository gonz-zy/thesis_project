#!/usr/bin/env bash
# Supplementary test WITHOUT CycleGAN: DenseNet-121 + CBAM and ConvNeXt V2-Tiny + CBAM on the untranslated 676 images,
# for two image paths (each in its own folder):
#   5_no_cyclegan_1024_to_224/          equalized 1024 -> 224, then augmentation at 224 during training
#   5_no_cyclegan_1024_to_256_to_224/   equalized 1024 -> 256 (Setup 3's files) -> 224, then augmentation (= S2/S4B path)
# Same recipe as the main study's Setup 2 / Setup 4B (20-trial search, seed 100; seeds 42-46; 65 validation; 124 test).
# Run from the project root in Git Bash:  bash supplementary_test_for_downsampling/run_no_cyclegan.sh
set -euo pipefail
source venv/Scripts/activate
S=supplementary_test_for_downsampling
LOG=$S/run_no_cyclegan_log.txt
T124=3_downsampling_branch/output/labels/test_balanced_124.csv
MAIN=5_training_and_classification
step() { echo -e "\n===== $(date '+%F %T') | $1" | tee -a "$LOG"; }

for V in 1024_to_224 1024_to_256_to_224; do
  D=$S/5_no_cyclegan_$V
  step "$V | images + statistics (skipped if present)"
  [ -f $D/images/manifest.csv ] || python -u $S/make_images_no_cyclegan.py --variant $V >> "$LOG" 2>&1
  [ -f $D/images/train_stats.json ] || python -u 4_image_transformation/make_stats.py --csv $D/images/manifest.csv \
       --split train --out $D/images/train_stats.json >> "$LOG" 2>&1

  step "$V | DenseNet-121 + CBAM: search + seeds 42-46"
  python -u $D/s2_densenet121_nocyclegan/run_all.py --run B > $D/s2_densenet121_nocyclegan/run_all_log.txt 2>&1

  step "$V | ConvNeXt V2 + CBAM: search + seeds 42-46"
  python -u $D/s4_convnextv2_nocyclegan/run_all.py --run B > $D/s4_convnextv2_nocyclegan/run_all_log.txt 2>&1

  step "$V | evaluation on the 124 test images"
  # declared: DenseNet vs ConvNeXt, both without CycleGAN (mirrors S2 vs S4B)
  python -u 6_testing_and_validation/evaluate.py \
    --setup S2noGAN=$D/s2_densenet121_nocyclegan/output/run_B/final/cbam_all \
    --setup S4BnoGAN=$D/s4_convnextv2_nocyclegan/output/run_B/final/cbam_all \
    --pair S2noGAN=S4BnoGAN --subset $T124 --out $D/evaluation_124 >> "$LOG" 2>&1
  # descriptive: with vs without CycleGAN (main-study S2 / S4B), Holm over these two
  python -u 6_testing_and_validation/evaluate.py \
    --setup S2=$MAIN/s2_densenet121_cyclegan/output/run_B/final/cbam_all \
    --setup S2noGAN=$D/s2_densenet121_nocyclegan/output/run_B/final/cbam_all \
    --setup S4B=$MAIN/s4_convnextv2_cyclegan/output/run_B/final/cbam_all \
    --setup S4BnoGAN=$D/s4_convnextv2_nocyclegan/output/run_B/final/cbam_all \
    --pair S2=S2noGAN --pair S4B=S4BnoGAN --subset $T124 --out $D/evaluation_124_vs_cyclegan >> "$LOG" 2>&1
done

step "DONE"
