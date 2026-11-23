#!/usr/bin/env bash
# Supplementary test: Chehade-style CBAM (one CBAM on the last block), 3 image sets x 2 networks x 2 recipes.
#   full_chehade_classifier/   Chehade's classifier recipe (frozen lower layers, dense 1024/512, dropout 0.5, fixed lr 1e-4),
#                              5 seeds, no search
#   cbam_last_our_recipe/      our recipe (full fine-tuning, 20-trial search, seeds 42-46), CBAM on the last block only
# Image sets (CXR_IMAGESET): cyclegan | nocyclegan_1024_to_256_to_224 | nocyclegan_1024_to_224
# Run from the project root in Git Bash:  bash supplementary_test_for_downsampling/5_chehade_cbam/run_chehade_cbam.sh
set -euo pipefail
source venv/Scripts/activate
CH=supplementary_test_for_downsampling/5_chehade_cbam
LOG=$CH/run_chehade_cbam_log.txt
T124=3_downsampling_branch/output/labels/test_balanced_124.csv
SETS="cyclegan nocyclegan_1024_to_256_to_224 nocyclegan_1024_to_224"
step() { echo -e "\n===== $(date '+%F %T') | $1" | tee -a "$LOG"; }

# ---- 1. full Chehade classifier (fast: no search) ----
for S in $SETS; do
  for NET in densenet convnext; do
    step "Chehade recipe | $NET | $S | seeds 42-46"
    for SEED in 42 43 44 45 46; do
      CXR_IMAGESET=$S python -u $CH/full_chehade_classifier/train_chehade.py --net $NET --seed $SEED --skip_if_done \
        > /dev/null 2>> "$LOG"
    done
  done
done

# ---- 2. CBAM last, our recipe (search + 5 seeds) ----
for S in $SETS; do
  for NET in s2_densenet121 s4_convnextv2; do
    step "our recipe, CBAM last | $NET | $S | search + seeds 42-46"
    mkdir -p $CH/cbam_last_our_recipe/$NET/output/$S
    CXR_IMAGESET=$S python -u $CH/cbam_last_our_recipe/$NET/run_all.py --run B \
      > $CH/cbam_last_our_recipe/$NET/output/$S/run_all_log.txt 2>&1
  done
done

# ---- 3. evaluation on the 124 test images ----
# setup names: <net>_<recipe>_<images>; net D = DenseNet-121, C = ConvNeXt V2; recipe all = CBAM everywhere (earlier runs),
# last = CBAM last + our recipe, chh = full Chehade classifier; images GAN = CycleGAN, N256 = 1024->256->224, N224 = 1024->224
declare -A DIR
SUPP=supplementary_test_for_downsampling
DIR[D_all_GAN]=5_training_and_classification/s2_densenet121_cyclegan/output/run_B/final/cbam_all
DIR[C_all_GAN]=5_training_and_classification/s4_convnextv2_cyclegan/output/run_B/final/cbam_all
DIR[D_all_N256]=$SUPP/5_no_cyclegan_1024_to_256_to_224/s2_densenet121_nocyclegan/output/run_B/final/cbam_all
DIR[C_all_N256]=$SUPP/5_no_cyclegan_1024_to_256_to_224/s4_convnextv2_nocyclegan/output/run_B/final/cbam_all
DIR[D_all_N224]=$SUPP/5_no_cyclegan_1024_to_224/s2_densenet121_nocyclegan/output/run_B/final/cbam_all
DIR[C_all_N224]=$SUPP/5_no_cyclegan_1024_to_224/s4_convnextv2_nocyclegan/output/run_B/final/cbam_all
for pair in "GAN:cyclegan" "N256:nocyclegan_1024_to_256_to_224" "N224:nocyclegan_1024_to_224"; do
  K=${pair%%:*}; S=${pair#*:}
  DIR[D_last_$K]=$CH/cbam_last_our_recipe/s2_densenet121/output/$S/run_B/final/cbam_last
  DIR[C_last_$K]=$CH/cbam_last_our_recipe/s4_convnextv2/output/$S/run_B/final/cbam_last
  DIR[D_chh_$K]=$CH/full_chehade_classifier/s2_densenet121/output/$S/run_B/final/cbam_last
  DIR[C_chh_$K]=$CH/full_chehade_classifier/s4_convnextv2/output/$S/run_B/final/cbam_last
done

evaluate() {   # evaluate <out folder> <pair> <pair> ...   (Holm over the pairs of one call)
  local out=$1; shift
  local args=() seen=" "
  for p in "$@"; do
    for n in ${p%%=*} ${p#*=}; do
      [[ $seen == *" $n "* ]] || { args+=(--setup "$n=${DIR[$n]}"); seen+="$n "; }
    done
    args+=(--pair "$p")
  done
  step "evaluation -> $out"
  python -u 6_testing_and_validation/evaluate.py "${args[@]}" --subset $T124 --out $CH/evaluation/$out >> "$LOG" 2>&1
}
evaluate densenet_vs_convnext_cbam_last D_last_GAN=C_last_GAN D_last_N256=C_last_N256 D_last_N224=C_last_N224
evaluate densenet_vs_convnext_chehade   D_chh_GAN=C_chh_GAN D_chh_N256=C_chh_N256 D_chh_N224=C_chh_N224
evaluate cbam_all_vs_cbam_last          D_all_GAN=D_last_GAN D_all_N256=D_last_N256 D_all_N224=D_last_N224 \
                                        C_all_GAN=C_last_GAN C_all_N256=C_last_N256 C_all_N224=C_last_N224
evaluate our_recipe_vs_chehade          D_last_GAN=D_chh_GAN D_last_N256=D_chh_N256 D_last_N224=D_chh_N224 \
                                        C_last_GAN=C_chh_GAN C_last_N256=C_chh_N256 C_last_N224=C_chh_N224
evaluate with_vs_without_cyclegan       D_last_GAN=D_last_N256 D_last_GAN=D_last_N224 C_last_GAN=C_last_N256 C_last_GAN=C_last_N224 \
                                        D_chh_GAN=D_chh_N256 D_chh_GAN=D_chh_N224 C_chh_GAN=C_chh_N256 C_chh_GAN=C_chh_N224
step "DONE"
