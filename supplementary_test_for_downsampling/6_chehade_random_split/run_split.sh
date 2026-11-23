#!/usr/bin/env bash
# Supplementary test: Chehade-style random IMAGE-LEVEL split (70 / 20 / 10, patients ignored), original K-Means and
# CycleGAN (fitted on all 676 images, so reused), 2 image sets x 2 networks x 2 recipes, for one split k.
#   our_recipe/      our original recipe: CBAM after every block, 20-trial search (seed 100), seeds 42-46
#   chehade_recipe/  Chehade's classifier: frozen lower layers, CBAM last, dense 1024/512, fixed lr 1e-4, seeds 42-46
# Image sets: cyclegan (main-study CycleGAN images) | nocyclegan_1024_to_224 (untranslated, one resize)
# Run from the project root in Git Bash:  bash supplementary_test_for_downsampling/6_chehade_random_split/run_split.sh 1
set -euo pipefail
K=${1:?usage: run_split.sh <split number 1-10>}
KK=$(printf "%02d" "$K")
source venv/Scripts/activate
RS=supplementary_test_for_downsampling/6_chehade_random_split
LOG=$RS/run_split_${KK}_log.txt
SETS="cyclegan nocyclegan_1024_to_224"
export CXR_SPLIT=$K
step() { echo -e "\n===== $(date '+%F %T') | split $KK | $1" | tee -a "$LOG"; }

step "1. split + normalization statistics"
python -u $RS/make_split.py --split $K >> "$LOG" 2>&1
for S in $SETS; do CXR_IMAGESET=$S python -u $RS/our_recipe/s2_densenet121/cxr2_data.py --run B >> "$LOG" 2>&1; done

for S in $SETS; do
  for NET in densenet convnext; do
    step "2. Chehade recipe | $NET | $S | seeds 42-46"
    for SEED in 42 43 44 45 46; do
      CXR_IMAGESET=$S python -u $RS/chehade_recipe/train_chehade.py --net $NET --seed $SEED --skip_if_done > /dev/null 2>> "$LOG"
    done
  done
done

for S in $SETS; do
  for NET in s2_densenet121 s4_convnextv2; do
    step "3. our recipe | $NET | $S | search + seeds 42-46"
    mkdir -p $RS/our_recipe/$NET/output/split_$KK/$S
    CXR_IMAGESET=$S python -u $RS/our_recipe/$NET/run_all.py --run B > $RS/our_recipe/$NET/output/split_$KK/$S/run_all_log.txt 2>&1
  done
done

# setup names: <net>_<recipe>_<images>; D = DenseNet-121, C = ConvNeXt V2; ours = CBAM all + our recipe,
# chh = Chehade's classifier; GAN = CycleGAN images, N224 = no CycleGAN (1024 -> 224)
declare -A DIR
for pair in "GAN:cyclegan" "N224:nocyclegan_1024_to_224"; do
  T=${pair%%:*}; S=${pair#*:}
  DIR[D_ours_$T]=$RS/our_recipe/s2_densenet121/output/split_$KK/$S/run_B/final/cbam_all
  DIR[C_ours_$T]=$RS/our_recipe/s4_convnextv2/output/split_$KK/$S/run_B/final/cbam_all
  DIR[D_chh_$T]=$RS/chehade_recipe/s2_densenet121/output/split_$KK/$S/run_B/final/cbam_last
  DIR[C_chh_$T]=$RS/chehade_recipe/s4_convnextv2/output/split_$KK/$S/run_B/final/cbam_last
done
evaluate() {   # evaluate <out folder> <pair> ...   (Holm over the pairs of one call)
  local out=$1; shift
  local args=() seen=" "
  for p in "$@"; do
    for n in ${p%%=*} ${p#*=}; do
      [[ $seen == *" $n "* ]] || { args+=(--setup "$n=${DIR[$n]}"); seen+="$n "; }
    done
    args+=(--pair "$p")
  done
  step "4. evaluation -> $out"
  python -u 6_testing_and_validation/evaluate.py "${args[@]}" --subset $RS/splits/split_$KK/test_images.csv \
    --out $RS/evaluation/split_$KK/$out >> "$LOG" 2>&1
}
evaluate densenet_vs_convnext_ours    D_ours_GAN=C_ours_GAN D_ours_N224=C_ours_N224
evaluate densenet_vs_convnext_chehade D_chh_GAN=C_chh_GAN D_chh_N224=C_chh_N224
evaluate with_vs_without_cyclegan     D_ours_GAN=D_ours_N224 C_ours_GAN=C_ours_N224 D_chh_GAN=D_chh_N224 C_chh_GAN=C_chh_N224
evaluate our_recipe_vs_chehade        D_ours_GAN=D_chh_GAN C_ours_GAN=C_chh_GAN D_ours_N224=D_chh_N224 C_ours_N224=C_chh_N224
step "DONE"
