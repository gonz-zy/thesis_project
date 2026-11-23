# Archive

Files kept for the record. None of them are used by the final pipeline or by the numbers in the paper.
Paths inside these files are the old paths (`data/2-processed/...`, `results/...`, `scripts/...`).

## 3_downsampling_branch/

| Folder | What it is |
|---|---|
| `cand_e075/` | Cleaning of the 676 images with the CycleGAN checkpoint at epoch 75. A candidate used only to choose the checkpoint. |
| `cand_e100/` | The same with the epoch-100 checkpoint, on the 676 images only. Epoch 100 was chosen. The final cleaned set (all 18,734 images) is in `3_downsampling_branch/output/cleaned/`. |
| `cyclegan_smoke/` | A 2-epoch test run of `train_cyclegan.py --smoke` (8 + 8 images) to check speed. Not a real model. |

The checkpoint was chosen on image criteria only (centre change and SSIM from `translation_change.py`, and visual
checks), not on classifier results. See `handoff/HANDOFF.md`.

## 6_testing_and_validation/

| Item | What it is |
|---|---|
| `evaluate_old_setup1.py` | The first version of `evaluate.py`, written for Setup 1 alone. Replaced by `6_testing_and_validation/evaluate.py`, which gives the same numbers when run without `--pair` and `--subset`. |
| `evaluation_old_setup1/` | Output of the old script for Setup 1 alone. |
| `evaluation_s1_vs_s3/` | An interim comparison of Setups 1 and 3, made before Setup 4 existed. |
| `evaluation_interim_S4B/` | An interim comparison made when only Setup 4 run B was finished (no run A). |

The final evaluation results are in `6_testing_and_validation/output/`.

## 7_util/

| File | What it is |
|---|---|
| `make_chapter4_sample_figures.py` | An earlier copy of `7_util/make_chapter4_results_figures.py` (figures IV21 to IV25 on the 12,262 images only). The newer script also covers the 124-image test set and writes the figure data. |
| `check_gpu.py` | Despite its name, it only creates folders from an early project layout (`models/`, `results/classification/`, `scripts/classification/`) that is no longer used. |

## rev13_declared_design/ (archived 2026-11-22, after the adviser meeting)

The design of manuscript Rev 13 was replaced. The new declared comparisons are only **Setup 1 vs Setup 3**
(12,262 test images) and **Setup 2 vs Setup 4B** (124 test images), where Setup 2 is now CycleGAN + DenseNet-121 + CBAM,
trained here with the Setup 4B recipe (`5_training_and_classification/s2_densenet121_cyclegan`).
Everything below was moved here and is no longer part of the paper.

| Item | What it is |
|---|---|
| `5_training_and_classification/s4_convnextv2_cyclegan/run_A/` | Setup 4 run A (tuned on all 5,985 validation images): search, five seeds, weights. |
| `6_testing_and_validation/test_12262/` | Rev 13 evaluation on 12,262 images: S1, S3, S4A, S4B, pairs S1=S3, S3=S4A, S3=S4B. |
| `6_testing_and_validation/test_124/` | Rev 13 evaluation on the 124 images: all four setups, all six pairs (supplementary). |
| `6_testing_and_validation/test_12203_without_failed/` | Sensitivity analysis without the 120 failed translations (supplementary). |
| `7_util/tables/tableIV15`-`IV18` | Supplementary tables (other test set, supplementary DeLong, without failed translations). Removed from the live folder. |
| `7_util/tables/tableIV11`-`IV14`, `figures/fig_IV22`-`IV24`, `figure_data/fig_IV22*`, `fig_IV23*` | Copies of the Rev 13 versions, kept before these are rebuilt for the new design. |
