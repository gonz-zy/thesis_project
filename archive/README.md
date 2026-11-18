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
