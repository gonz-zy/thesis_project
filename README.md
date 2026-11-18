# Pneumonia Binary Classification on NIH ChestX-ray14 via StyleGAN2-ADA, DenseNet-121 and ConvNeXt V2

Code for the thesis (Gonzales, Samson, Santos). Binary task: Pneumonia vs No Finding on NIH ChestX-ray14.
The folders follow the system architecture diagram. Each stage has its scripts at the top of the folder and its
results in an `output/` folder. A stage reads the `output/` folder of the stage before it directly (no copies).

| Setup | Method | Code |
|---|---|---|
| 1 | StyleGAN2-ADA upsampling + DenseNet-121 | `5_training_and_classification/s1_densenet121_stylegan2-ada` |
| 2 | Chehade et al. (2025), AUC 0.8387 ± 0.014 | Reported result only. Not trained here. |
| 3 | StyleGAN2-ADA upsampling + ConvNeXt V2 | `5_training_and_classification/s3_convnextv2_stylegan2-ada` |
| 4A | CycleGAN downsampling + ConvNeXt V2, tuned on all 5,985 validation images | `5_training_and_classification/s4_convnextv2_cyclegan` (`run_A`) |
| 4B | Same as 4A, tuned on the 65-image validation list | `5_training_and_classification/s4_convnextv2_cyclegan` (`run_B`) |

## Folder structure

```
0_dataset/                      Data_Entry_2017_v2020.csv, images/ (NIH images, not in the repo), docs/
1_global_pre-processing/        label filtering + histogram equalization
2_upsampling_branch/            patient-level split, resize 1024 -> 256, StyleGAN2-ADA, synthetic metadata
3_downsampling_branch/          676-image set, Sobel/HOG features, K-Means, CycleGAN cleaning
4_image_transformation/         normalization statistics (mean, SD) for the classifiers
5_training_and_classification/  s1, s3, s4: training, Optuna search, final 5-seed runs
6_testing_and_validation/       AUC, CI, DeLong, ANOVA on the saved test predictions
7_util/                         checks and the Chapter 4 tables and figures
archive/                        superseded runs and scripts (see archive/README.md)
handoff/                        reasons behind the settings, and notes for the next person
```

Not in the repo (too large): every image folder, `features_676.npy`, and all model weights
(`*.pt`, `*.pkl`). Their folders are kept so you can see where they go.

## Input and output of each stage

| Stage | Main input | Main output |
|---|---|---|
| 0 | NIH download | `0_dataset/Data_Entry_2017_v2020.csv`, `0_dataset/images/*.png` |
| 1 | stage 0 | `1_global_pre-processing/output/filtered_labels.csv` (60,683 rows), `output/normalized_images/` (1024 x 1024) |
| 2 | stage 1 | `2_upsampling_branch/output/labels/{training,validation,testing}_labels.csv` (42,436 / 5,985 / 12,262 rows), `labels/synthetic_labels.csv` (41,972), `labels/combined_training_labels.csv` (84,408), image folders at 256 x 256, `output/stylegan2-ada/` (training log and FID) |
| 3 | stages 1 and 2 | `3_downsampling_branch/output/labels/subset_676.csv`, `test_balanced_124.csv`, `validation_small_65.csv`, `test_not_failed.csv`; `clusters/clustered_676.csv`, `clustered_all.csv`; `cleaned/cleaned_manifest.csv` + `cleaned/224/`; `fid_kid/`, `translation_change/` |
| 4 | stages 2 and 3 | `4_image_transformation/output/train_stats_setup1_setup3.json`, `train_stats_setup4.json` |
| 5 | stages 2, 3, 4 | per setup: `output/search/` (Optuna) and `output/final/cbam_all/seed42..46/` with `test_predictions_cbam-all_seed<N>.csv` (columns: `Image Index, Patient ID, Finding Labels, label, prob_pneumonia`), `val_predictions_*`, `history_*`, `metrics_*` |
| 6 | stage 5 | `6_testing_and_validation/output/test_12262/`, `test_124/`, `test_12203_without_failed/`: `ensemble_metrics.csv`, `per_seed_test_metrics.csv`, `delong.csv`, `anova_seeds.csv`, `confusion_matrices.csv`, ... |
| 7 | all | `7_util/output/chapter4/` tables, figures and figure data |

The `ensemble_metrics.csv` files in stage 6 hold the final performance metrics (the diagram's `final_performance_metrics.csv`).

## Run order

Run every command from this folder (the project root). Python 3.10+.
Packages: `pandas numpy scipy scikit-learn scikit-image opencv-python pillow tqdm torch torchvision timm optuna joblib matplotlib pytorch-fid`.

```bash
# 1. Global pre-processing
python 1_global_pre-processing/1_histogram_normalization.py

# 2. Upsampling branch
python 2_upsampling_branch/1_datasplit_for_stylegan2-ada.py
#    StyleGAN2-ADA: see "StyleGAN2-ADA commands" below
python 2_upsampling_branch/2_generate_synthetic_metadata.py --rename --view_mode sample
python 2_upsampling_branch/3_combine_training_labels.py

# 3. Downsampling branch
python 3_downsampling_branch/make_subset.py
python 3_downsampling_branch/features.py --workers 8
python 3_downsampling_branch/cluster_kmeans.py
python 3_downsampling_branch/label_grid.py            # mark manual_labels.csv by hand, then:
python 3_downsampling_branch/label_grid.py --score
python 3_downsampling_branch/assign_domains.py --workers 8
python 3_downsampling_branch/train_cyclegan.py
python 3_downsampling_branch/clean_images.py --manifest 3_downsampling_branch/output/clusters/clustered_all.csv
python 3_downsampling_branch/translation_change.py --cleaned_dir 3_downsampling_branch/output/cleaned
python 3_downsampling_branch/make_flag_list.py
python 3_downsampling_branch/fid_kid.py --backbone fid

# 4. Image transformation (normalization statistics; the resize/standardize/augment code is in cxr_data.py)
python 4_image_transformation/make_stats.py --csv 2_upsampling_branch/output/labels/combined_training_labels.csv --out 4_image_transformation/output/train_stats_setup1_setup3.json
python 4_image_transformation/make_stats.py --csv 3_downsampling_branch/output/cleaned/cleaned_manifest.csv --split training --out 4_image_transformation/output/train_stats_setup4.json

# 5. Training and classification (each = Optuna search, then seeds 42-46 with --eval_test)
python 5_training_and_classification/s1_densenet121_stylegan2-ada/run_all.py
python 5_training_and_classification/s3_convnextv2_stylegan2-ada/run_all.py
python 5_training_and_classification/s4_convnextv2_cyclegan/run_all.py --run both

# 6. Testing and validation (full commands in the header of evaluate.py)
python 6_testing_and_validation/evaluate.py --setup S1=... --setup S3=... --setup S4A=... --setup S4B=... --pair S1=S3 --pair S3=S4A --pair S3=S4B --out 6_testing_and_validation/output/test_12262

# 7. Chapter 4 tables and figures
python 7_util/make_chapter4_tables.py              # Tables IV1-IV4, IV6-IV8 (+ supporting tables)
python 7_util/make_chapter4_result_tables.py       # Tables IV5, IV9-IV18, IV20
python 7_util/make_chapter4_training_figures.py    # Figures IV19, IV20
python 7_util/make_chapter4_results_figures.py     # Figures IV21-IV25
```

### StyleGAN2-ADA commands

Run inside `2_upsampling_branch/stylegan2-ada-pytorch`. The training settings come from the saved
`output/stylegan2-ada/training_options.json` and `log.txt` (one run, 0 to 5,000 kimg, 2 GPUs, batch 32, 17.9 hours),
which is also what the manuscript reports. The exact command line was not saved, so check it before reuse.

```bash
python dataset_tool.py --source=../output/training_pneumonia_only --dest=stylegan_dataset.zip
python train.py --outdir=training-runs --data=stylegan_dataset.zip --gpus=2 --batch=32 --kimg=5000 \
    --metrics=none --seed=42 --target=0.6 --augpipe=bg --gamma=0.8192
# FID of each snapshot afterwards (metrics were off during training):
python calc_metrics.py --metrics=fid50k_full --data=stylegan_dataset.zip --network=training-runs/<run>/network-snapshot-004000.pkl
# 41,972 images from the 4,000 kimg snapshot (FID 62.46, the lowest):
python generate.py --outdir=../output/generated-pneumonia --seeds=0-41971 --network=training-runs/<run>/network-snapshot-004000.pkl
```

### Chapter 4 tables and figures

All in `7_util/output/chapter4/`. Tables: `tables/tableIV1_*.csv` to `tableIV20_*.csv` (no file for Table IV19,
which was written by hand). Figures: `figures/fig_IV19_*.png` to `fig_IV25_*.png`. Files starting with
`a`, `b` or `c` are supporting tables that are not in the manuscript.

## Results in this repo (ensemble of seeds 42-46)

| Setup | Test set | AUC (95% CI) | Recall |
|---|---|---|---|
| 1 | 12,262 images | 0.653 (0.584–0.723) | 0.559 |
| 3 | 12,262 images | 0.691 (0.614–0.768) | 0.559 |
| 4A | 12,262 images | 0.665 (0.584–0.745) | 0.542 |
| 4B | 124-image test set | 0.735 (0.648–0.822) | 0.678 |

DeLong, declared comparisons (Table IV13, Holm over the three): S1 vs S3 and S3 vs S4A on the 12,262 images,
p = 0.569 and 0.569; S3 vs S4B on the 124-image test set, p = 0.655.
Seed ANOVA on AUC (Holm-adjusted): 0.019, 0.039, 0.0018. FID of the synthetic images: 62.46.
These values were re-computed from the saved predictions after the reorganization and match the saved files.
