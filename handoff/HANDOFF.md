# HANDOFF: code reorganization and the reasons behind the settings

**Date:** November 19, 2026
**Authors:** Zyro Gonzales, Francis Gerard Samson, Joey Alfonso Santos

The paper (manuscript Rev 13) is the authority for every number. This file holds the reasoning that used to sit
in long script headers. The original headers are kept word for word in `ORIGINAL_SCRIPT_NOTES.md` (same folder).

---

## 1. What changed in the reorganization

- Files were moved into the stage folders `0_dataset` to `7_util`, following the architecture diagram.
  Each stage writes to its own `output/` folder, and the next stage reads that folder directly.
- Only paths were changed in the code. No step of the study was changed. Exceptions, all small:
  - `1_histogram_normalization.py` had broken paths (`data/raw`, `data/processed/...`) and wrote `filtered.csv`,
    while every later script reads `filtered_labels.csv`. It now writes `1_global_pre-processing/output/filtered_labels.csv`.
  - `clean_images.py` defaulted to the epoch-200 CycleGAN checkpoint, but the paper used epoch 100.
    The default is now epoch 100 (`GAN_EPOCH_USED = 100` in `setup4_config.py`).
  - `2_generate_synthetic_metadata.py`: a broken print (`"\Sex share"`) was fixed to `"\nSex share"`.
  - `1_dataspilt_for_stylegan2-ada.py` was renamed `1_datasplit_for_stylegan2-ada.py` (spelling).
- The file-path columns in these CSVs were updated to the new folders (only the folder part of each path changed):
  `combined_training_labels.csv`, `subset_676.csv`, `test_balanced_124.csv`, `validation_small_65.csv`,
  `clustered_676.csv`, `cleaned_manifest.csv`.
- Normalization files were renamed: `data/2-processed/train_stats.json` → `4_image_transformation/output/train_stats_setup1_setup3.json`,
  `2B-input_for_cyclegan/train_stats.json` → `train_stats_setup4.json`.
- Evaluation folders were renamed: `evaluation_final` → `test_12262`, `evaluation_final_124` → `test_124`,
  `evaluation_sensitivity` → `test_12203_without_failed`.
- Long headers and comments were cut. Comments marked `WEIGHTS:` show where model weights are loaded or saved,
  and `TUNING:` shows where the learning rate and decay floor are tuned or used.
- Old and interim runs were moved to `archive/` (see `archive/README.md`).
- Chapter 4 tables are named by their manuscript number (`tableIV<N>_*.csv`). Tables IV1-IV4 and IV6-IV8 were
  renamed in `make_chapter4_tables.py`; the others are built by the new `make_chapter4_result_tables.py`.
- Console messages printed by the scripts were not changed.

**Records with old paths.** `run_settings.json`, `run_log.txt`, `metrics_*.json`, `clean_report.json` and
`train_stats_setup4.json` (its `source_csv` field) still show the old paths. They record how the runs were done
at the time, so they were not edited.

### Check after the reorganization

`evaluate.py` was re-run from the new folders on the saved predictions, for the 12,262 images, the 124-image test set
and the 12,203 images without failed translations. Every output file matches the saved one (differences below 1e-6,
floating-point rounding only). `make_chapter4_results_figures.py` and `make_chapter4_tables.py` were also re-run and
give the same figure data and tables. Nothing was retrained.

| Setup | Test set | AUC (95% CI) |
|---|---|---|
| 1 | 12,262 | 0.653 (0.584–0.723) |
| 3 | 12,262 | 0.691 (0.614–0.768) |
| 4A | 12,262 | 0.665 (0.584–0.745) |
| 4B | 124 | 0.735 (0.648–0.822) |

Note: the saved `test_124` results were made without `--pair` (all six pairs). The saved `test_12262` and
`test_12203_without_failed` results used the three pairs `S1=S3`, `S3=S4A`, `S3=S4B`.
Table IV13 (declared comparisons) takes S1 vs S3 and S3 vs S4A from `test_12262` and S3 vs S4B from `test_124`,
then applies Holm over these three (0.569, 0.569, 0.655). No single `delong.csv` holds this;
`7_util/make_chapter4_result_tables.py` builds it.

---

## 2. Missing pieces (not in the repo)

These were missing and have been rebuilt after the fact:
1. **`combined_training_labels.csv`** (how it was made was not recorded): `2_upsampling_branch/3_combine_training_labels.py`
   now builds it. Its output is identical to the saved file (byte for byte once paths use `\`, as on Windows).
2. **`train_stats_setup1_setup3.json`** (how it was made was not recorded): `4_image_transformation/make_stats.py`
   on `combined_training_labels.csv` gives the same mean and SD (difference about 1e-14). The original file had only
   `mean`, `std`, `n_images`; the training code reads only `mean` and `std`.
3. **StyleGAN2-ADA commands**: written in `README.md`. The commands remembered by the team used `--gpus=1 --batch=16`
   and a `--resume` from run `00002`. The saved `training_options.json` and `log.txt` are from a different run
   (`00000-...-auto2-...-batch32`): 2 GPUs, batch 32, no resume, one continuous run from 0 to 5,000 kimg in 17.9 hours.
   The manuscript also says batch size 32 and two GPUs, so the README uses those values. The batch-16 run was
   probably an earlier attempt. FID was computed after training (metrics were off). The images are named
   `SYNTH_4000_*`, so snapshot 4000 (FID 62.46) was used, and the metadata script was run with `--rename`.
   `synthetic_labels.csv` has both AP and PA view positions, so it was run with `--view_mode sample`
   (the default `PA` would give PA only).
4. **Figures IV19 and IV20**: `7_util/make_chapter4_training_figures.py` redraws them from the saved histories and
   predictions, and rewrites `table_IV_sample_predictions.csv` and `table_IV_sample_counts.csv` (identical to the
   saved ones). The four sample images are fixed in the script (the same images as Figures IV11 to IV18).

Still missing:
- Code for Figures IV11 to IV18 (the sample X-ray images) and for Table IV19 (written by hand).
- Images, `features_676.npy` and all weights (`*.pt`, `*.pkl`) are too large for the repo.

---

## 3. Reasons behind the settings

### Stage 1 and 2: pre-processing and split
- Histogram equalization is done once, on the 1024 x 1024 images. It is not applied again later.
- The split is patient-level, 70/20/10 (train/test/validation), seed 42. Every setup uses the same split.
- Images are resized 1024 → 256 with LANCZOS because StyleGAN2-ADA was trained at 256.
- Synthetic clinical values: each synthetic image copies age, sex and view position from one randomly chosen real
  training Pneumonia image (232 rows, seed 42), so age and sex stay paired.

### Stage 3: CycleGAN downsampling (Setup 4)
- **676-image set.** All 322 Pneumonia images (232 / 31 / 59 per split) plus 354 No Finding images (255 / 34 / 65),
  the 354 : 322 ratio of Chehade et al., applied per split. No Finding images come from the same split's patients,
  at most one image per patient, seed 7 (not a training seed).
- **Our split, not Chehade's.** Chehade et al. split by image; we keep our patient-level split.
- **Features.** HOG at 1024 x 1024 with scikit-image defaults gives 580,644 values; with 4 values (variance of gradients,
  lung sharpness, black ratio, white ratio) this is the 580,648 Chehade et al. report. That is how we inferred that
  HOG was computed at full resolution. The Sobel size (3 x 3), the black and white cut-offs (0.2, 0.8) and the
  lung mask are our assumptions, because Chehade et al. do not publish them. HOG makes up almost the whole vector,
  so the clustering is in practice a HOG clustering.
- **Lung mask.** Otsu does not work on equalized images (it lands at about grey 128 on every image). The mask is the
  darkest 25% of pixels in a central window, then the two largest regions. If it collapses, the whole image is used
  and this is recorded.
- **K-Means per class** (k = 2 on Pneumonia, k = 2 on No Finding), following Chehade et al. This keeps the artifact
  share decided within each class, so "was cleaned" is not a cue for the label.
- **Which cluster is the artifact cluster** was decided by a manual check of 200 images (`label_grid.py`), not by an
  automatic rule. K-Means mostly separated view position: 79.6% of the Pneumonia artifact cluster are AP images.
- **Images outside the 676** (full validation and test sets) get their domain from the saved per-class K-Means models.
  The image's true label picks the model, so a test image's cleaning depends on its label. This is a limitation.
- **Leak.** Features, K-Means and CycleGAN are fitted on all 676 images, test images included, following Chehade et al.
  This is a known pre-processing leak and is reported as a limitation.
- **CycleGAN.** Zhu et al. defaults: 200 epochs (constant for 100, then linear decay), Adam 2e-4 (0.5, 0.999), batch 1,
  LSGAN, cycle weight 10, identity weight 5, image pool 50, ResNet-9 generators, 70 x 70 PatchGAN, horizontal flip only.
  It works at 256 x 256, so every image takes the same chain as Setup 3 (1024 → 256 → 224), and artifact and clean
  images differ only in whether G_AB ran.
- **Epoch 100, not 200.** Chosen only on image criteria (centre change, SSIM, visual checks). Epoch 200 changed the lungs
  strongly. Classifier results were not used.
- **Failed translations.** An artifact image is flagged if its mean absolute change in the centre 70% of the image is above
  20 grey levels (120 of 9,182). The rule was fixed before any Setup 4 result was seen. All images stay in the main
  results; the analysis without them is in `6_testing_and_validation/output/test_12203_without_failed/`.
- **FID/KID** compare features only. They cannot show whether anatomy was changed, so images were also checked by eye.

### Stage 4: image transformation
- Resize 256 → 224 (torchvision, bilinear), scale to [0, 1], standardize with the training-set mean and SD.
- Setups 1 and 3 share one set of statistics; Setup 4 has its own (487 cleaned training images).
- Augmentation (Chehade et al.): rotation 10°, horizontal flip, width/height shift 0.1, zoom 0.1, nearest fill.
  Training images only. The horizontal flip moves the heart to the right side (stated limitation).

### Stage 5: training
- Image + 3 clinical values (sex and view position as 0/1, age / 100) concatenated after global average pooling,
  then one output unit. No dropout, no extra clinical layer.
- CBAM after each of the four blocks/stages. CBAM weights start small (N(0, 0.01²)) and the output is multiplied by 4,
  so each CBAM starts close to identity and does not shrink the pretrained features.
- ConvNeXt V2-Tiny: `convnextv2_tiny.fcmae_ft_in1k` from timm, drop-path 0.2 (fixed).
- Adam, weight decay 1e-4, batch 16, early stopping on validation AUC (patience 5), linear warmup then cosine decay.
  Setups 1 and 3: up to 15 epochs, 3 warmup. Setup 4: up to 50 epochs, 10 warmup (the same 20% share),
  positive-class weight 255 / 232 = 1.099.
- **Tuning.** Only the learning rate (1e-5 to 3e-4, log) and the decay floor (0 to 0.5 of the learning rate) are tuned,
  with Optuna TPE: 20 trials, 5 random start-up trials, seed 100 (not a final seed). The objective is the mean of the
  3 highest validation AUCs of a trial, because one best epoch is too noisy with 31 validation Pneumonia images.
  The test set is never opened during the search. Setup 4 runs A and B are tuned separately.
- **Shortcut check.** Training recall on real vs synthetic Pneumonia is logged (Setups 1 and 3). For Setup 4, recall on
  artifact vs clean images is logged instead.

### Stage 6: evaluation
- Ensemble = mean probability of seeds 42–46. Threshold = Youden index on the validation predictions.
- 95% CI = estimate ± 1.96 × SD of 1,000 stratified bootstrap samples (class counts kept), seed 42.
- DeLong test between setups on the same test images, Holm adjustment over the declared pairs.
- One-way ANOVA across the five seeds (training randomness) and across bootstrap samples (with eta-squared).
- Chehade et al. (2025) is shown as a reported reference only. They published no per-image predictions,
  so DeLong and ANOVA cannot include Setup 2.
- No control without synthetic images was trained, so no causal claim about the synthetic images can be made.

---

## 4. Open items

- [ ] Update the architecture diagram: 41,972 synthetic scans (not 41,775); Optuna tunes the learning rate and decay floor;
      stage numbering 2A/2B → 2 and 3.
- [ ] Decide whether to add the missing scripts in Section 2 (at least the one that builds `combined_training_labels.csv`).
- [ ] Where the raw NIH images, generated images and weights are stored (to be filled in by the team).
