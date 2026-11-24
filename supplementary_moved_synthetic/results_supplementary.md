# Supplementary results: synthetic Pneumonia moved into validation and test (Setups 1 and 3)

Supplementary only. It does not replace the main real-only results (Tables IV11-IV14).
Setup 1 = StyleGAN2-ADA + DenseNet-121 + CBAM + clinical values. Setup 3 = StyleGAN2-ADA + ConvNeXt V2-Tiny + CBAM + clinical values.
All numbers are five-seed mean-probability ensembles; the threshold is the Youden point on the mean ensemble validation
probability, applied unchanged. CI = estimate ± 1.96 × SD of 1,000 stratified bootstrap samples (seed 42).

## 1. Design (what was changed)

| | Main study | This experiment |
|---|---|---|
| Real images | 232 / 31 / 59 Pneumonia; 42,204 / 5,954 / 12,203 No Finding | identical, same partitions |
| Synthetic Pneumonia | 41,972, all in training | 29,422 train / 4,281 val / 8,269 test (random, seed 20261124, each used once) |
| Train / val / test size | 84,408 / 5,985 / 12,262 | 71,858 / 10,266 / 20,531 |
| Synthetic : real Pneumonia in training | 181 : 1 | 127 : 1 |
| pos_weight | 1.000 | **1.423** (original rule, No Finding ÷ Pneumonia = 42,204 ÷ 29,654) |
| Training mean / SD | 0.49285 / 0.29342 | 0.49261 / 0.29346 (recomputed) |
| Hyperparameters, code, seeds, epoch cap, patience | - | unchanged (lr and min_lr_ratio from the main search) |

Overlap check (as Table IV2): zero real-patient, real-image and synthetic-image overlap between train-val, train-test and val-test
(`output/manifests/overlap_check.csv`). Slice (b) contains exactly the original 12,262 test images.

## 2. Training

| Setup | Seed | Epochs run | Best epoch | Best validation AUC | Train recall, real Pneumonia | Train recall, synthetic Pneumonia |
|---|---|---|---|---|---|---|
| 1 | 42 / 43 / 44 / 45 / 46 | 7 / 11 / 8 / 9 / 9 | 2 / 6 / 3 / 4 / 4 | 0.998 / 0.997 / 0.998 / 0.998 / 0.998 | 0.004 / 0.004 / 0.004 / 0.015 / 0.004 | 1.000 (all) |
| 3 | 42 / 43 / 44 / 45 / 46 | 15 / 15 / 15 / 11 / 15 | 13 / 12 / 11 / 6 / 12 | 0.997 (all) | 0.004 / 0.007 / 0.000 / 0.000 / 0.007 | 1.000 (all) |

Main study for comparison: best validation AUC 0.693 (Setup 1) and 0.628 (Setup 3). The validation AUC of about 0.997 here reflects
detection of synthetic versus real images, not Pneumonia.

## 3. Test results

### Table S1. AUC with 95% CI and precision-recall AUC

| Slice | n (Pneumonia) | Setup 1 AUC [CI] | Setup 3 AUC [CI] | Setup 1 PR-AUC [CI] | Setup 3 PR-AUC [CI] | Prevalence |
|---|---|---|---|---|---|---|
| (a) Mixed | 20,531 (8,328) | 0.9975 [0.9967, 0.9983] | 0.9978 [0.9971, 0.9986] | 0.998 [0.997, 0.999] | 0.998 [0.998, 0.999] | 0.406 |
| **(b) Real only (headline)** | 12,262 (59) | **0.645 [0.573, 0.717]** | **0.690 [0.611, 0.769]** | 0.014 [0.000, 0.037] | 0.013 [0.007, 0.019] | 0.005 |
| (c) Synthetic only | 20,472 (8,269) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.404 |

### Table S2. Threshold metrics at the validation Youden threshold

| Slice | Setup (threshold) | Accuracy | Precision | Recall | Specificity | F1 | TP / FN / FP / TN |
|---|---|---|---|---|---|---|---|
| (a) Mixed | S1 (0.6835) | 0.9971 | 1.0000 | 0.9929 | 1.0000 | 0.9964 | 8,269 / 59 / 0 / 12,203 |
| (a) Mixed | S3 (0.9824) | 0.9971 | 1.0000 | 0.9928 | 1.0000 | 0.9964 | 8,268 / 60 / 0 / 12,203 |
| (b) Real only | S1 | 0.9952 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0 / 59 / 0 / 12,203 |
| (b) Real only | S3 | 0.9952 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0 / 59 / 0 / 12,203 |
| (c) Synthetic only | S1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 8,269 / 0 / 0 / 12,203 |
| (c) Synthetic only | S3 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 0.9999 | 8,268 / 1 / 0 / 12,203 |

Accuracy 0.9952 in slice (b) is the share of No Finding images (12,203 / 12,262); it is not a sign of skill. Confusion-matrix plots:
`output/evaluation/<slice>/confusion_S1.png`, `confusion_S3.png`.

### Table S3. DeLong test, Setup 1 vs Setup 3 (ensemble AUC), Holm-corrected over the three slices

| Slice | AUC S1 | AUC S3 | Difference [95% CI] | z | p | p (Holm, 3 slices) | Significant |
|---|---|---|---|---|---|---|---|
| (a) Mixed | 0.9975 | 0.9978 | -0.0003 [-0.0007, 0.0001] | -1.48 | 0.138 | 0.401 | No |
| (b) Real only | 0.6449 | 0.6900 | -0.0451 [-0.1040, 0.0138] | -1.50 | 0.134 | 0.401 | No |
| (c) Synthetic only | 1.0000 | 1.0000 | 0.0000 | 0.00 | 1.000 | 1.000 | No |

Seed ANOVA (five seeds each, mixed slice) on AUC: p = 0.0007, η² = 0.78; this is a difference in seed-to-seed behaviour at AUC about 0.997 and is not a Pneumonia effect.

### Table S4. Slice (b) against the main study (same 12,262 images)

| Setup | Main AUC [CI] | Moved-synthetic AUC [CI] | Difference |
|---|---|---|---|
| 1 | 0.653 [0.584, 0.723] | 0.645 [0.573, 0.717] | -0.008 |
| 3 | 0.691 [0.614, 0.768] | 0.690 [0.611, 0.769] | -0.001 |

## 4. Shortcut check (recall on real vs synthetic Pneumonia)

| Setup | Partition | Real Pneumonia recall | Synthetic Pneumonia recall | No Finding specificity | AUC real-P vs NF | AUC synthetic-P vs NF |
|---|---|---|---|---|---|---|
| 1 | Train (best epoch, t = 0.5, mean of seeds) | 0.006 | 1.000 | 1.000 | - | - |
| 1 | Validation (ensemble threshold) | 0.000 | 1.000 | 1.000 | 0.698 | 1.000 |
| 1 | Test (ensemble threshold) | 0.000 | 1.000 | 1.000 | 0.645 | 1.000 |
| 3 | Train (best epoch, t = 0.5, mean of seeds) | 0.004 | 1.000 | 1.000 | - | - |
| 3 | Validation (ensemble threshold) | 0.000 | 1.000 | 1.000 | 0.628 | 1.000 |
| 3 | Test (ensemble threshold) | 0.000 | 1.000 | 1.000 | 0.690 | 1.000 |

The per-seed values are in `output/evaluation/shortcut_check.csv`. Every model flags every synthetic image and none of the real Pneumonia
images, in training, validation and test alike.

## 5. Leakage diagnostics

Encoder: frozen torchvision DenseNet-121 (ImageNet weights, not an evaluated model), global-average-pooled, L2-normalised; cosine distance.
Threshold: the 1st percentile (0.0424) and 5th percentile (0.0472) of the nearest-real-training distance of the real val/test images, whose
patients are disjoint from training, so that distribution is the "no relationship to training" baseline.

| Group | Reference | n | Median distance | 1st pct | Fraction below 1st-pct threshold | Fraction below 5th-pct threshold |
|---|---|---|---|---|---|---|
| Real val/test | real training | 18,247 | 0.064 | 0.042 | 1.0% (by construction) | 5.0% (by construction) |
| Real val/test | synthetic training | 18,247 | 0.076 | 0.048 | 0.1% | 0.9% |
| Synthetic val | real training | 4,281 | 0.071 | 0.046 | 0.1% | 1.4% |
| Synthetic test | real training | 8,269 | 0.071 | 0.046 | 0.2% | 1.6% |
| Synthetic val | synthetic training | 4,281 | **0.037** | 0.020 | **70.6%** | 83.4% |
| Synthetic test | synthetic training | 8,269 | **0.037** | 0.020 | **70.5%** | 83.4% |
| Synthetic training (5,000 sampled, self excluded) | synthetic training | 5,000 | 0.037 | 0.019 | 70.5% | 83.7% |

- Synthetic val/test images sit as close to their nearest synthetic training image as synthetic training images sit to each other
  (median 0.037 for both). They are about half as far from synthetic training images as real images are (0.076).
- They are not closer to the 232 real training Pneumonia images than genuinely independent real images are (0.1-1.6% below the thresholds).
- Interpretation: the moved images are not copies of the real training images, but they come from the same generator and form the same
  tight cluster as the synthetic training set. They are not independent of training.
- One real val/test image has distance 0.0000 to a real training image (minimum column); a single exact-duplicate pair, not examined further.

Metadata tuples (age, sex, view position), `output/leakage/metadata_summary.json`:
- 12,550 synthetic val/test rows contain only **157 distinct tuples**.
- All 157 occur in the 232 real training Pneumonia rows, so **100% of the synthetic val/test rows (12,550 / 12,550) are exact copies of training clinical vectors**. The same tuples also occur among the synthetic training rows.

## 6. Interpretation

1. **Headline, slice (b):** moving synthetic images changes nothing for real Pneumonia. AUC is 0.645 (Setup 1) and 0.690 (Setup 3), against
   0.653 and 0.691 in the main study, with CIs about ±0.07-0.08. The Setup 1 vs 3 difference stays non-significant (p = 0.134, Holm 0.401).
2. **Slices (a) and (c) measure synthetic-image detection.** AUC 0.998-1.000 with no overlap between the groups is what a model that tells
   synthetic from real would score; the near-perfect validation AUC shows the same thing. These slices say nothing about clinical performance.
3. **No real Pneumonia is detected at the chosen threshold.** The validation set is 99.3% synthetic Pneumonia, so the Youden threshold (0.68 and 0.98)
   is set by the synthetic-versus-real boundary and sits above every real image. In the main study the threshold was about 0.003-0.004 and gave
   recall 0.56. A threshold chosen on this validation set is therefore unusable for real Pneumonia, even though the ranking (AUC) on real images is unchanged.
4. **Early stopping** now picks epochs 2-6 for Setup 1 because validation AUC saturates immediately; the epoch choice is driven by
   synthetic detection, not by learning Pneumonia.

## 7. Limitations

1. The synthetic val/test images come from a generator trained on the 232 training Pneumonia images and are not independent of training
   (Section 5: they sit as close to the synthetic training set as it sits to itself).
2. Their metadata was copied from training rows: all 12,550 val/test synthetic clinical vectors duplicate training rows (157 distinct tuples).
3. Only 59 real test Pneumonia images exist, so the effective sample size for real performance is unchanged (CIs about ±0.07-0.08), and slices (a) and (c) are
   inflated by synthetic-image detection.
4. The synthetic-to-real ratio in training changed from about 181:1 to about 127:1, and pos_weight changed from 1.000 to 1.423 (original rule applied to
   the new counts), so this is not an isolated change from the main experiment.
5. Supplementary: it does not replace the main real-only results.
6. Additional notes: the learning rate and cosine floor were taken from the main search and not re-tuned; cuDNN is non-deterministic as in the main runs;
   leakage distances use one frozen ImageNet encoder, and absolute values depend on that choice.

## 8. Files

`output/evaluation/` (per-slice `ensemble_metrics.csv`, `confusion_matrices.csv`, `delong.csv`, `anova_*.csv`, confusion PNGs; `delong_holm_over_slices.csv`,
`pr_auc.csv`, `roc_curves.png`, `pr_curves.png`, `shortcut_check.csv`, `compare_with_main_slice_b.csv`), `output/leakage/`, `output/manifests/`
(split manifests with `source`, slice lists, `split_log.json`, `overlap_check.csv`), `output/s1/` and `output/s3/` (per-seed runs).
