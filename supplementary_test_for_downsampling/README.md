# Supplementary test: a corrected K-Means artifact clustering for the downsampling branch

Started 2026-11-23 (after the adviser meeting). **Nothing in the original folders is changed.** Inputs are read from
`3_downsampling_branch/output/` (the 676-image list, the 124 / 65 lists, `features_676.npy`, the 200 manual marks);
every output is written inside this folder.

| Folder | What it is |
|---|---|
| `3_new_downsampling_branch/` | new K-Means (`cluster_kmeans_v2.py`), then copies of the CycleGAN, cleaning and quality-check scripts. Its `setup4_config.py` sends all outputs here. |
| `5_new_training_and_classification/` | copies of Setup 2 (CycleGAN + DenseNet-121) and Setup 4B (CycleGAN + ConvNeXt V2) that read the new cleaned images. Test = the 124-image test set only. Evaluations: `evaluation_124/`, `evaluation_124_old_vs_new/`. |
| `run_after_cyclegan.sh` | every step after the CycleGAN training, in order. |

## Why
The original clusters followed view position (AP / PA) more than artifacts (Table IV20), because the 580,644 HOG
values outweighed the 4 scalar features. Agreement with the manual marks was 65% (Pneumonia) and 56% (No Finding).

## Design (unchanged from the main study unless listed)
- Same 676 images, same split (487 / 65 / 124), K-Means and CycleGAN fitted on all 676 images (Chehade's procedure,
  the same leakage as in the main study).
- CycleGAN: same recipe; the schedule of a 200-epoch run, stopped at epoch 100 (= the epoch-100 generator rule).
- Setup 2 and Setup 4B: same code, search (20 trials, seed 100) and seeds (42-46) as the main study.
- Only the 676 images are translated (run A is archived, so nothing outside the 676 is needed).

## How the clustering was chosen (protocol in the docstring of `cluster_kmeans_v2.py`)
The 200 manual marks were split once, stratified by class and original cluster: 100 to **select** the version and
the artifact direction, 100 held out to **report** agreement. Criterion: balanced accuracy on the select half,
averaged over the classes; each cluster must hold at least 10% of its class.

- **Round 1** (standardized features, 7 versions incl. the original): the original (C0) was the only eligible
  version. Every rebalanced version put 15-18 extreme Pneumonia images in one cluster (outliers of the 4 scalars).
  For No Finding the original clusters tied on the select half, so the direction was arbitrary. Files: `3_new_downsampling_branch/output/clusters/round1/`.
- **Amendment 1 / round 2** (declared before scoring): rank-based (quantile) scaling instead of standardization,
  same rule; the run would stop if the original won again. **Chosen: Q1, the 4 scalars quantile-transformed.**

| Held-out half (50 marks per class) | Pneumonia | No Finding |
|---|---|---|
| Original clustering (thesis, all 200 marks) | 65% | 56% |
| New Q1: agreement (95% CI) | 70% (56-81) | 58% (44-71) |
| New Q1: balanced accuracy | 0.715 | 0.689 |
| AP share, artifact / clean cluster (original -> new) | 80% / 18% -> 67% / 23% | 23% / 5% -> 20% / 6% |

The gain is small and within the uncertainty of 50 marks per class; only 6 of the 50 held-out No Finding marks are artifacts.

## Results (run 2026-11-23, 04:33-05:31; log `run_after_cyclegan_log.txt`)

### 1. Cleaning quality (676 images, KID x1000, 10 repeats; lower = artifact and clean groups more alike)
| Group | Original clusters: before -> after | New clusters: before -> after | KID improved (new) |
|---|---|---|---|
| All 676 | 19.4 -> 21.1 | 9.6 -> 18.5 | 0/10 |
| No Finding | 10.2 -> 16.1 | 6.0 -> 14.3 | 0/10 |
| Pneumonia | 40.2 -> 36.1 | 14.6 -> 23.0 | 0/10 |

The new groups are already much closer before cleaning (KID 9.6 vs 19.4), i.e. the new split separates the images less
strongly. After cleaning they are *further apart* in every group and every repeat: the translation adds its own
signature instead of removing artifacts. FID shows the same (71.1 -> 79.8).
Translation change (341 artifact images): centre MAE 5.3 grey levels (original 4.2), centre SSIM 0.972,
12 images change > 10 levels in the centre, 1 image > 40 levels on average (script warning).

### 2. Search (20 trials each, 65 validation images)
| | lr | floor | best top-3 val AUC |
|---|---|---|---|
| Setup 2 new (DenseNet) | 4.11e-05 | 0.138 | 0.818 (old 0.839) |
| Setup 4B new (ConvNeXt V2) | 3.31e-05 | 0.268 | 0.759 (old 0.813) |

### 3. Test, 124 images, five-seed ensemble (95% CI)
| | AUC | Accuracy | Recall | Specificity | F1 |
|---|---|---|---|---|---|
| Setup 2 old | 0.725 (0.635-0.814) | 0.573 | 0.814 | 0.354 | 0.644 |
| **Setup 2 new** | **0.737 (0.651-0.823)** | 0.653 | 0.712 | 0.600 | 0.661 |
| Setup 4B old | 0.735 (0.648-0.822) | 0.677 | 0.678 | 0.677 | 0.667 |
| **Setup 4B new** | **0.738 (0.651-0.824)** | 0.661 | 0.661 | 0.662 | 0.650 |

- **Declared comparison, Setup 2 new vs Setup 4B new:** DeLong diff -0.001 (95% CI -0.045 to 0.043), p = 0.972;
  seed ANOVA on AUC 0.724 vs 0.732, p = 0.558, eta^2 = 0.045. No difference: same conclusion as the main study.
- **Old vs new (descriptive, Holm over the two pairs):** DeLong p = 0.509 (S2) and 0.878 (S4B), both adjusted 1.0.
  AUC unchanged. Setup 2's threshold moved (0.217 -> 0.514), so its accuracy, precision and specificity rose
  (seed ANOVA adjusted p 0.015, 0.010, 0.016); this is a threshold effect from the 65 validation images, not better ranking.

### Conclusion
The corrected clustering agrees slightly better with the manual marks, but it does not change the downstream results:
cleaning still does not make the artifact and clean groups alike, AUC moves by at most 0.012, and DenseNet-121 and
ConvNeXt V2 remain tied (~0.74). The bottleneck is the 487 / 65 / 124 sample size, not the clustering.
