# Supplementary tests — handoff (analysis per finished setup)

Updated automatically after each setup finishes; newest entry at the bottom. All tests use the 676-image set
(487 train / 65 validation / 124 test, patient-level), and are evaluated on the same 124 test images
(five-seed ensemble, Youden threshold from the 65 validation images, DeLong + Holm, seed ANOVA).
Main-study reference (CycleGAN, CBAM after every block): **Setup 2 DenseNet 0.725, Setup 4B ConvNeXt 0.735**
(DeLong p = 0.596). Detailed tables: `README.md` in this folder. Model weights are not in git (too large).

| # | Test | Status |
|---|---|---|
| 1 | Corrected K-Means (new clustering + new CycleGAN) | done |
| 2 | Without CycleGAN (two resize paths) | done |
| 3a | Chehade CBAM — full Chehade classifier recipe (3 image sets x 2 networks) | done |
| 3b | Chehade CBAM — CBAM last, our recipe (3 image sets x 2 networks) | running (1 of 6 done) |
| 3c | Chehade CBAM — comparisons | queued |

---

## 1. Corrected K-Means (folders `3_new_downsampling_branch/`, `5_new_training_and_classification/`)
**What:** the original clusters followed view position (AP/PA) more than artifacts. A new clustering (4 scalar
features, rank-scaled, no HOG) was chosen on 100 of the 200 manual marks and checked on the other 100; a new CycleGAN
was trained on the new groups (100 epochs), and Setups 2 / 4B were retrained.

| | Original | New |
|---|---|---|
| Agreement with manual marks (held-out half) | 65% Pneumonia / 56% No Finding | 70% / 58% |
| KID x1000 artifact vs clean, before -> after cleaning | 19.4 -> 21.1 | 9.6 -> 18.5 |
| DenseNet test AUC | 0.725 | 0.737 |
| ConvNeXt test AUC | 0.735 | 0.738 |

- DenseNet vs ConvNeXt (new): DeLong p = 0.97. Old vs new: p = 0.51 (DenseNet), 0.88 (ConvNeXt).
- **Analysis:** the clustering improved a little (within the uncertainty of 50 marks per class), but the CycleGAN still
  pushes the two groups apart instead of together, and classification does not change. The clustering was not the
  bottleneck.

## 2. Without CycleGAN (folders `5_no_cyclegan_1024_to_224/`, `5_no_cyclegan_1024_to_256_to_224/`)
**What:** the same two networks and recipe on the untranslated images. 1024->224 = one resize; 1024->256->224 = the
exact S2/S4B path (clean images byte-identical to the main study, artifact images simply not translated).

| Test AUC (95% CI) | With CycleGAN | No CycleGAN 1024->224 | No CycleGAN 1024->256->224 |
|---|---|---|---|
| DenseNet-121 + CBAM | 0.725 (0.635-0.814) | 0.725 (0.638-0.812) | 0.713 (0.620-0.806) |
| ConvNeXt V2 + CBAM | 0.735 (0.648-0.822) | 0.748 (0.662-0.833) | 0.728 (0.639-0.817) |

- With vs without CycleGAN: DeLong p = 0.97 / 0.50 (1024->224) and 0.64 / 0.69 (1024->256->224); all Holm-adjusted 1.0.
- DenseNet vs ConvNeXt without CycleGAN: p = 0.26 and 0.47.
- **Analysis:** removing the CycleGAN moves AUC by at most 0.013 in either direction. Together with the KID result,
  this means the CycleGAN cleaning step contributes nothing measurable in our pipeline — the opposite of Chehade's
  +3.98 points (their Table 2). The resize path does not matter. Thesis sentence: "An ablation without CycleGAN gave
  the same test AUC (DenseNet 0.725 vs 0.725; ConvNeXt 0.748 vs 0.735; DeLong p >= 0.50)."

## 3a. Full Chehade classifier (folder `5_chehade_cbam/full_chehade_classifier/`, finished 14:02)
**What:** Chehade et al.'s classifier recipe as written in their Sec. 3.6: ImageNet backbone with the lower layers
frozen (assumption: everything before the last block), one CBAM on the last block, dense 1024 -> 512 + dropout 0.5,
clinical values joined before the output, Adam with a fixed lr 1e-4 (no search), plain BCE, inputs in [0, 1],
patience 5; seeds 42-46. Same 487 / 65 / 124 images. (Ensemble numbers below were computed from the finished runs;
the script's own `evaluation/` folders are written at the end of the whole test.)

| Test AUC (95% CI) | CycleGAN | No CycleGAN 1024->256->224 | No CycleGAN 1024->224 |
|---|---|---|---|
| DenseNet-121, Chehade recipe | 0.694 (0.602-0.786) | 0.715 (0.627-0.802) | 0.717 (0.631-0.804) |
| ConvNeXt V2, Chehade recipe | 0.694 (0.602-0.786) | 0.694 (0.603-0.785) | 0.713 (0.625-0.801) |
| *Our recipe, CBAM everywhere (reference)* | *D 0.725 / C 0.735* | *D 0.713 / C 0.728* | *D 0.725 / C 0.748* |

- DenseNet vs ConvNeXt (Chehade recipe): DeLong p = 0.99 / 0.50 / 0.89 — tied in every image set.
- Our recipe vs Chehade recipe (CycleGAN images): DenseNet 0.725 vs 0.694 (DeLong p = 0.30), ConvNeXt 0.735 vs 0.694
  (p = 0.16). Seed ANOVA on AUC: DenseNet p = 0.27; **ConvNeXt 0.727 vs 0.677, p = 0.0016 (Holm 0.011)** — the only
  significant result here: per seed, our recipe is better for ConvNeXt.
- With vs without CycleGAN (Chehade recipe): DenseNet 0.694 vs 0.715 (p = 0.30), ConvNeXt 0.694 vs 0.694 (p = 0.99).
- Training: best epoch 2-19 (median ~8); training Pneumonia recall reaches ~0.90 by epoch 9 while validation AUC peaks
  around epoch 4 (DenseNet seed 42), so freezing the lower layers slows the overfitting but does not stop it.
  Validation AUCs are 0.69-0.76 (vs 0.81-0.84 in our searched runs), and the validation-to-test drop is smaller
  (~0.03 vs ~0.09-0.13) because there is no search picking the luckiest settings on the 65 images.

**Analysis:** reproducing Chehade's classifier recipe does **not** bring us closer to their 0.8387; it is slightly
worse than our recipe (0.69-0.72 vs 0.71-0.75), significantly so per seed for ConvNeXt. So the gap to Chehade is not
explained by our fine-tuning choices (full fine-tuning, CBAM everywhere, tuned lr). The CycleGAN again adds nothing.
The remaining explanations are outside the classifier: their split (image-level, 10 random splits; ours is one fixed
patient-level split), their test-set size and variance (their SD 1.4% is unusually small for ~135-image test sets),
and details they do not report. Next: 3b separates the effect of CBAM placement alone.

## 3b-1. CBAM last, our recipe — DenseNet-121, CycleGAN images (finished 14:28)
**What:** identical to Setup 2 (CycleGAN images, full fine-tuning, 20-trial search, seeds 42-46) except that there is
only **one CBAM, after the last dense block** (as Chehade). Search optimum lr 1.17e-4, floor 0.38 (best top-3
validation AUC 0.828, trial 8, 0 diverged).

| DenseNet-121, CycleGAN images | CBAM after every block (Setup 2) | CBAM last only | Full Chehade recipe |
|---|---|---|---|
| Test AUC (95% CI) | 0.725 (0.635-0.814) | 0.725 (0.635-0.814) | 0.694 (0.602-0.786) |
| Mean AUC of the 5 seeds | 0.706 | 0.711 | 0.687 |
| Accuracy / recall / specificity | 0.573 / 0.814 / 0.354 | 0.613 / 0.661 / 0.569 | 0.645 / 0.712 / 0.585 |
| Threshold (from validation) | 0.217 | 0.437 | 0.345 |

- CBAM everywhere vs CBAM last: identical ensemble AUC (both rank exactly 2,779 of the 3,835 Pneumonia / No Finding
  pairs correctly; the predictions themselves differ, see the thresholds and seed means). DeLong p = 1.0; seed ANOVA
  p = 0.65.
- CBAM last (our recipe) vs full Chehade recipe: 0.725 vs 0.694, DeLong p = 0.30, seed ANOVA p = 0.18.
- Per seed: best epoch 5-10, validation 0.80-0.86 vs test 0.70-0.73 (the same ~0.11 validation-to-test drop as
  Setup 2). Domain check: training Pneumonia recall 0.86-0.94 on artifact images vs 0.65-0.82 on clean ones.

**Analysis:** where the CBAM sits makes no difference for DenseNet (0.725 either way). The CBAM placement is not why
our results differ from Chehade's, and their placement is not better than ours.
