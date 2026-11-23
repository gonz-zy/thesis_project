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
| 3b | Chehade CBAM — CBAM last, our recipe (3 image sets x 2 networks) | running (5 of 6 done) |
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

## 3b-2. CBAM last, our recipe — ConvNeXt V2, CycleGAN images (finished 15:00)
**What:** identical to Setup 4B except for one CBAM after the last stage. Search optimum lr 1.66e-4 (higher than
Setup 4B's), floor 0.05; best top-3 validation AUC 0.839 (trial 17, 0 diverged).

| ConvNeXt V2, CycleGAN images | CBAM after every stage (Setup 4B) | CBAM last only | Full Chehade recipe |
|---|---|---|---|
| Test AUC (95% CI) | 0.735 (0.648-0.822) | 0.701 (0.610-0.793) | 0.694 (0.602-0.786) |
| Mean AUC of the 5 seeds | 0.727 | 0.706 | 0.677 |
| Accuracy / recall / specificity | 0.677 / 0.678 / 0.677 | 0.581 / 0.627 / 0.538 | 0.621 / 0.593 / 0.646 |

- CBAM everywhere vs CBAM last: 0.735 vs 0.701, DeLong p = 0.15 (Holm 0.31); seed ANOVA p = 0.064.
- CBAM last vs full Chehade recipe: 0.701 vs 0.694, DeLong p = 0.82; seed ANOVA p = 0.46.
- Per seed: validation 0.78-0.82 vs test 0.69-0.72 (drop ~0.09). Domain check: training Pneumonia recall 0.82-0.93
  on artifact vs 0.63-0.88 on clean images.

**Analysis:** for ConvNeXt, one CBAM at the end is a little worse than a CBAM after every stage (-0.034), but not
significantly. Together with 3b-1 (DenseNet: no change), Chehade's CBAM placement does not help either network; if
anything our placement is slightly better for ConvNeXt.

## 3b-3. CBAM last, our recipe — DenseNet-121, no CycleGAN (1024->256->224) (finished 15:29)
**What:** DenseNet with one CBAM after the last block, untranslated images (clean images identical to the main study).
Search optimum lr 2.36e-5 (low), floor 0.39; best top-3 validation AUC 0.781 (trial 10, 0 diverged).

| DenseNet-121, no CycleGAN (1024->256->224) | CBAM after every block | CBAM last only | Full Chehade recipe |
|---|---|---|---|
| Test AUC (95% CI) | 0.713 (0.620-0.806) | 0.742 (0.655-0.829) | 0.715 (0.627-0.802) |
| Mean AUC of the 5 seeds | 0.703 | 0.729 | 0.708 |
| Accuracy / recall / specificity | 0.540 / 0.780 / 0.323 | 0.669 / 0.729 / 0.615 | 0.653 / 0.763 / 0.554 |

- CBAM everywhere vs CBAM last: 0.713 vs 0.742, DeLong p = 0.090 (Holm 0.18); seed ANOVA p = 0.30.
- CBAM last vs full Chehade recipe: 0.742 vs 0.715, DeLong p = 0.32; seed ANOVA p = 0.46.
- Per seed: test 0.72-0.74 (the most stable set so far); validation 0.75-0.79, so the validation-to-test drop is only
  ~0.04 — the low learning rate gave longer runs (best epoch 12-23) and less selection optimism than in 3b-1 / 3b-2.

**Analysis:** the highest DenseNet AUC so far (0.742), but within noise of every other DenseNet run (0.69-0.74). Here
CBAM-last is slightly better than CBAM-everywhere, while for ConvNeXt on CycleGAN images (3b-2) it was slightly worse:
the CBAM placement effect changes sign between runs and is never significant — it is noise around ~0.72.

## 3b-4. CBAM last, our recipe — ConvNeXt V2, no CycleGAN (1024->256->224) (finished 15:59)
**What:** ConvNeXt with one CBAM after the last stage, untranslated images. Search optimum lr 4.24e-5, floor 0.42
(best top-3 validation AUC 0.802). Note: this is random start-up trial 1, the same settings that won the CBAM-everywhere
search on these images (the search seed 100 draws the same 5 start-up trials every time).

| ConvNeXt V2, no CycleGAN (1024->256->224) | CBAM after every stage | CBAM last only | Full Chehade recipe |
|---|---|---|---|
| Test AUC (95% CI) | 0.728 (0.639-0.817) | 0.726 (0.638-0.815) | 0.694 (0.603-0.785) |
| Mean AUC of the 5 seeds | 0.729 | 0.723 | 0.689 |
| Accuracy / recall / specificity | 0.637 / 0.678 / 0.600 | 0.661 / 0.678 / 0.646 | 0.637 / 0.712 / 0.569 |

- CBAM everywhere vs CBAM last: 0.728 vs 0.726, DeLong p = 0.88; seed ANOVA p = 0.47.
- CBAM last vs full Chehade recipe: 0.726 vs 0.694, DeLong p = 0.13; **seed ANOVA 0.723 vs 0.689, p = 0.014
  (Holm 0.027)** — per seed our recipe beats Chehade's again (as for ConvNeXt in 3a).
- Per seed: test 0.71-0.74, validation 0.75-0.80 (drop ~0.06).

**Analysis:** CBAM placement: no difference (0.728 vs 0.726). Recipe: our full fine-tuning with a searched lr is
better than Chehade's frozen-backbone recipe for ConvNeXt (now significant per seed in two image sets). A frozen
backbone suits DenseNet better than ConvNeXt, plausibly because ConvNeXt's ImageNet features need more adaptation to
grey-scale X-rays.

## 3b-5. CBAM last, our recipe — DenseNet-121, no CycleGAN (1024->224) (finished 16:23)
**What:** DenseNet with one CBAM after the last block, untranslated images resized 1024->224 in one step. Search
optimum lr 1.80e-4, floor 0.16; best top-3 validation AUC 0.785 (trial 17, 0 diverged).

| DenseNet-121, no CycleGAN (1024->224) | CBAM after every block | CBAM last only | Full Chehade recipe |
|---|---|---|---|
| Test AUC (95% CI) | 0.725 (0.638-0.812) | 0.725 (0.636-0.813) | 0.717 (0.631-0.804) |
| Mean AUC of the 5 seeds | 0.701 | 0.720 | 0.706 |
| Accuracy / recall / specificity | 0.645 / 0.797 / 0.508 | 0.645 / 0.746 / 0.554 | 0.629 / 0.780 / 0.492 |

- CBAM everywhere vs CBAM last: 0.725 vs 0.725, DeLong p = 0.96; seed ANOVA p = 0.39.
- CBAM last vs full Chehade recipe: 0.725 vs 0.717, DeLong p = 0.79; seed ANOVA p = 0.39.
- Per seed: test 0.69-0.75, validation 0.75-0.82 (drop ~0.07).

**Analysis:** no difference anywhere: CBAM placement, recipe, and (vs 3b-1, CycleGAN images 0.725) the CycleGAN —
all three DenseNet CBAM-last runs land at 0.725-0.742.
