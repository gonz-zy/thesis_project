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
| 3a | Chehade CBAM — full Chehade classifier recipe (3 image sets x 2 networks) | running |
| 3b | Chehade CBAM — CBAM last, our recipe (3 image sets x 2 networks) | queued |
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
