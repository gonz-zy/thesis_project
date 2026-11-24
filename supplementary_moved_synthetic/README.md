# Supplementary: synthetic Pneumonia moved into validation and test (Setups 1 and 3)

Supplementary experiment only. It does **not** replace the main real-only results and does not touch any
main-experiment file. Everything it writes goes to `supplementary_moved_synthetic/output/`.

## What changes from the main study
- Real images stay in their original partition (232 / 31 / 59 Pneumonia; 42,204 / 5,954 / 12,203 No Finding).
- The 41,972 existing StyleGAN2-ADA images, all used for training in the main study, are reassigned at random
  without replacement: 29,422 train / 4,281 val / 8,269 test (split seed 20261124). No new images are generated.
- New partitions: train 71,858, val 10,266, test 20,531.
- Training-set mean/std recomputed on the new training set: 0.49261 / 0.29346 (main: 0.49285 / 0.29342).
- **pos_weight = 1.423.** This is the original rule in `train.py` (No Finding ÷ Pneumonia) applied to the new
  counts, 42,204 ÷ 29,654. In the main Setups 1 and 3 it was 1.0. It is kept on purpose and listed as a limitation.
- Unchanged: the original `train.py` / `cxr_data.py` / model code of each setup (only path constants are
  redirected by `train_moved.py`), the main-study tuned lr and min_lr_ratio (`best_params_cbam-all.json`, no new
  search), CBAM after every block, clinical features, batch 16, 15 epochs (3 warmup), patience 5 on validation AUC,
  weight decay 1e-4, samples_per_epoch 84,384 (now about 1.17 passes over the training set), seeds 42-46, Youden
  threshold on the five-seed mean validation probability, the original `evaluate.py` (1,000 stratified bootstrap
  samples, seed 42).

## Seeds
| What | Seed |
|---|---|
| Synthetic reassignment | 20261124 (`numpy.random.default_rng`) |
| Training runs | 42, 43, 44, 45, 46 (Python, NumPy, torch, CUDA, sampler generator; set by the original `train.py`) |
| Bootstrap | 42 |
| Leakage: synthetic self-distance sample | 0 |

cuDNN is left in its default (non-deterministic) mode, as in the main runs, so GPU results can differ in the
last decimals between re-runs.

## Files
| File | Purpose |
|---|---|
| `make_split.py` | Step 0 count checks, Step 1 split, manifests (`source` = real / synthetic), slice lists, overlap check |
| `train_moved.py` | runs the original Setup 1 / Setup 3 `train.py` on the new manifests |
| `leakage.py` | Step 3: nearest-neighbour embedding distances (frozen ImageNet DenseNet-121) and metadata-tuple copies |
| `evaluate_slices.py` | Step 2: original `evaluate.py` per slice, Holm over slices, PR-AUC, ROC/PR curves, confusion plots, shortcut check, comparison with the main study |
| `run_all.sh` | the whole pipeline |
| `results_supplementary.md` | results tables, interpretation, limitations |

Outputs: `output/manifests/`, `output/s1/`, `output/s3/` (runs; weights `*.pt` are git-ignored),
`output/leakage/`, `output/evaluation/` (`a_mixed/`, `b_real_only/`, `c_synthetic_only/`), `output/logs/`,
`output/run_log.txt`.

## Reproduce
From the repository root, with the project venv active:

```bash
bash supplementary_moved_synthetic/run_all.sh
```

or step by step:

```bash
python supplementary_moved_synthetic/make_split.py
python 4_image_transformation/make_stats.py --csv supplementary_moved_synthetic/output/manifests/combined_training_labels.csv \
       --out supplementary_moved_synthetic/output/train_stats_moved.json
for seed in 42 43 44 45 46; do
  python supplementary_moved_synthetic/train_moved.py --setup s1 --cbam all --seed $seed --eval_test \
         --params 5_training_and_classification/s1_densenet121_stylegan2-ada/output/search/best_params_cbam-all.json
  python supplementary_moved_synthetic/train_moved.py --setup s3 --cbam all --seed $seed --eval_test \
         --params 5_training_and_classification/s3_convnextv2_stylegan2-ada/output/search/best_params_cbam-all.json
done
python supplementary_moved_synthetic/leakage.py
python supplementary_moved_synthetic/evaluate_slices.py
```

GPU time on an RTX 5070: about 12-14 h in total.
