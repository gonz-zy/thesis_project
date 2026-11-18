# Original script notes (before the cleanup)

The long header notes that were removed from each script are kept here word for word, so no reasoning is lost.
Paths inside these notes are the OLD paths (`data/2-processed/...`, `results/...`, `scripts/...`).


## `2_upsampling_branch/2_generate_synthetic_metadata.py`

```text
2_generate_synthetic_metadata.py

Creates synthetic_labels.csv for the 41,972 StyleGAN2-ADA pneumonia images,
matching the NIH ChestX-ray14 CSV column format exactly.

- Patient Age / Sex (and optionally View Position) are sampled from the REAL
  pneumonia rows of the TRAINING split only (232 rows). Each synthetic row copies
  the values from ONE randomly chosen real row, so age/Sex correlations are kept.
- Patient IDs are SYNTH_00001 ... SYNTH_41972 and are checked against real IDs.
- Random seed = 42.

Usage (from the project root):
    python 2_generate_synthetic_metadata.py
    python 2_generate_synthetic_metadata.py --rename          # rename files to SYNTH_4000_XXXXX.png
    python 2_generate_synthetic_metadata.py --view_mode sample
```


## `3_downsampling_branch/assign_domains.py`

```text
assign_domains.py -- gives every validation and test image outside the 676 a domain (artifact | clean).

Why: Setup 4's classifier is trained on the 487 cleaned training images, but its full-size evaluation (run A: 5,985
validation images; primary comparison with Setup 3: 12,262 test images) needs every one of those images to go through
the SAME cleaning. Only 65 + 124 of them are among the 676 whose domains K-Means and the manual check already settled.

How: for each image the exact feature code of features.py (same mask, same HOG, same four scalars) -> the saved per-class
StandardScaler -> the saved per-class K-Means -> cluster -> domain, using the cluster -> domain map of the 676.
Per-class means the image's TRUE label picks which saved model is used, as in Chehade (they cluster per class). A test image's
cleaning therefore depends on its label. State this as a limitation. The two classes have similar artifact shares (44% / 46%).

Safety: before touching any new image the script runs the 676 through the same code and requires that it reproduces their
four scalars and every cluster in clustered_676.csv. If it does not, the extraction differs from the one that made the
clusters and the script stops. Only the cluster id and two distances are kept per image (the 580,648-value vectors are
never stored: 18,000 of them would be about 42 GB).

Reads   clusters\clustered_676.csv, clusters\kmeans_pneumonia.joblib, kmeans_no_finding.joblib, features\scalars_676.csv,
        labels\validation_labels.csv and testing_labels.csv (2A), the 1024 images in 1-normalized_images
Writes  clusters\assigned_outside.csv   one row per new image (resume-safe: rerun continues where it stopped)
        clusters\clustered_all.csv      the 676 plus the new images: Image Index, Finding Labels, split, cluster, domain, origin
        clusters\assign_report.json

CPU only. Roughly 0.5 s per image per core: about 40 min with --workers 4 (Setup 3 running), about 20 min with 8.

Run (project root):
    venv\Scripts\python.exe scripts\2b-cyclegan\assign_domains.py --workers 4
    venv\Scripts\python.exe scripts\2b-cyclegan\assign_domains.py --selfcheck_only        (about 2 min)
```


## `3_downsampling_branch/clean_images.py`

```text
clean_images.py -- writes the 224 x 224 images the Setup 4 classifier reads.

For every row of the manifest (default: clustered_676.csv):
    artifact domain : Setup 3's 256 image -> G_AB -> rounded to 8 bit -> torchvision Resize((224, 224)) -> PNG
    clean domain    : Setup 3's 256 image ----------------------------> torchvision Resize((224, 224)) -> PNG
The last step is the SAME function for both (setup4_images.to_224), and it is the transform CXRDataset applies, so a
clean image comes out byte-identical to what Setup 3's pipeline feeds its network. --verify_setup3_dir proves that
against the real cxr_data.py.

Any CSV with Image Index, split and domain works, so the images outside the 676 (full validation and test sets, for the
full-size evaluation) go through the same script once a domain has been assigned to them.

Writes  cleaned\224\<Image Index>              the classifier input
        cleaned\translated_256\<Image Index>   G_AB output before the 224 step (artifact images only; used by FID/KID)
        cleaned\cleaned_manifest.csv            the manifest plus `translated` and `path` (the 224 file)
        cleaned\clean_report.json               counts and how much the generator changed each image

Run (project root):
    venv\Scripts\python.exe scripts\2b-cyclegan\clean_images.py
    venv\Scripts\python.exe scripts\2b-cyclegan\clean_images.py --verify_setup3_dir scripts\3b-convnextv2_stylegan2-ada
Needs only the epoch-200 checkpoint; the GPU is optional (the 676 take about a minute on the CPU).
```


## `3_downsampling_branch/cluster_kmeans.py`

```text
cluster_kmeans.py -- per-class K-Means on the 580,648-value vectors. Step 3 of Setup 4.

CPU only, a few minutes, about 6-8 GB peak RAM. Safe to run while a GPU job trains.

Following Chehade et al.: "k-means was applied to both no finding and pneumonia images, resulting in two
clusters for each class". So K-Means is fitted TWICE, k=2 each time:
    run 1   the 322 pneumonia images
    run 2   the 354 No Finding images
Their Table 1 reports the four resulting cells (152 / 133 artifact, 170 / 221 clean).

Fitting per class also means the artifact share is decided WITHIN each class, so "was cleaned by CycleGAN"
cannot become a cue for the label. Fitting once on all 676 could produce class-skewed clusters.

Standardisation is per class (StandardScaler on that class's 676-subset rows), matching the per-class fit.
No PCA: the paper standardises the features and clusters them directly.

This script does NOT decide which cluster holds the artifacts. That comes from the manual check in
label_grid.py, because the automatic rules are unreliable: the friend's code used "higher Sobel energy",
while the paper's own description implies artifacts REDUCE lung sharpness, so the direction may invert.
A provisional guess is printed (lower mean lung sharpness = artifacts) and recorded, clearly marked as
provisional, so label_grid.py can confirm or overturn it.

Also computed, as a cheap diagnostic: K-Means on the FOUR SCALARS alone, and its agreement with the main
clustering. The main vector is 580,644 HOG values to 4 scalars, so after standardisation the scalars
contribute about 0.0007% of the Euclidean distance. If the two clusterings agree, the scalars and HOG see
the same structure; if not, the thesis should say the partition is driven by HOG orientation structure.

Outputs (data\2-processed\2B-input_for_cyclegan\clusters\):
    clustered_676.csv     subset_676.csv plus cluster, cluster_key, provisional_domain, scalar_cluster
    kmeans_<class>.joblib  the fitted scaler and K-Means per class (used later for images outside the 676)
    cluster_report.json    sizes, quality metrics, per-cluster scalar means, the provisional guess

Run (project root):
    python scripts\2b-cyclegan\cluster_kmeans.py
    python scripts\2b-cyclegan\cluster_kmeans.py --quality   (adds silhouette etc.; slower on 580k dims)
```


## `3_downsampling_branch/cyclegan_model.py`

```text
cyclegan_model.py -- CycleGAN (Zhu et al. 2017) for single-channel radiographs. Pure model code: no paths, no data.

Architectures, LSGAN loss, image pool and loss weights are the reusable parts of the friend's `CycleGAN v3.py`
(ResNet-9 generators, 70 x 70 PatchGAN, identity weight 5.0 = the reference implementation's 0.5 x lambda_cycle 10).

What differs from the friend's file, and why:
  * the image pool stores DETACHED copies (the friend's pool kept tensors attached to the autograd graph)
  * checkpoints also hold both LR schedulers, the pools and the sampler state, so a resumed run continues
    exactly where it stopped (the friend's `load` restarted the LR schedule from epoch 0)
  * the per-iteration losses stay on the device (no .item() sync every step); the trainer reads them once per epoch
  * no dataset, no config dataclass, no file logic here: train_cyclegan.py and clean_images.py own those

Domain A = artifact images, domain B = clean images.  G_AB: artifact -> clean (the cleaner), G_BA: clean -> artifact.
```


## `3_downsampling_branch/features.py`

```text
features.py -- Chehade's contour/texture feature vector for the 676 subset. Step 2 of Setup 4.

CPU only, no GPU, but it is the long job: HOG at 1024 x 1024 takes roughly 0.5-1.5 s per image per core.
Safe to run while a GPU job trains, but use --workers 4 then so it does not starve the trainer's dataloader.

The vector (580,648 values per image), our reading of Chehade et al. Sec. 3.4.1:
    HOG at 1024 x 1024, scikit-image defaults   127 x 127 blocks x (2 x 2 cells x 9 orientations) = 580,644
    variance of gradients                       variance of the Sobel gradient magnitude over the whole image
    lung sharpness                              mean gradient magnitude inside a lung mask
    black ratio                                 share of pixels below BLACK_THRESHOLD
    white ratio                                 share of pixels above WHITE_THRESHOLD
                                                                                        total = 580,648
580,648 is the number the paper reports, which is how we inferred that HOG was computed at full resolution
and dominates the vector. The paper does not publish the Sobel kernel, the lung-mask method or the
black/white cut-offs, so those are OUR assumptions (setup4_config.py) and the thesis must say so.

Lung mask (our assumption): the darkest 25% of the pixels inside a central window (rows 15-88%, columns
12-88%), small holes closed, thin bridges opened, then the two largest components, which are the two lung
fields. Otsu cannot be used because histogram equalisation makes it split every image at about grey 128.
If the mask collapses (< 1% of the image) the whole image is used instead and the fallback is recorded in
the manifest, so these images can be checked.

Note on scale: HOG is 580,644 of the 580,648 values, so after per-feature standardisation the four scalars
contribute about 0.0007% of the Euclidean distance K-Means works with. The clustering is therefore a
HOG clustering in practice. This follows the paper's described vector and is reported as such; the four
scalars are still saved separately so cluster_kmeans.py can report them per cluster.

Outputs (data\2-processed\2B-input_for_cyclegan\features\):
    features_676.npy       float32 memmap, 676 x 580,648 (about 1.6 GB), row order = subset_676.csv
    scalars_676.csv        the 4 scalars per image, readable, plus the mask fallback flag
    feature_manifest.csv   Image Index, class, split, row index, timing
    feature_report.json    settings, dimensions and timing, for the thesis

Run (project root):
    python scripts\2b-cyclegan\features.py --workers 4        (while the GPU is busy)
    python scripts\2b-cyclegan\features.py --workers 8        (when nothing else is running)
    python scripts\2b-cyclegan\features.py --probe            (one image: check the dimension first)
Resume-safe: rows already written are skipped unless --restart is given.
```


## `3_downsampling_branch/fid_kid.py`

```text
fid_kid.py -- did the CycleGAN make the artifact-domain images look like the clean-domain images?

Compares Inception features of three image sets (all at the 224 x 224 the classifier sees):
    R         real CLEAN-domain images (never translated; identical before and after "cleaning")
    A_before  ARTIFACT-domain images as Setup 3 would feed them (256 -> 224 only, no G_AB)
    A_after   the same artifact-domain images after G_AB (cleaned/224, what Setup 4 feeds)
and reports FID and KID of each against R, plus a NOISE FLOOR: two disjoint random halves of R against each other
(what "same domain" looks like at this sample size). A_after is only "clean-looking" if it falls to that floor.

How to read it
  * KID (unbiased, polynomial kernel) carries the interpretation: it is unbiased, so it can be ~0 or slightly negative
    for equal distributions. FID is biased upward for small samples and is printed for comparability with the literature.
  * Per repeat, A and the two R halves are random subsets of the SAME size n, so the three numbers are comparable.
  * Groups: "seen" = the 676 images the CycleGAN was trained on (all / No Finding / Pneumonia separately);
    "unseen" = the 18,058 No Finding images of the validation/test split that G_AB never saw (origin 'assigned').
    Only the unseen group says anything about how G_AB behaves on new images.
  * LIMIT: FID/KID measure whether the two feature distributions match. They cannot see whether the anatomy was
    changed (a generator can lower FID/KID by hallucinating a "clean look"). Look at the images too.
  * Inception was trained on colour photographs: use the numbers to compare before vs after, not as absolute quality.

Backbones
  fid          the standard FID Inception (pip install pytorch-fid; weights download on first use). DEFAULT.
  torchvision  torchvision's ImageNet Inception-v3 (weights download on first use), if pytorch-fid is not an option.
  random       a tiny random network, ONLY to test this script without downloads. Never for results.

Run (project root; needs the cleaned images and clustered_all.csv):
    python scripts\2b-cyclegan\fid_kid.py
    python scripts\2b-cyclegan\fid_kid.py --n_per_set 1000 --repeats 10 --backbone fid
Output: results\cyclegan_fid_kid\fid_kid_<backbone>.csv and .json (settings, group sizes, resize check)
```


## `3_downsampling_branch/label_grid.py`

```text
label_grid.py -- the manual check that decides which cluster CycleGAN cleans. Step 4 of Setup 4.

CPU only, seconds. Two modes:

    --make    (default) draws 4 contact sheets, 25 images each, one per cluster, and writes a CSV for
              your marks, plus lung-mask overlays so the mask assumption can be checked by eye
    --more    draws ANOTHER round of unseen images and appends them to the same CSV, keeping the marks
              already made. Use it when the first round left the direction uncertain; more marks per
              cluster is the only way to tell a real difference from sampling noise.
    --masks   only rewrites the lung-mask overlays (+ mask area stats); never touches your marks
    --score   re-merges each image's CURRENT cluster from clustered_676.csv (so a refit cannot leave stale
              numbers), then reads the filled-in CSV and reports the agreement %, a 95% CI and a Fisher exact test of
              whether the two clusters differ at all, then FIXES the artifact cluster

Why this step exists
  K-Means gives two clusters per class but no names. CycleGAN must be pointed at the artifact cluster, and
  getting the direction wrong would train it to ADD artifacts. cluster_kmeans.py made a provisional guess
  from mean lung sharpness, but the friend's code used the opposite rule, so the direction is decided here
  by your eyes instead. Chehade did the same thing: they labelled all 676 manually and reported 83.54%
  (pneumonia) and 82.49% (No Finding) agreement with the clustering.

What counts as an artifact (Chehade Fig. 1 and Sec. 3.4): pacemakers, implants, ECG leads, wires, catheters,
surgical clips, burned-in text or position markers (L, R, PORTABLE, AP), and clearly bad captures - heavy
blur, extreme over- or under-exposure, large cut-off regions. A normal sharp radiograph with no hardware is
clean. If genuinely unsure, leave the row blank: blanks are excluded rather than guessed.

Your labels are used ONLY to (1) name the clusters and (2) report an agreement %. They never enter training,
so the pipeline stays unsupervised, exactly as in the paper.

Outputs (data\2-processed\2B-input_for_cyclegan\label_grids\):
    grid_<class>_cluster<k>.png   25 numbered thumbnails per cluster
    manual_labels.csv             the file you fill in: write a or c in the 'manual' column
    masks\mask_<image>.png        20 lung-mask overlays (red = mask) to sanity-check the assumption
    label_report.json             (after --score) agreement per class and the FINAL artifact cluster

Run (project root):
    python scripts\2b-cyclegan\label_grid.py                  # make the sheets
    ... fill in manual_labels.csv in Excel: a = artifact, c = clean ...
    python scripts\2b-cyclegan\label_grid.py --score          # agreement + fix the direction
```


## `3_downsampling_branch/make_flag_list.py`

```text
make_flag_list.py -- list the images whose CycleGAN translation FAILED, and write the test list without them.

Rule (fixed before any Setup 4 result was seen): an artifact-domain image is "failed" if the mean absolute difference
between its before and after versions, measured in the CENTRE of the image (rows/cols 15 %-85 %), is above 20 grey levels
(translation_change.py, column mae_centre). Clean-domain images are never translated, so they are never flagged.
The rule uses only the images, never labels or classifier results.

Reads   <cleaned folder>\translation_change.csv   (written by translation_change.py)
Writes  <labels folder>\flagged_failed.csv        every flagged image (split, class, mae_centre, ssim_centre)
        <labels folder>\test_not_failed.csv       testing_labels.csv without the flagged images, same columns
Use test_not_failed.csv with  evaluate.py --subset  for the sensitivity result. The main results keep all images.

Run (project root):
    python scripts\2b-cyclegan\make_flag_list.py
```


## `3_downsampling_branch/make_subset.py`

```text
make_subset.py -- builds the 676-image Setup 4 subset from the fixed patient-level split. Step 1 of Setup 4.

CPU only, a few seconds, writes only CSVs. Safe to run while a GPU job is training.

What it does
  * keeps EVERY pneumonia image of each split (232 / 31 / 59 = 322, Chehade's total)
  * draws No Finding images per split to reach the 354 : 322 ratio (255 / 34 / 65 = 354)
  * draws them from THAT SPLIT'S OWN patients, so the patient-level split is preserved
  * draws at most one image per patient, so no patient contributes several near-identical radiographs
  * writes the shared 676 CSV, the balanced 124-image test list and the 65-image small validation list
  * prints a patient summary and the checks the later scripts rely on

Outputs (all under data\2-processed\2B-input_for_cyclegan\labels\):
    subset_676.csv              the 676 images: Image Index, Finding Labels, Patient ID, clinical values,
                                split, src_path
    test_balanced_124.csv       the 59 + 65 test images (secondary table, comparable to Chehade)
    validation_small_65.csv     the 31 + 34 validation images (run B)

Run (project root):
    python scripts\2b-cyclegan\make_subset.py
    python scripts\2b-cyclegan\make_subset.py --seed 7 --dry_run

The draw is reproducible: the same --seed always selects the same images.
```


## `3_downsampling_branch/setup4_config.py`

```text
setup4_config.py -- every path, count and constant the Setup 4 (CycleGAN downsampling) scripts share.

Defined once here so the subset builder, the feature extractor, K-Means, the label grid, CycleGAN and the
cleaning script cannot drift apart. Import it; do not copy values out of it.

Run every script from the project root (thesis_project).

Protocol decisions recorded here (see HANDOFF_SETUP4.md for the reasoning):
  * class ratio 354 : 322, applied per split (Chehade et al. 2025)
  * our fixed PATIENT-LEVEL split is reused; Chehade's own split was image-level and is not reproduced
  * features are computed on the 1024 x 1024 equalised images (HOG dominates the 580,648-value vector)
  * K-Means is fitted PER CLASS (k=2 on the 322 pneumonia, k=2 on the 354 No Finding), following
    "k-means was applied to both no finding and pneumonia images, resulting in two clusters for each class"
  * resize chain 1024 -> 256 -> 224 for EVERY image, exactly as Setup 3: 1024 -> 256 is the LANCZOS step already
    stored in 2A-input_for_stylegan2ada (reused, not redone), 256 -> 224 is torchvision Resize((224, 224)).
    CycleGAN trains and translates at 256; the generator output is rounded to 8 bit, then takes the same
    256 -> 224 step as an untranslated image. Artifact and clean images therefore differ ONLY by whether G_AB ran.
  * CycleGAN augmentation: horizontal flip only, on the full 256 x 256 image (no crop, no resize jitter)
  * everything (features, K-Means, CycleGAN) is fitted on all 676 images, test images included, following
    Chehade; this is a known preprocessing leak and is reported as a limitation

DEVIATION LOG (for the thesis; each line is a choice of ours, not of Chehade et al.):
  * 2026-10-14  CycleGAN works at 256 x 256 and the classifier input is 224 (earlier plan: 224 direct). Reason: the
                chain must equal Setup 3's, so that the data method is the only difference between Setups 3 and 4.
  * 2026-10-14  CycleGAN images are held in memory and sampled by one seeded generator (no DataLoader workers):
                676 grayscale images are 44 MB, so workers add nothing and cannot repeat each other's samples.             
  *2026-10-14   The CycleGAN checkpoint at epoch 100 is used, not the final epoch 200. Chosen only on image criteria: 
                anatomy change (centre MAE, SSIM) and visual checks. Epoch 200 changed the lungs strongly. Classifier results were not used.
  *2026-10-14   Translation failures: images whose centre change (mean absolute difference, centre 70% of the image, after vs before G_AB)
                exceeds 20 grey levels are flagged (120 of 9,182 artifact-domain images). All images stay in the main results; a sensitivity result
                without them is reported. The rule was fixed before any Setup 4 result was seen
```


## `3_downsampling_branch/setup4_images.py`

```text
setup4_images.py -- the ONE place where Setup 4 reads a 256 image and takes the 256 -> 224 step.

train_cyclegan.py and clean_images.py both import this, so artifact images, clean images and the Setup 3 pipeline
cannot end up on different resize paths. A different path would let the classifier tell the two domains apart by
resampling blur alone, which is a class cue unrelated to the artifacts.

Chain (identical for every image):  1024 --LANCZOS--> 256 (already stored by Setup 3) --[G_AB if artifact]-->
8-bit --torchvision Resize((224, 224))--> 224 PNG.
```


## `3_downsampling_branch/train_cyclegan.py`

```text
train_cyclegan.py -- trains ONE CycleGAN on the pooled artifact domain (A) and clean domain (B) of the 676 images.

Reads   clusters\clustered_676.csv  (needs the final `domain` column; run label_grid.py --score first)
        Setup 3's 256 x 256 images (2A-input_for_stylegan2ada\{training,validation,testing}), reused as they are
Writes  cyclegan\cyclegan_latest.pt        every epoch (atomic overwrite)
        cyclegan\cyclegan_epoch025.pt ...  every 25th epoch and the last one (the epoch-200 file is the one used)
        cyclegan\training_history.csv, run_config.json, samples\epoch_XXX.png

Recipe (Zhu et al. defaults): 200 epochs, lr 2e-4 constant for 100 then linear decay, Adam(0.5, 0.999), batch 1,
LSGAN, lambda_cycle 10, lambda_identity 5, image pool 50, ResNet-9 generators, 70 x 70 PatchGAN.
Augmentation: horizontal flip only, on the whole 256 x 256 image. One epoch = max(|A|, |B|) iterations; A is walked in a
shuffled order, B is drawn at random for each step (the reference recipe's unaligned sampling).
All 676 images are used, test images included, following Chehade et al. (stated limitation).

The 676 images live in memory (44 MB), so there are no DataLoader workers: a single seeded generator makes every
choice, which makes the run reproducible and removes the repeated-sample problem of the friend's per-worker RNG.

Run (project root, GPU free):
    venv\Scripts\python.exe scripts\2b-cyclegan\train_cyclegan.py --smoke        (8 + 8 images, 2 epochs, own folder)
    venv\Scripts\python.exe scripts\2b-cyclegan\train_cyclegan.py                (the real run)
    venv\Scripts\python.exe scripts\2b-cyclegan\train_cyclegan.py --resume auto   (continue after an interruption)
```


## `3_downsampling_branch/translation_change.py`

```text
translation_change.py -- how much did G_AB change each artifact-domain image, and WHERE?

For every artifact-domain image in one or more cleaned folders (output of clean_images.py --out <folder>) it compares
    before  Setup 3's 224 image (the 256 image -> torchvision Resize((224, 224)), no G_AB)
    after   <folder>\224\<Image Index>   (what Setup 4 would feed the classifier)
and measures, in grey levels (0-255) unless stated:
    mae_all      mean absolute difference over the whole image
    mean_shift   signed mean difference (after - before): a global brightness shift, not a structural change
    mae_centre   mean absolute difference in the CENTRE (rows/cols 15 %-85 %): lungs, heart, spine = the anatomy
    mae_border   mean absolute difference in the outer 15 % frame: where markers, tubes and edges usually sit
    ssim_centre  structural similarity (1 = identical structure) of the centre
Two readings matter:
  * mae_centre should be SMALL: the cleaning should not touch the anatomy. mae_border may be larger: that is where
    the artifacts are. (The centre/border split is a crude stand-in for a lung mask; markers can sit inside the centre.)
  * pneumonia images must not be changed MORE than No Finding images. A Mann-Whitney test on mae_centre is printed;
    a small p means the generator treats the classes differently, which would be a new shortcut.
Thresholds (10 and 20 grey levels in the centre) are rules of thumb, not standards: use them to compare candidates.

Run (project root), several candidates at once:
    python scripts\2b-cyclegan\translation_change.py --cleaned_dir data\2-processed\2B-input_for_cyclegan\cand_e075 ^
        data\2-processed\2B-input_for_cyclegan\cand_e100
Output per folder: <folder>\translation_change.csv, most_changed.png, pneumonia_most_changed.png (before | after | |difference| x4);
and results\cyclegan_translation_change\summary.csv for the comparison table.
Needs: pip install scikit-image scipy
```


## `4_image_transformation/make_stats.py`

```text
make_stats.py -- pixel mean / std of a TRAINING set, as used for T.Normalize in the classifier.

Pipeline per image = exactly cxr_data.make_transform(train=False): grayscale -> Resize((224,224)) -> ToTensor [0,1].
Mean and std are pooled over every pixel of every image (population std, float64 sums).

Two uses
  1) CHECK on Setup 3 (does this script reproduce the old file? expected mean 0.4929, std 0.2934, n 84408):
       python scripts\2b-cyclegan\make_stats.py --csv data\2-processed\2A-input_for_stylegan2ada\labels\combined_training_labels.csv ^
              --out results\stats_check_setup3.json --compare data\2-processed\train_stats.json
     (the CSV has a 'path' column; the old file is only READ, never written)
  2) Setup 4 (487 cleaned training images):
       python scripts\2b-cyclegan\make_stats.py --csv data\2-processed\2B-input_for_cyclegan\cleaned_manifest.csv ^
              --split training --out data\2-processed\2B-input_for_cyclegan\train_stats.json
```


## `5_training_and_classification/s1_densenet121_stylegan2-ada/bo_search.py`

```text
bo_search.py -- Bayesian Optimization of the Adam learning rate and the decay schedule (one search per setup).

Manuscript: Optuna Bayesian Optimization supplies Adam's initial learning rate and decay scheduling, chosen on the
VALIDATION set only. This script tunes exactly two values:
    lr            1e-5 .. 3e-4   (log scale)      initial Adam learning rate
    min_lr_ratio  0 .. 0.5                         cosine-decay floor, as a fraction of lr
Everything else is fixed (batch 16, weight decay 1e-4, head, 3 warmup epochs) and reported as fixed.

Search design
  sampler     Optuna TPE (a Bayesian-style sampler); the first 5 trials are random "startup" trials, the other 15
              are guided by what the earlier trials showed
  trials      20, each a real training run of up to 15 epochs with patience 5 (train.py, --trial N)
  objective   mean of the 3 highest validation AUCs of the trial (validation has only 31 pneumonia cases, so a
              single best epoch is too noisy); a diverged trial (NaN loss) scores 0.5 so the search keeps going
  seed        100 -- deliberately NOT one of the final seeds 42-46, so no final run is favoured by the tuning
  test set    never opened during the search
  resume      the study lives in an SQLite file; rerun the same command to continue after an interruption

If more than 5 of the 20 trials diverge, lower --lr_max to 1e-4 and rerun (the script prints this advice).
Note: the best validation AUC found by a search is optimistic (it is the best of many noisy scores). Only the
5-seed test results of the final runs are reported as performance.

Run (project root):
    python scripts\3a-densenet121_stylegan2-ada\bo_search.py --cbam all
Output: results\densenet121_stylegan2-ada\search\best_params_cbam-all.json, trials_cbam-all.csv, optuna_cbam-all.db
```


## `5_training_and_classification/s1_densenet121_stylegan2-ada/cxr_data.py`

```text
cxr_data.py -- shared paths, run-folder naming, augmentation and the Dataset for the DenseNet-121 setup.

Everything that must be identical between train.py, bo_search.py, run_experiments.py and the other scripts
lives here, so a path or a preprocessing rule is defined exactly once.

Run every script from the project root (thesis_project).
```


## `5_training_and_classification/s1_densenet121_stylegan2-ada/densenet_cbam.py`

```text
densenet_cbam.py -- DenseNet-121 + CBAM + clinical variables (Setup 1 classifier).

Architecture (manuscript, Developmental Method):
  ImageNet-pretrained DenseNet-121 -> CBAM after EACH of the four dense blocks -> global average pooling
  -> concatenate the 3 clinical values (sex, view, age) -> ONE fully connected unit -> sigmoid (applied in the loss).
  No dropout, no extra clinical layer: the head is "GAP then one FC", as the manuscript and Chehade describe.

cbam = "all"   CBAM after all four dense blocks  (thesis method, Setup 1)
cbam = "last"  CBAM only after the last block     (Chehade's placement; exploratory check only)
cbam = "none"  plain DenseNet-121                 (exploratory check only)

CBAM initialisation (manuscript: "small random weights"):
  every CBAM convolution ~ N(0, 0.01^2). The two gates then start near sigmoid(0) = 0.5, so a plain CBAM would
  shrink the pretrained features to about 0.5 x 0.5 = 0.25 at the start. The module output is therefore multiplied
  by a fixed factor of 4, which makes each CBAM start as an approximate identity (0.25 x 4 = 1) and keeps the
  pretrained features intact. Training then learns where to emphasise. The attention maths runs in float32.

Running this file prints the parameter counts (useful for the CBAM-cost sentence in the manuscript):
    python scripts\3a-densenet121_stylegan2-ada\densenet_cbam.py
```


## `5_training_and_classification/s1_densenet121_stylegan2-ada/run_all.py`

```text
run_all.py -- runs the whole Setup 1 pipeline unattended, one step after the other:
    1. bo_search.py        Bayesian Optimization (lr + decay floor) on the validation set
    2. run_experiments.py  five final seeds (42-46) with the tuned values, test set scored once (--eval_test)
    3. shortcut_check.py   real vs synthetic pneumonia recall of every final run
The chain stops at the first step that fails, so the final runs never start from a broken search.
Every step is resume-safe: if the PC restarts, run the same command again and finished work is skipped.

Run (project root):
    python scripts\3a-densenet121_stylegan2-ada\run_all.py
Extra options go to bo_search.py, e.g. a lower learning-rate ceiling:
    python scripts\3a-densenet121_stylegan2-ada\run_all.py --lr_max 1e-4
```


## `5_training_and_classification/s1_densenet121_stylegan2-ada/run_experiments.py`

```text
run_experiments.py -- final runs of Setup 1 with the TUNED hyper-parameters: five seeds, one after another.

Reads lr and min_lr_ratio from the search result (best_params_cbam-<cbam>.json written by bo_search.py), then runs
train.py for seeds 42-46 with the same values, so the five runs differ only by their random seed (this is what the
across-seed ANOVA needs). Finished runs are skipped on restart; a crashed run does not stop the rest.
Add --eval_test to score the test set (one scoring per run; predictions are saved for evaluate.py).

Run (project root):
    python scripts\3a-densenet121_stylegan2-ada\run_experiments.py --cbam all --eval_test
    python scripts\3a-densenet121_stylegan2-ada\run_experiments.py --summarize_only

Exploratory check (not in the thesis): --cbam last or --cbam none. They use their own search file if it exists;
to reuse the values tuned on CBAM all, add:  --params results\densenet121_stylegan2-ada\search\best_params_cbam-all.json
```


## `5_training_and_classification/s1_densenet121_stylegan2-ada/shortcut_check.py`

```text
shortcut_check.py -- is the model recognising REAL pneumonia, or only GAN-generated pneumonia?

Reads every history_*.csv and metrics_*.json of the FINAL runs and reports, per run, the training recall
(threshold 0.5) on real vs synthetic pneumonia at the best-validation epoch and at the last epoch.

How to read it:
  real recall near 0 and synthetic recall near 1  -> shortcut ("looks generated" = pneumonia)
  both high                                        -> the model learned something about pneumonia
(Real recall near 1 can also mean memorisation of the 232 real images; compare with the validation AUC.)

    python scripts\3a-densenet121_stylegan2-ada\shortcut_check.py            (add --phase search for the trials)
Writes <results>\final\shortcut_check.csv
```


## `5_training_and_classification/s1_densenet121_stylegan2-ada/train.py`

```text
train.py -- trains ONE run of Setup 1: StyleGAN2-ADA combined training set + DenseNet-121 + CBAM.

Protocol (identical for every setup so the comparison is fair):
  optimizer  Adam, L2 weight decay 1e-4, ONE learning rate for all parameters
  schedule   3 warmup epochs (lr rises linearly from ~0), then cosine decay down to min_lr_ratio x lr
             (lr and min_lr_ratio are the two values tuned by Bayesian Optimization -- manuscript:
              "initial learning rate and decay scheduling")
  batch      16, up to 15 epochs, early stopping on validation AUC with patience 5
  loss       binary cross-entropy weighted by class frequency (pos_weight = n_neg / n_pos; = 1.0 on the
             balanced combined training set)
  epoch      84,384 samples drawn without replacement = one pass of the combined set = 5,274 batches of 16
  model      DenseNet-121 + CBAM + clinical values (densenet_cbam.py)
  no gradient clipping (not in the manuscript or in Chehade et al.)

A run is either
  * a SEARCH TRIAL  (--trial N, called by bo_search.py): validation only, the test set is never opened;
  * a FINAL RUN     (default; called by run_experiments.py): add --eval_test to score the test set once.

Training-set shortcut check: every epoch logs the training recall on REAL vs SYNTHETIC pneumonia. Real recall near 0
with synthetic recall near 1 means the network learned "looks generated" instead of "has pneumonia".

Examples (project root):
    python scripts\3a-densenet121_stylegan2-ada\train.py --cbam all --lr 1e-4 --min_lr_ratio 0.1 --seed 42
    python scripts\3a-densenet121_stylegan2-ada\train.py --cbam all --params results\densenet121_stylegan2-ada\search\best_params_cbam-all.json --seed 42 --eval_test
```


## `5_training_and_classification/s3_convnextv2_stylegan2-ada/bo_search.py`

```text
bo_search.py -- Bayesian Optimization of the Adam learning rate and the decay schedule (one search per setup).

Manuscript: Optuna Bayesian Optimization supplies Adam's initial learning rate and decay scheduling, chosen on the
VALIDATION set only. This script tunes exactly two values:
    lr            1e-5 .. 3e-4   (log scale)      initial Adam learning rate
    min_lr_ratio  0 .. 0.5                         cosine-decay floor, as a fraction of lr
Everything else is fixed (batch 16, weight decay 1e-4, head, 3 warmup epochs) and reported as fixed.

Search design
  sampler     Optuna TPE (a Bayesian-style sampler); the first 5 trials are random "startup" trials, the other 15
              are guided by what the earlier trials showed
  trials      20, each a real training run of up to 15 epochs with patience 5 (train.py, --trial N)
  objective   mean of the 3 highest validation AUCs of the trial (validation has only 31 pneumonia cases, so a
              single best epoch is too noisy); a diverged trial (NaN loss) scores 0.5 so the search keeps going
  seed        100 -- deliberately NOT one of the final seeds 42-46, so no final run is favoured by the tuning
  test set    never opened during the search
  resume      the study lives in an SQLite file; rerun the same command to continue after an interruption

If more than 5 of the 20 trials diverge, lower --lr_max to 1e-4 and rerun (the script prints this advice).
Note: the best validation AUC found by a search is optimistic (it is the best of many noisy scores). Only the
5-seed test results of the final runs are reported as performance.

Run (project root):
    python scripts\3b-convnextv2_stylegan2-ada\bo_search.py --cbam all
Output: results\convnextv2_stylegan2-ada\search\best_params_cbam-all.json, trials_cbam-all.csv, optuna_cbam-all.db
```


## `5_training_and_classification/s3_convnextv2_stylegan2-ada/convnext_cbam.py`

```text
convnext_cbam.py -- ConvNeXt V2-Tiny + CBAM + clinical variables (Setup 3 classifier; Setup 4 uses the same model).

Architecture (manuscript, Developmental Method; same recipe as DenseNet-121 in 3a so the two are comparable):
  ImageNet-1k-pretrained ConvNeXt V2-Tiny (timm 'convnextv2_tiny.fcmae_ft_in1k': FCMAE self-supervised
  pre-training, then supervised fine-tuning on ImageNet-1k)
  -> CBAM after EACH of the four stages (ConvNeXt's counterpart of DenseNet's four dense blocks)
  -> global average pooling -> concatenate the 3 clinical values (sex, view, age)
  -> ONE fully connected unit -> sigmoid (applied in the loss).
  No dropout and no extra clinical layer in the head.

cbam = "all"   CBAM after all four stages  (thesis method)
cbam = "last"  CBAM only after the last stage (exploratory check only)
cbam = "none"  plain ConvNeXt V2-Tiny       (exploratory check only)

CBAM initialisation is identical to the DenseNet model: every CBAM convolution ~ N(0, 0.01^2) ("small random
weights"), and the module output is multiplied by 4 so each CBAM starts as an approximate identity
(gates ~0.5 x 0.5 = 0.25, times 4 = 1). The attention maths runs in float32.

Fixed backbone setting (reported in Methods): stochastic depth (drop-path) rate 0.2, the usual fine-tuning value for
ConvNeXt; it is not tuned. It is ConvNeXt's own regulariser inside the blocks, not a dropout layer in the head.

Needs:  pip install timm      (weights are downloaded from Hugging Face on first use, so internet is required once)

Running this file prints the parameter counts (for the CBAM-cost sentence in the manuscript) and the detected
stage channels, which is also a quick check that the installed timm version is handled:
    python scripts\3b-convnextv2_stylegan2-ada\convnext_cbam.py
```


## `5_training_and_classification/s3_convnextv2_stylegan2-ada/cxr_data.py`

```text
cxr_data.py -- shared paths, run-folder naming, augmentation and the Dataset for the ConvNeXt V2 setup.

Everything that must be identical between train.py, bo_search.py, run_experiments.py and the other scripts
lives here, so a path or a preprocessing rule is defined exactly once.

Run every script from the project root (thesis_project).
```


## `5_training_and_classification/s3_convnextv2_stylegan2-ada/run_all.py`

```text
run_all.py -- runs the whole Setup 3 pipeline unattended, one step after the other:
    1. bo_search.py        Bayesian Optimization (lr + decay floor) on the validation set
    2. run_experiments.py  five final seeds (42-46) with the tuned values, test set scored once (--eval_test)
    3. shortcut_check.py   real vs synthetic pneumonia recall of every final run
The chain stops at the first step that fails, so the final runs never start from a broken search.
Every step is resume-safe: if the PC restarts, run the same command again and finished work is skipped.

Run (project root):
    python scripts\3b-convnextv2_stylegan2-ada\run_all.py
Extra options go to bo_search.py, e.g. a lower learning-rate ceiling:
    python scripts\3b-convnextv2_stylegan2-ada\run_all.py --lr_max 1e-4
```


## `5_training_and_classification/s3_convnextv2_stylegan2-ada/run_experiments.py`

```text
run_experiments.py -- final runs of Setup 3 with the TUNED hyper-parameters: five seeds, one after another.

Reads lr and min_lr_ratio from the search result (best_params_cbam-<cbam>.json written by bo_search.py), then runs
train.py for seeds 42-46 with the same values, so the five runs differ only by their random seed (this is what the
across-seed ANOVA needs). Finished runs are skipped on restart; a crashed run does not stop the rest.
Add --eval_test to score the test set (one scoring per run; predictions are saved for evaluate.py).

Run (project root):
    python scripts\3b-convnextv2_stylegan2-ada\run_experiments.py --cbam all --eval_test
    python scripts\3b-convnextv2_stylegan2-ada\run_experiments.py --summarize_only

Exploratory check (not in the thesis): --cbam last or --cbam none. They use their own search file if it exists;
to reuse the values tuned on CBAM all, add:  --params results\convnextv2_stylegan2-ada\search\best_params_cbam-all.json
```


## `5_training_and_classification/s3_convnextv2_stylegan2-ada/shortcut_check.py`

```text
shortcut_check.py -- is the model recognising REAL pneumonia, or only GAN-generated pneumonia?

Reads every history_*.csv and metrics_*.json of the FINAL runs and reports, per run, the training recall
(threshold 0.5) on real vs synthetic pneumonia at the best-validation epoch and at the last epoch.

How to read it:
  real recall near 0 and synthetic recall near 1  -> shortcut ("looks generated" = pneumonia)
  both high                                        -> the model learned something about pneumonia
(Real recall near 1 can also mean memorisation of the 232 real images; compare with the validation AUC.)

    python scripts\3b-convnextv2_stylegan2-ada\shortcut_check.py            (add --phase search for the trials)
Writes <results>\final\shortcut_check.csv
```


## `5_training_and_classification/s3_convnextv2_stylegan2-ada/train.py`

```text
train.py -- trains ONE run of Setup 3: StyleGAN2-ADA combined training set + ConvNeXt V2-Tiny + CBAM.

Protocol (identical for every setup so the comparison is fair):
  optimizer  Adam, L2 weight decay 1e-4, ONE learning rate for all parameters
  schedule   3 warmup epochs (lr rises linearly from ~0), then cosine decay down to min_lr_ratio x lr
             (lr and min_lr_ratio are the two values tuned by Bayesian Optimization -- manuscript:
              "initial learning rate and decay scheduling")
  batch      16, up to 15 epochs, early stopping on validation AUC with patience 5
  loss       binary cross-entropy weighted by class frequency (pos_weight = n_neg / n_pos; = 1.0 on the
             balanced combined training set)
  epoch      84,384 samples drawn without replacement = one pass of the combined set = 5,274 batches of 16
  model      ConvNeXt V2-Tiny + CBAM + clinical values (convnext_cbam.py)
  no gradient clipping (not in the manuscript or in Chehade et al.)

A run is either
  * a SEARCH TRIAL  (--trial N, called by bo_search.py): validation only, the test set is never opened;
  * a FINAL RUN     (default; called by run_experiments.py): add --eval_test to score the test set once.

Training-set shortcut check: every epoch logs the training recall on REAL vs SYNTHETIC pneumonia. Real recall near 0
with synthetic recall near 1 means the network learned "looks generated" instead of "has pneumonia".

Examples (project root):
    python scripts\3b-convnextv2_stylegan2-ada\train.py --cbam all --lr 1e-4 --min_lr_ratio 0.1 --seed 42
    python scripts\3b-convnextv2_stylegan2-ada\train.py --cbam all --params results\convnextv2_stylegan2-ada\search\best_params_cbam-all.json --seed 42 --eval_test
```


## `5_training_and_classification/s4_convnextv2_cyclegan/bo_search.py`

```text
bo_search.py -- Bayesian Optimization of the Adam learning rate and the decay floor for Setup 4 (one search per run).

Identical to Setup 3's search except for the data: each trial is a real training run of train.py on the 487 cleaned
images and is scored on THIS run's validation set (run B: 65 images, run A: 5,985 images). The test set is never opened.
    lr            1e-5 .. 3e-4   (log)    min_lr_ratio  0 .. 0.5
    sampler       Optuna TPE, 5 random startup trials, 20 trials, seed 100 (not a final seed)
    trial         up to 50 epochs, patience 5 (train.py defaults)
    objective     mean of the 3 highest validation AUCs of the trial (a diverged trial scores 0.5)
    resume        the study lives in an SQLite file; rerun the same command to continue
Run B has only 31 + 34 validation images, so its objective is noisy; that is a property of the design, not a bug.

Run (project root):
    python scripts\3c-convnextv2_cyclegan\bo_search.py --run B
Output: results\convnextv2_cyclegan\run_B\search\best_params_cbam-all.json, trials_cbam-all.csv, optuna_cbam-all.db
```


## `5_training_and_classification/s4_convnextv2_cyclegan/cxr4_data.py`

```text
cxr4_data.py -- Setup 4 classifier: where the data is, how the three tables are built, and a self-check.

Setup 4 = the Setup 3 classifier (ConvNeXt V2-Tiny + CBAM, same recipe) trained on the 487 CYCLEGAN-CLEANED training
images instead of the StyleGAN2-ADA set. Everything that must be identical between train.py, bo_search.py,
run_experiments.py and run_all.py lives here, so a path or a split rule is defined exactly once.

It reuses, unchanged, from Setup 3 (scripts\3b-convnextv2_stylegan2-ada): CXRDataset, ChehadeAugment, run_paths and the
model (convnext_cbam.py). Paths and counts come from scripts\2b-cyclegan\setup4_config.py.
Do NOT add a file named cxr_data.py or convnext_cbam.py to this folder: it would shadow the 3b modules.

Tables
  train       487 cleaned images (232 pneumonia + 255 No Finding), from training_labels.csv
  validation  run B: the 65-image list validation_small_65.csv (31 + 34)      -> Chehade-style small validation
              run A: all 5,985 validation images
  test        always all 12,262 images, in EXACTLY the row order of testing_labels.csv (evaluate.py pairs the
              predictions of the three setups on 'Image Index')
Clinical values (sex, view, age) and Patient ID come from the official split CSVs; the pixels come from the cleaned
224 x 224 PNGs written by clean_images.py. 'is_synthetic' is re-used to carry the ARTIFACT-DOMAIN flag of the
training images (1 = the image went through G_AB), so train.py can log recall per domain.

Self-check (project root):
    python scripts\3c-convnextv2_cyclegan\cxr4_data.py --run B
    python scripts\3c-convnextv2_cyclegan\cxr4_data.py --run A
```


## `5_training_and_classification/s4_convnextv2_cyclegan/run_all.py`

```text
run_all.py -- runs the whole Setup 4 classifier pipeline unattended, one step after the other, for run B, run A or both:
    1. bo_search.py        Bayesian Optimization (lr + decay floor) on THIS run's validation set
    2. run_experiments.py  five final seeds (42-46) with the tuned values, test set scored once (--eval_test)
The chain stops at the first step that fails, so the final runs never start from a broken search. Every step is
resume-safe: if the PC restarts, run the same command again and finished work is skipped.
Run it in a standalone PowerShell window (not inside VS Code).

Run (project root), B first, then A:
    python scripts\3c-convnextv2_cyclegan\run_all.py --run B
    python scripts\3c-convnextv2_cyclegan\run_all.py --run A
    (or  --run both  = B then A in one go)
Extra options go to bo_search.py, e.g.  --lr_max 1e-4
Results: results\convnextv2_cyclegan\run_B\   and   run_A\
```


## `5_training_and_classification/s4_convnextv2_cyclegan/run_experiments.py`

```text
run_experiments.py -- final runs of Setup 4 with the TUNED hyper-parameters: five seeds, one after another.

Reads lr and min_lr_ratio from this run's search result (run_<A|B>\search\best_params_cbam-all.json), then runs train.py
for seeds 42-46 with the same values, so the five runs differ only by their random seed. Finished runs are skipped on
restart; a crashed run does not stop the rest. --eval_test scores the full 12,262-image test set (once per run).

Run (project root):
    python scripts\3c-convnextv2_cyclegan\run_experiments.py --run B --eval_test
    python scripts\3c-convnextv2_cyclegan\run_experiments.py --run B --summarize_only
```


## `5_training_and_classification/s4_convnextv2_cyclegan/train.py`

```text
train.py -- trains ONE run of Setup 4: CycleGAN-cleaned images + ConvNeXt V2-Tiny + CBAM.

Same protocol as Setup 3 (scripts\3b-convnextv2_stylegan2-ada\train.py), so the data method is the only thing that
differs. The differences, all forced by the 487-image training set, are written down here and saved in metrics_*.json:
  training set  487 cleaned images (232 pneumonia + 255 No Finding); one epoch = one pass over them
                = 31 batches of 16 (the last batch has 7 images; ConvNeXt has no batch-norm, so that is harmless)
  epochs        cap 50 (Setup 3: 15), warmup 10 epochs (Setup 3: 3 of 15 = 20 %, here 10 of 50 = 20 %), patience 5
  loss          BCE with pos_weight = n_neg / n_pos = 255 / 232 = 1.099 (computed from the table, not typed in)
  normalise     Setup 4's OWN mean/std (make_stats.py on the 487 cleaned training images)
  validation    run B: the 65-image list (31 + 34)      run A: all 5,985 validation images
  test          always all 12,262 images, row order = testing_labels.csv, scored once per final run (--eval_test)
Unchanged: Adam, weight decay 1e-4, one learning rate, batch 16, linear warmup then cosine to min_lr_ratio x lr
(stepped per batch, defined over the epoch cap), bf16, Chehade augmentation on training images only, no gradient
clipping, Youden threshold chosen on the validation set, the model (convnext_cbam.py).

Domain check: every epoch logs training recall / accuracy separately for artifact-domain images (translated by
G_AB) and clean-domain images. If one domain is much worse than the other, the cleaning did not make them equal.

A run is a SEARCH TRIAL (--trial N, called by bo_search.py; never opens the test set) or a FINAL RUN (default).

Examples (project root):
    python scripts\3c-convnextv2_cyclegan\train.py --run B --lr 1e-4 --min_lr_ratio 0.1 --seed 42
    python scripts\3c-convnextv2_cyclegan\train.py --run B --params results\convnextv2_cyclegan\run_B\search\best_params_cbam-all.json --seed 42 --eval_test
--smoke = CPU test on synthetic data (no pretrained weights, counts not asserted). Never use it for results.
```


## `6_testing_and_validation/evaluate.py`

```text
evaluate.py -- statistics on the SAVED test predictions. No training, no GPU, takes minutes.
(Extended version: drop-in for the old one. Without --pair, --subset the numbers are identical to the old script.)

For every setup (a folder holding seed*/test_predictions_*.csv from run_experiments.py --eval_test) it reports:
  1. per-seed test metrics                      (AUC, accuracy, precision, recall, specificity, F1) + confusion counts
  2. the 5-seed ENSEMBLE (mean probability) with a STRATIFIED bootstrap: 1,000 resamples, pneumonia and
     No Finding images drawn separately so every resample keeps the real class counts (59 / 12,203).
     CI = estimate +/- 1.96 x SD of the resamples (manuscript). Every resample's metrics are saved.
     Confusion counts (tp, fn, fp, tn) for every seed and for the ensemble -> confusion_matrices.csv
  3. DeLong test for AUC between setups (ensemble predictions), Holm-corrected p-values
  4. ANOVA, both versions the manuscript reports:
       (a) across the 5 SEEDS of each setup  (+ Tukey post-hoc when 3 or more setups) -- training randomness
       (b) across the 1,000 BOOTSTRAP resamples, with eta-squared -- test-set sampling noise.
           Resamples overlap heavily, so p-values here are tiny for small differences: read eta-squared and the
           CIs, not just p.
  5. Chehade et al. (2025) printed beside the results as a DESCRIPTIVE reference only (they published no
     per-image predictions, so neither DeLong nor ANOVA can include them).

NEW
  --pair A=B   (repeatable) test ONLY the declared comparisons. DeLong, the seed ANOVA and the bootstrap ANOVA are run
               for these pairs only, and Holm covers exactly these pairs (per metric for the ANOVAs). The difference
               reported is the FIRST minus the SECOND setup. Without --pair every pair of setups is tested, as before.
               Declared family for this thesis:  --pair S1=S3  --pair S3=S4A  --pair S3=S4B
  --subset CSV only the test images listed in the CSV's `Image Index` column are scored (e.g. test_balanced_124.csv,
               59 pneumonia + 65 No Finding). The thresholds still come from the VALIDATION predictions. Every listed
               image must be present in every setup, otherwise the script stops.

Threshold: Youden's J on VALIDATION predictions only (the ensemble uses the mean of the seeds' validation
probabilities; if val_predictions_*.csv are missing, the mean of the seeds' thresholds is used and a note is printed).
All setups are aligned by Image Index; the script stops if their test cases or labels differ (DeLong would be invalid).
The prediction columns may be named  Image Index|image_id,  label,  prob_pneumonia|prob.

Run (project root), full test set, declared comparisons:
    venv\Scripts\python.exe scripts\3c-convnextv2_cyclegan\evaluate.py ^
        --setup S1=results\densenet121_stylegan2-ada\final\cbam_all ^
        --setup S3=results\convnextv2_stylegan2-ada\final\cbam_all ^
        --setup S4A=results\convnextv2_cyclegan\run_A\final\cbam_all ^
        --setup S4B=results\convnextv2_cyclegan\run_B\final\cbam_all ^
        --pair S1=S3 --pair S3=S4A --pair S3=S4B --out results\evaluation\full_test
Same on the 124-image list:  add  --subset data\2-processed\2B-input_for_cyclegan\labels\test_balanced_124.csv
and  --out results\evaluation\balanced_124
```


## `7_util/compare_clusters.py`

```text
compare_clusters.py -- how much did the clustering move after the lung-mask fix? Read-only.

    python scripts\util\compare_clusters.py <old clustered_676.csv> <new clustered_676.csv>

K-Means cluster numbers are arbitrary, so within each class the new numbering is matched to the old one
(either as-is or swapped, whichever agrees more) before counting images that changed cluster.
```


## `7_util/inspect_labels.py`

```text
inspect_labels.py -- prints the structure of the label CSVs so the Setup 4 subset builder can be written correctly.

Reads nothing but CSVs, uses no GPU and takes seconds, so it is safe to run while Setup 1 is training.
Prints, for every label file found: its columns and dtypes, row and class counts, patient counts, a few example
rows (with the Patient ID column blanked), and the checks the Setup 4 subset builder depends on:
  * is there a Patient ID column, and is it the same type everywhere
  * do train / validation / test share any patient
  * how many No Finding images and patients each split has (enough to draw 255 / 34 / 65 from)
  * how many patients carry BOTH Pneumonia and No Finding images (mixed labels)
  * whether Image Index values are unique and carry a .png extension

Run from the project root:
    python scripts\util\inspect_labels.py
Paste the whole output into the chat. No file is written and nothing is modified.
```


## `7_util/make_chapter4_results_figures.py`

```text
make_chapter4_results_figures.py -- figures for Chapter IV, Section D (Figures IV21 to IV25), main-experiment layout.

Replaces the earlier, unsaved figure code. Reads only files that already exist in results\ (nothing is retrained):
    stylegan2-ada\metric-fid50k_full.jsonl                       FID of every saved StyleGAN2-ADA snapshot
    <setup>\final\cbam_all\seed42..46\test_predictions_*.csv     Image Index, label, prob_pneumonia
    evaluation_final\ensemble_metrics.csv                          AUC, 95% CI and the EXACT Youden threshold (12,262 images)
    evaluation_final\confusion_matrices.csv                        ensemble TP / FN / FP / TN (used as a check)
    evaluation_final_124\ensemble_metrics.csv, confusion_matrices.csv   the same on the 124-image test set
    ..\data\2-processed\2B-input_for_cyclegan\labels\test_balanced_124.csv   the 124 test images of the 676-image set
    densenet121_stylegan2-ada\final\shortcut_check.csv            training recall on real vs synthetic Pneumonia
    convnextv2_stylegan2-ada\final\shortcut_check.csv

Writes into --out (default results\chapter4\figures):
    fig_IV21_fid_by_snapshot.png
    fig_IV22_roc_curves.png
    fig_IV23_auc_ci.png
    fig_IV24_confusion_matrices.png
    fig_IV25_shortcut_check.png
and into --data_out (default results\chapter4\figure_data) one CSV per figure with the plotted numbers:
    fig_IV19_training_curves_data.csv, fig_IV20_sample_probabilities_data.csv, fig_IV21_fid_data.csv,
    fig_IV22_IV24_test_probabilities.csv, fig_IV22_roc_points.csv, fig_IV23_IV24_auc_thresholds_counts.csv,
    fig_IV25_shortcut_check_data.csv

Figures IV22 to IV24 show the MAIN experiment of each setup (MAIN_TEST below): Setups 1 and 3 and Setup 4 run A on
the full 12,262-image test set, Setup 4 run B on the 124-image test set (65-image validation, Chehade-style design).
The 124 images are a subset of the 12,262; their predictions are the same files, filtered on Image Index.

The ensemble probability is the mean of the five seeds' test probabilities, the same rule as evaluate.py.
The thresholds are read at full precision from ensemble_metrics.csv and are NEVER rounded before use; rounding
only happens in the printed labels. The script stops if its confusion counts differ from confusion_matrices.csv.

Run (project root):
    venv\Scripts\python.exe scripts\util\make_chapter4_results_figures.py
Needs: pandas, numpy, matplotlib, scikit-learn.
```


## `7_util/make_chapter4_tables.py`

```text
make_chapter4_tables.py -- builds the Chapter 4 tables and figures for the FINISHED parts of the thesis:
    A  data splits           (NIH metadata, the StyleGAN split files, the 676-image Setup 4 subset)
    B  StyleGAN2-ADA training (options, FID curve, loss / ADA curves, samples, synthetic set)
    C  K-clustering          (cluster sizes, quality indices, manual check, final domains, pool purity)

Read-only on your data. Everything is written to  results\chapter4\  :
    tables\*.csv        one CSV per table  (open in Excel; paste into the manuscript)
    figures\*.png       figures
    chapter4_notes.md   what was produced, what was skipped and why, and the assumptions

Run from the project root (second terminal if Setup 3 is training; this is CPU-only and takes seconds):
    venv\Scripts\python.exe scripts\util\make_chapter4_tables.py
    venv\Scripts\python.exe scripts\util\make_chapter4_tables.py --only C        (one part)
    venv\Scripts\python.exe scripts\util\make_chapter4_tables.py --root D:\path\to\thesis_project

Every step is independent: a missing file or an unexpected column makes THAT step print SKIPPED with the
reason and the script carries on. Nothing is typed by hand; every number comes from your files.
Needs pandas and numpy; figures need matplotlib and Pillow (pip install matplotlib pillow).
```


## `7_util/verify_synthetic.py`

```text
verify_synthetic.py  --  READ-ONLY checks before Step 2 (combine real + synthetic).
Run from the project root (thesis_project):
    python scripts-stylegan2-adaerify_synthetic.py
Changes nothing on disk.
```
