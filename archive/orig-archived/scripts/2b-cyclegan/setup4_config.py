"""
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
"""
import os

# ---------------------------------------------------------------- source data (Setups 1 and 3; read only)
SRC_ROOT = os.path.join("data", "2-processed", "2A-input_for_stylegan2ada")
SRC_LABELS = os.path.join(SRC_ROOT, "labels")
SRC_SPLIT_CSV = {                       # the fixed patient-level split, shared by every setup
    "train": os.path.join(SRC_LABELS, "training_labels.csv"),       # 42,436 rows, 232 pneumonia
    "validation": os.path.join(SRC_LABELS, "validation_labels.csv"),  # 5,985 rows, 31 pneumonia
    "test": os.path.join(SRC_LABELS, "testing_labels.csv"),         # 12,262 rows, 59 pneumonia
}

# IMPORTANT: the images inside 2A-input_for_stylegan2ada\{training,validation,testing} were resized to
# 256 x 256 with LANCZOS by scripts\2-stylegan2-ada\1_dataspilt_for_stylegan2-ada.py, because StyleGAN2-ADA
# trained at 256. Setup 4 therefore reads the ORIGINAL 1024 x 1024 histogram-equalised PNGs written by
# scripts\1-preprocessing\1_histogram_normalization.py (cv2.equalizeHist, 8-bit grayscale, no resize),
# which is also what that split script itself consumed. All 60,683 filtered images live in one flat folder.
SRC_IMG_1024 = os.path.join("data", "2-processed", "1-normalized_images")
SRC_IMG_DIR = {split: SRC_IMG_1024 for split in ("train", "validation", "test")}
SRC_IMG_256 = {                         # Setup 3's images; used only to document the resize chains
    "train": os.path.join(SRC_ROOT, "training"),
    "validation": os.path.join(SRC_ROOT, "validation"),
    "test": os.path.join(SRC_ROOT, "testing"),
}

# ---------------------------------------------------------------- Setup 4 data and results
ROOT = os.path.join("data", "2-processed", "2B-input_for_cyclegan")
LABELS = os.path.join(ROOT, "labels")
SUBSET_CSV = os.path.join(LABELS, "subset_676.csv")          # the 676 images, with split and class
BALANCED_TEST_CSV = os.path.join(LABELS, "test_balanced_124.csv")   # 59 + 65, the Chehade-comparable list
SMALL_VAL_CSV = os.path.join(LABELS, "validation_small_65.csv")     # 31 + 34, run B's validation
FEATURES_DIR = os.path.join(ROOT, "features")
CLUSTERS_DIR = os.path.join(ROOT, "clusters")
LABELGRID_DIR = os.path.join(ROOT, "label_grids")
GAN_DIR = os.path.join(ROOT, "cyclegan")
CLEAN_DIR = os.path.join(ROOT, "cleaned")                    # cleaned/224 = what the classifier reads
CLEAN_224_DIR = os.path.join(CLEAN_DIR, "224")               # every image, artifact or clean, 224 x 224 PNG
CLEAN_256_DIR = os.path.join(CLEAN_DIR, "translated_256")    # G_AB output before the 256 -> 224 step (artifact images only)
CLEANED_CSV = os.path.join(CLEAN_DIR, "cleaned_manifest.csv")
CLUSTERED_CSV = os.path.join(CLUSTERS_DIR, "clustered_676.csv")  # written by cluster_kmeans.py, `domain` added by label_grid.py --score
STATS_JSON = os.path.join(ROOT, "train_stats.json")          # Setup 4's OWN mean/std (not Setup 3's)
RESULTS_DIR = os.path.join("results", "convnextv2_cyclegan")  # run_A / run_B subfolders

# ---------------------------------------------------------------- counts (354 : 322 per split)
PNEUMONIA_PER_SPLIT = {"train": 232, "validation": 31, "test": 59}   # all of them; nothing is dropped
NOFINDING_PER_SPLIT = {"train": 255, "validation": 34, "test": 65}   # = round(pneumonia x 354 / 322)
TOTAL_676 = 676

# ---------------------------------------------------------------- resolution and seeds
SRC_SIZE = 1024          # equalised images on disk
FEATURE_SIZE = 1024      # features are computed at full resolution (Chehade's 580,648-value vector)
GAN_SIZE = 256           # CycleGAN training and translation resolution (= Setup 3's intermediate size)
IMG = 224                # classifier input resolution
SUBSET_SEED = 7          # draws the No Finding images; deliberately not a training seed (42-46) or 100
KMEANS_SEED = 42
GAN_SEED = 42
SEEDS = [42, 43, 44, 45, 46]
BO_SEED = 100

# ---------------------------------------------------------------- features (our reading of Chehade Sec. 3.4.1)
# HOG at 1024 with scikit-image defaults gives 127 x 127 x 36 = 580,644 values; plus the 4 scalars below
# that is 580,648, the number the paper reports. The paper does not publish the parameters below, so these
# are OUR assumptions and the thesis must say so.
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
SOBEL_KSIZE = 3          # assumption: 3 x 3 Sobel
BLACK_THRESHOLD = 0.2    # assumption: pixels below this count as "black"
WHITE_THRESHOLD = 0.8    # assumption: pixels above this count as "white"
HOG_DIM = 580_644
N_SCALARS = 4            # variance of gradients, lung sharpness, black ratio, white ratio
FEATURE_DIM = HOG_DIM + N_SCALARS   # 580,648

# ---------------------------------------------------------------- K-Means
N_CLUSTERS = 2
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 500
LABEL_GRID_PER_CLUSTER = 50   # 4 clusters x 50 = 200 marks (25 first, then 25 more; the 200 marks exist)

# ---------------------------------------------------------------- CycleGAN (Zhu et al. defaults)
GAN_EPOCHS = 200
GAN_DECAY_START = 100         # 100 constant, then linear decay to zero
GAN_BATCH = 1
GAN_LR = 2e-4
GAN_BETAS = (0.5, 0.999)
GAN_LAMBDA_CYCLE = 10.0
GAN_LAMBDA_IDENTITY = 5.0
GAN_BLOCKS = 9                # ResNet-9 generators
GAN_NDF_LAYERS = 3            # 70 x 70 PatchGAN
GAN_POOL = 50
GAN_WORKERS = 0               # unused by train_cyclegan.py: images are preloaded (kept so old imports still work)
GAN_SAVE_EVERY = 25           # plus the latest: 8 numbered files + latest at about 340 MB each = about 3 GB, not 68 GB

# ---------------------------------------------------------------- classifier (differences from Setup 3)
EPOCH_CAP = 50                # 487 images per epoch instead of 84,384 samples
WARMUP_EPOCHS = 10            # same 20% warmup share as Setup 3's 3 of 15
PATIENCE = 5
BATCH = 16
WEIGHT_DECAY = 1e-4

COLUMNS = ["Image Index", "Finding Labels", "Patient ID", "Patient Age", "Patient Sex", "View Position"]


def run_dir(run):
    """run is 'A' (full 5,985-image validation) or 'B' (small 65-image validation)."""
    assert run in ("A", "B"), run
    return os.path.join(RESULTS_DIR, f"run_{run}")


def make_dirs():
    for d in (LABELS, FEATURES_DIR, CLUSTERS_DIR, LABELGRID_DIR, GAN_DIR, CLEAN_DIR, CLEAN_224_DIR, CLEAN_256_DIR):
        os.makedirs(d, exist_ok=True)
