"""
Stage 3: shared paths, counts and settings of the CycleGAN branch and Setup 4.
Imported by the other scripts in this folder and by 5_training_and_classification/s4_convnextv2_cyclegan.
Run every script from the project folder. Reasons for the settings: handoff/HANDOFF.md.
"""
import os

# inputs from stages 1 and 2 (read only)
SRC_ROOT = os.path.join("2_upsampling_branch", "output")
SRC_LABELS = os.path.join(SRC_ROOT, "labels")
SRC_SPLIT_CSV = {                       # the fixed patient-level split, shared by every setup
    "train": os.path.join(SRC_LABELS, "training_labels.csv"),       # 42,436 rows, 232 pneumonia
    "validation": os.path.join(SRC_LABELS, "validation_labels.csv"),  # 5,985 rows, 31 pneumonia
    "test": os.path.join(SRC_LABELS, "testing_labels.csv"),         # 12,262 rows, 59 pneumonia
}

# features use the 1024 x 1024 images of stage 1, not the 256 x 256 images of stage 2
SRC_IMG_1024 = os.path.join("1_global_pre-processing", "output", "normalized_images")
SRC_IMG_DIR = {split: SRC_IMG_1024 for split in ("train", "validation", "test")}
SRC_IMG_256 = {                         # 256 x 256 images of stage 2 (CycleGAN input)
    "train": os.path.join(SRC_ROOT, "training"),
    "validation": os.path.join(SRC_ROOT, "validation"),
    "test": os.path.join(SRC_ROOT, "testing"),
}

# outputs of stage 3, Setup 4 normalization stats and Setup 4 results
ROOT = os.path.join("3_downsampling_branch", "output")
LABELS = os.path.join(ROOT, "labels")
SUBSET_CSV = os.path.join(LABELS, "subset_676.csv")          # the 676 images, with split and class
BALANCED_TEST_CSV = os.path.join(LABELS, "test_balanced_124.csv")   # 124-image test set: 59 + 65
SMALL_VAL_CSV = os.path.join(LABELS, "validation_small_65.csv")     # 65-image validation list (Setup 4B): 31 + 34
FEATURES_DIR = os.path.join(ROOT, "features")
CLUSTERS_DIR = os.path.join(ROOT, "clusters")
LABELGRID_DIR = os.path.join(ROOT, "label_grids")
GAN_DIR = os.path.join(ROOT, "cyclegan")                    # CycleGAN weights (cyclegan_epochXXX.pt)
CLEAN_DIR = os.path.join(ROOT, "cleaned")                    # cleaned/224 = what the classifier reads
CLEAN_224_DIR = os.path.join(CLEAN_DIR, "224")               # every image, artifact or clean, 224 x 224 PNG
CLEAN_256_DIR = os.path.join(CLEAN_DIR, "translated_256")    # G_AB output before the 256 -> 224 step (artifact images only)
CLEANED_CSV = os.path.join(CLEAN_DIR, "cleaned_manifest.csv")
CLUSTERED_CSV = os.path.join(CLUSTERS_DIR, "clustered_676.csv")  # written by cluster_kmeans.py, `domain` added by label_grid.py --score
STATS_JSON = os.path.join("4_image_transformation", "output", "train_stats_setup4.json")  # Setup 4's own mean/std
RESULTS_DIR = os.path.join("5_training_and_classification", "s4_convnextv2_cyclegan", "output")  # run_A / run_B subfolders

# counts (354 : 322 per split)
PNEUMONIA_PER_SPLIT = {"train": 232, "validation": 31, "test": 59}   # all of them; nothing is dropped
NOFINDING_PER_SPLIT = {"train": 255, "validation": 34, "test": 65}   # = round(pneumonia x 354 / 322)
TOTAL_676 = 676

# resolution and seeds
SRC_SIZE = 1024          # equalised images on disk
FEATURE_SIZE = 1024      # features are computed at full resolution (Chehade's 580,648-value vector)
GAN_SIZE = 256           # CycleGAN training and translation resolution (= Setup 3's intermediate size)
IMG = 224                # classifier input resolution
SUBSET_SEED = 7          # draws the No Finding images; deliberately not a training seed (42-46) or 100
KMEANS_SEED = 42
GAN_SEED = 42
SEEDS = [42, 43, 44, 45, 46]
BO_SEED = 100

# features: HOG (580,644 values at 1024) + 4 scalars = 580,648, as in Chehade et al.
# the values marked "assumption" are not published by Chehade et al.
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

# K-Means
N_CLUSTERS = 2
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 500
LABEL_GRID_PER_CLUSTER = 50   # images marked by hand per cluster (4 x 50 = 200)

# CycleGAN (Zhu et al. defaults)
GAN_EPOCHS = 200
GAN_EPOCH_USED = 100          # checkpoint used for the cleaned images in the paper
GAN_DECAY_START = 100         # 100 constant, then linear decay to zero
GAN_BATCH = 1
GAN_LR = 2e-4
GAN_BETAS = (0.5, 0.999)
GAN_LAMBDA_CYCLE = 10.0
GAN_LAMBDA_IDENTITY = 5.0
GAN_BLOCKS = 9                # ResNet-9 generators
GAN_NDF_LAYERS = 3            # 70 x 70 PatchGAN
GAN_POOL = 50
GAN_WORKERS = 0               # unused: images are preloaded
GAN_SAVE_EVERY = 25           # save weights every 25 epochs

# classifier (differences from Setup 3)
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
